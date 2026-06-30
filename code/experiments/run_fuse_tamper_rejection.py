#!/usr/bin/env python3
"""Verify that an altered persistent file-key envelope fails closed.

This is a final-binary integration test.  It mounts the FUSE filesystem, writes
and fsyncs random data, unmounts, changes one byte of the physical marker's
``user.pqc_metadata`` xattr, remounts, and verifies that opening the logical
file fails.  The script records the observed errno; it never treats an
unavailable mount or an unexpected errno as a pass.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FUSE_BIN = ROOT / "build" / "pqc_fuse"
METADATA_XATTR = "user.pqc_metadata"


def start_fuse(storage_dir: Path, mount_dir: Path, password: str) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env["PQC_MASTER_PASSWORD"] = password
    proc = subprocess.Popen(
        [str(FUSE_BIN), str(storage_dir), str(mount_dir), "-f"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if subprocess.run(["mountpoint", "-q", str(mount_dir)], check=False).returncode == 0:
            return proc
        if proc.poll() is not None:
            raise RuntimeError(f"FUSE exited before mount: rc={proc.returncode}")
        time.sleep(0.05)
    raise TimeoutError("timed out waiting for FUSE mount")


def stop_fuse(proc: subprocess.Popen[bytes] | None, mount_dir: Path) -> None:
    subprocess.run(
        ["fusermount3", "-u", str(mount_dir)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if proc is not None and proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)


def write_and_sync(path: Path, payload: bytes) -> None:
    with path.open("wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())


def run_once(password: str) -> dict[str, object]:
    storage_dir = Path(tempfile.mkdtemp(prefix="aegis_tamper_store_"))
    mount_dir = Path(tempfile.mkdtemp(prefix="aegis_tamper_mnt_"))
    proc: subprocess.Popen[bytes] | None = None
    logical_name = "tamper.bin"
    result: dict[str, object] = {
        "test": "authenticated_persistent_envelope_tamper_rejection",
        "metadata_xattr": METADATA_XATTR,
        "expected_errno": errno.EKEYREJECTED,
        "open_rejected": False,
        "observed_errno": None,
    }

    try:
        proc = start_fuse(storage_dir, mount_dir, password)
        write_and_sync(mount_dir / logical_name, os.urandom(128 * 1024))
        stop_fuse(proc, mount_dir)
        proc = None

        marker = storage_dir / logical_name
        envelope = bytearray(os.getxattr(marker, METADATA_XATTR))
        if not envelope:
            raise RuntimeError("persistent key envelope was empty")
        # Alter the last HMAC byte; changing it cannot preserve the original tag.
        envelope[-1] ^= 0x01
        os.setxattr(marker, METADATA_XATTR, bytes(envelope))
        result["envelope_bytes"] = len(envelope)

        proc = start_fuse(storage_dir, mount_dir, password)
        try:
            with (mount_dir / logical_name).open("rb") as f:
                f.read(1)
        except OSError as exc:
            result["observed_errno"] = exc.errno
            result["open_rejected"] = exc.errno == errno.EKEYREJECTED
        else:
            result["open_rejected"] = False
            result["detail"] = "tampered envelope opened successfully"
        return result
    finally:
        stop_fuse(proc, mount_dir)
        shutil.rmtree(storage_dir, ignore_errors=True)
        shutil.rmtree(mount_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--password",
        default=os.environ.get("PQC_MASTER_PASSWORD", "validation-password"),
        help="test-only mount password (default: PQC_MASTER_PASSWORD or validation-password)",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "artifacts" / "validation" / "fuse_tamper_rejection.json"),
    )
    args = parser.parse_args()

    if not FUSE_BIN.is_file():
        raise FileNotFoundError(f"missing final binary: {FUSE_BIN}")

    result = run_once(args.password)
    result["pass"] = bool(result["open_rejected"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
