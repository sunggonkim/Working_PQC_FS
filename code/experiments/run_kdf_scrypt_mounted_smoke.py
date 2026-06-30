#!/usr/bin/env python3
"""Mounted smoke for the D1-S1 scrypt KDF metadata path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import struct
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FUSE_BIN = ROOT / "build" / "pqc_fuse"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "kdf_crypto_plane"

KDF_MAGIC = 0x5051434B44463131
KDF_VERSION = 1
KDF_ALG_SCRYPT = 2
KDF_STRUCT = struct.Struct("<QIIIIQIIQ32s32s")


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def mount_is_visible(mount_dir: Path) -> bool:
    mount_path = mount_dir.resolve()
    try:
        with open("/proc/mounts", "r", encoding="utf-8",
                  errors="replace") as proc_mounts:
            for line in proc_mounts:
                fields = line.split()
                if len(fields) >= 3 and fields[2].startswith("fuse"):
                    try:
                        if Path(fields[1]).resolve() == mount_path:
                            return True
                    except OSError:
                        pass
    except FileNotFoundError:
        pass
    return subprocess.run(
        ["mountpoint", "-q", str(mount_dir)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def start_fuse(storage_dir: Path, mount_dir: Path, out_dir: Path,
               label: str, password: str) -> tuple[subprocess.Popen[bytes], Path, Path]:
    stdout_path = out_dir / f"{label}.stdout.txt"
    stderr_path = out_dir / f"{label}.stderr.txt"
    env = os.environ.copy()
    env.update({
        "PQC_MASTER_PASSWORD": password,
        "PQC_ADMISSION_TRACE_PATH": str(out_dir / f"{label}.admission.jsonl"),
    })
    proc = subprocess.Popen(
        [str(FUSE_BIN), str(storage_dir), str(mount_dir), "-f"],
        cwd=ROOT,
        env=env,
        stdout=stdout_path.open("wb"),
        stderr=stderr_path.open("wb"),
    )
    return proc, stdout_path, stderr_path


def wait_for_mount(proc: subprocess.Popen[bytes], mount_dir: Path,
                   timeout_s: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if mount_is_visible(mount_dir):
            return True
        if proc.poll() is not None:
            return False
        time.sleep(0.05)
    return False


def stop_fuse(proc: subprocess.Popen[bytes] | None, mount_dir: Path) -> int | None:
    if mount_is_visible(mount_dir):
        for tool in ("fusermount3", "fusermount"):
            exe = shutil.which(tool)
            if exe and subprocess.run(
                [exe, "-uz", str(mount_dir)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode == 0:
                break
    if proc is not None and proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    return proc.returncode if proc is not None else None


def parse_kdf_metadata(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if len(data) != KDF_STRUCT.size:
        return {"exists": True, "size": len(data), "valid": False}
    (magic, version, algorithm, salt_len, reserved, scrypt_n, scrypt_r,
     scrypt_p, scrypt_maxmem, salt, digest) = KDF_STRUCT.unpack(data)
    computed = hashlib.sha256(data[:KDF_STRUCT.size - 32]).digest()
    return {
        "exists": True,
        "size": len(data),
        "valid": (
            magic == KDF_MAGIC and version == KDF_VERSION and
            algorithm == KDF_ALG_SCRYPT and salt_len == 32 and
            digest == computed
        ),
        "magic_hex": f"0x{magic:016x}",
        "version": version,
        "algorithm": algorithm,
        "salt_len": salt_len,
        "reserved": reserved,
        "scrypt_n": scrypt_n,
        "scrypt_r": scrypt_r,
        "scrypt_p": scrypt_p,
        "scrypt_maxmem": scrypt_maxmem,
        "salt_sha256": hashlib.sha256(salt[:salt_len]).hexdigest(),
        "salt_nonzero": any(salt[:salt_len]),
        "digest_valid": digest == computed,
    }


def source_checks() -> dict[str, bool]:
    keyring = (ROOT / "code" / "crypto" / "pqc_keyring.c").read_text(
        encoding="utf-8", errors="replace"
    )
    fmt = (ROOT / "code" / "common" / "pqc_format.h").read_text(
        encoding="utf-8", errors="replace"
    )
    posix = (ROOT / "code" / "fs" / "pqc_posix.c").read_text(
        encoding="utf-8", errors="replace"
    )
    namespace = (ROOT / "code" / "fs" / "pqc_namespace.c").read_text(
        encoding="utf-8", errors="replace"
    )
    return {
        "scrypt_derivation_present": "EVP_PBE_scrypt" in keyring,
        "kdf_metadata_format_present": "pqc_kdf_metadata_t" in fmt,
        "kdf_filename_constant_present": "PQC_KDF_METADATA_FILENAME" in fmt,
        "scrypt_algorithm_constant_present": "PQC_KDF_ALG_SCRYPT" in fmt,
        "legacy_pbkdf2_compatibility_present":
            "derive_pbkdf2_legacy" in keyring and
            "PBKDF2-HMAC-SHA256-legacy" in keyring,
        "metadata_hidden_from_path_ops":
            "PQC_KDF_METADATA_FILENAME" in posix,
        "metadata_hidden_from_readdir":
            "PQC_KDF_METADATA_FILENAME" in namespace,
        "runtime_reports_kdf_name": "pqc_keyring_kdf_name" in keyring,
    }


def write_read_fsync(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        written = os.write(fd, data)
        if written != len(data):
            raise OSError(f"short write {written}/{len(data)}")
        os.fsync(fd)
    finally:
        os.close(fd)


def run_smoke(out_dir: Path) -> dict[str, Any]:
    if not FUSE_BIN.exists():
        raise SystemExit(f"missing build artifact: {relpath(FUSE_BIN)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="aegisq_kdf_scrypt_"))
    storage_dir = work_dir / "store"
    mount_dir = work_dir / "mnt"
    storage_dir.mkdir()
    mount_dir.mkdir()
    password = "kdf-smoke-correct-password"
    wrong_password = "kdf-smoke-wrong-password"
    payload = b"aegis-q scrypt kdf mounted smoke\n" * 64
    processes: list[subprocess.Popen[bytes]] = []
    runs: list[dict[str, Any]] = []

    try:
        proc, stdout, stderr = start_fuse(storage_dir, mount_dir, out_dir,
                                          "initial", password)
        processes.append(proc)
        initial_mounted = wait_for_mount(proc, mount_dir)
        if initial_mounted:
            write_read_fsync(mount_dir / "secret.bin", payload)
            initial_listing = sorted(os.listdir(mount_dir))
        else:
            initial_listing = []
        initial_rc = stop_fuse(proc, mount_dir)
        runs.append({
            "label": "initial",
            "mounted": initial_mounted,
            "returncode": initial_rc,
            "stdout": relpath(stdout),
            "stderr": relpath(stderr),
        })

        metadata_path = storage_dir / ".pqc_kdf"
        metadata = parse_kdf_metadata(metadata_path) if metadata_path.exists() else {
            "exists": False,
            "valid": False,
        }

        proc, stdout, stderr = start_fuse(storage_dir, mount_dir, out_dir,
                                          "remount_correct", password)
        processes.append(proc)
        remount_mounted = wait_for_mount(proc, mount_dir)
        remount_data = b""
        remount_error = ""
        if remount_mounted:
            try:
                remount_data = (mount_dir / "secret.bin").read_bytes()
            except OSError as exc:
                remount_error = f"errno={exc.errno}"
            remount_listing = sorted(os.listdir(mount_dir))
        else:
            remount_listing = []
        remount_rc = stop_fuse(proc, mount_dir)
        runs.append({
            "label": "remount_correct",
            "mounted": remount_mounted,
            "returncode": remount_rc,
            "stdout": relpath(stdout),
            "stderr": relpath(stderr),
            "read_error": remount_error,
        })

        proc, stdout, stderr = start_fuse(storage_dir, mount_dir, out_dir,
                                          "remount_wrong_password",
                                          wrong_password)
        processes.append(proc)
        wrong_mounted = wait_for_mount(proc, mount_dir)
        wrong_read_failed = False
        wrong_read_error = ""
        if wrong_mounted:
            try:
                _ = (mount_dir / "secret.bin").read_bytes()
            except OSError as exc:
                wrong_read_failed = True
                wrong_read_error = f"errno={exc.errno}"
        else:
            wrong_read_failed = True
            wrong_read_error = "mount rejected"
        wrong_rc = stop_fuse(proc, mount_dir)
        runs.append({
            "label": "remount_wrong_password",
            "mounted": wrong_mounted,
            "returncode": wrong_rc,
            "stdout": relpath(stdout),
            "stderr": relpath(stderr),
            "read_failed": wrong_read_failed,
            "read_error": wrong_read_error,
        })

        checks = {
            "initial_mount_started": initial_mounted,
            "kdf_metadata_exists": metadata.get("exists") is True,
            "kdf_metadata_valid": metadata.get("valid") is True,
            "kdf_metadata_uses_scrypt":
                metadata.get("algorithm") == KDF_ALG_SCRYPT,
            "kdf_salt_nonzero": metadata.get("salt_nonzero") is True,
            "kdf_hidden_initial": ".pqc_kdf" not in initial_listing,
            "kdf_hidden_remount": ".pqc_kdf" not in remount_listing,
            "correct_remount_read_matches": remount_data == payload,
            "wrong_password_fails_closed": wrong_read_failed,
            "source_checks_pass": all(source_checks().values()),
        }
        return {
            "schema_version": 1,
            "generated_by": "code/experiments/run_kdf_scrypt_mounted_smoke.py",
            "generated_utc": now_utc(),
            "scope": (
                "D1-S1 mounted smoke for production scrypt KDF metadata.  It "
                "does not close the full D1 paper/security gate."
            ),
            "storage_dir": str(storage_dir),
            "mount_dir": str(mount_dir),
            "kdf_metadata_path": str(metadata_path),
            "kdf_metadata": metadata,
            "runs": runs,
            "initial_listing": initial_listing,
            "remount_listing": remount_listing,
            "source_checks": source_checks(),
            "checks": checks,
            "overall_pass": all(checks.values()),
        }
    finally:
        for proc in processes:
            if proc.poll() is None:
                stop_fuse(proc, mount_dir)
        shutil.rmtree(work_dir, ignore_errors=True)


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
    json_path = out_dir / "kdf_scrypt_mounted_smoke.json"
    md_path = out_dir / "kdf_scrypt_mounted_smoke.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    lines = [
        "# KDF Scrypt Mounted Smoke",
        "",
        f"- Overall pass: `{str(payload['overall_pass']).lower()}`",
        f"- KDF metadata valid: `{str(payload['checks']['kdf_metadata_valid']).lower()}`",
        f"- Correct remount read matches: `{str(payload['checks']['correct_remount_read_matches']).lower()}`",
        f"- Wrong password fails closed: `{str(payload['checks']['wrong_password_fails_closed']).lower()}`",
        "",
        "## Checks",
        "",
    ]
    for key, value in payload["checks"].items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")
    lines.extend(["", "## Boundary", "", payload["scope"], ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    payload = run_smoke(args.out_dir)
    write_outputs(args.out_dir, payload)
    print(json.dumps({
        "overall_pass": payload["overall_pass"],
        "json": relpath(args.out_dir / "kdf_scrypt_mounted_smoke.json"),
        "failed_checks": [
            key for key, value in payload["checks"].items() if not value
        ],
    }, indent=2, sort_keys=True))
    return 0 if payload["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
