#!/usr/bin/env python3
"""Run a hardware-backed monotonic freshness replay on the current Thor box.

The intent here is narrow: prove or disprove that restoring only the file-backed
storage directory cannot replay an already-advanced hardware anchor.  This script
does not invent a stronger claim than the retained logs support.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FUSE_BIN = ROOT / "build" / "pqc_fuse"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "tpm_monotonic_replay"


@dataclass
class ReplayResult:
    mode: str
    detail: str
    returncode: int | None
    stderr_tail: str


def sudo_password() -> str:
    password = os.environ.get("PQC_SUDO_PASSWORD")
    if not password:
        raise RuntimeError("PQC_SUDO_PASSWORD is required for the hardware-backed replay run")
    return password


def run_sudo(cmd: list[str], *, cwd: Path, stdout_path: Path, stderr_path: Path) -> subprocess.CompletedProcess:
    password = sudo_password()
    proc = subprocess.run(
        ["sudo", "-S", "-p", "", *cmd],
        cwd=cwd,
        input=password + "\n",
        text=True,
        stdout=stdout_path.open("w", encoding="utf-8"),
        stderr=stderr_path.open("w", encoding="utf-8"),
        check=False,
    )
    return proc


def start_fuse(storage_dir: Path, mount_dir: Path, log_dir: Path) -> subprocess.Popen[str]:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "pqc_fuse.stdout.txt"
    stderr_path = log_dir / "pqc_fuse.stderr.txt"
    env = os.environ.copy()
    env["PQC_MASTER_PASSWORD"] = env.get("PQC_MASTER_PASSWORD", "benchmark-password")
    env["PQC_FRESHNESS_ANCHOR_BACKEND"] = "hardware"
    env.setdefault("PQC_FRESHNESS_ANCHOR_PATH", str(storage_dir / ".anchor"))
    env.setdefault("PQC_TPM_TCTI", "device:/dev/tpmrm0")
    env.setdefault("PQC_FRESHNESS_WINDOW_N", "1")

    cmd = [
        "sudo",
        "-S",
        "-p",
        "",
        "env",
        f"PQC_MASTER_PASSWORD={env['PQC_MASTER_PASSWORD']}",
        "PQC_FRESHNESS_ANCHOR_BACKEND=hardware",
        f"PQC_FRESHNESS_ANCHOR_PATH={env['PQC_FRESHNESS_ANCHOR_PATH']}",
        f"PQC_TPM_TCTI={env['PQC_TPM_TCTI']}",
        f"PQC_FRESHNESS_WINDOW_N={env['PQC_FRESHNESS_WINDOW_N']}",
        str(FUSE_BIN),
        str(storage_dir),
        str(mount_dir),
        "-f",
    ]

    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdin=subprocess.PIPE,
        stdout=stdout_path.open("w", encoding="utf-8"),
        stderr=stderr_path.open("w", encoding="utf-8"),
        text=True,
    )
    proc.stdin.write(sudo_password() + "\n")
    proc.stdin.flush()
    proc.stdin.close()
    return proc


def stop_fuse(proc: subprocess.Popen[str], mount_dir: Path, log_dir: Path) -> None:
    stdout_path = log_dir / "unmount.stdout.txt"
    stderr_path = log_dir / "unmount.stderr.txt"
    if proc.poll() is None:
        try:
            run_sudo(["fusermount3", "-u", str(mount_dir)], cwd=ROOT, stdout_path=stdout_path, stderr_path=stderr_path)
        except Exception:
            run_sudo(["umount", str(mount_dir)], cwd=ROOT, stdout_path=stdout_path, stderr_path=stderr_path)
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def wait_for_mount(proc: subprocess.Popen[str], mount_dir: Path, timeout_s: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        if mount_is_visible(mount_dir):
            return True
        time.sleep(0.05)
    return False


def mount_is_visible(mount_dir: Path) -> bool:
    mount_path = mount_dir.resolve()
    try:
        with open("/proc/mounts", "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                fields = line.split()
                if len(fields) < 3:
                    continue
                mounted_at = Path(fields[1]).resolve()
                fs_type = fields[2]
                if mounted_at == mount_path and fs_type.startswith("fuse"):
                    return True
    except FileNotFoundError:
        pass
    return subprocess.run(["mountpoint", "-q", str(mount_dir)], check=False).returncode == 0


def write_payload(path: Path, payload: bytes) -> None:
    payload_b64 = base64.b64encode(payload).decode("ascii")
    password = sudo_password()
    subprocess.run(
        [
            "sudo",
            "-S",
            "-p",
            "",
            "env",
            f"PAYLOAD_B64={payload_b64}",
            "python3",
            "-c",
            (
                "import base64, os, pathlib; "
                f"path = pathlib.Path({str(path)!r}); "
                "data = base64.b64decode(os.environ['PAYLOAD_B64']); "
                "path.write_bytes(data)"
            ),
        ],
        input=password + "\n",
        text=True,
        check=True,
    )


def read_file(path: Path) -> bytes:
    password = sudo_password()
    proc = subprocess.run(
        [
            "sudo",
            "-S",
            "-p",
            "",
            "python3",
            "-c",
            (
                "import base64, pathlib, sys; "
                f"path = pathlib.Path({str(path)!r}); "
                "sys.stdout.write(base64.b64encode(path.read_bytes()).decode('ascii'))"
            ),
        ],
        input=password + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    return base64.b64decode(proc.stdout)


def try_read_file(path: Path) -> tuple[int, bytes | None, str]:
    password = sudo_password()
    proc = subprocess.run(
        [
            "sudo",
            "-S",
            "-p",
            "",
            "python3",
            "-c",
            (
                "import base64, pathlib, sys; "
                f"path = pathlib.Path({str(path)!r}); "
                "sys.stdout.write(base64.b64encode(path.read_bytes()).decode('ascii'))"
            ),
        ],
        input=password + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return proc.returncode, None, proc.stderr.strip()
    return 0, base64.b64decode(proc.stdout), ""


def sudo_copytree(src: Path, dst: Path) -> None:
    password = sudo_password()
    subprocess.run(
        [
            "sudo",
            "-S",
            "-p",
            "",
            "python3",
            "-c",
            (
                "import pathlib, shutil; "
                f"src = pathlib.Path({str(src)!r}); "
                f"dst = pathlib.Path({str(dst)!r}); "
                "shutil.copytree(src, dst, symlinks=True)"
            ),
        ],
        input=password + "\n",
        text=True,
        check=True,
    )


def sudo_rmtree(path: Path) -> None:
    password = sudo_password()
    subprocess.run(
        ["sudo", "-S", "-p", "", "rm", "-rf", str(path)],
        input=password + "\n",
        text=True,
        check=True,
    )


def tail_text(path: Path, limit: int = 8000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-limit:]


def run_replay(out_dir: Path) -> dict[str, object]:
    storage_dir = Path(tempfile.mkdtemp(prefix="tpm_mono_store_", dir="/tmp"))
    mount_dir = Path(tempfile.mkdtemp(prefix="tpm_mono_mnt_", dir="/tmp"))
    snapshot_dir = Path(tempfile.mkdtemp(prefix="tpm_mono_snapshot_", dir="/tmp"))
    live_log = out_dir / "live_mount"
    replay_log = out_dir / "replay_mount"
    proc: subprocess.Popen[str] | None = None
    replay_result: ReplayResult | None = None

    try:
        proc = start_fuse(storage_dir, mount_dir, live_log)
        if not wait_for_mount(proc, mount_dir):
            raise RuntimeError("initial hardware-backed mount did not come up")

        test_file = mount_dir / "payload.bin"
        write_payload(test_file, b"monotonic-v1\n")
        stop_fuse(proc, mount_dir, live_log)
        proc = None

        shutil.rmtree(snapshot_dir, ignore_errors=True)
        sudo_copytree(storage_dir, snapshot_dir)

        proc = start_fuse(storage_dir, mount_dir, live_log)
        if not wait_for_mount(proc, mount_dir):
            raise RuntimeError("second hardware-backed mount did not come up")


        write_payload(test_file, b"monotonic-v2\n" * 1024)
        stop_fuse(proc, mount_dir, live_log)
        proc = None

        sudo_rmtree(storage_dir)
        sudo_copytree(snapshot_dir, storage_dir)

        proc = start_fuse(storage_dir, mount_dir, replay_log)
        mounted = wait_for_mount(proc, mount_dir)
        stderr_tail = tail_text(replay_log / "pqc_fuse.stderr.txt")
        if mounted:
            read_rc, observed, read_stderr = try_read_file(test_file)
            if read_rc != 0:
                replay_result = ReplayResult(
                    "fail_closed",
                    f"restored snapshot mounted but payload read failed closed (rc={read_rc}): {read_stderr}",
                    proc.returncode,
                    stderr_tail,
                )
            elif observed == b"monotonic-v1\n":
                replay_result = ReplayResult("rollback_visible", "restored baseline remained readable", proc.returncode, stderr_tail)
            elif observed.startswith(b"monotonic-v2\n"):
                replay_result = ReplayResult("rollback_visible", "restored snapshot unexpectedly exposed the newer payload", proc.returncode, stderr_tail)
            else:
                replay_result = ReplayResult("rollback_visible", "restored snapshot mounted with unexpected content", proc.returncode, stderr_tail)
            stop_fuse(proc, mount_dir, replay_log)
            proc = None
        else:
            rc = proc.returncode if proc else None
            replay_result = ReplayResult("fail_closed", "restored snapshot was rejected against the advanced hardware anchor", rc, stderr_tail)
            if proc and proc.poll() is None:
                stop_fuse(proc, mount_dir, replay_log)
                proc = None

        return {
            "storage_dir": str(storage_dir),
            "mount_dir": str(mount_dir),
            "snapshot_dir": str(snapshot_dir),
            "replay_result": replay_result.__dict__ if replay_result else None,
            "replay_visible": replay_result.mode == "rollback_visible" if replay_result else None,
            "fail_closed": replay_result.mode == "fail_closed" if replay_result else None,
            "live_mount_logs": {
                "stdout": str(live_log / "pqc_fuse.stdout.txt"),
                "stderr": str(live_log / "pqc_fuse.stderr.txt"),
                "unmount_stdout": str(live_log / "unmount.stdout.txt"),
                "unmount_stderr": str(live_log / "unmount.stderr.txt"),
            },
            "replay_mount_logs": {
                "stdout": str(replay_log / "pqc_fuse.stdout.txt"),
                "stderr": str(replay_log / "pqc_fuse.stderr.txt"),
                "unmount_stdout": str(replay_log / "unmount.stdout.txt"),
                "unmount_stderr": str(replay_log / "unmount.stderr.txt"),
            },
        }
    finally:
        if proc is not None:
            try:
                stop_fuse(proc, mount_dir, replay_log)
            except Exception:
                pass
        sudo_rmtree(storage_dir)
        shutil.rmtree(mount_dir, ignore_errors=True)
        sudo_rmtree(snapshot_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Hardware-backed monotonic freshness replay")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not FUSE_BIN.exists():
        raise FileNotFoundError(f"pqc_fuse binary not found: {FUSE_BIN}")

    payload = {
        "note": (
            "Hardware-backed replay harness only; it retains a concrete fail-closed or rollback-visible "
            "outcome for a storage-snapshot restore after the TPM-backed anchor has advanced."
        ),
        "command": ["experiments/run_tpm_monotonic_replay.py"],
        "result": None,
    }
    try:
        payload["result"] = run_replay(args.out_dir)
    except Exception as exc:
        payload["error"] = str(exc)
        (args.out_dir / "tpm_monotonic_replay.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (args.out_dir / "tpm_monotonic_replay.md").write_text(
            "\n".join([
                "# TPM monotonic replay",
                "",
                "The hardware-backed monotonic replay harness did not complete.",
                "",
                f"- Error: `{exc}`",
                f"- Phase: `{payload.get('phase')}`",
                "",
                "This artifact is still useful as a narrow harness definition, but it is not evidence.",
            ]) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2))
        return 1

    (args.out_dir / "tpm_monotonic_replay.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = [
        "# TPM monotonic replay",
        "",
        "This run snapshots the file-backed storage directory, advances the hardware anchor, restores the stale snapshot, and then attempts a remount.",
        "",
        f"- Replay mode: `{payload['result']['replay_result']['mode']}`",
        f"- Detail: `{payload['result']['replay_result']['detail']}`",
        f"- Return code: `{payload['result']['replay_result']['returncode']}`",
        "",
        f"- Live mount logs: `{payload['result']['live_mount_logs']['stderr']}`",
        f"- Replay mount logs: `{payload['result']['replay_mount_logs']['stderr']}`",
        "",
        "This artifact is conservative: it only supports the exact replay classification captured in the logs.",
    ]
    (args.out_dir / "tpm_monotonic_replay.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
