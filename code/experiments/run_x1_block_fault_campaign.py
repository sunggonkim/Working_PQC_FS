#!/usr/bin/env python3
"""Run a focused X1 lower-block fault campaign for AEGIS-Q.

The campaign is intentionally narrow and local to a loopback device.  It does
not reboot the host, cut physical power, or certify drive-cache behavior.  It
does exercise a stronger failure plane than daemon SIGKILL: an ext4 lower
filesystem mounted on a device-mapper target whose table is switched from
``linear`` to ``error`` and back while the final ``pqc_fuse`` binary is used.
It also includes a post-fsync row: once the mounted write/fsync has returned
success, a later lower-block interruption must recover the latest accepted
payload or fail closed, not silently roll back to an older plaintext.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FUSE_BIN = ROOT / "build" / "pqc_fuse"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "x1_block_fault_campaign"

PAYLOAD_BYTES = 4096
ACCEPTABLE_VERDICTS = {"previous_committed", "latest_committed", "fail_closed"}

WRITE_CLIENT = r"""
import hashlib
import json
import os
import pathlib
import time
import sys

path = pathlib.Path(os.environ["TARGET_PATH"])
payload = bytes.fromhex(os.environ["PAYLOAD_HEX"])
mode = os.environ.get("OPEN_MODE", "r+b")
pause_file = os.environ.get("PAUSE_BEFORE_FSYNC_FILE")
release_file = os.environ.get("RELEASE_FSYNC_FILE")
try:
    with path.open(mode) as f:
        f.seek(0)
        f.write(payload)
        f.flush()
        if pause_file and release_file:
            pathlib.Path(pause_file).write_text("ready\n", encoding="utf-8")
            deadline = time.monotonic() + float(os.environ.get("PAUSE_TIMEOUT_S", "20"))
            while time.monotonic() < deadline:
                if pathlib.Path(release_file).exists():
                    break
                time.sleep(0.02)
            else:
                raise TimeoutError("timed out waiting for fsync release file")
        os.fsync(f.fileno())
    print(json.dumps({
        "operation": "write_fsync",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }))
except OSError as exc:
    print(json.dumps({
        "operation": "write_fsync",
        "error": "OSError",
        "errno": exc.errno,
        "message": str(exc),
    }), file=sys.stderr)
    raise
"""

READ_CLIENT = r"""
import hashlib
import json
import os
import pathlib
import sys

path = pathlib.Path(os.environ["TARGET_PATH"])
try:
    data = path.read_bytes()
    print(json.dumps({
        "operation": "read_all",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }))
except OSError as exc:
    print(json.dumps({
        "operation": "read_all",
        "error": "OSError",
        "errno": exc.errno,
        "message": str(exc),
    }), file=sys.stderr)
    raise
"""


@dataclass
class FuseProc:
    proc: subprocess.Popen[bytes]
    stdout: Any
    stderr: Any


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def payload(label: str) -> bytes:
    seed = (label + ":").encode("ascii")
    return (seed * ((PAYLOAD_BYTES // len(seed)) + 1))[:PAYLOAD_BYTES]


def run_logged(command: list[str], out_dir: Path, name: str, *,
               stdin: bytes | None = None, timeout_s: float = 60.0,
               env: dict[str, str] | None = None) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / f"{name}.stdout.txt"
    stderr_path = out_dir / f"{name}.stderr.txt"
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        try:
            proc = subprocess.run(
                command,
                cwd=ROOT,
                input=stdin,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout_s,
                check=False,
                env=full_env,
            )
            returncode: int | None = proc.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            returncode = None
            timed_out = True
    result: dict[str, Any] = {
        "command": command,
        "returncode": returncode,
        "timeout": timed_out,
        "stdout": rel(stdout_path),
        "stderr": rel(stderr_path),
    }
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace").strip()
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace").strip()
    if stdout_text:
        result["stdout_tail"] = stdout_text[-500:]
    if stderr_text:
        result["stderr_tail"] = stderr_text[-500:]
    return result


def sudo_prefix() -> list[str]:
    return ["sudo", "-S", "-p", ""]


def sudo_logged(command: list[str], out_dir: Path, name: str,
                password: str | None, timeout_s: float = 60.0) -> dict[str, Any]:
    stdin = (password + "\n").encode("utf-8") if password else None
    return run_logged(sudo_prefix() + command, out_dir, name, stdin=stdin, timeout_s=timeout_s)


def require_tools() -> dict[str, Any]:
    tools = {tool: shutil.which(tool) for tool in ("sudo", "dmsetup", "losetup", "mkfs.ext4", "mount", "umount", "e2fsck")}
    return {
        "tools": tools,
        "all_present": all(tools.values()),
        "missing": sorted(tool for tool, path in tools.items() if not path),
        "fuse_binary": rel(FUSE_BIN),
        "fuse_binary_exists": FUSE_BIN.exists(),
    }


def parse_last_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return {}
    try:
        return json.loads(text.splitlines()[-1])
    except json.JSONDecodeError:
        return {}


def run_client(script: str, env: dict[str, str], out_dir: Path, name: str,
               timeout_s: float = 20.0) -> dict[str, Any]:
    result = run_logged(["python3", "-c", script], out_dir, name, timeout_s=timeout_s, env=env)
    stdout_path = ROOT / result["stdout"]
    stderr_path = ROOT / result["stderr"]
    if stdout_path.exists():
        parsed = parse_last_json(stdout_path)
        if parsed:
            result["stdout_json"] = parsed
    if stderr_path.exists():
        parsed = parse_last_json(stderr_path)
        if parsed:
            result["stderr_json"] = parsed
    return result


def start_client(script: str, env: dict[str, str], out_dir: Path,
                 name: str) -> tuple[subprocess.Popen[bytes], Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / f"{name}.stdout.txt"
    stderr_path = out_dir / f"{name}.stderr.txt"
    full_env = os.environ.copy()
    full_env.update(env)
    stdout = stdout_path.open("wb")
    stderr = stderr_path.open("wb")
    proc = subprocess.Popen(
        ["python3", "-c", script],
        cwd=ROOT,
        env=full_env,
        stdout=stdout,
        stderr=stderr,
    )
    stdout.close()
    stderr.close()
    return proc, stdout_path, stderr_path


def finish_client(proc: subprocess.Popen[bytes], stdout_path: Path,
                  stderr_path: Path, timeout_s: float) -> dict[str, Any]:
    timed_out = False
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        proc.wait(timeout=5)
    result: dict[str, Any] = {
        "command": ["python3", "-c", "WRITE_CLIENT"],
        "returncode": proc.returncode,
        "timeout": timed_out,
        "stdout": rel(stdout_path),
        "stderr": rel(stderr_path),
    }
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace").strip()
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace").strip()
    if stdout_text:
        result["stdout_tail"] = stdout_text[-500:]
        parsed = parse_last_json(stdout_path)
        if parsed:
            result["stdout_json"] = parsed
    if stderr_text:
        result["stderr_tail"] = stderr_text[-500:]
        parsed = parse_last_json(stderr_path)
        if parsed:
            result["stderr_json"] = parsed
    return result


def wait_for_file(path: Path, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.02)
    return False


def mount_is_visible(mount_dir: Path) -> bool:
    proc = subprocess.run(["mountpoint", "-q", str(mount_dir)], check=False)
    return proc.returncode == 0


def start_fuse(storage_dir: Path, mount_dir: Path, password: str, out_dir: Path,
               label: str) -> FuseProc:
    log_dir = out_dir / "mount_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout = (log_dir / f"{label}.stdout.txt").open("wb")
    stderr = (log_dir / f"{label}.stderr.txt").open("wb")
    env = os.environ.copy()
    env.update({
        "PQC_MASTER_PASSWORD": password,
        "PQC_FRESHNESS_ANCHOR_BACKEND": "file",
        "PQC_FRESHNESS_ANCHOR_PATH": str(storage_dir / ".anchor"),
        "PQC_FRESHNESS_WINDOW_N": "1",
        "PQC_KEY_ROTATION_INTERVAL_S": "0",
        "PQC_ADMISSION_TRACE_PATH": str(out_dir / f"{label}.admission_trace.jsonl"),
    })
    proc = subprocess.Popen(
        [str(FUSE_BIN), str(storage_dir), str(mount_dir), "-f"],
        cwd=ROOT,
        env=env,
        stdout=stdout,
        stderr=stderr,
    )
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if mount_is_visible(mount_dir):
            return FuseProc(proc=proc, stdout=stdout, stderr=stderr)
        if proc.poll() is not None:
            stdout.close()
            stderr.close()
            raise RuntimeError(f"FUSE exited before mount for {label}: rc={proc.returncode}")
        time.sleep(0.05)
    stdout.close()
    stderr.close()
    raise TimeoutError(f"timed out waiting for FUSE mount for {label}")


def stop_fuse(handle: FuseProc | None, mount_dir: Path, out_dir: Path, label: str) -> dict[str, Any]:
    result: dict[str, Any] = {"label": label}
    if mount_is_visible(mount_dir):
        for idx, command in enumerate((["fusermount3", "-uz", str(mount_dir)], ["fusermount", "-uz", str(mount_dir)])):
            if shutil.which(command[0]):
                attempt = run_logged(command, out_dir, f"{label}_unmount_{idx}", timeout_s=10)
                result.setdefault("unmount_attempts", []).append(attempt)
                if attempt["returncode"] == 0:
                    break
    if handle is not None:
        if handle.proc.poll() is None:
            handle.proc.send_signal(signal.SIGINT)
            try:
                handle.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                handle.proc.kill()
                handle.proc.wait(timeout=3)
        result["daemon_returncode"] = handle.proc.returncode
        handle.stdout.close()
        handle.stderr.close()
    return result


def dm_table(sectors: int, target: str, loop_dev: str | None = None) -> str:
    if target == "linear":
        if not loop_dev:
            raise ValueError("linear target requires loop device")
        return f"0 {sectors} linear {loop_dev} 0"
    if target == "error":
        return f"0 {sectors} error"
    raise ValueError(target)


def switch_dm(name: str, sectors: int, target: str, loop_dev: str | None,
              out_dir: Path, label: str, password: str | None) -> list[dict[str, Any]]:
    table = dm_table(sectors, target, loop_dev)
    return [
        sudo_logged(["dmsetup", "suspend", name], out_dir, f"{label}_suspend", password, timeout_s=20),
        sudo_logged(["dmsetup", "load", name, "--table", table], out_dir, f"{label}_load", password, timeout_s=20),
        sudo_logged(["dmsetup", "resume", name], out_dir, f"{label}_resume", password, timeout_s=20),
    ]


def classify_recovery(storage_dir: Path, mount_dir: Path, password: str,
                      out_dir: Path, previous: bytes, latest: bytes) -> dict[str, Any]:
    handle: FuseProc | None = None
    try:
        handle = start_fuse(storage_dir, mount_dir, password, out_dir, "recovery")
    except Exception as exc:
        return {"verdict": "fail_closed", "acceptable": True, "detail": f"remount rejected state: {exc}"}
    try:
        result = run_client(READ_CLIENT, {"TARGET_PATH": str(mount_dir / "probe.bin")}, out_dir, "recovery_read")
        if result.get("returncode") != 0:
            return {"verdict": "fail_closed", "acceptable": True, "detail": "recovery read failed", "read": result}
        digest = (result.get("stdout_json") or {}).get("sha256")
        size = (result.get("stdout_json") or {}).get("bytes")
        if digest == sha256(latest) and size == len(latest):
            verdict = "latest_committed"
            detail = "recovered latest payload"
        elif digest == sha256(previous) and size == len(previous):
            verdict = "previous_committed"
            detail = "recovered previous committed payload"
        else:
            verdict = "silent_corruption"
            detail = f"recovered unknown digest bytes={size} sha256={digest}"
        return {
            "verdict": verdict,
            "acceptable": verdict in ACCEPTABLE_VERDICTS,
            "detail": detail,
            "read": result,
        }
    finally:
        stop_fuse(handle, mount_dir, out_dir, "recovery")


def setup_loop_dm(work_dir: Path, out_dir: Path, password: str | None,
                  size_mb: int, dm_name: str) -> dict[str, Any]:
    image = work_dir / "lower.img"
    image_result = run_logged(["truncate", "-s", f"{size_mb}M", str(image)], out_dir, "truncate_image")
    loop_result = sudo_logged(["losetup", "--find", "--show", str(image)], out_dir, "losetup", password)
    loop_dev = (loop_result.get("stdout_tail") or "").strip().splitlines()[-1] if loop_result.get("stdout_tail") else ""
    sectors = (size_mb * 1024 * 1024) // 512
    create = sudo_logged(["dmsetup", "create", dm_name, "--table", dm_table(sectors, "linear", loop_dev)], out_dir, "dmsetup_create", password)
    mapper = f"/dev/mapper/{dm_name}"
    mkfs = sudo_logged(["mkfs.ext4", "-F", "-q", mapper], out_dir, "mkfs_ext4", password, timeout_s=60)
    lower_mount = work_dir / "lower_mount"
    lower_mount.mkdir()
    mount = sudo_logged(["mount", "-o", "sync", mapper, str(lower_mount)], out_dir, "mount_lower", password)
    chown = sudo_logged(["chown", f"{os.getuid()}:{os.getgid()}", str(lower_mount)], out_dir, "chown_lower", password)
    return {
        "image": str(image),
        "loop_dev": loop_dev,
        "dm_name": dm_name,
        "mapper": mapper,
        "sectors": sectors,
        "lower_mount": str(lower_mount),
        "steps": {
            "truncate": image_result,
            "losetup": loop_result,
            "dmsetup_create": create,
            "mkfs_ext4": mkfs,
            "mount": mount,
            "chown": chown,
        },
    }


def cleanup_loop_dm(state: dict[str, Any], work_dir: Path, out_dir: Path, password: str | None) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    lower_mount = state.get("lower_mount")
    if lower_mount:
        steps.append(sudo_logged(["umount", "-l", str(lower_mount)], out_dir, "cleanup_umount_lower", password, timeout_s=20))
    dm_name = state.get("dm_name")
    if dm_name:
        steps.append(sudo_logged(["dmsetup", "remove", "-f", str(dm_name)], out_dir, "cleanup_dm_remove", password, timeout_s=20))
    loop_dev = state.get("loop_dev")
    if loop_dev:
        steps.append(sudo_logged(["losetup", "-d", str(loop_dev)], out_dir, "cleanup_losetup_detach", password, timeout_s=20))
    shutil.rmtree(work_dir, ignore_errors=True)
    return steps


def remount_lower_after_fault(state: dict[str, Any], out_dir: Path, password: str | None) -> list[dict[str, Any]]:
    lower_mount = state["lower_mount"]
    mapper = state["mapper"]
    dm_name = state["dm_name"]
    table = dm_table(int(state["sectors"]), "linear", str(state["loop_dev"]))
    return [
        sudo_logged(["umount", "-l", lower_mount], out_dir, "restore_umount_lower", password, timeout_s=20),
        sudo_logged(["dmsetup", "remove", "-f", dm_name], out_dir, "restore_dm_remove", password, timeout_s=20),
        sudo_logged(["dmsetup", "create", dm_name, "--table", table], out_dir, "restore_dm_create_linear", password, timeout_s=20),
        sudo_logged(["e2fsck", "-fy", mapper], out_dir, "restore_e2fsck", password, timeout_s=60),
        sudo_logged(["mount", "-o", "sync", mapper, lower_mount], out_dir, "restore_mount_lower", password, timeout_s=20),
        sudo_logged(["chown", f"{os.getuid()}:{os.getgid()}", lower_mount], out_dir, "restore_chown_lower", password, timeout_s=20),
    ]


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    readiness = require_tools()
    password = os.environ.get(args.sudo_password_env)
    report: dict[str, Any] = {
        "artifact": "x1_block_fault_campaign",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": [
            "Loopback ext4 lower filesystem on a device-mapper target.",
            "Fault injection switches the lower block device from linear to error and back.",
            "This is block-device interruption evidence, not physical power-loss, kernel-crash, or drive-cache certification.",
        ],
        "readiness": readiness,
        "sudo_password_env": args.sudo_password_env,
        "rows": [],
        "overall_pass": False,
    }
    if not readiness["all_present"] or not readiness["fuse_binary_exists"]:
        report["blocked"] = True
        report["blocker"] = "missing required tool or FUSE binary"
        return report
    if not password and os.geteuid() != 0:
        report["blocked"] = True
        report["blocker"] = f"missing sudo password in {args.sudo_password_env}"
        return report

    work_dir = Path(tempfile.mkdtemp(prefix="aegisq_x1_block_fault_", dir="/tmp"))
    dm_name = f"aegisq_x1_{os.getpid()}"
    state: dict[str, Any] = {}
    handle: FuseProc | None = None
    mount_dir = work_dir / "fuse_mount"
    previous = payload("previous")
    latest = payload("latest")
    try:
        state = setup_loop_dm(work_dir, out_dir, password, args.size_mb, dm_name)
        lower_mount = Path(state["lower_mount"])
        storage_dir = lower_mount / "store"
        storage_dir.mkdir()
        mount_dir.mkdir()

        handle = start_fuse(storage_dir, mount_dir, args.password, out_dir, "baseline")
        baseline_write = run_client(
            WRITE_CLIENT,
            {
                "TARGET_PATH": str(mount_dir / "probe.bin"),
                "PAYLOAD_HEX": previous.hex(),
                "OPEN_MODE": "wb",
            },
            out_dir,
            "baseline_write",
        )
        baseline_stop = stop_fuse(handle, mount_dir, out_dir, "baseline")
        handle = None

        handle = start_fuse(storage_dir, mount_dir, args.password, out_dir, "fault")
        switch_to_error = switch_dm(state["dm_name"], state["sectors"], "error", None, out_dir, "fault_error", password)
        fault_write = run_client(
            WRITE_CLIENT,
            {
                "TARGET_PATH": str(mount_dir / "probe.bin"),
                "PAYLOAD_HEX": latest.hex(),
                "OPEN_MODE": "r+b",
            },
            out_dir,
            "fault_write",
            timeout_s=20,
        )
        fault_stop = stop_fuse(handle, mount_dir, out_dir, "fault")
        handle = None
        restore_steps = remount_lower_after_fault(state, out_dir, password)
        recovery = classify_recovery(storage_dir, mount_dir, args.password, out_dir, previous, latest)
        committed_after_first = latest if recovery.get("verdict") == "latest_committed" else previous

        row = {
            "case": "lower_block_error_during_latest_write",
            "fault_model": "device-mapper error target while final pqc_fuse writes and fsyncs a mounted file",
            "baseline_write_returncode": baseline_write.get("returncode"),
            "fault_write_returncode": fault_write.get("returncode"),
            "fault_write_failed": fault_write.get("returncode") != 0,
            "verdict": recovery.get("verdict"),
            "acceptable": recovery.get("acceptable") is True and recovery.get("verdict") in ACCEPTABLE_VERDICTS,
            "baseline_write": baseline_write,
            "baseline_stop": baseline_stop,
            "switch_to_error": switch_to_error,
            "fault_write": fault_write,
            "fault_stop": fault_stop,
            "restore_steps": restore_steps,
            "recovery": recovery,
        }
        latest_prefsync = payload("latest_prefsync")
        handle = start_fuse(storage_dir, mount_dir, args.password, out_dir, "fault_prefsync")
        ready_file = out_dir / "prefsync_ready"
        release_file = out_dir / "prefsync_release"
        prefsync_proc, prefsync_stdout, prefsync_stderr = start_client(
            WRITE_CLIENT,
            {
                "TARGET_PATH": str(mount_dir / "probe.bin"),
                "PAYLOAD_HEX": latest_prefsync.hex(),
                "OPEN_MODE": "r+b",
                "PAUSE_BEFORE_FSYNC_FILE": str(ready_file),
                "RELEASE_FSYNC_FILE": str(release_file),
                "PAUSE_TIMEOUT_S": "30",
            },
            out_dir,
            "prefsync_fault_write",
        )
        prefsync_ready = wait_for_file(ready_file, 30.0)
        prefsync_switch_to_error = switch_dm(
            state["dm_name"], state["sectors"], "error", None,
            out_dir, "prefsync_fault_error", password)
        release_file.write_text("go\n", encoding="utf-8")
        prefsync_write = finish_client(
            prefsync_proc, prefsync_stdout, prefsync_stderr, timeout_s=30.0)
        prefsync_stop = stop_fuse(handle, mount_dir, out_dir, "fault_prefsync")
        handle = None
        prefsync_restore_steps = remount_lower_after_fault(state, out_dir, password)
        prefsync_recovery = classify_recovery(
            storage_dir, mount_dir, args.password, out_dir,
            committed_after_first, latest_prefsync)
        committed_after_prefsync = (
            latest_prefsync
            if prefsync_recovery.get("verdict") == "latest_committed"
            else committed_after_first
        )

        row_prefsync = {
            "case": "lower_block_error_after_write_before_fsync",
            "fault_model": "client writes and flushes a mounted file, campaign switches device-mapper to error before fsync returns",
            "prefsync_ready": prefsync_ready,
            "fault_write_returncode": prefsync_write.get("returncode"),
            "fault_write_failed": prefsync_write.get("returncode") != 0,
            "verdict": prefsync_recovery.get("verdict"),
            "acceptable": (
                prefsync_ready is True
                and all(step.get("returncode") == 0 for step in prefsync_switch_to_error)
                and prefsync_recovery.get("acceptable") is True
                and prefsync_recovery.get("verdict") in ACCEPTABLE_VERDICTS
            ),
            "switch_to_error": prefsync_switch_to_error,
            "fault_write": prefsync_write,
            "fault_stop": prefsync_stop,
            "restore_steps": prefsync_restore_steps,
            "recovery": prefsync_recovery,
        }

        latest_postfsync = payload("latest_postfsync")
        handle = start_fuse(storage_dir, mount_dir, args.password, out_dir, "fault_postfsync")
        postfsync_write = run_client(
            WRITE_CLIENT,
            {
                "TARGET_PATH": str(mount_dir / "probe.bin"),
                "PAYLOAD_HEX": latest_postfsync.hex(),
                "OPEN_MODE": "r+b",
            },
            out_dir,
            "postfsync_write",
            timeout_s=20,
        )
        postfsync_switch_to_error = switch_dm(
            state["dm_name"], state["sectors"], "error", None,
            out_dir, "postfsync_fault_error", password)
        postfsync_stop = stop_fuse(handle, mount_dir, out_dir, "fault_postfsync")
        handle = None
        postfsync_restore_steps = remount_lower_after_fault(state, out_dir, password)
        postfsync_recovery = classify_recovery(
            storage_dir, mount_dir, args.password, out_dir,
            committed_after_prefsync, latest_postfsync)

        row_postfsync = {
            "case": "lower_block_error_after_successful_fsync",
            "fault_model": "mounted write and fsync return success, then device-mapper switches to error before recovery",
            "fault_write_returncode": postfsync_write.get("returncode"),
            "fault_write_failed": postfsync_write.get("returncode") != 0,
            "verdict": postfsync_recovery.get("verdict"),
            "acceptable": (
                postfsync_write.get("returncode") == 0
                and all(step.get("returncode") == 0 for step in postfsync_switch_to_error)
                and postfsync_recovery.get("acceptable") is True
                and postfsync_recovery.get("verdict") in {"latest_committed", "fail_closed"}
            ),
            "write": postfsync_write,
            "switch_to_error": postfsync_switch_to_error,
            "fault_stop": postfsync_stop,
            "restore_steps": postfsync_restore_steps,
            "recovery": postfsync_recovery,
        }
        report["setup"] = state
        report["rows"] = [row, row_prefsync, row_postfsync]
        report["failure_taxonomy"] = [
            {
                "verdict": "previous_committed",
                "meaning": "latest write was not accepted across the lower-block interruption; previous committed state remains readable",
                "claim_status": "closed by this campaign if observed",
            },
            {
                "verdict": "latest_committed",
                "meaning": "latest write reached an accepted durable state before the interruption was observed",
                "claim_status": "acceptable but not a physical power-loss proof",
            },
            {
                "verdict": "fail_closed",
                "meaning": "remount or read rejects the state instead of exposing unauthenticated data",
                "claim_status": "acceptable fail-closed outcome",
            },
            {
                "verdict": "silent_corruption",
                "meaning": "recovered bytes match no oracle state",
                "claim_status": "unacceptable",
            },
        ]
        restore_mount_ok = any(
            len(step.get("command", [])) >= 3
            and step.get("command", [])[-2:] == [state["mapper"], state["lower_mount"]]
            and step.get("returncode") == 0
            for step in restore_steps
        )
        prefsync_restore_mount_ok = any(
            len(step.get("command", [])) >= 3
            and step.get("command", [])[-2:] == [state["mapper"], state["lower_mount"]]
            and step.get("returncode") == 0
            for step in prefsync_restore_steps
        )
        postfsync_restore_mount_ok = any(
            len(step.get("command", [])) >= 3
            and step.get("command", [])[-2:] == [state["mapper"], state["lower_mount"]]
            and step.get("returncode") == 0
            for step in postfsync_restore_steps
        )
        report["overall_pass"] = (
            baseline_write.get("returncode") == 0
            and all(step.get("returncode") == 0 for step in switch_to_error)
            and restore_mount_ok
            and row["acceptable"] is True
            and prefsync_restore_mount_ok
            and row_prefsync["acceptable"] is True
            and postfsync_restore_mount_ok
            and row_postfsync["acceptable"] is True
        )
        report["paper_claim_boundary"] = (
            "This campaign supports lower-block interruption recovery on a loopback ext4/device-mapper stack. "
            "It includes a post-fsync accepted-state row, but still does not certify physical power loss, kernel crash, drive write-cache loss, arbitrary workloads, or full POSIX crash semantics."
        )
    finally:
        stop_fuse(handle, mount_dir, out_dir, "cleanup_fuse")
        if state:
            report["cleanup"] = cleanup_loop_dm(state, work_dir, out_dir, password)
        else:
            shutil.rmtree(work_dir, ignore_errors=True)
    return report


def write_outputs(out_dir: Path, report: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "x1_block_fault_campaign.json"
    md_path = out_dir / "x1_block_fault_campaign.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# X1 Block-Fault Campaign",
        "",
        f"- Overall pass: `{str(report.get('overall_pass')).lower()}`",
        f"- Blocked: `{str(report.get('blocked', False)).lower()}`",
        f"- Scope: {'; '.join(report.get('scope', []))}",
        "",
        "## Rows",
        "",
    ]
    for row in report.get("rows", []):
        lines.append(
            f"- `{row.get('case')}`: verdict `{row.get('verdict')}`, "
            f"acceptable `{row.get('acceptable')}`, fault write rc `{row.get('fault_write_returncode')}`"
        )
    if report.get("blocker"):
        lines.extend(["", "## Blocker", "", str(report["blocker"])])
    lines.extend(["", "## Claim Boundary", "", str(report.get("paper_claim_boundary", "No claim upgrade."))])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--size-mb", type=int, default=128)
    parser.add_argument("--password", default=os.environ.get("PQC_MASTER_PASSWORD", "x1-block-fault-password"))
    parser.add_argument("--sudo-password-env", default="SUDO_PASSWORD")
    args = parser.parse_args()

    report = run_campaign(args)
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    write_outputs(out_dir, report)
    print(json.dumps({
        "overall_pass": report.get("overall_pass"),
        "blocked": report.get("blocked", False),
        "blocker": report.get("blocker"),
        "rows": [
            {
                "case": row.get("case"),
                "verdict": row.get("verdict"),
                "acceptable": row.get("acceptable"),
                "fault_write_returncode": row.get("fault_write_returncode"),
            }
            for row in report.get("rows", [])
        ],
        "json": rel(out_dir / "x1_block_fault_campaign.json"),
    }, indent=2, sort_keys=True))
    return 0 if report.get("overall_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
