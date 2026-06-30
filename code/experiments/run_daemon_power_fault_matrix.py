#!/usr/bin/env python3
"""Daemon-crash and power-loss-equivalent fault matrix for pqc_fuse.

This campaign exercises the final ``build/pqc_fuse`` binary with opt-in
cutpoint hooks compiled into the daemon.  Each daemon row kills the FUSE process
at a named internal persistence boundary, remounts the same backing store, and
classifies the result with the paper's recovery-oracle vocabulary.

The scope is intentionally narrow: these are daemon SIGKILL cutpoints and
file-state recovery oracles, not a physical power-loss or kernel-crash
certification.
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
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FUSE_BIN = ROOT / "build" / "pqc_fuse"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "daemon_power_fault_campaign"

PAYLOAD_BYTES = 4096
ALLOWED_VERDICTS = {
    "previous_committed",
    "latest_committed",
    "fail_closed",
    "silent_corruption",
    "unexpected_liveness_failure",
}
ACCEPTABLE_VERDICTS = {
    "previous_committed",
    "latest_committed",
    "fail_closed",
}

DAEMON_CUTPOINTS = [
    {
        "case": "data_write",
        "required_cut_point": "data write",
        "hook": "data_write_after_pwrite",
        "operation": "write_latest",
    },
    {
        "case": "journal_append",
        "required_cut_point": "journal append",
        "hook": "journal_append_after",
        "operation": "write_latest",
    },
    {
        "case": "journal_fsync",
        "required_cut_point": "journal fsync barrier",
        "hook": "journal_fsync_after",
        "operation": "write_latest",
    },
    {
        "case": "xattr_checkpoint_update",
        "required_cut_point": "xattr/checkpoint update",
        "hook": "logical_size_xattr_after",
        "operation": "write_latest",
    },
    {
        "case": "checkpoint_write",
        "required_cut_point": "checkpoint write",
        "hook": "checkpoint_xattr_after",
        "operation": "write_latest",
    },
    {
        "case": "anchor_update",
        "required_cut_point": "anchor update",
        "hook": "anchor_update_before",
        "operation": "write_latest",
    },
    {
        "case": "fsync",
        "required_cut_point": "fsync",
        "hook": "fsync_before_return",
        "operation": "write_latest",
    },
    {
        "case": "remount",
        "required_cut_point": "remount",
        "hook": "remount_after_checkpoint_load",
        "operation": "read_precommitted_latest",
    },
    {
        "case": "application_read",
        "required_cut_point": "application read",
        "hook": "read_after_auth",
        "operation": "read_precommitted_latest",
    },
]

REQUIRED_DAEMON_CUTPOINTS = {
    "data write",
    "journal append",
    "checkpoint write",
    "anchor update",
    "xattr/checkpoint update",
    "fsync",
    "remount",
    "application read",
}

WRITE_CLIENT = r"""
import hashlib
import json
import os
import pathlib
import sys

path = pathlib.Path(os.environ["TARGET_PATH"])
payload = bytes.fromhex(os.environ["PAYLOAD_HEX"])
mode = os.environ.get("OPEN_MODE", "r+b")
try:
    with path.open(mode) as f:
        f.seek(0)
        f.write(payload)
        f.flush()
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


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def payload(label: str) -> bytes:
    seed = (label + ":").encode("ascii")
    return (seed * ((PAYLOAD_BYTES // len(seed)) + 1))[:PAYLOAD_BYTES]


def run_capture(command: list[str], out_dir: Path, name: str) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / f"{name}.stdout.txt"
    stderr_path = out_dir / f"{name}.stderr.txt"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        proc = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
    return {
        "command": command,
        "returncode": proc.returncode,
        "stdout": relpath(stdout_path),
        "stderr": relpath(stderr_path),
    }


def run_client(script: str, env: dict[str, str], out_dir: Path, name: str,
               timeout_s: float = 12.0) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / f"{name}.stdout.txt"
    stderr_path = out_dir / f"{name}.stderr.txt"
    full_env = os.environ.copy()
    full_env.update(env)
    timed_out = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        proc = subprocess.Popen(
            ["python3", "-c", script],
            cwd=ROOT,
            env=full_env,
            stdout=stdout,
            stderr=stderr,
        )
        try:
            returncode = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            try:
                returncode = proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                returncode = None
    result: dict[str, Any] = {
        "command": ["python3", "-c", "<inline>"],
        "returncode": returncode,
        "timeout": timed_out,
        "stdout": relpath(stdout_path),
        "stderr": relpath(stderr_path),
    }
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace").strip()
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace").strip()
    if stdout_text:
        try:
            result["stdout_json"] = json.loads(stdout_text.splitlines()[-1])
        except json.JSONDecodeError:
            result["stdout_tail"] = stdout_text[-500:]
    if stderr_text:
        result["stderr_tail"] = stderr_text[-500:]
    return result


def mount_is_visible(mount_dir: Path) -> bool:
    mount_path = mount_dir.resolve()
    try:
        with open("/proc/mounts", "r", encoding="utf-8", errors="replace") as f:
            for line in f:
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
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def start_fuse(storage_dir: Path, mount_dir: Path, password: str, out_dir: Path,
               label: str, extra_env: dict[str, str] | None = None) -> FuseProc:
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
    if extra_env:
        env.update(extra_env)

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


def force_unmount(mount_dir: Path, out_dir: Path, label: str) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for idx, command in enumerate((
        ["fusermount3", "-uz", str(mount_dir)],
        ["fusermount", "-uz", str(mount_dir)],
        ["umount", "-l", str(mount_dir)],
    )):
        if shutil.which(command[0]) is None:
            continue
        result = run_capture(command, out_dir, f"{label}_unmount_{idx}")
        attempts.append(result)
        if result["returncode"] == 0:
            break
    return attempts


def stop_fuse(handle: FuseProc | None, mount_dir: Path, out_dir: Path,
              label: str, lazy: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {"label": label, "unmount_attempts": []}
    if lazy or mount_is_visible(mount_dir):
        result["unmount_attempts"] = force_unmount(mount_dir, out_dir, label)
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


def parse_marker(marker_path: Path) -> list[dict[str, Any]]:
    entries = []
    if not marker_path.exists():
        return entries
    for line in marker_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            entries.append({"malformed": line})
    return entries


def scan_storage(storage_dir: Path, out_path: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(storage_dir.rglob("*")):
        rel = path.relative_to(storage_dir)
        item: dict[str, Any] = {"path": str(rel)}
        try:
            st = path.lstat()
        except OSError as exc:
            item.update({"error": f"lstat errno={exc.errno}"})
            entries.append(item)
            continue
        item["mode"] = st.st_mode
        item["size"] = st.st_size
        if path.is_file():
            try:
                data = path.read_bytes()
                item["sha256"] = sha256_bytes(data)
            except OSError as exc:
                item["read_error"] = f"errno={exc.errno}"
        xattrs: dict[str, Any] = {}
        try:
            for name in sorted(os.listxattr(path)):
                try:
                    value = os.getxattr(path, name)
                    xattrs[name] = {"bytes": len(value), "sha256": sha256_bytes(value)}
                except OSError as exc:
                    xattrs[name] = {"error": f"errno={exc.errno}"}
        except OSError as exc:
            xattrs["_list_error"] = f"errno={exc.errno}"
        if xattrs:
            item["xattrs"] = xattrs
        entries.append(item)
    report = {"storage_dir": str(storage_dir), "entries": entries}
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"path": relpath(out_path), "entries": len(entries)}


def classify_recovery(storage_dir: Path, mount_dir: Path, password: str,
                      case_dir: Path, previous: bytes, latest: bytes,
                      logical_name: str) -> dict[str, Any]:
    handle: FuseProc | None = None
    try:
        handle = start_fuse(storage_dir, mount_dir, password, case_dir, "recovery")
    except Exception as exc:
        return {
            "verdict": "fail_closed",
            "acceptable": True,
            "detail": f"clean recovery remount rejected state: {exc}",
        }

    try:
        result = run_client(
            READ_CLIENT,
            {"TARGET_PATH": str(mount_dir / logical_name)},
            case_dir,
            "recovery_read",
        )
        if result.get("timeout"):
            verdict = "unexpected_liveness_failure"
            acceptable = False
            detail = "recovery read timed out"
        elif result.get("returncode") != 0:
            verdict = "fail_closed"
            acceptable = True
            detail = f"recovery read failed rc={result.get('returncode')}"
        else:
            stdout_json = result.get("stdout_json") or {}
            digest = stdout_json.get("sha256")
            size = stdout_json.get("bytes")
            if digest == sha256_bytes(latest) and size == len(latest):
                verdict = "latest_committed"
                acceptable = True
                detail = "recovery digest matches latest committed payload"
            elif digest == sha256_bytes(previous) and size == len(previous):
                verdict = "previous_committed"
                acceptable = True
                detail = "recovery digest matches previous committed payload"
            else:
                verdict = "silent_corruption"
                acceptable = False
                detail = f"recovery digest matched no oracle state: bytes={size} sha256={digest}"
        return {
            "verdict": verdict,
            "acceptable": acceptable,
            "detail": detail,
            "read": result,
        }
    finally:
        stop_fuse(handle, mount_dir, case_dir, "recovery")


def run_daemon_case(case: dict[str, str], out_dir: Path, password: str) -> dict[str, Any]:
    case_dir = out_dir / case["case"]
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    storage_dir = Path(tempfile.mkdtemp(prefix=f"daemon_fault_{case['case']}_store_", dir="/tmp"))
    mount_dir = Path(tempfile.mkdtemp(prefix=f"daemon_fault_{case['case']}_mnt_", dir="/tmp"))
    logical_name = "probe.bin"
    previous = payload("previous")
    latest = payload("latest")
    marker_path = case_dir / "fault_marker.jsonl"

    setup_steps: list[dict[str, Any]] = []
    handle: FuseProc | None = None
    fault_handle: FuseProc | None = None
    try:
        handle = start_fuse(storage_dir, mount_dir, password, case_dir, "baseline")
        baseline = run_client(
            WRITE_CLIENT,
            {
                "TARGET_PATH": str(mount_dir / logical_name),
                "PAYLOAD_HEX": previous.hex(),
                "OPEN_MODE": "wb",
            },
            case_dir,
            "baseline_write",
        )
        setup_steps.append({"phase": "baseline_write", **baseline})
        stop_fuse(handle, mount_dir, case_dir, "baseline")
        handle = None
        if baseline.get("returncode") != 0:
            raise RuntimeError("baseline write failed")

        if case["operation"] == "read_precommitted_latest":
            handle = start_fuse(storage_dir, mount_dir, password, case_dir, "precommit_latest")
            precommit = run_client(
                WRITE_CLIENT,
                {
                    "TARGET_PATH": str(mount_dir / logical_name),
                    "PAYLOAD_HEX": latest.hex(),
                    "OPEN_MODE": "r+b",
                },
                case_dir,
                "precommit_latest_write",
            )
            setup_steps.append({"phase": "precommit_latest_write", **precommit})
            stop_fuse(handle, mount_dir, case_dir, "precommit_latest")
            handle = None
            if precommit.get("returncode") != 0:
                raise RuntimeError("precommit latest write failed")

        fault_env = {
            "PQC_FAULT_CUTPOINT": case["hook"],
            "PQC_FAULT_MARKER_PATH": str(marker_path),
        }
        fault_handle = start_fuse(storage_dir, mount_dir, password, case_dir, "fault", fault_env)
        if case["operation"] == "read_precommitted_latest":
            fault_client = run_client(
                READ_CLIENT,
                {"TARGET_PATH": str(mount_dir / logical_name)},
                case_dir,
                "fault_read",
            )
        else:
            fault_client = run_client(
                WRITE_CLIENT,
                {
                    "TARGET_PATH": str(mount_dir / logical_name),
                    "PAYLOAD_HEX": latest.hex(),
                    "OPEN_MODE": "r+b",
                },
                case_dir,
                "fault_write",
            )

        time.sleep(0.2)
        daemon_returncode = fault_handle.proc.poll()
        if daemon_returncode is None:
            try:
                daemon_returncode = fault_handle.proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                daemon_returncode = None
        fault_stop = stop_fuse(fault_handle, mount_dir, case_dir, "fault", lazy=True)
        fault_handle = None
        if daemon_returncode is None:
            daemon_returncode = fault_stop.get("daemon_returncode")

        storage_scan = scan_storage(storage_dir, case_dir / "faulted_storage_scan.json")
        recovery = classify_recovery(
            storage_dir,
            mount_dir,
            password,
            case_dir,
            previous,
            latest,
            logical_name,
        )
        marker_entries = parse_marker(marker_path)
        fault_triggered = any(entry.get("name") == case["hook"] for entry in marker_entries)
        daemon_killed = daemon_returncode in (-signal.SIGKILL, 128 + signal.SIGKILL, 137)
        verdict = recovery["verdict"]
        acceptable = (
            fault_triggered
            and daemon_killed
            and verdict in ACCEPTABLE_VERDICTS
            and recovery.get("acceptable") is True
        )
        return {
            "case": case["case"],
            "required_cut_point": case["required_cut_point"],
            "hook": case["hook"],
            "operation": case["operation"],
            "verdict": verdict,
            "acceptable": acceptable,
            "detail": recovery["detail"],
            "fault_triggered": fault_triggered,
            "daemon_killed": daemon_killed,
            "daemon_returncode": daemon_returncode,
            "marker": relpath(marker_path),
            "marker_entries": marker_entries,
            "setup_steps": setup_steps,
            "fault_client": fault_client,
            "fault_stop": fault_stop,
            "faulted_storage_scan": storage_scan,
            "recovery": recovery,
            "scope": "final-binary daemon SIGKILL at named cutpoint; not physical power loss",
        }
    except Exception as exc:
        return {
            "case": case["case"],
            "required_cut_point": case["required_cut_point"],
            "hook": case["hook"],
            "operation": case["operation"],
            "verdict": "unexpected_liveness_failure",
            "acceptable": False,
            "detail": f"campaign setup or execution failed: {exc}",
            "fault_triggered": False,
            "daemon_killed": False,
            "setup_steps": setup_steps,
            "scope": "final-binary daemon SIGKILL at named cutpoint; not physical power loss",
        }
    finally:
        stop_fuse(handle, mount_dir, case_dir, "cleanup_baseline", lazy=True)
        stop_fuse(fault_handle, mount_dir, case_dir, "cleanup_fault", lazy=True)
        shutil.rmtree(storage_dir, ignore_errors=True)
        shutil.rmtree(mount_dir, ignore_errors=True)


def load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"missing artifact {relpath(path)}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON in {relpath(path)}: {exc}"


def count_verdicts(verdicts: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for verdict in verdicts:
        counts[verdict] = counts.get(verdict, 0) + 1
    return counts


def normalize_verdict(verdict: str) -> str:
    if verdict == "previous":
        return "previous_committed"
    if verdict == "latest":
        return "latest_committed"
    return verdict


def build_application_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    sqlite_wal_path = ROOT / "artifacts" / "validation" / "sqlite_fault_campaign" / "sqlite_fault_campaign.json"
    sqlite_wal, error = load_json(sqlite_wal_path)
    if sqlite_wal is None:
        rows.append({
            "mode": "SQLite WAL/FULL",
            "coverage": "application-level commit boundary",
            "source": relpath(sqlite_wal_path),
            "verdict": "unexpected_liveness_failure",
            "acceptable": False,
            "detail": error,
        })
    else:
        trial_rows = sqlite_wal.get("rows", [])
        verdicts = [str(row.get("verdict")) for row in trial_rows]
        acceptable = bool(trial_rows) and all(
            row.get("acceptable") is True and row.get("verdict") in ALLOWED_VERDICTS
            for row in trial_rows
        )
        rows.append({
            "mode": "SQLite WAL/FULL",
            "coverage": "application-level commit boundary",
            "source": relpath(sqlite_wal_path),
            "trials": len(trial_rows),
            "verdict_counts": count_verdicts(verdicts),
            "verdict": "previous_committed" if acceptable else "unexpected_liveness_failure",
            "acceptable": acceptable,
            "detail": "deterministic SQLite WAL/FULL file-state fault campaign",
        })

    sqlite_syscall_path = ROOT / "artifacts" / "validation" / "sqlite_syscall_crash_tpm" / "sqlite_syscall_crash_tpm.json"
    sqlite_syscall, error = load_json(sqlite_syscall_path)
    if sqlite_syscall is None:
        rows.append({
            "mode": "SQLite rollback DELETE/EXTRA",
            "coverage": "application-level commit boundary",
            "source": relpath(sqlite_syscall_path),
            "verdict": "unexpected_liveness_failure",
            "acceptable": False,
            "detail": error,
        })
    else:
        trials = sqlite_syscall.get("trials", [])
        verdicts = [
            normalize_verdict(str((trial.get("replay") or {}).get("verdict")))
            for trial in trials
        ]
        acceptable = bool(trials) and (sqlite_syscall.get("summary") or {}).get("all_acceptable") is True
        acceptable = acceptable and all(verdict in ALLOWED_VERDICTS for verdict in verdicts)
        rows.append({
            "mode": "SQLite rollback DELETE/EXTRA",
            "coverage": "fdatasync SIGKILL application crash boundary",
            "source": relpath(sqlite_syscall_path),
            "trials": len(trials),
            "verdict_counts": count_verdicts(verdicts),
            "verdict": "previous_committed" if acceptable else "unexpected_liveness_failure",
            "acceptable": acceptable,
            "detail": "SQLite writer killed by strace fdatasync SIGKILL on TPM-backed FUSE",
        })

    combined_path = ROOT / "artifacts" / "validation" / "combined_durability_bundle" / "combined_durability_bundle.json"
    combined, error = load_json(combined_path)
    if combined is None:
        rows.append({
            "mode": "non-SQLite key-value",
            "coverage": "stale backing-store replay after anchor advance",
            "source": relpath(combined_path),
            "verdict": "unexpected_liveness_failure",
            "acceptable": False,
            "detail": error,
        })
    else:
        dbm_replay = ((combined.get("unified_dbm_campaign") or {}).get("replay") or {})
        dbm_verdict = normalize_verdict(str(dbm_replay.get("verdict")))
        dbm_acceptable = dbm_replay.get("acceptable") is True and dbm_verdict in ALLOWED_VERDICTS
        rows.append({
            "mode": "dbm.dumb key-value",
            "coverage": "non-SQLite append/key-value workload",
            "source": relpath(combined_path),
            "trials": 1,
            "verdict_counts": count_verdicts([dbm_verdict]),
            "verdict": dbm_verdict,
            "acceptable": dbm_acceptable,
            "detail": str(dbm_replay.get("detail")),
        })

        sqlite_replay = ((combined.get("unified_campaign") or {}).get("replay") or {})
        sqlite_verdict = normalize_verdict(str(sqlite_replay.get("verdict")))
        sqlite_acceptable = sqlite_replay.get("acceptable") is True and sqlite_verdict in ALLOWED_VERDICTS
        rows.append({
            "mode": "SQLite DELETE/EXTRA TPM replay",
            "coverage": "same-backing-store stale snapshot after hardware anchor advance",
            "source": relpath(combined_path),
            "trials": 1,
            "verdict_counts": count_verdicts([sqlite_verdict]),
            "verdict": sqlite_verdict,
            "acceptable": sqlite_acceptable,
            "detail": str(sqlite_replay.get("detail")),
        })

    return rows


def build_coverage(daemon_rows: list[dict[str, Any]],
                   application_rows: list[dict[str, Any]]) -> dict[str, Any]:
    covered_daemon = {
        row["required_cut_point"]
        for row in daemon_rows
        if row.get("acceptable") and row.get("required_cut_point") in REQUIRED_DAEMON_CUTPOINTS
    }
    app_modes = {
        "sqlite_wal_full": any(row.get("mode") == "SQLite WAL/FULL" and row.get("acceptable") for row in application_rows),
        "sqlite_rollback_or_persistent": any(
            row.get("mode") in ("SQLite rollback DELETE/EXTRA", "SQLite DELETE/EXTRA TPM replay")
            and row.get("acceptable")
            for row in application_rows
        ),
        "non_sqlite_append_or_kv": any(row.get("mode") == "dbm.dumb key-value" and row.get("acceptable") for row in application_rows),
    }
    return {
        "required_daemon_cutpoints": sorted(REQUIRED_DAEMON_CUTPOINTS),
        "covered_daemon_cutpoints": sorted(covered_daemon),
        "missing_daemon_cutpoints": sorted(REQUIRED_DAEMON_CUTPOINTS - covered_daemon),
        "application_modes": app_modes,
        "missing_application_modes": sorted(name for name, ok in app_modes.items() if not ok),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Daemon power-fault campaign",
        "",
        "Scope: final-binary daemon SIGKILL cutpoints plus retained application recovery artifacts. This is not physical power-loss or kernel-crash certification.",
        "",
        f"- Overall pass: `{report['overall_pass']}`",
        f"- Daemon rows: `{len(report['daemon_rows'])}`",
        f"- Application rows: `{len(report['application_rows'])}`",
        f"- Missing daemon cut points: `{', '.join(report['coverage']['missing_daemon_cutpoints']) or 'none'}`",
        f"- Missing application modes: `{', '.join(report['coverage']['missing_application_modes']) or 'none'}`",
        "",
        "## Daemon cutpoints",
        "",
    ]
    for row in report["daemon_rows"]:
        lines.append(
            f"- {row['required_cut_point']} (`{row['hook']}`): verdict `{row['verdict']}`, "
            f"acceptable `{row['acceptable']}`, marker `{row['fault_triggered']}`, "
            f"daemon_killed `{row['daemon_killed']}`"
        )
    lines.extend(["", "## Application modes", ""])
    for row in report["application_rows"]:
        lines.append(
            f"- {row['mode']}: verdict `{row['verdict']}`, acceptable `{row['acceptable']}`, "
            f"trials `{row.get('trials', 0)}`, source `{row['source']}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--password", default=os.environ.get("PQC_MASTER_PASSWORD", "daemon-power-fault-password"))
    parser.add_argument("--skip-daemon-runs", action="store_true",
                        help="Only rebuild the manifest from existing application artifacts.")
    args = parser.parse_args()

    if not FUSE_BIN.exists() and not args.skip_daemon_runs:
        raise SystemExit(f"missing FUSE binary: {FUSE_BIN}")

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    daemon_rows = [] if args.skip_daemon_runs else [
        run_daemon_case(case, out_dir, args.password)
        for case in DAEMON_CUTPOINTS
    ]
    application_rows = build_application_rows()
    coverage = build_coverage(daemon_rows, application_rows)
    daemon_ok = bool(daemon_rows) and all(row.get("acceptable") is True for row in daemon_rows)
    app_ok = bool(application_rows) and all(row.get("acceptable") is True for row in application_rows)
    coverage_ok = not coverage["missing_daemon_cutpoints"] and not coverage["missing_application_modes"]
    report = {
        "note": "Daemon SIGKILL and power-loss-equivalent recovery matrix for AEGIS-Q durability claims.",
        "scope": [
            "Daemon rows use final build/pqc_fuse with PQC_FAULT_CUTPOINT hooks.",
            "Application rows are retained SQLite/dbm artifacts with external state or digest oracles.",
            "Daemon rows use one logical block; multi-block application atomicity is covered only by application journal rows.",
            "The matrix does not certify physical power loss, kernel crash, drive write-cache behavior, or arbitrary workloads.",
        ],
        "allowed_verdicts": sorted(ALLOWED_VERDICTS),
        "acceptable_verdicts": sorted(ACCEPTABLE_VERDICTS),
        "daemon_rows": daemon_rows,
        "application_rows": application_rows,
        "coverage": coverage,
        "overall_pass": daemon_ok and app_ok and coverage_ok,
    }

    json_path = out_dir / "daemon_power_fault_campaign.json"
    md_path = out_dir / "daemon_power_fault_campaign.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({
        "out_dir": relpath(out_dir),
        "daemon_rows": len(daemon_rows),
        "application_rows": len(application_rows),
        "overall_pass": report["overall_pass"],
        "missing_daemon_cutpoints": coverage["missing_daemon_cutpoints"],
        "missing_application_modes": coverage["missing_application_modes"],
    }, indent=2))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
