#!/usr/bin/env python3
"""Mounted-path proof for Gate 0.16-S2.

The smoke keeps strict durability semantics intact.  It enables
``PQC_PARALLEL_COMMIT_MODE=epoch-gated-strict`` for one mounted run and checks
that authenticated writeback reaches the runtime parallel-commit coordinator.
It also runs a strict-mode mounted write/read/fsync control.
"""

from __future__ import annotations

import argparse
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
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "parallel_commit_contract"


@dataclass
class FuseHandle:
    proc: subprocess.Popen[bytes]
    stdout: Any
    stderr: Any


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def fusermount_command() -> str:
    for name in ("fusermount3", "fusermount"):
        if shutil.which(name):
            return name
    return "fusermount3"


def start_fuse(
    storage_dir: Path,
    mount_dir: Path,
    out_dir: Path,
    password: str,
    extra_env: dict[str, str] | None = None,
) -> FuseHandle:
    log_dir = out_dir / "mount_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout = (log_dir / "pqc_fuse.stdout.txt").open("wb")
    stderr = (log_dir / "pqc_fuse.stderr.txt").open("wb")
    env = os.environ.copy()
    env.update(
        {
            "PQC_MASTER_PASSWORD": password,
            "PQC_FRESHNESS_ANCHOR_BACKEND": "file",
            "PQC_FRESHNESS_ANCHOR_PATH": str(storage_dir / ".anchor"),
            "PQC_ADMISSION_TRACE_PATH": str(out_dir / "admission_trace.jsonl"),
            "PQC_ENABLE_ADMISSION_ON_WRITE": "0",
            "PQC_ENABLE_QOS_THROTTLE_ON_WRITE": "0",
            "PQC_KEY_ROTATION_INTERVAL_S": "0",
        }
    )
    if extra_env:
        env.update(extra_env)

    proc = subprocess.Popen(
        [str(FUSE_BIN), str(storage_dir), str(mount_dir), "-f"],
        cwd=ROOT,
        env=env,
        stdout=stdout,
        stderr=stderr,
    )
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if subprocess.run(["mountpoint", "-q", str(mount_dir)],
                          check=False).returncode == 0:
            return FuseHandle(proc=proc, stdout=stdout, stderr=stderr)
        if proc.poll() is not None:
            stdout.close()
            stderr.close()
            raise RuntimeError(f"FUSE exited before mount: rc={proc.returncode}")
        time.sleep(0.05)
    stdout.close()
    stderr.close()
    raise TimeoutError("timed out waiting for AEGIS-Q FUSE mount")


def stop_fuse(handle: FuseHandle | None, mount_dir: Path, out_dir: Path) -> dict[str, Any]:
    log_dir = out_dir / "mount_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    unmount = subprocess.run(
        [fusermount_command(), "-u", str(mount_dir)],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    (log_dir / "unmount.stdout.txt").write_text(unmount.stdout, encoding="utf-8")
    (log_dir / "unmount.stderr.txt").write_text(unmount.stderr, encoding="utf-8")
    result: dict[str, Any] = {
        "argv": [fusermount_command(), "-u", str(mount_dir)],
        "returncode": unmount.returncode,
    }
    if handle is None:
        return result
    if handle.proc.poll() is None:
        handle.proc.send_signal(signal.SIGINT)
        try:
            handle.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            handle.proc.kill()
            handle.proc.wait(timeout=5)
    handle.stdout.close()
    handle.stderr.close()
    result["fuse_returncode"] = handle.proc.returncode
    return result


def mounted_write_read(mount_dir: Path, name: str, payload: bytes) -> dict[str, Any]:
    path = mount_dir / name
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
    try:
        written = os.write(fd, payload)
        os.fdatasync(fd)
        os.lseek(fd, 0, os.SEEK_SET)
        recovered = os.read(fd, len(payload))
    finally:
        os.close(fd)
    return {
        "path": str(path),
        "payload_len": len(payload),
        "written": written,
        "read_len": len(recovered),
        "matches": recovered == payload,
    }


def parse_trace(trace_path: Path) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    malformed = 0
    if trace_path.exists():
        for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(payload, dict):
                events.append(payload)
    event_names = [str(event.get("event", "")) for event in events]
    begin_events = [event for event in events if event.get("event") == "begin"]
    finish_events = [event for event in events if event.get("event") == "finish"]
    return {
        "path": relpath(trace_path),
        "exists": trace_path.exists(),
        "event_count": len(events),
        "malformed_line_count": malformed,
        "events": event_names,
        "begin_count": len(begin_events),
        "finish_count": len(finish_events),
        "roles": sorted({str(event.get("role", "")) for event in events if event.get("role")}),
        "max_group_size": max(
            [int(event.get("group_size", 0) or 0) for event in events],
            default=0,
        ),
    }


def run_case(case_dir: Path, mode: str) -> dict[str, Any]:
    storage_dir = Path(tempfile.mkdtemp(prefix=f"aegisq_{mode}_storage_"))
    mount_dir = Path(tempfile.mkdtemp(prefix=f"aegisq_{mode}_mnt_"))
    case_dir.mkdir(parents=True, exist_ok=True)
    password = f"parallel-commit-{mode}"
    extra_env: dict[str, str] = {}
    trace_path = case_dir / "parallel_commit_trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()
    if mode == "epoch_gated_strict":
        extra_env.update(
            {
                "PQC_PARALLEL_COMMIT_MODE": "epoch-gated-strict",
                "PQC_PARALLEL_COMMIT_SHARDS": "4",
                "PQC_PARALLEL_COMMIT_GROUP_MAX": "1",
                "PQC_PARALLEL_COMMIT_WAIT_NS": "0",
                "PQC_PARALLEL_COMMIT_TRACE_PATH": str(trace_path),
            }
        )

    fuse: FuseHandle | None = None
    unmount: dict[str, Any] = {}
    client: dict[str, Any] = {}
    error: str | None = None
    try:
        fuse = start_fuse(storage_dir, mount_dir, case_dir, password, extra_env)
        payload = (f"parallel-commit-smoke:{mode}:".encode("ascii") +
                   bytes((i % 251 for i in range(8192))))
        client = mounted_write_read(mount_dir, "smoke.dat", payload)
    except Exception as exc:  # noqa: BLE001 - retained as artifact evidence
        error = repr(exc)
    finally:
        try:
            unmount = stop_fuse(fuse, mount_dir, case_dir)
        finally:
            shutil.rmtree(mount_dir, ignore_errors=True)
            shutil.rmtree(storage_dir, ignore_errors=True)

    trace = parse_trace(trace_path)
    return {
        "mode": mode,
        "client": client,
        "trace": trace,
        "unmount": unmount,
        "error": error,
        "pass": (
            error is None and
            client.get("matches") is True and
            unmount.get("returncode") == 0 and
            (mode == "strict" or (
                trace["begin_count"] >= 1 and
                trace["finish_count"] >= 1 and
                "leader" in trace["roles"]
            ))
        ),
    }


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "epoch_path_smoke.json"
    md_path = out_dir / "epoch_path_smoke.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    lines = [
        "# Parallel Commit Epoch Path Smoke",
        "",
        f"- Generated: `{payload['generated_utc']}`",
        f"- Overall pass: `{str(payload['overall_pass']).lower()}`",
        "",
    ]
    for case in payload["cases"]:
        lines.append(f"## {case['mode']}")
        lines.append(f"- Pass: `{str(case['pass']).lower()}`")
        lines.append(f"- Read matches write: `{str(case['client'].get('matches')).lower()}`")
        lines.append(f"- Trace events: `{case['trace'].get('events')}`")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    strict = run_case(args.out_dir / "strict_path_smoke", "strict")
    epoch = run_case(args.out_dir / "epoch_path_smoke", "epoch_gated_strict")
    payload = {
        "schema_version": 1,
        "generated_by": "experiments/run_parallel_commit_epoch_path_smoke.py",
        "generated_utc": now_utc(),
        "scope": "Gate 0.16-S2 mounted-path proof for explicit parallel commit mode.",
        "cases": [strict, epoch],
        "overall_pass": strict["pass"] and epoch["pass"],
        "negative_claim_guard": (
            "This smoke proves that epoch-gated-strict writeback reaches the "
            "parallel-commit coordinator. It does not prove sync reduction, "
            "epoch redo-log durability, throughput improvement, or Gate 0.16 closure."
        ),
    }
    write_outputs(args.out_dir, payload)
    print(json.dumps({
        "overall_pass": payload["overall_pass"],
        "json": relpath(args.out_dir / "epoch_path_smoke.json"),
        "epoch_trace": epoch["trace"],
    }, indent=2, sort_keys=True))
    return 0 if payload["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
