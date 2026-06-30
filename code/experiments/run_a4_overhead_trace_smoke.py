#!/usr/bin/env python3
"""Gate A4 mounted overhead trace smoke.

The runner exercises the production FUSE path once and retains only the narrow
overhead traces needed by A4: FUSE operation latency counters, crypto-plane
route counters, publication timing, lock profile, and durability-site counters.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FUSE_BIN = Path(os.environ.get("PQC_FUSE_BIN", ROOT / "build" / "pqc_fuse"))
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "a4_hidden_overhead_accounting"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def mount_is_visible(mount_dir: Path) -> bool:
    return subprocess.run(
        ["mountpoint", "-q", str(mount_dir)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def fusermount_command() -> str:
    for name in ("fusermount3", "fusermount"):
        if shutil.which(name):
            return name
    return "fusermount3"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"exists": True, "json_error": str(exc)}
    if isinstance(payload, dict):
        payload["exists"] = True
        return payload
    return {"exists": True, "json_error": "top-level value is not an object"}


def parse_jsonl(path: Path) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    malformed = 0
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(payload, dict):
                events.append(payload)
    publication = [event for event in events
                   if event.get("event") == "publication_dispatch"]
    return {
        "exists": path.exists(),
        "path": rel(path),
        "event_count": len(events),
        "malformed_line_count": malformed,
        "publication_count": len(publication),
        "publication_elapsed_ns_total": sum(
            int(event.get("elapsed_ns", 0) or 0) for event in publication
        ),
        "publication_sync_count_total": sum(
            int(event.get("sync_count", 0) or 0) for event in publication
        ),
        "publication_data_fsync_count_total": sum(
            int(event.get("data_fsync_count", 0) or 0)
            for event in publication
        ),
        "publication_journal_fsync_count_total": sum(
            int(event.get("journal_fsync_count", 0) or 0)
            for event in publication
        ),
        "publication_epoch_log_fsync_count_total": sum(
            int(event.get("epoch_log_fsync_count", 0) or 0)
            for event in publication
        ),
    }


def parse_durability(stderr_path: Path) -> dict[str, int]:
    if not stderr_path.exists():
        return {}
    target = ""
    for line in stderr_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Durability mounted-operation sync stats:" in line:
            target = line
    if not target:
        for line in stderr_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "Durability sync stats:" in line:
                target = line
    return {
        key: int(value)
        for key, value in re.findall(r"([a-zA-Z_]+)=([0-9]+)", target)
    }


def fuse_ops_by_name(trace: dict[str, Any]) -> dict[str, dict[str, Any]]:
    ops = trace.get("operations", [])
    if not isinstance(ops, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for op in ops:
        if isinstance(op, dict) and isinstance(op.get("op"), str):
            out[str(op["op"])] = op
    return out


def source_checks() -> dict[str, bool]:
    fuse = (ROOT / "code" / "frontend" / "pqc_fuse.c").read_text(
        encoding="utf-8", errors="replace"
    )
    trace = (ROOT / "code" / "runtime" / "pqc_fuse_trace.c").read_text(
        encoding="utf-8", errors="replace"
    )
    runtime = (ROOT / "code" / "runtime" / "pqc_runtime.c").read_text(
        encoding="utf-8", errors="replace"
    )
    return {
        "fuse_operations_wrapped": "trace_write" in fuse
        and ".write      = trace_write" in fuse
        and ".fsync      = trace_fsync" in fuse,
        "fuse_trace_dump_env": "PQC_FUSE_TRACE_PATH" in trace,
        "fuse_trace_latency_counters": "total_ns" in trace
        and "max_ns" in trace
        and "pqc_fuse_trace_end" in trace,
        "runtime_dumps_fuse_trace": "pqc_fuse_trace_dump_if_requested" in runtime,
    }


def start_fuse(storage_dir: Path, mount_dir: Path, out_dir: Path
               ) -> tuple[subprocess.Popen[bytes], Path, Path]:
    stdout_path = out_dir / "pqc_fuse.stdout.txt"
    stderr_path = out_dir / "pqc_fuse.stderr.txt"
    env = os.environ.copy()
    env.update({
        "PQC_MASTER_PASSWORD": "a4-overhead-smoke",
        "PQC_FRESHNESS_ANCHOR_BACKEND": "file",
        "PQC_FRESHNESS_ANCHOR_PATH": str(storage_dir / ".anchor"),
        "PQC_ENABLE_ADMISSION_ON_WRITE": "0",
        "PQC_ENABLE_QOS_THROTTLE_ON_WRITE": "0",
        "PQC_KEY_ROTATION_INTERVAL_S": "0",
        "PQC_FUSE_TRACE_PATH": str(out_dir / "fuse_trace.json"),
        "PQC_PLANE_TRACE_PATH": str(out_dir / "plane_trace.json"),
        "PQC_PUBLICATION_TRACE_PATH": str(out_dir / "publication_trace.jsonl"),
        "PQC_LOCK_PROFILE_PATH": str(out_dir / "lock_profile.jsonl"),
    })
    stdout = stdout_path.open("wb")
    stderr = stderr_path.open("wb")
    try:
        proc = subprocess.Popen(
            [str(FUSE_BIN), str(storage_dir), str(mount_dir), "-f"],
            cwd=ROOT,
            env=env,
            stdout=stdout,
            stderr=stderr,
        )
    finally:
        stdout.close()
        stderr.close()
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if mount_is_visible(mount_dir):
            return proc, stdout_path, stderr_path
        if proc.poll() is not None:
            raise RuntimeError(f"FUSE exited before mount: rc={proc.returncode}")
        time.sleep(0.05)
    raise TimeoutError("timed out waiting for AEGIS-Q FUSE mount")


def stop_fuse(proc: subprocess.Popen[bytes] | None,
              mount_dir: Path) -> dict[str, Any]:
    unmount_rc: int | None = None
    if mount_is_visible(mount_dir):
        unmount = subprocess.run(
            [fusermount_command(), "-u", str(mount_dir)],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        unmount_rc = unmount.returncode
    if proc is not None and proc.poll() is None and unmount_rc == 0:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    if proc is not None and proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    return {
        "unmount_rc": unmount_rc,
        "fuse_rc": None if proc is None else proc.returncode,
    }


def run_workload(mount_dir: Path) -> dict[str, Any]:
    path = mount_dir / "overhead.dat"
    payload = (b"aegisq-a4-overhead:" * 4096)[:65536]
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
    try:
        written = os.write(fd, payload)
        os.fsync(fd)
        os.lseek(fd, 0, os.SEEK_SET)
        observed = os.read(fd, len(payload))
    finally:
        os.close(fd)
    listed = "overhead.dat" in os.listdir(mount_dir)
    return {
        "payload_bytes": len(payload),
        "written": written,
        "read_len": len(observed),
        "readback_matches": observed == payload,
        "listed": listed,
    }


def build_report(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="aegisq_a4_overhead_"))
    storage_dir = work_dir / "storage"
    mount_dir = work_dir / "mnt"
    storage_dir.mkdir()
    mount_dir.mkdir()
    proc: subprocess.Popen[bytes] | None = None
    workload: dict[str, Any] = {}
    stop: dict[str, Any] = {}
    error: str | None = None
    try:
        proc, _, stderr_path = start_fuse(storage_dir, mount_dir, out_dir)
        workload = run_workload(mount_dir)
        stop = stop_fuse(proc, mount_dir)
        proc = None
    except Exception as exc:  # noqa: BLE001 - retained as smoke evidence
        error = repr(exc)
        stderr_path = out_dir / "pqc_fuse.stderr.txt"
    finally:
        if proc is not None:
            stop = stop_fuse(proc, mount_dir)
        shutil.rmtree(work_dir, ignore_errors=True)

    fuse_trace = read_json(out_dir / "fuse_trace.json")
    plane_trace = read_json(out_dir / "plane_trace.json")
    publication_trace = parse_jsonl(out_dir / "publication_trace.jsonl")
    durability = parse_durability(stderr_path)
    ops = fuse_ops_by_name(fuse_trace)
    required_ops = ("create", "write", "fsync", "read", "release", "readdir")
    proof_checks = {
        "workload_passed": error is None
        and workload.get("readback_matches") is True
        and workload.get("listed") is True
        and stop.get("unmount_rc") == 0,
        "fuse_trace_has_required_ops": all(
            int(ops.get(name, {}).get("calls", 0) or 0) > 0
            for name in required_ops
        ),
        "fuse_trace_has_latency": all(
            int(ops.get(name, {}).get("total_ns", 0) or 0) > 0
            for name in ("write", "fsync", "read")
        ),
        "plane_trace_has_data_path": int(
            plane_trace.get("data_aes_gcm_encrypt_blocks", 0) or 0
        ) > 0
        and int(plane_trace.get("data_aes_gcm_decrypt_blocks", 0) or 0) > 0,
        "publication_trace_has_timing": publication_trace["publication_count"] > 0
        and publication_trace["publication_elapsed_ns_total"] > 0,
        "durability_has_sites": int(durability.get("data_sidecar", 0)) > 0
        and int(durability.get("journal_sidecar", 0)) > 0
        and int(durability.get("marker_metadata", 0)) > 0,
    }
    checks = source_checks()
    proof_checks.update(checks)
    report = {
        "overall_pass": all(proof_checks.values()),
        "schema": "a4-overhead-trace-smoke-v1",
        "generated_utc": now_utc(),
        "fuse_binary": rel(FUSE_BIN),
        "workload": workload,
        "stop": stop,
        "error": error,
        "source_checks": checks,
        "proof_checks": proof_checks,
        "fuse_trace": fuse_trace,
        "plane_trace": plane_trace,
        "publication_trace": publication_trace,
        "durability_mounted_operation_stats": durability,
        "scope": (
            "This is daemon-side mounted-path overhead accounting. FUSE "
            "operation latency is a daemon-side proxy for the FUSE crossing; "
            "it is not a kernel scheduler context-switch trace."
        ),
        "non_claims": [
            "not proof of eBPF/io_uring bypass",
            "not a kernel context-switch count",
            "not a CUDA optimization claim by itself",
            "not a throughput ranking",
        ],
    }
    (out_dir / "a4_overhead_trace_smoke.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def markdown(report: dict[str, Any]) -> str:
    ops = fuse_ops_by_name(report["fuse_trace"])
    lines = [
        "# A4 Hidden Overhead Trace Smoke",
        "",
        f"- Overall pass: `{report['overall_pass']}`",
        f"- Scope: {report['scope']}",
        "",
        "## FUSE Operation Counters",
        "",
    ]
    for name in ("create", "write", "fsync", "read", "release", "readdir"):
        op = ops.get(name, {})
        lines.append(
            f"- `{name}` calls `{op.get('calls', 0)}`, "
            f"errors `{op.get('errors', 0)}`, "
            f"total_ns `{op.get('total_ns', 0)}`, "
            f"max_ns `{op.get('max_ns', 0)}`"
        )
    lines.extend(["", "## Proof Checks", ""])
    for key, value in report["proof_checks"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    report = build_report(out_dir)
    md_path = out_dir / "a4_overhead_trace_smoke.md"
    md_path.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({
        "overall_pass": report["overall_pass"],
        "json": str(out_dir / "a4_overhead_trace_smoke.json"),
        "markdown": str(md_path),
        "fuse_write_calls": fuse_ops_by_name(report["fuse_trace"]).get(
            "write", {}
        ).get("calls", 0),
        "publication_count": report["publication_trace"]["publication_count"],
    }, indent=2, sort_keys=True))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
