#!/usr/bin/env python3
"""Mounted-path telemetry sweep for Gate 0.16-S3.

This runner does not claim throughput improvement or group-commit closure.  It
exercises the explicit ``epoch-gated-strict`` runtime mode through the mounted
FUSE path and retains raw trace evidence for group size, wait time, queue
depth, shard count, and concurrent client coverage.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import signal
import statistics
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FUSE_BIN = ROOT / "build" / "pqc_fuse"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "parallel_commit_contract"

CASE_MATRIX = [
    {
        "name": "single_client_s1_g1",
        "shards": 1,
        "group_max": 1,
        "wait_ns": 0,
        "clients": 1,
        "ops_per_client": 2,
        "payload_size": 4096,
    },
    {
        "name": "dual_client_s1_g2_wait",
        "shards": 1,
        "group_max": 2,
        "wait_ns": 5_000_000,
        "clients": 2,
        "ops_per_client": 2,
        "payload_size": 4096,
    },
    {
        "name": "quad_client_s4_g2_wait",
        "shards": 4,
        "group_max": 2,
        "wait_ns": 5_000_000,
        "clients": 4,
        "ops_per_client": 2,
        "payload_size": 4096,
    },
]


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


def percentile(values: list[int], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * pct
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    weight = rank - lo
    return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)


def summarize_numbers(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def start_fuse(
    storage_dir: Path,
    mount_dir: Path,
    out_dir: Path,
    password: str,
    env_overrides: dict[str, str],
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
    env.update(env_overrides)
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


def client_worker(
    mount_dir: str,
    result_path: str,
    client_id: int,
    ops_per_client: int,
    payload_size: int,
    start_time: float,
) -> None:
    result: dict[str, Any] = {
        "client_id": client_id,
        "ops": [],
        "pass": False,
        "error": None,
    }
    try:
        while time.time() < start_time:
            time.sleep(0.0005)
        mount = Path(mount_dir)
        for op_id in range(ops_per_client):
            payload = (
                f"client={client_id}:op={op_id}:".encode("ascii") +
                bytes(((client_id * 31 + op_id + i) % 251 for i in range(payload_size)))
            )
            path = mount / f"client_{client_id:02d}_op_{op_id:02d}.dat"
            start_ns = time.monotonic_ns()
            fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
            try:
                written = os.write(fd, payload)
                os.fdatasync(fd)
                os.lseek(fd, 0, os.SEEK_SET)
                recovered = os.read(fd, len(payload))
            finally:
                os.close(fd)
            end_ns = time.monotonic_ns()
            result["ops"].append({
                "op_id": op_id,
                "path": str(path),
                "payload_len": len(payload),
                "written": written,
                "read_len": len(recovered),
                "matches": recovered == payload,
                "latency_ns": end_ns - start_ns,
            })
        result["pass"] = all(op["matches"] for op in result["ops"])
    except Exception as exc:  # noqa: BLE001 - retained as artifact evidence
        result["error"] = repr(exc)
    Path(result_path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")


def load_client_results(result_dir: Path, client_count: int) -> list[dict[str, Any]]:
    results = []
    for client_id in range(client_count):
        path = result_dir / f"client_{client_id:02d}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            payload = {
                "client_id": client_id,
                "pass": False,
                "error": f"missing-or-invalid-result:{exc!r}",
            }
        payload["result_path"] = relpath(path)
        results.append(payload)
    return results


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
    begin_events = [event for event in events if event.get("event") == "begin"]
    finish_events = [event for event in events if event.get("event") == "finish"]
    group_sizes = [int(event.get("group_size", 0) or 0) for event in begin_events]
    queue_depths = [int(event.get("queue_depth", 0) or 0) for event in begin_events]
    wait_ns = [int(event.get("wait_ns", 0) or 0) for event in begin_events]
    shards = [int(event.get("shard", 0) or 0) for event in begin_events]
    shard_counts = [
        int(event.get("config_shard_count", 0) or 0)
        for event in events
        if event.get("config_shard_count") is not None
    ]
    return {
        "path": relpath(trace_path),
        "exists": trace_path.exists(),
        "event_count": len(events),
        "malformed_line_count": malformed,
        "begin_count": len(begin_events),
        "finish_count": len(finish_events),
        "roles": sorted({str(event.get("role", "")) for event in events if event.get("role")}),
        "observed_shards": sorted(set(shards)),
        "observed_shard_count_configs": sorted(set(shard_counts)),
        "group_size_distribution": dict(sorted(Counter(group_sizes).items())),
        "queue_depth_distribution": dict(sorted(Counter(queue_depths).items())),
        "group_size_summary": summarize_numbers(group_sizes),
        "queue_depth_summary": summarize_numbers(queue_depths),
        "wait_ns_summary": summarize_numbers(wait_ns),
    }


def run_case(out_dir: Path, case: dict[str, Any]) -> dict[str, Any]:
    case_dir = out_dir / "telemetry_sweep" / case["name"]
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    storage_dir = Path(tempfile.mkdtemp(prefix=f"aegisq_{case['name']}_storage_"))
    mount_dir = Path(tempfile.mkdtemp(prefix=f"aegisq_{case['name']}_mnt_"))
    result_dir = case_dir / "client_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    trace_path = case_dir / "parallel_commit_trace.jsonl"
    env = {
        "PQC_PARALLEL_COMMIT_MODE": "epoch-gated-strict",
        "PQC_PARALLEL_COMMIT_SHARDS": str(case["shards"]),
        "PQC_PARALLEL_COMMIT_GROUP_MAX": str(case["group_max"]),
        "PQC_PARALLEL_COMMIT_WAIT_NS": str(case["wait_ns"]),
        "PQC_PARALLEL_COMMIT_TRACE_PATH": str(trace_path),
    }

    fuse: FuseHandle | None = None
    unmount: dict[str, Any] = {}
    error: str | None = None
    processes: list[mp.Process] = []
    try:
        fuse = start_fuse(storage_dir, mount_dir, case_dir,
                          f"parallel-commit-telemetry-{case['name']}", env)
        start_time = time.time() + 0.25
        for client_id in range(int(case["clients"])):
            result_path = result_dir / f"client_{client_id:02d}.json"
            proc = mp.Process(
                target=client_worker,
                args=(
                    str(mount_dir),
                    str(result_path),
                    client_id,
                    int(case["ops_per_client"]),
                    int(case["payload_size"]),
                    start_time,
                ),
            )
            proc.start()
            processes.append(proc)
        for proc in processes:
            proc.join(timeout=30)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)
    except Exception as exc:  # noqa: BLE001 - retained as artifact evidence
        error = repr(exc)
    finally:
        for proc in processes:
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)
        try:
            unmount = stop_fuse(fuse, mount_dir, case_dir)
        finally:
            shutil.rmtree(mount_dir, ignore_errors=True)
            shutil.rmtree(storage_dir, ignore_errors=True)

    client_results = load_client_results(result_dir, int(case["clients"]))
    trace = parse_trace(trace_path)
    expected_ops = int(case["clients"]) * int(case["ops_per_client"])
    worker_exitcodes = [proc.exitcode for proc in processes]
    case_pass = (
        error is None and
        unmount.get("returncode") == 0 and
        all(code == 0 for code in worker_exitcodes) and
        all(result.get("pass") is True for result in client_results) and
        trace["exists"] and
        trace["begin_count"] >= int(case["clients"]) and
        trace["finish_count"] >= 1 and
        trace["queue_depth_summary"].get("count", 0) > 0 and
        trace["wait_ns_summary"].get("count", 0) > 0
    )
    op_latencies = [
        int(op.get("latency_ns", 0) or 0)
        for result in client_results
        for op in result.get("ops", [])
    ]
    return {
        "name": case["name"],
        "config": case,
        "expected_ops": expected_ops,
        "worker_exitcodes": worker_exitcodes,
        "client_results": client_results,
        "client_latency_ns_summary": summarize_numbers(op_latencies),
        "trace": trace,
        "unmount": unmount,
        "error": error,
        "pass": case_pass,
    }


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "parallel_commit_telemetry_sweep.json"
    md_path = out_dir / "parallel_commit_telemetry_sweep.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    lines = [
        "# Parallel Commit Telemetry Sweep",
        "",
        f"- Generated: `{payload['generated_utc']}`",
        f"- Overall pass: `{str(payload['overall_pass']).lower()}`",
        f"- Case count: `{len(payload['cases'])}`",
        "",
    ]
    for case in payload["cases"]:
        trace = case["trace"]
        lines.append(f"## {case['name']}")
        lines.append(f"- Pass: `{str(case['pass']).lower()}`")
        lines.append(f"- Config: `{case['config']}`")
        lines.append(f"- Trace: `{trace['path']}`")
        lines.append(f"- Begin/finish: `{trace['begin_count']}` / `{trace['finish_count']}`")
        lines.append(f"- Group sizes: `{trace['group_size_distribution']}`")
        lines.append(f"- Queue depths: `{trace['queue_depth_distribution']}`")
        lines.append(f"- Wait ns summary: `{trace['wait_ns_summary']}`")
        lines.append("")
    lines.extend([
        "## Negative Claim Guard",
        "",
        payload["negative_claim_guard"],
        "",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    cases = [run_case(args.out_dir, case) for case in CASE_MATRIX]
    payload = {
        "schema_version": 1,
        "generated_by": "experiments/run_parallel_commit_telemetry_sweep.py",
        "generated_utc": now_utc(),
        "scope": "Gate 0.16-S3 mounted-path parallel-commit telemetry sweep.",
        "cases": cases,
        "overall_pass": all(case["pass"] for case in cases),
        "coverage": {
            "shard_counts": sorted({case["config"]["shards"] for case in cases}),
            "group_max_values": sorted({case["config"]["group_max"] for case in cases}),
            "wait_ns_values": sorted({case["config"]["wait_ns"] for case in cases}),
            "client_counts": sorted({case["config"]["clients"] for case in cases}),
        },
        "negative_claim_guard": (
            "This sweep proves mounted-path telemetry coverage for group size, "
            "wait time, queue depth, shard count, and concurrent clients. It "
            "does not prove throughput improvement, fdatasync reduction, "
            "fairness, replay ordering, or Gate 0.16 closure."
        ),
    }
    write_outputs(args.out_dir, payload)
    print(json.dumps({
        "overall_pass": payload["overall_pass"],
        "json": relpath(args.out_dir / "parallel_commit_telemetry_sweep.json"),
        "coverage": payload["coverage"],
    }, indent=2, sort_keys=True))
    return 0 if payload["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
