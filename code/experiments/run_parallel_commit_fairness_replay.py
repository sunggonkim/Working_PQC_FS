#!/usr/bin/env python3
"""Fairness/starvation and trace replay-order evidence for Gate 0.16-S4.

This runner exercises the production mounted path with explicit
``epoch-gated-strict`` parallel-commit mode.  It retains two kinds of evidence:

* a bounded starvation negative test where every client must complete all
  writes through the same commit shard; and
* a trace-level replay-order reconstruction that checks per-shard epochs are
  uniquely completed in increasing order and records reconstruction time.

The replay evidence is deliberately trace-level.  It is not redo-log recovery
evidence and it does not close Gate 0.9 or Gate 0.16.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_parallel_commit_telemetry_sweep import (
    DEFAULT_OUT,
    ROOT,
    client_worker,
    load_client_results,
    relpath,
    start_fuse,
    stop_fuse,
    summarize_numbers,
)


CASES = [
    {
        "name": "starvation_same_shard",
        "purpose": "bounded starvation negative test under one shared commit shard",
        "shards": 1,
        "group_max": 4,
        "wait_ns": 10_000_000,
        "clients": 8,
        "ops_per_client": 4,
        "payload_size": 4096,
        "join_timeout_s": 45,
    },
    {
        "name": "replay_order_multi_shard",
        "purpose": "per-shard epoch replay-order reconstruction",
        "shards": 4,
        "group_max": 2,
        "wait_ns": 5_000_000,
        "clients": 4,
        "ops_per_client": 3,
        "payload_size": 4096,
        "join_timeout_s": 45,
    },
]


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def latency_fairness(client_results: list[dict[str, Any]]) -> dict[str, Any]:
    per_client = []
    for result in client_results:
        latencies = [
            int(op.get("latency_ns", 0) or 0)
            for op in result.get("ops", [])
        ]
        per_client.append({
            "client_id": result.get("client_id"),
            "op_count": len(result.get("ops", [])),
            "pass": result.get("pass"),
            "error": result.get("error"),
            "latency_ns_summary": summarize_numbers(latencies),
        })
    op_counts = [int(item["op_count"]) for item in per_client]
    max_latencies = [
        int(item["latency_ns_summary"].get("max", 0) or 0)
        for item in per_client
        if item["latency_ns_summary"].get("count", 0)
    ]
    return {
        "per_client": per_client,
        "op_count_distribution": dict(sorted(Counter(op_counts).items())),
        "min_ops": min(op_counts) if op_counts else 0,
        "max_ops": max(op_counts) if op_counts else 0,
        "client_max_latency_ns_summary": summarize_numbers(max_latencies),
        "max_to_min_ops_equal": len(set(op_counts)) == 1 if op_counts else False,
    }


def load_trace(trace_path: Path) -> tuple[list[dict[str, Any]], int]:
    events: list[dict[str, Any]] = []
    malformed = 0
    if trace_path.exists():
        for line_no, line in enumerate(
            trace_path.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(payload, dict):
                payload["_line_no"] = line_no
                events.append(payload)
    return events, malformed


def reconstruct_replay_order(trace_path: Path) -> dict[str, Any]:
    start_ns = time.perf_counter_ns()
    events, malformed = load_trace(trace_path)
    begin_by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    finish_by_shard: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("event") not in {"begin", "finish"}:
            continue
        shard = int(event.get("shard", 0) or 0)
        epoch = int(event.get("epoch", 0) or 0)
        if event.get("event") == "begin":
            begin_by_key[(shard, epoch)].append(event)
        elif event.get("event") == "finish":
            finish_by_shard[shard].append(event)

    replay_plan = []
    duplicate_finishes = []
    missing_begin_epochs = []
    non_monotonic_shards = []
    for shard, finishes in sorted(finish_by_shard.items()):
        ordered = sorted(finishes, key=lambda event: int(event.get("_line_no", 0)))
        epochs = [int(event.get("epoch", 0) or 0) for event in ordered]
        if len(epochs) != len(set(epochs)):
            duplicate_finishes.append(shard)
        if epochs != sorted(epochs):
            non_monotonic_shards.append(shard)
        for epoch in epochs:
            key = (shard, epoch)
            begins = begin_by_key.get(key, [])
            if not begins:
                missing_begin_epochs.append({"shard": shard, "epoch": epoch})
            replay_plan.append({
                "shard": shard,
                "epoch": epoch,
                "begin_count": len(begins),
                "finish_line": next(
                    int(event.get("_line_no", 0) or 0)
                    for event in ordered
                    if int(event.get("epoch", 0) or 0) == epoch
                ),
            })
    elapsed_ns = time.perf_counter_ns() - start_ns
    finish_count = sum(len(values) for values in finish_by_shard.values())
    begin_count = sum(len(values) for values in begin_by_key.values())
    return {
        "trace_path": relpath(trace_path),
        "trace_exists": trace_path.exists(),
        "event_count": len(events),
        "malformed_line_count": malformed,
        "begin_count": begin_count,
        "finish_count": finish_count,
        "observed_shards": sorted(finish_by_shard.keys()),
        "replay_plan": replay_plan,
        "replay_plan_length": len(replay_plan),
        "reconstruction_time_ns": elapsed_ns,
        "duplicate_finish_shards": duplicate_finishes,
        "non_monotonic_shards": non_monotonic_shards,
        "missing_begin_epochs": missing_begin_epochs,
        "pass": (
            trace_path.exists() and
            malformed == 0 and
            finish_count > 0 and
            not duplicate_finishes and
            not non_monotonic_shards and
            not missing_begin_epochs
        ),
    }


def run_case(out_dir: Path, case: dict[str, Any]) -> dict[str, Any]:
    case_dir = out_dir / "fairness_replay" / case["name"]
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    result_dir = case_dir / "client_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    storage_dir = Path(tempfile.mkdtemp(prefix=f"aegisq_{case['name']}_storage_"))
    mount_dir = Path(tempfile.mkdtemp(prefix=f"aegisq_{case['name']}_mnt_"))
    trace_path = case_dir / "parallel_commit_trace.jsonl"
    env = {
        "PQC_PARALLEL_COMMIT_MODE": "epoch-gated-strict",
        "PQC_PARALLEL_COMMIT_SHARDS": str(case["shards"]),
        "PQC_PARALLEL_COMMIT_GROUP_MAX": str(case["group_max"]),
        "PQC_PARALLEL_COMMIT_WAIT_NS": str(case["wait_ns"]),
        "PQC_PARALLEL_COMMIT_TRACE_PATH": str(trace_path),
    }

    fuse = None
    unmount: dict[str, Any] = {}
    error: str | None = None
    processes: list[mp.Process] = []
    try:
        fuse = start_fuse(storage_dir, mount_dir, case_dir,
                          f"parallel-commit-s4-{case['name']}", env)
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
            proc.join(timeout=float(case["join_timeout_s"]))
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
    fairness = latency_fairness(client_results)
    replay = reconstruct_replay_order(trace_path)
    expected_ops = int(case["clients"]) * int(case["ops_per_client"])
    worker_exitcodes = [proc.exitcode for proc in processes]
    completed_ops = sum(len(result.get("ops", [])) for result in client_results)
    starvation_pass = (
        error is None and
        unmount.get("returncode") == 0 and
        all(code == 0 for code in worker_exitcodes) and
        all(result.get("pass") is True for result in client_results) and
        completed_ops == expected_ops and
        fairness["min_ops"] == int(case["ops_per_client"]) and
        fairness["max_ops"] == int(case["ops_per_client"])
    )
    return {
        "name": case["name"],
        "purpose": case["purpose"],
        "config": case,
        "worker_exitcodes": worker_exitcodes,
        "expected_ops": expected_ops,
        "completed_ops": completed_ops,
        "client_results": client_results,
        "fairness": fairness,
        "replay_order": replay,
        "unmount": unmount,
        "error": error,
        "starvation_negative_pass": starvation_pass,
        "replay_order_pass": replay["pass"],
        "pass": starvation_pass and replay["pass"],
    }


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "parallel_commit_fairness_replay.json"
    md_path = out_dir / "parallel_commit_fairness_replay.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    lines = [
        "# Parallel Commit Fairness and Replay-Order Evidence",
        "",
        f"- Generated: `{payload['generated_utc']}`",
        f"- Overall pass: `{str(payload['overall_pass']).lower()}`",
        "",
    ]
    for case in payload["cases"]:
        lines.append(f"## {case['name']}")
        lines.append(f"- Purpose: {case['purpose']}")
        lines.append(f"- Pass: `{str(case['pass']).lower()}`")
        lines.append(f"- Starvation negative pass: `{str(case['starvation_negative_pass']).lower()}`")
        lines.append(f"- Replay-order pass: `{str(case['replay_order_pass']).lower()}`")
        lines.append(f"- Completed ops: `{case['completed_ops']}` / `{case['expected_ops']}`")
        lines.append(f"- Op count distribution: `{case['fairness']['op_count_distribution']}`")
        lines.append(f"- Replay reconstruction time ns: `{case['replay_order']['reconstruction_time_ns']}`")
        lines.append(f"- Replay plan length: `{case['replay_order']['replay_plan_length']}`")
        lines.append(f"- Observed replay shards: `{case['replay_order']['observed_shards']}`")
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
    cases = [run_case(args.out_dir, case) for case in CASES]
    payload = {
        "schema_version": 1,
        "generated_by": "experiments/run_parallel_commit_fairness_replay.py",
        "generated_utc": now_utc(),
        "scope": "Gate 0.16-S4 fairness/starvation and trace replay-order evidence.",
        "cases": cases,
        "overall_pass": all(case["pass"] for case in cases),
        "coverage": {
            "case_names": [case["name"] for case in cases],
            "client_counts": sorted({case["config"]["clients"] for case in cases}),
            "shard_counts": sorted({case["config"]["shards"] for case in cases}),
            "group_max_values": sorted({case["config"]["group_max"] for case in cases}),
            "wait_ns_values": sorted({case["config"]["wait_ns"] for case in cases}),
        },
        "negative_claim_guard": (
            "This artifact proves bounded mounted-path starvation-negative "
            "behavior and trace-level per-shard epoch replay-order "
            "reconstruction for the current coordinator. It does not prove "
            "redo-log recovery, crash replay, fdatasync reduction, throughput "
            "improvement, or Gate 0.16 closure."
        ),
    }
    write_outputs(args.out_dir, payload)
    print(json.dumps({
        "overall_pass": payload["overall_pass"],
        "json": relpath(args.out_dir / "parallel_commit_fairness_replay.json"),
        "coverage": payload["coverage"],
    }, indent=2, sort_keys=True))
    return 0 if payload["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
