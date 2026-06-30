#!/usr/bin/env python3
"""Gate 0.9-S4 strict versus epoch publication comparison."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_parallel_commit_epoch_path_smoke import (  # noqa: E402
    FUSE_BIN,
    ROOT,
    relpath,
    start_fuse,
    stop_fuse,
)


DEFAULT_OUT = ROOT / "artifacts" / "validation" / "publication_protocol_fault_matrix"


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


def summarize_ns(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min_ns": min(values),
        "max_ns": max(values),
        "mean_ns": sum(values) / len(values),
        "p50_ns": percentile(values, 0.50),
        "p95_ns": percentile(values, 0.95),
        "p99_ns": percentile(values, 0.99),
        "p999_ns": percentile(values, 0.999),
    }


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
    publication = [
        event for event in events
        if event.get("event") == "publication_dispatch"
    ]
    epoch_append = [
        event for event in events
        if event.get("event") == "epoch_redo_log_append"
    ]
    return {
        "path": relpath(path),
        "exists": path.exists(),
        "event_count": len(events),
        "malformed_line_count": malformed,
        "publication_events": publication,
        "epoch_append_events": epoch_append,
        "publication_count": len(publication),
        "commit_latency_ns": summarize_ns([
            int(event.get("elapsed_ns", 0) or 0)
            for event in publication
        ]),
        "sync_count_total": sum(
            int(event.get("sync_count", 0) or 0) for event in publication
        ),
        "data_fsync_count_total": sum(
            int(event.get("data_fsync_count", 0) or 0) for event in publication
        ),
        "journal_fsync_count_total": sum(
            int(event.get("journal_fsync_count", 0) or 0)
            for event in publication
        ),
        "epoch_log_fsync_count_total": sum(
            int(event.get("epoch_log_fsync_count", 0) or 0)
            for event in publication
        ),
        "modes": sorted({
            str(event.get("mode", "")) for event in publication
            if event.get("mode")
        }),
        "rc_values": [int(event.get("rc", 0) or 0) for event in publication],
        "epoch_append_group_size_max": max(
            [int(event.get("group_size", 0) or 0) for event in epoch_append],
            default=0,
        ),
        "epoch_append_sync_primitives": sorted({
            str(event.get("sync_primitive", "")) for event in epoch_append
            if event.get("sync_primitive")
        }),
}


def parse_lock_profile(path: Path) -> dict[str, Any]:
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

    summaries = [
        event for event in events
        if event.get("event") == "lock_profile_summary"
    ]
    hot_locks = {
        str(event.get("lock")): {
            "samples": int(event.get("samples", 0) or 0),
            "wait_p99_le_ns": int(event.get("wait_p99_le_ns", 0) or 0),
            "hold_p99_le_ns": int(event.get("hold_p99_le_ns", 0) or 0),
            "cond_wait_samples": int(event.get("cond_wait_samples", 0) or 0),
            "cond_wait_p99_le_ns": int(event.get("cond_wait_p99_le_ns", 0) or 0),
            "order_violations": int(event.get("order_violations", 0) or 0),
        }
        for event in summaries
        if event.get("lock")
    }
    return {
        "path": relpath(path),
        "exists": path.exists(),
        "event_count": len(events),
        "malformed_line_count": malformed,
        "lock_hold_count": sum(
            1 for event in events if event.get("event") == "lock_hold"
        ),
        "summary_count": len(summaries),
        "lock_order_violations": sum(
            item["order_violations"] for item in hot_locks.values()
        ),
        "hot_locks": hot_locks,
    }


def summarize_publication_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "publication_count": len(events),
        "commit_latency_ns": summarize_ns([
            int(event.get("elapsed_ns", 0) or 0) for event in events
        ]),
        "sync_count_total": sum(
            int(event.get("sync_count", 0) or 0) for event in events
        ),
        "data_fsync_count_total": sum(
            int(event.get("data_fsync_count", 0) or 0) for event in events
        ),
        "journal_fsync_count_total": sum(
            int(event.get("journal_fsync_count", 0) or 0) for event in events
        ),
        "epoch_log_fsync_count_total": sum(
            int(event.get("epoch_log_fsync_count", 0) or 0) for event in events
        ),
    }


def write_fsync_read(mount_dir: Path, name: str, payload: bytes) -> dict[str, Any]:
    path = mount_dir / name
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
    write_start = time.perf_counter_ns()
    try:
        written = os.write(fd, payload)
        os.fdatasync(fd)
        write_fsync_ns = time.perf_counter_ns() - write_start
        read_start = time.perf_counter_ns()
        os.lseek(fd, 0, os.SEEK_SET)
        recovered = os.read(fd, len(payload))
        read_ns = time.perf_counter_ns() - read_start
    finally:
        os.close(fd)
    return {
        "path": str(path),
        "payload_len": len(payload),
        "written": written,
        "write_fsync_ns": write_fsync_ns,
        "read_ns": read_ns,
        "matches": recovered == payload,
    }


def run_case(case_dir: Path, label: str, mode: str | None,
             repetitions: int, warmup: int,
             payload_bytes: int) -> dict[str, Any]:
    storage_dir = Path(tempfile.mkdtemp(prefix=f"aegisq_{label}_storage_"))
    mount_dir = Path(tempfile.mkdtemp(prefix=f"aegisq_{label}_mnt_"))
    case_dir.mkdir(parents=True, exist_ok=True)
    trace_path = case_dir / "publication_trace.jsonl"
    lock_profile_path = case_dir / "lock_profile.jsonl"
    if trace_path.exists():
        trace_path.unlink()
    if lock_profile_path.exists():
        lock_profile_path.unlink()
    env = {
        "PQC_PUBLICATION_TRACE_PATH": str(trace_path.resolve()),
        "PQC_LOCK_PROFILE_PATH": str(lock_profile_path.resolve()),
    }
    if mode is not None:
        env["PQC_PUBLICATION_MODE"] = mode

    fuse = None
    unmount: dict[str, Any] = {}
    measured: list[dict[str, Any]] = []
    warmups: list[dict[str, Any]] = []
    error: str | None = None
    try:
        fuse = start_fuse(storage_dir, mount_dir, case_dir,
                          f"epoch-publication-{label}", env)
        for i in range(warmup + repetitions):
            payload = (
                f"publication-comparison:{label}:{i}:".encode("ascii") +
                bytes((j % 251 for j in range(payload_bytes)))
            )[:payload_bytes]
            result = write_fsync_read(mount_dir, f"op_{i:03d}.dat", payload)
            result["op_index"] = i
            if i < warmup:
                warmups.append(result)
            else:
                measured.append(result)
    except Exception as exc:  # noqa: BLE001 - retained as artifact evidence
        error = repr(exc)
    finally:
        try:
            unmount = stop_fuse(fuse, mount_dir, case_dir)
        finally:
            shutil.rmtree(mount_dir, ignore_errors=True)
            shutil.rmtree(storage_dir, ignore_errors=True)

    trace = parse_jsonl(trace_path)
    lock_profile = parse_lock_profile(lock_profile_path)
    measured_publications = trace["publication_events"][-repetitions:]
    measured_trace = summarize_publication_events(measured_publications)
    measured_latencies = [
        int(op["write_fsync_ns"]) for op in measured if op.get("matches") is True
    ]
    total_payload_bytes = sum(int(op.get("payload_len", 0)) for op in measured)
    total_write_fsync_ns = sum(measured_latencies)
    throughput_mib_s = None
    if total_write_fsync_ns > 0:
        throughput_mib_s = (
            total_payload_bytes / (1024.0 * 1024.0) /
            (total_write_fsync_ns / 1_000_000_000.0)
        )
    pass_condition = (
        error is None and
        unmount.get("returncode") == 0 and
        len(measured) == repetitions and
        all(op.get("matches") is True for op in measured) and
        trace["publication_count"] >= warmup + repetitions and
        measured_trace["publication_count"] == repetitions and
        trace["malformed_line_count"] == 0 and
        lock_profile["exists"] is True and
        lock_profile["malformed_line_count"] == 0 and
        lock_profile["lock_order_violations"] == 0 and
        all(int(event.get("rc", 0) or 0) == 0
            for event in measured_publications)
    )
    return {
        "label": label,
        "mode": mode or "strict",
        "warmup": warmups,
        "measured_ops": measured,
        "trace": trace,
        "lock_profile": lock_profile,
        "measured_publication_events": measured_publications,
        "measured_publication_summary": measured_trace,
        "unmount": unmount,
        "error": error,
        "client_write_fsync_latency_ns": summarize_ns(measured_latencies),
        "throughput_mib_s": throughput_mib_s,
        "payload_bytes_measured": total_payload_bytes,
        "total_write_fsync_ns": total_write_fsync_ns,
        "sync_count_total": measured_trace["sync_count_total"],
        "sync_count_per_measured_op": (
            measured_trace["sync_count_total"] / repetitions if repetitions else None
        ),
        "pass": pass_condition,
    }


def run_concurrent_case(case_dir: Path, label: str, mode: str | None,
                        clients: int, payload_bytes: int,
                        epoch_group_max: int = 1,
                        epoch_group_wait_ns: int = 0) -> dict[str, Any]:
    storage_dir = Path(tempfile.mkdtemp(prefix=f"aegisq_{label}_storage_"))
    mount_dir = Path(tempfile.mkdtemp(prefix=f"aegisq_{label}_mnt_"))
    case_dir.mkdir(parents=True, exist_ok=True)
    trace_path = case_dir / "publication_trace.jsonl"
    lock_profile_path = case_dir / "lock_profile.jsonl"
    if trace_path.exists():
        trace_path.unlink()
    if lock_profile_path.exists():
        lock_profile_path.unlink()
    env = {
        "PQC_PUBLICATION_TRACE_PATH": str(trace_path.resolve()),
        "PQC_LOCK_PROFILE_PATH": str(lock_profile_path.resolve()),
    }
    if mode is not None:
        env["PQC_PUBLICATION_MODE"] = mode
    if mode == "epoch-redo-log" and epoch_group_max > 1:
        env.update({
            "PQC_EPOCH_GROUP_MAX": str(epoch_group_max),
            "PQC_EPOCH_GROUP_WAIT_NS": str(epoch_group_wait_ns),
        })

    fuse = None
    unmount: dict[str, Any] = {}
    results: list[dict[str, Any] | None] = [None] * clients
    error: str | None = None
    start_barrier = threading.Barrier(clients)

    def worker(index: int) -> None:
        payload = (
            f"publication-group:{label}:{index}:".encode("ascii") +
            bytes((j % 251 for j in range(payload_bytes)))
        )[:payload_bytes]
        try:
            start_barrier.wait(timeout=10.0)
            result = write_fsync_read(
                mount_dir, f"group_op_{index:03d}.dat", payload)
            result["op_index"] = index
            results[index] = result
        except Exception as exc:  # noqa: BLE001 - retained as artifact evidence
            results[index] = {
                "op_index": index,
                "error": repr(exc),
                "matches": False,
            }

    wall_start_ns = 0
    wall_elapsed_ns = 0
    try:
        fuse = start_fuse(storage_dir, mount_dir, case_dir,
                          f"epoch-publication-{label}", env)
        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(clients)]
        wall_start_ns = time.perf_counter_ns()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30.0)
        wall_elapsed_ns = time.perf_counter_ns() - wall_start_ns
        alive = [idx for idx, thread in enumerate(threads) if thread.is_alive()]
        if alive:
            error = f"worker timeout: {alive}"
    except Exception as exc:  # noqa: BLE001 - retained as artifact evidence
        error = repr(exc)
    finally:
        try:
            unmount = stop_fuse(fuse, mount_dir, case_dir)
        finally:
            shutil.rmtree(mount_dir, ignore_errors=True)
            shutil.rmtree(storage_dir, ignore_errors=True)

    measured = [result for result in results if result is not None]
    trace = parse_jsonl(trace_path)
    lock_profile = parse_lock_profile(lock_profile_path)
    measured_publications = trace["publication_events"][-len(measured):]
    measured_trace = summarize_publication_events(measured_publications)
    measured_latencies = [
        int(op["write_fsync_ns"]) for op in measured
        if op.get("matches") is True and "write_fsync_ns" in op
    ]
    total_payload_bytes = sum(int(op.get("payload_len", 0))
                              for op in measured)
    throughput_mib_s = None
    if wall_elapsed_ns > 0:
        throughput_mib_s = (
            total_payload_bytes / (1024.0 * 1024.0) /
            (wall_elapsed_ns / 1_000_000_000.0)
        )
    pass_condition = (
        error is None and
        unmount.get("returncode") == 0 and
        len(measured) == clients and
        all(op.get("matches") is True for op in measured) and
        trace["publication_count"] >= clients and
        measured_trace["publication_count"] == clients and
        trace["malformed_line_count"] == 0 and
        lock_profile["exists"] is True and
        lock_profile["malformed_line_count"] == 0 and
        lock_profile["lock_order_violations"] == 0 and
        all(int(event.get("rc", 0) or 0) == 0
            for event in measured_publications)
    )
    return {
        "label": label,
        "mode": mode or "strict",
        "workload_kind": "concurrent_unique_file_fdatasync",
        "client_count": clients,
        "epoch_group_max": epoch_group_max,
        "epoch_group_wait_ns": epoch_group_wait_ns,
        "measured_ops": measured,
        "trace": trace,
        "lock_profile": lock_profile,
        "measured_publication_events": measured_publications,
        "measured_publication_summary": measured_trace,
        "unmount": unmount,
        "error": error,
        "client_write_fsync_latency_ns": summarize_ns(measured_latencies),
        "throughput_mib_s": throughput_mib_s,
        "payload_bytes_measured": total_payload_bytes,
        "wall_elapsed_ns": wall_elapsed_ns,
        "sync_count_total": measured_trace["sync_count_total"],
        "sync_count_per_measured_op": (
            measured_trace["sync_count_total"] / clients if clients else None
        ),
        "epoch_append_group_size_max": trace["epoch_append_group_size_max"],
        "epoch_append_sync_primitives": trace["epoch_append_sync_primitives"],
        "pass": pass_condition,
    }


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "epoch_publication_comparison.json"
    md_path = out_dir / "epoch_publication_comparison.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    lines = [
        "# Epoch Publication Comparison",
        "",
        f"- Generated: `{payload['generated_utc']}`",
        f"- Overall pass: `{str(payload['overall_pass']).lower()}`",
        f"- Workload contract: `{payload['workload_contract']}`",
        "",
    ]
    for case in payload["cases"]:
        lines.append(f"## {case['label']}")
        lines.append(f"- Pass: `{str(case['pass']).lower()}`")
        lines.append(f"- Throughput MiB/s: `{case['throughput_mib_s']}`")
        lines.append(f"- Client p99 ns: `{case['client_write_fsync_latency_ns'].get('p99_ns')}`")
        lines.append(f"- Client p99.9 ns: `{case['client_write_fsync_latency_ns'].get('p999_ns')}`")
        lines.append(f"- Commit p99 ns: `{case['measured_publication_summary']['commit_latency_ns'].get('p99_ns')}`")
        lines.append(f"- Sync count total: `{case['sync_count_total']}`")
        lines.append(f"- Lock hold events: `{case['lock_profile'].get('lock_hold_count')}`")
        lines.append(f"- Lock-order violations: `{case['lock_profile'].get('lock_order_violations')}`")
        for lock_name in ("fd_lock", "commit_lock", "epoch_barrier_lock", "parallel_runtime_lock"):
            lock = case["lock_profile"].get("hot_locks", {}).get(lock_name)
            if lock:
                lines.append(
                    f"- {lock_name} hold p99 <= ns: `{lock.get('hold_p99_le_ns')}`"
                )
        if case.get("workload_kind"):
            lines.append(f"- Workload kind: `{case['workload_kind']}`")
            lines.append(f"- Client count: `{case.get('client_count')}`")
            lines.append(f"- Max epoch append group size: `{case.get('epoch_append_group_size_max')}`")
            lines.append(f"- Epoch sync primitives: `{case.get('epoch_append_sync_primitives')}`")
        lines.append("")
    lines.extend([
        "## Verdict",
        "",
        payload["comparison_verdict"],
        "",
        "## Negative Claim Guard",
        "",
        payload["negative_claim_guard"],
        "",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--payload-bytes", type=int, default=16384)
    parser.add_argument("--group-clients", type=int, default=4)
    parser.add_argument("--group-wait-ns", type=int, default=100_000)
    args = parser.parse_args()
    if not FUSE_BIN.exists():
        raise SystemExit(f"missing build artifact: {relpath(FUSE_BIN)}")

    strict = run_case(args.out_dir / "comparison_strict", "strict", "strict",
                      args.repetitions, args.warmup, args.payload_bytes)
    epoch = run_case(args.out_dir / "comparison_epoch_redo_log",
                     "epoch_redo_log", "epoch-redo-log",
                     args.repetitions, args.warmup, args.payload_bytes)
    strict_grouped = run_concurrent_case(
        args.out_dir / "comparison_strict_grouped",
        "strict_grouped", "strict",
        args.group_clients, args.payload_bytes,
    )
    epoch_grouped = run_concurrent_case(
        args.out_dir / "comparison_epoch_redo_log_grouped",
        "epoch_redo_log_grouped", "epoch-redo-log",
        args.group_clients, args.payload_bytes,
        epoch_group_max=args.group_clients,
        epoch_group_wait_ns=args.group_wait_ns,
    )
    strict_tput = strict.get("throughput_mib_s") or 0.0
    epoch_tput = epoch.get("throughput_mib_s") or 0.0
    strict_sync = strict.get("sync_count_total")
    epoch_sync = epoch.get("sync_count_total")
    strict_group_sync = strict_grouped.get("sync_count_total")
    epoch_group_sync = epoch_grouped.get("sync_count_total")
    strict_group_tput = strict_grouped.get("throughput_mib_s") or 0.0
    epoch_group_tput = epoch_grouped.get("throughput_mib_s") or 0.0
    strict_group_p99 = (
        strict_grouped.get("client_write_fsync_latency_ns", {}).get("p99_ns")
        or 0.0
    )
    epoch_group_p99 = (
        epoch_grouped.get("client_write_fsync_latency_ns", {}).get("p99_ns")
        or 0.0
    )
    if (epoch_grouped.get("pass") is True and
            strict_grouped.get("pass") is True and
            epoch_group_sync is not None and
            strict_group_sync is not None and
            epoch_group_sync < strict_group_sync):
        throughput_word = (
            "higher" if epoch_group_tput > strict_group_tput
            else "lower_or_equal"
        )
        p99_word = (
            "lower" if epoch_group_p99 < strict_group_p99
            else "higher_or_equal"
        )
        verdict = (
            "Grouped epoch redo-log amortized the metadata barrier in the "
            "concurrent mounted workload: total traced sync count is lower "
            "than strict while journal fsync remains off the foreground epoch "
            f"path. Throughput is {throughput_word} and client p99 is "
            f"{p99_word} versus strict in this run, so claims must be scoped "
            "to the metrics that actually improve."
        )
    elif epoch_tput > strict_tput and epoch_sync <= strict_sync:
        verdict = (
            "Epoch redo-log improved throughput without increasing the traced "
            "publication sync count in this narrow workload."
        )
    else:
        verdict = (
            "Epoch redo-log did not yet demonstrate the desired amortization: "
            "the current implementation removes the strict journal fdatasync "
            "from the foreground write path but replaces it with an epoch-log "
            "barrier, so it remains a correctness scaffold rather than a "
            "group-commit fast path."
        )
    payload = {
        "schema_version": 1,
        "generated_by": "experiments/run_epoch_publication_comparison.py",
        "generated_utc": now_utc(),
        "scope": "Gate 0.9-S4 strict versus epoch publication measurement.",
        "workload_contract": {
            "repetitions": args.repetitions,
            "warmup": args.warmup,
            "payload_bytes": args.payload_bytes,
            "operation": "unique-file write + fdatasync + readback",
            "mount_options": "-f foreground FUSE via helper",
            "cache_state": "not dropped; same mounted-path contract for both modes",
            "modes": ["strict", "epoch-redo-log"],
            "group_clients": args.group_clients,
            "group_wait_ns": args.group_wait_ns,
        },
        "cases": [strict, epoch, strict_grouped, epoch_grouped],
        "overall_pass": (
            strict["pass"] and epoch["pass"] and
            strict_grouped["pass"] and epoch_grouped["pass"]
        ),
        "comparison_verdict": verdict,
        "negative_claim_guard": (
            "This artifact is a narrow mounted-path comparison. It does not "
            "support any claim that epoch mode reduces total sync count, "
            "improves throughput, improves p99 latency, or is SOSP-ready "
            "unless the reported metrics show that result under the stated "
            "workload. A zero journal-fsync count only supports the narrower "
            "claim that strict journal publication has moved out of the "
            "foreground epoch-mode write path."
        ),
    }
    write_outputs(args.out_dir, payload)
    print(json.dumps({
        "overall_pass": payload["overall_pass"],
        "json": relpath(args.out_dir / "epoch_publication_comparison.json"),
        "strict_throughput_mib_s": strict.get("throughput_mib_s"),
        "epoch_throughput_mib_s": epoch.get("throughput_mib_s"),
        "strict_sync_count": strict_sync,
        "epoch_sync_count": epoch_sync,
        "strict_grouped_throughput_mib_s": strict_group_tput,
        "epoch_grouped_throughput_mib_s": epoch_group_tput,
        "strict_grouped_sync_count": strict_group_sync,
        "epoch_grouped_sync_count": epoch_group_sync,
        "strict_grouped_p99_ms": strict_group_p99 / 1_000_000.0,
        "epoch_grouped_p99_ms": epoch_group_p99 / 1_000_000.0,
    }, indent=2, sort_keys=True))
    return 0 if payload["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
