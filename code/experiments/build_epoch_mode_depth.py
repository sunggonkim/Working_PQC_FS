#!/usr/bin/env python3
"""Build the R5 epoch-mode depth closeout artifact.

The artifact is intentionally a claim guard as much as a summary: epoch mode
must show both what improved and what regressed under multi-writer pressure.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PUBLICATION = (
    ROOT / "artifacts" / "validation" / "epoch_mode_depth" /
    "publication_depth" / "epoch_publication_comparison.json"
)
DEFAULT_REPLAY = (
    ROOT / "artifacts" / "validation" / "parallel_commit_contract" /
    "parallel_commit_fairness_replay.json"
)
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "epoch_mode_depth"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def case_by_label(data: dict[str, Any], label: str) -> dict[str, Any]:
    for case in data.get("cases", []):
        if case.get("label") == label:
            return case
    return {}


def metric(case: dict[str, Any], key: str) -> float | None:
    value = (case.get("client_write_fsync_latency_ns") or {}).get(key)
    return float(value) if value is not None else None


def throughput(case: dict[str, Any]) -> float | None:
    value = case.get("throughput_mib_s")
    return float(value) if value is not None else None


def syncs(case: dict[str, Any]) -> int | None:
    value = case.get("sync_count_total")
    return int(value) if value is not None else None


def replay_summary(data: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for case in data.get("cases", []):
        replay = case.get("replay_order") or {}
        rows.append({
            "name": case.get("name"),
            "pass": bool(case.get("pass")),
            "clients": (case.get("config") or {}).get("clients"),
            "completed_ops": case.get("completed_ops"),
            "replay_plan_length": replay.get("replay_plan_length"),
            "reconstruction_time_ns": replay.get("reconstruction_time_ns"),
            "observed_shards": replay.get("observed_shards"),
        })
    return {
        "overall_pass": bool(data.get("overall_pass")),
        "coverage": data.get("coverage") or {},
        "rows": rows,
        "max_clients": max((int(row.get("clients") or 0) for row in rows), default=0),
        "max_replay_time_ns": max(
            (int(row.get("reconstruction_time_ns") or 0) for row in rows),
            default=0,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publication", type=Path, default=DEFAULT_PUBLICATION)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    publication = load_json(args.publication)
    replay = load_json(args.replay)
    strict = case_by_label(publication, "strict_grouped")
    epoch = case_by_label(publication, "epoch_redo_log_grouped")
    replay_info = replay_summary(replay)

    strict_p99 = metric(strict, "p99_ns")
    strict_p999 = metric(strict, "p999_ns")
    epoch_p99 = metric(epoch, "p99_ns")
    epoch_p999 = metric(epoch, "p999_ns")
    strict_sync = syncs(strict)
    epoch_sync = syncs(epoch)
    strict_tput = throughput(strict)
    epoch_tput = throughput(epoch)
    client_count = int(epoch.get("client_count") or 0)
    max_group_size = int(epoch.get("epoch_append_group_size_max") or 0)

    sync_win = (
        strict_sync is not None and epoch_sync is not None
        and epoch_sync < strict_sync
    )
    latency_loss = (
        strict_p99 is not None and epoch_p99 is not None
        and epoch_p99 >= strict_p99
    )
    throughput_loss = (
        strict_tput is not None and epoch_tput is not None
        and epoch_tput <= strict_tput
    )
    p999_present = strict_p999 is not None and epoch_p999 is not None

    overall_pass = (
        bool(publication.get("overall_pass"))
        and bool(replay.get("overall_pass"))
        and client_count >= 8
        and max_group_size >= 8
        and p999_present
        and sync_win
        and (latency_loss or throughput_loss)
        and replay_info["max_clients"] >= 8
        and replay_info["max_replay_time_ns"] > 0
    )
    report = {
        "artifact": "epoch_mode_depth",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": overall_pass,
        "source_artifacts": {
            "publication_depth": rel(args.publication),
            "fairness_replay": rel(args.replay),
        },
        "depth_comparison": {
            "client_count": client_count,
            "strict_grouped": {
                "p99_ms": strict_p99 / 1_000_000.0 if strict_p99 is not None else None,
                "p999_ms": strict_p999 / 1_000_000.0 if strict_p999 is not None else None,
                "sync_count": strict_sync,
                "throughput_mib_s": strict_tput,
            },
            "epoch_grouped": {
                "p99_ms": epoch_p99 / 1_000_000.0 if epoch_p99 is not None else None,
                "p999_ms": epoch_p999 / 1_000_000.0 if epoch_p999 is not None else None,
                "sync_count": epoch_sync,
                "throughput_mib_s": epoch_tput,
                "max_group_size": max_group_size,
                "sync_primitives": epoch.get("epoch_append_sync_primitives") or [],
            },
            "sync_win": sync_win,
            "latency_loss": latency_loss,
            "throughput_loss": throughput_loss,
        },
        "replay_evidence": replay_info,
        "verdict": (
            "At eight concurrent writers, epoch-redo-log grouping reduces "
            f"traced publication syncs from {strict_sync} to {epoch_sync} and "
            f"forms an {max_group_size}-writer group, but the client tail does "
            f"not improve: p99.9 is {epoch_p999 / 1_000_000.0:.2f} ms for "
            f"epoch versus {strict_p999 / 1_000_000.0:.2f} ms for strict. "
            "The replay-order artifact reconstructs committed epochs in "
            f"{replay_info['max_replay_time_ns']} ns at up to "
            f"{replay_info['max_clients']} clients. This supports an "
            "amortization mechanism with explicit tail-latency loss cases, "
            "not a universal fast path."
        ),
        "claim_guard": (
            "Paper text may claim only that higher-depth epoch grouping "
            "amortizes sync count and has replay-order evidence. It must also "
            "state the observed p99/p99.9 or throughput loss and must not claim "
            "that epoch mode universally improves latency, throughput, or "
            "general filesystem scalability."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "epoch_mode_depth.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    print(json.dumps({
        "overall_pass": overall_pass,
        "json": rel(json_path),
        "sync_win": sync_win,
        "latency_loss": latency_loss,
        "throughput_loss": throughput_loss,
        "max_replay_time_ns": replay_info["max_replay_time_ns"],
    }, indent=2, sort_keys=True))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
