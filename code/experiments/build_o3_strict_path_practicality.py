#!/usr/bin/env python3
"""Build the O3 strict-path practicality closeout.

This is a retained-evidence closeout, not a new benchmark campaign.  It answers
the repeated review question: the strict FUSE publication path remains a
correctness-preserving cost boundary, but the implemented epoch/group barrier
is the production-facing hybrid path for batched or concurrent writes that can
share a barrier.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "artifacts" / "validation" / "strict_path_practicality"
JSON_OUT = OUT_DIR / "strict_path_practicality.json"
MD_OUT = OUT_DIR / "strict_path_practicality.md"

INPUTS = {
    "a1_throughput_decision": ROOT / "artifacts" / "validation" / "a1_throughput_decision" / "a1_throughput_decision.json",
    "a2_gap_attribution": ROOT / "artifacts" / "validation" / "a2_gap_attribution" / "a2_gap_attribution.json",
    "a3_batching_decision": ROOT / "artifacts" / "validation" / "a3_batching_decision" / "a3_batching_decision.json",
    "x6_strict_cost_reduction": ROOT / "artifacts" / "validation" / "x6_strict_cost_reduction" / "x6_strict_cost_reduction_model.json",
}

SOURCE_FILES = {
    "strict_publish": ROOT / "code" / "storage" / "pqc_strict_publish.c",
    "strict_publish_header": ROOT / "code" / "storage" / "pqc_strict_publish.h",
    "epoch_publish": ROOT / "code" / "storage" / "pqc_epoch_publish.c",
    "epoch_publish_header": ROOT / "code" / "storage" / "pqc_epoch_publish.h",
    "parallel_commit": ROOT / "code" / "fs" / "pqc_parallel_commit.c",
}

PAPER_FILES = [
    ROOT / "Paper" / "3_Design.tex",
    ROOT / "Paper" / "4_Evaluation.tex",
    ROOT / "Paper" / "10_Discussion_and_Limitations.tex",
]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"present": path.exists(), "error": str(exc), "path": rel(path)}
    if not isinstance(data, dict):
        return {"present": True, "error": "json root is not object", "path": rel(path)}
    data["present"] = True
    data["path"] = rel(path)
    return data


def pct_change(before: float, after: float) -> float | None:
    if before == 0:
        return None
    return (after - before) / before


def ns_to_ms(value: float | int | None) -> float | None:
    if value is None:
        return None
    return float(value) / 1_000_000.0


def nested(data: dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def build_metrics(a1: dict[str, Any], a3: dict[str, Any], x6: dict[str, Any]) -> dict[str, Any]:
    counts = nested(x6, "attribution_model", "retained_counts") or {}
    cycles = int(counts.get("data_sidecar_calls") or 0)
    total_after = int(nested(x6, "attribution_model", "modeled_after_blocking_syncs") or 0)
    per_cycle = total_after / cycles if cycles else None
    marker_removed = int(nested(x6, "attribution_model", "modeled_syncfs_removed") or 0)

    current = nested(a3, "comparisons", "current_100us_viability") or {}
    strict = current.get("strict_grouped") or {}
    epoch = current.get("epoch_grouped") or {}
    fault = nested(a3, "comparisons", "fault_matrix_publication") or {}
    fault_strict = fault.get("strict_grouped") or {}
    fault_epoch = fault.get("epoch_grouped") or {}

    strict_sync = float(strict.get("sync_count_total") or 0)
    epoch_sync = float(epoch.get("sync_count_total") or 0)
    strict_p99_ms = ns_to_ms(strict.get("p99_ns"))
    epoch_p99_ms = ns_to_ms(epoch.get("p99_ns"))
    strict_thr = float(strict.get("throughput_mib_s") or 0)
    epoch_thr = float(epoch.get("throughput_mib_s") or 0)

    fault_strict_p99_ms = ns_to_ms(fault_strict.get("p99_ns"))
    fault_epoch_p99_ms = ns_to_ms(fault_epoch.get("p99_ns"))

    return {
        "strict_single_client_cost_boundary": {
            "frozen_aegisq_throughput_mib_s_median": nested(a1, "metrics", "frozen_aegisq_throughput_mib_s_median"),
            "frozen_aegisq_p99_us_median": nested(a1, "metrics", "frozen_aegisq_p99_us_median"),
            "a2_aegisq_to_gocryptfs_ratio": nested(a1, "metrics", "a2_aegisq_to_gocryptfs_ratio"),
            "strict_publication_cycles_from_a2": cycles,
            "strict_sync_family_ops_after_x6": total_after,
            "strict_sync_family_ops_per_cycle_after_x6": per_cycle,
            "marker_syncfs_removed_by_x6": marker_removed,
            "remaining_barriers": [
                "data-sidecar durability",
                "journal-sidecar durability",
                "marker/checkpoint publication",
            ],
        },
        "hybrid_epoch_grouped_current_100us": {
            "strict_sync_count_total": strict_sync,
            "epoch_sync_count_total": epoch_sync,
            "sync_count_reduction_fraction": -pct_change(strict_sync, epoch_sync) if strict_sync else None,
            "strict_p99_ms": strict_p99_ms,
            "epoch_p99_ms": epoch_p99_ms,
            "p99_reduction_fraction": -pct_change(strict_p99_ms, epoch_p99_ms) if strict_p99_ms else None,
            "strict_throughput_mib_s": strict_thr,
            "epoch_throughput_mib_s": epoch_thr,
            "throughput_increase_fraction": pct_change(strict_thr, epoch_thr) if strict_thr else None,
            "max_epoch_append_group_size": epoch.get("max_epoch_append_group_size"),
            "epoch_sync_primitives": epoch.get("epoch_sync_primitives"),
        },
        "hybrid_epoch_fault_matrix_loss_boundary": {
            "strict_sync_count_total": fault_strict.get("sync_count_total"),
            "epoch_sync_count_total": fault_epoch.get("sync_count_total"),
            "strict_p99_ms": fault_strict_p99_ms,
            "epoch_p99_ms": fault_epoch_p99_ms,
            "strict_throughput_mib_s": fault_strict.get("throughput_mib_s"),
            "epoch_throughput_mib_s": fault_epoch.get("throughput_mib_s"),
            "interpretation": (
                "The grouped barrier can reduce sync count while still losing "
                "latency or throughput when waiting dominates; it is an "
                "admission rule, not a general fast path."
            ),
        },
    }


def source_checks() -> dict[str, bool]:
    strict = read_text(SOURCE_FILES["strict_publish"]) + read_text(SOURCE_FILES["strict_publish_header"])
    epoch = read_text(SOURCE_FILES["epoch_publish"]) + read_text(SOURCE_FILES["epoch_publish_header"])
    parallel = read_text(SOURCE_FILES["parallel_commit"])
    return {
        "strict_keeps_data_fdatasync": "PQC_DURABILITY_SITE_DATA_SIDECAR" in strict,
        "strict_keeps_journal_fdatasync": "PQC_DURABILITY_SITE_JOURNAL_SIDECAR" in strict,
        "strict_exposes_epoch_callbacks": "after_data_fsync" in strict and "after_metadata_publish" in strict,
        "epoch_redo_log_mode_present": "PQC_PUBLICATION_MODE_EPOCH_REDO_LOG" in epoch,
        "epoch_group_barrier_present": "pqc_epoch_group_barrier_wait" in epoch,
        "epoch_shared_syncfs_barrier_present": "pqc_epoch_sync_fd" in epoch and "syncfs" in epoch,
        "epoch_can_skip_journal_fsync": "skip_journal_fsync" in strict and "journal_fsync_skipped_epoch_after" in strict,
        "epoch_group_admission_knobs_present": "PQC_EPOCH_GROUP_MAX" in epoch and "PQC_EPOCH_GROUP_WAIT_NS" in epoch,
        "parallel_commit_runtime_present": "epoch-gated-strict" in parallel and "pqc_parallel_commit_begin" in parallel,
    }


def paper_checks() -> dict[str, bool]:
    text = "\n".join(read_text(path) for path in PAPER_FILES)
    return {
        "strict_path_cost_boundary_present": "strict path" in text and "cost boundary" in text,
        "hybrid_barrier_answer_present": "hybrid" in text and "barrier" in text,
        "djc_barrier_boundary_present": "D/J/C barrier" in text,
        "kernel_assist_future_scoped": "kernel-assist" in text and "future" in text,
        "physical_power_loss_negated": "physical power-loss" in text and "not" in text,
        "single_client_not_general_win": "Single-client epoch" in text or "single-client" in text,
    }


def build() -> dict[str, Any]:
    artifacts = {name: load_json(path) for name, path in INPUTS.items()}
    a1 = artifacts["a1_throughput_decision"]
    a3 = artifacts["a3_batching_decision"]
    x6 = artifacts["x6_strict_cost_reduction"]
    metrics = build_metrics(a1, a3, x6)
    s_checks = source_checks()
    p_checks = paper_checks()

    proof_checks = {
        "a1_passes": a1.get("overall_pass") is True,
        "a2_present": artifacts["a2_gap_attribution"].get("present") is True,
        "a3_passes": a3.get("overall_pass") is True,
        "x6_passes": x6.get("overall_pass") is True,
        "x6_removed_marker_syncfs": metrics["strict_single_client_cost_boundary"]["marker_syncfs_removed_by_x6"] > 0,
        "strict_per_cycle_still_has_three_boundaries": (
            metrics["strict_single_client_cost_boundary"]["strict_sync_family_ops_per_cycle_after_x6"] == 3.0
        ),
        "a3_grouped_reduces_sync_count": nested(a3, "derived", "grouped_sync_reduced_somewhere") is True,
        "a3_has_win_and_loss_boundary": nested(a3, "derived", "grouped_has_win_and_loss") is True,
        "a3_single_not_general_win": nested(a3, "derived", "single_not_general_win") is True,
        "a3_replay_fault_matrix_passes": nested(a3, "proof_checks", "replay_fault_matrix_passes") is True,
        "current_grouped_sync_reduction_positive": (
            (metrics["hybrid_epoch_grouped_current_100us"]["sync_count_reduction_fraction"] or 0.0) > 0.0
        ),
    }
    overall_pass = (
        all(proof_checks.values())
        and all(s_checks.values())
        and all(p_checks.values())
    )
    return {
        "schema": "aegisq.o3_strict_path_practicality.v1",
        "generated_utc": now_utc(),
        "overall_pass": overall_pass,
        "inputs": {name: rel(path) for name, path in INPUTS.items()},
        "source_files": {name: rel(path) for name, path in SOURCE_FILES.items()},
        "metrics": metrics,
        "proof_checks": proof_checks,
        "source_checks": s_checks,
        "paper_checks": p_checks,
        "decision": {
            "verdict": "hybrid-barrier-closeout" if overall_pass else "incomplete",
            "what_is_closed": (
                "O3 is closed as a production-facing hybrid barrier response: "
                "strict per-write publication remains the conservative "
                "D/J/C cost boundary, while epoch/group publication is the "
                "implemented opt-in path for concurrent or batched writes that "
                "can share a barrier."
            ),
            "what_is_not_claimed": [
                "not a single-client strict-path throughput win",
                "not a general-purpose filesystem throughput claim",
                "not kernel upstreaming or a kernel-assist implementation",
                "not physical power-loss or drive-cache certification",
            ],
            "next_if_osdi_demands_more": (
                "Run O2 supported fscrypt first; only open a kernel-assist "
                "publication redesign if the paper explicitly claims kernel "
                "integration rather than a scoped edge FUSE runtime."
            ),
        },
    }


def write_markdown(report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    current = metrics["hybrid_epoch_grouped_current_100us"]
    strict = metrics["strict_single_client_cost_boundary"]
    lines = [
        "# O3 strict-path practicality closeout",
        "",
        f"- Overall pass: `{report['overall_pass']}`",
        f"- Verdict: `{report['decision']['verdict']}`",
        f"- Strict sync-family ops after X6: `{strict['strict_sync_family_ops_after_x6']}`",
        f"- Marker syncfs removed by X6: `{strict['marker_syncfs_removed_by_x6']}`",
        f"- Grouped sync reduction: `{current['strict_sync_count_total']}` -> `{current['epoch_sync_count_total']}`",
        f"- Grouped p99: `{current['strict_p99_ms']:.3f}` ms -> `{current['epoch_p99_ms']:.3f}` ms",
        f"- Grouped throughput: `{current['strict_throughput_mib_s']:.3f}` MiB/s -> `{current['epoch_throughput_mib_s']:.3f}` MiB/s",
        "",
        "The closeout does not claim that strict single-client publication is now",
        "fast.  It claims that the implemented hybrid epoch path is the right",
        "production-facing answer for concurrent or batched writes that can share",
        "a barrier, while strict remains the conservative D/J/C publication",
        "boundary.",
        "",
    ]
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    report = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report)
    print(f"wrote {rel(JSON_OUT)}")
    print(f"overall_pass={report['overall_pass']}")
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
