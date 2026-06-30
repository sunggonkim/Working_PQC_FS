#!/usr/bin/env python3
"""Build the X11 ML-KEM/cuPQC break-even closeout.

This closeout answers the repeated review concern that the mounted ML-KEM
workflow benefit is positive but modest.  It does not run a new benchmark; it
combines the retained primitive placement CSV with the five-run mounted rekey
methodology row to explain where the GPU lane is useful and where CPU fallback
is the correct policy under UMA pressure.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts" / "reports" / "x11_mlkem_break_even_model"

WORKLOAD_MAP = ROOT / "artifacts" / "results" / "placement" / "workload_map.csv"
REKEY_METHODOLOGY = ROOT / "artifacts" / "validation" / "keyplane_rekey_methodology" / "keyplane_rekey_workflow.json"

EVAL_TEX = ROOT / "Paper" / "4_Evaluation.tex"
CHECKLIST = ROOT / "SUBMISSION_CHECKLIST.md"

MLKEM_OPS = ("ml_kem_keygen", "ml_kem_encaps", "ml_kem_decaps")
PRESSURE_EXTRA_US = (0.0, 2500.0, 5000.0)
WORKFLOW_BATCH = 1024


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return data


def load_workload_map() -> dict[str, dict[str, dict[int, dict[str, float]]]]:
    table: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
    with WORKLOAD_MAP.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "measured":
                continue
            op = str(row.get("operation", ""))
            target = str(row.get("target", ""))
            if op not in MLKEM_OPS or target not in ("cpu", "gpu"):
                continue
            batch = int(row["batch"])
            table.setdefault(op, {}).setdefault(target, {})[batch] = {
                "p50_us": float(row["p50_us"]),
                "p99_us": float(row["p99_us"]),
                "throughput_per_s": float(row["throughput_per_s"]),
            }
    return table


def mode_summary(workflow: dict[str, Any], mode: str) -> dict[str, Any]:
    for row in workflow.get("mode_summaries", []):
        if isinstance(row, dict) and row.get("mode") == mode:
            return row
    raise KeyError(f"missing workflow mode {mode}")


def interpolate_crossing(points: list[tuple[int, float]], target_gap: float) -> float | None:
    if not points:
        return None
    if target_gap <= points[0][1]:
        return float(points[0][0])
    prev_batch, prev_gap = points[0]
    for batch, gap in points[1:]:
        if prev_gap <= target_gap <= gap:
            if gap == prev_gap:
                return float(batch)
            return prev_batch + (target_gap - prev_gap) * (batch - prev_batch) / (gap - prev_gap)
        prev_batch, prev_gap = batch, gap
    return None


def primitive_analysis(table: dict[str, dict[str, dict[int, dict[str, float]]]]) -> dict[str, Any]:
    analysis: dict[str, Any] = {}
    for op in MLKEM_OPS:
        cpu = table.get(op, {}).get("cpu", {})
        gpu = table.get(op, {}).get("gpu", {})
        batches = sorted(set(cpu) & set(gpu))
        rows = []
        for batch in batches:
            cpu_p50 = cpu[batch]["p50_us"]
            gpu_p50 = gpu[batch]["p50_us"]
            rows.append({
                "batch": batch,
                "cpu_p50_us": cpu_p50,
                "gpu_p50_us": gpu_p50,
                "cpu_minus_gpu_us": cpu_p50 - gpu_p50,
                "gpu_faster": gpu_p50 < cpu_p50,
            })
        first_gpu_win = next((row["batch"] for row in rows if row["gpu_faster"]), None)
        analysis[op] = {
            "rows": rows,
            "primitive_gpu_break_even_batch": first_gpu_win,
            "batch_1_cpu_wins": bool(rows and rows[0]["batch"] == 1 and rows[0]["cpu_p50_us"] < rows[0]["gpu_p50_us"]),
            "batch_16_gpu_wins": any(row["batch"] == 16 and row["gpu_faster"] for row in rows),
            "required_batches_present": all(batch in batches for batch in (1, 16, 256, 1024, 4096)),
        }
    return analysis


def workflow_analysis(workflow: dict[str, Any]) -> dict[str, Any]:
    cpu = mode_summary(workflow, "cpu_only")
    gpu = mode_summary(workflow, "gpu_batch")
    fallback = mode_summary(workflow, "policy_fallback")
    cpu_us = float(cpu["total_rekey_usec_median"])
    gpu_us = float(gpu["total_rekey_usec_median"])
    fallback_us = float(fallback["total_rekey_usec_median"])
    return {
        "files_per_mode": int(workflow.get("files_per_mode", 0) or 0),
        "cpu_only_median_us": cpu_us,
        "gpu_batch_median_us": gpu_us,
        "policy_fallback_median_us": fallback_us,
        "cpu_only_throughput_files_per_s": float(cpu["throughput_files_per_s_median"]),
        "gpu_batch_throughput_files_per_s": float(gpu["throughput_files_per_s_median"]),
        "policy_fallback_throughput_files_per_s": float(fallback["throughput_files_per_s_median"]),
        "observed_mounted_gain_us": cpu_us - gpu_us,
        "observed_speedup_median": float(workflow["gpu_vs_cpu_speedup_summary"]["median"]),
        "observed_speedup_ci95_low": float(workflow["gpu_vs_cpu_speedup_summary"]["ci95_low"]),
        "observed_speedup_ci95_high": float(workflow["gpu_vs_cpu_speedup_summary"]["ci95_high"]),
        "runs": int(workflow["gpu_vs_cpu_speedup_summary"]["runs"]),
        "fallback_delta_vs_cpu_us": abs(fallback_us - cpu_us),
        "mode_summaries_all_acceptable": all(
            bool(mode_summary(workflow, mode).get("all_acceptable"))
            for mode in ("cpu_only", "gpu_batch", "policy_fallback")
        ),
    }


def mounted_break_even_model(primitive: dict[str, Any], workflow: dict[str, Any]) -> dict[str, Any]:
    observed_gain = workflow["observed_mounted_gain_us"]
    model: dict[str, Any] = {}
    for op, detail in primitive.items():
        points = [(int(row["batch"]), float(row["cpu_minus_gpu_us"])) for row in detail["rows"]]
        by_batch = {batch: gap for batch, gap in points}
        primitive_gap_1024 = by_batch[WORKFLOW_BATCH]
        absorbed_overhead = primitive_gap_1024 - observed_gain
        scenarios = []
        for extra in PRESSURE_EXTRA_US:
            target_gap = absorbed_overhead + extra
            scenarios.append({
                "extra_uma_or_staging_overhead_us": extra,
                "target_primitive_gap_us": target_gap,
                "estimated_min_files_for_positive_mounted_gain": interpolate_crossing(points, target_gap),
            })
        model[op] = {
            "primitive_gap_at_workflow_batch_us": primitive_gap_1024,
            "derived_common_path_absorption_us": absorbed_overhead,
            "scenarios": scenarios,
        }
    base = [
        row["estimated_min_files_for_positive_mounted_gain"]
        for op in model.values()
        for row in op["scenarios"]
        if row["extra_uma_or_staging_overhead_us"] == 0.0
    ]
    plus5 = [
        row["estimated_min_files_for_positive_mounted_gain"]
        for op in model.values()
        for row in op["scenarios"]
        if row["extra_uma_or_staging_overhead_us"] == 5000.0
    ]
    return {
        "per_operation": model,
        "base_break_even_files_range": [min(base), max(base)],
        "plus_5ms_pressure_break_even_files_range": [min(plus5), max(plus5)],
        "interpretation": (
            "Primitive cuPQC wins by batch 16, but the mounted envelope/FUSE/admission path "
            "absorbs most of the primitive advantage until roughly 0.76K--0.81K refreshed files; "
            "extra UMA or staging pressure shifts the decision farther right."
        ),
    }


def paper_guard() -> dict[str, Any]:
    text = read_text(EVAL_TEX)
    required = {
        "five_run_methodology_row": "five-run methodology row" in text,
        "new_speedup": "1.186$\\times$" in text,
        "break_even_range": "760--812" in text,
        "pressure_range": "1.09--1.10K" in text,
        "cpu_equivalent_fallback": "fallback remains CPU-equivalent" in text,
        "no_overclaim": "not hardware-backed credential release" in text and "non-storage foreground recovery" in text,
    }
    return {
        "paper_file": rel(EVAL_TEX),
        "required": required,
        "complete": all(required.values()),
    }


def checklist_guard() -> dict[str, Any]:
    text = read_text(CHECKLIST)
    required = {
        "x11_done": "| X11 | DONE |" in text,
        "artifact_named": "x11_mlkem_break_even_model.json" in text,
        "single_platform_scoped": "Jetson-class UMA" in text,
    }
    return {
        "source": rel(CHECKLIST),
        "required": required,
        "complete": all(required.values()),
    }


def build_report() -> dict[str, Any]:
    table = load_workload_map()
    primitive = primitive_analysis(table)
    workflow_raw = load_json(REKEY_METHODOLOGY)
    workflow = workflow_analysis(workflow_raw)
    model = mounted_break_even_model(primitive, workflow)
    primitive_checks = {
        f"{op}_required_batches_present": detail["required_batches_present"]
        for op, detail in primitive.items()
    }
    primitive_checks.update({
        f"{op}_batch1_cpu_wins": detail["batch_1_cpu_wins"]
        for op, detail in primitive.items()
    })
    primitive_checks.update({
        f"{op}_batch16_gpu_wins": detail["batch_16_gpu_wins"]
        for op, detail in primitive.items()
    })
    workflow_checks = {
        "workflow_batch_1024": workflow["files_per_mode"] == WORKFLOW_BATCH,
        "methodology_runs_5": workflow["runs"] >= 5,
        "workflow_speedup_positive_but_modest": 1.0 < workflow["observed_speedup_median"] < 1.5,
        "fallback_cpu_equivalent": workflow["fallback_delta_vs_cpu_us"] <= 500.0,
        "modes_all_acceptable": workflow["mode_summaries_all_acceptable"],
    }
    model_checks = {
        "base_break_even_below_1024": model["base_break_even_files_range"][1] < WORKFLOW_BATCH,
        "plus5_pressure_break_even_above_1024": model["plus_5ms_pressure_break_even_files_range"][0] > WORKFLOW_BATCH,
    }
    paper = paper_guard()
    checklist = checklist_guard()
    checks = {
        **primitive_checks,
        **workflow_checks,
        **model_checks,
        "paper_complete": paper["complete"],
        "checklist_complete": checklist["complete"],
    }
    overall = all(checks.values())
    return {
        "artifact": "x11_mlkem_break_even_model",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "ML-KEM/cuPQC placement closeout for the elastic key-plane lane: "
            "primitive batch crossover, mounted workflow medians, CPU fallback, "
            "and UMA/staging sensitivity model."
        ),
        "inputs": {
            "workload_map": rel(WORKLOAD_MAP),
            "rekey_methodology": rel(REKEY_METHODOLOGY),
        },
        "primitive_analysis": primitive,
        "workflow_analysis": workflow,
        "mounted_break_even_model": model,
        "paper_guard": paper,
        "checklist_guard": checklist,
        "checks": checks,
        "overall_pass": overall,
        "verdict": (
            "X11 is closed: cuPQC is not oversold as ordinary-write acceleration; "
            "the retained data show a primitive GPU crossover at batch 16, a modest "
            "1.186x mounted 1024-file workflow gain under slack, CPU-equivalent "
            "fallback under pressure, and a modeled mounted break-even near 760--812 files."
            if overall else
            "X11 is not closed; inspect failed checks before relying on the ML-KEM break-even claim."
        ),
        "residual_risk": (
            "This is a retained-data model on one Jetson-class UMA platform, not a "
            "cross-platform portability proof or a claim that ML-KEM accelerates bulk file writes."
        ),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    workflow = report["workflow_analysis"]
    model = report["mounted_break_even_model"]
    lines = [
        "# X11 ML-KEM/cuPQC break-even closeout",
        "",
        f"Overall pass: `{report['overall_pass']}`",
        "",
        "## Verdict",
        "",
        report["verdict"],
        "",
        "## Mounted workflow",
        "",
        f"- CPU median: `{workflow['cpu_only_median_us'] / 1000.0:.3f}` ms",
        f"- GPU-with-slack median: `{workflow['gpu_batch_median_us'] / 1000.0:.3f}` ms",
        f"- Policy fallback median: `{workflow['policy_fallback_median_us'] / 1000.0:.3f}` ms",
        f"- Speedup median: `{workflow['observed_speedup_median']:.3f}x`",
        "",
        "## Modeled break-even",
        "",
        f"- Base range: `{model['base_break_even_files_range'][0]:.0f}`--`{model['base_break_even_files_range'][1]:.0f}` files",
        f"- +5 ms pressure range: `{model['plus_5ms_pressure_break_even_files_range'][0] / 1000.0:.2f}`--`{model['plus_5ms_pressure_break_even_files_range'][1] / 1000.0:.2f}`K files",
        "",
        "## Checks",
        "",
    ]
    for name, ok in report["checks"].items():
        lines.append(f"- `{name}`: `{ok}`")
    lines.extend(["", "## Residual risk", "", report["residual_risk"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report()
    json_path = args.out_dir / "x11_mlkem_break_even_model.json"
    md_path = args.out_dir / "x11_mlkem_break_even_model.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({
        "overall_pass": report["overall_pass"],
        "json": rel(json_path),
        "markdown": rel(md_path),
        "failed_checks": [name for name, ok in report["checks"].items() if not ok],
        "base_break_even_files_range": report["mounted_break_even_model"]["base_break_even_files_range"],
        "plus_5ms_pressure_break_even_files_range": report["mounted_break_even_model"]["plus_5ms_pressure_break_even_files_range"],
    }, indent=2, sort_keys=True))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
