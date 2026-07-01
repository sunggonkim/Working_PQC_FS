#!/usr/bin/env python3
"""Build the X9 QoS/admission design-space closeout.

This report does not rerun the mounted workload. It combines retained SQLite
QoS, kernel-control, sensitivity, and admission-interface evidence into one
paper-facing guard for the repeated review concern: AEGIS-Q recovers a bounded
foreground tail only by spending elastic background throughput, so the paper
must present the result as a Pareto point with explicit thresholds and
sensitivity rather than as a free or application-wide QoS win.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts" / "reports" / "x9_qos_admission_closeout"

COMPARISON = ROOT / "artifacts" / "validation" / "sqlite_kernel_qos_comparison" / "sqlite_kernel_qos_comparison.json"
SENSITIVITY_CSV = ROOT / "artifacts" / "validation" / "qos_sensitivity_analysis" / "qos_sensitivity_summary.csv"
SENSITIVITY_JSON = ROOT / "artifacts" / "validation" / "qos_sensitivity_analysis" / "qos_sensitivity_analysis.json"
ADMISSION_AUDIT = ROOT / "artifacts" / "validation" / "admission_interface_audit" / "admission_interface_audit.json"
ADMISSION_SOURCE = ROOT / "code" / "runtime" / "pqc_admission.c"
QOS_SOURCE = ROOT / "code" / "runtime" / "pqc_qos.c"
EVAL_TEX = ROOT / "Paper" / "4_Evaluation.tex"
INTRO_TEX = ROOT / "Paper" / "1_Introduction.tex"
DISCUSSION_TEX = ROOT / "Paper" / "10_Discussion_and_Limitations.tex"
CHECKLIST = ROOT / "SUBMISSION_CHECKLIST.md"

REQUIRED_HERO_MODES = [
    "app_only",
    "unthrottled_storage",
    "simple_controller",
    "aegis_policy",
]
REQUIRED_KERNEL_CONTROLS = [
    "ionice",
    "systemd_io_weight",
]
REQUIRED_SENSITIVITY_CASES = [
    "baseline",
    "slow_sampling",
    "high_threshold",
    "queue_depth_2",
    "background_128k",
    "hysteresis_wave",
    "low_pressure_no_throttle",
    "no_slack_mounted",
]
REQUIRED_PAPER_PHRASES = [
    "Pareto point rather than a free throughput improvement",
    "The default storage-pressure controller uses 20~ms telemetry sampling",
    "high threshold disables throttling",
    "two writers is fragile",
    "This keeps the claim on mounted storage-visible control, not an external application scheduler",
]
FORBIDDEN_UNGUARDED = [
    re.compile(r"application-wide QoS win", re.I),
    re.compile(r"external application scheduler", re.I),
    re.compile(r"free throughput improvement", re.I),
]
GUARD_TERMS = (
    "not ",
    "no ",
    "without ",
    "rather than ",
    "non-claim",
    "motivation",
    "forbidden",
    "does not",
)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return data


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def metric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, dict):
        value = value.get("median")
    return float(value) if value is not None else None


def summarize_comparison(data: dict[str, Any]) -> dict[str, Any]:
    rows = data.get("comparison_rows")
    if not isinstance(rows, list):
        rows = []

    hero: dict[str, dict[str, Any]] = {}
    kernel: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        summary = {
            "p99_ms": metric(row, "foreground_p99_ms"),
            "deadline_misses": metric(row, "deadline_misses"),
            "background_mb_s": metric(row, "background_throughput_mb_s"),
            "policy_throttle_rows": metric(row, "policy_throttle_rows"),
            "daemon_throttle_rows": metric(row, "daemon_throttle_rows"),
            "runs": row.get("runs") or row.get("successful_rows"),
        }
        if row.get("row_type") == "hero_mode":
            hero[str(row.get("mode"))] = summary
        elif row.get("row_type") == "kernel_baseline":
            kernel[str(row.get("kernel_control"))] = summary | {
                "mode": row.get("mode"),
                "detail": row.get("kernel_control_detail"),
            }

    aegis = hero.get("aegis_policy", {})
    unthrottled = hero.get("unthrottled_storage", {})
    simple = hero.get("simple_controller", {})

    def diff(left: dict[str, Any], right: dict[str, Any], key: str) -> float | None:
        if left.get(key) is None or right.get(key) is None:
            return None
        return float(left[key]) - float(right[key])

    tradeoff = {
        "aegis_vs_unthrottled_p99_ms": diff(aegis, unthrottled, "p99_ms"),
        "aegis_vs_unthrottled_background_mb_s": diff(aegis, unthrottled, "background_mb_s"),
        "aegis_vs_simple_p99_ms": diff(aegis, simple, "p99_ms"),
        "aegis_vs_simple_background_mb_s": diff(aegis, simple, "background_mb_s"),
    }
    checks = {
        "required_hero_modes_present": all(mode in hero for mode in REQUIRED_HERO_MODES),
        "required_kernel_controls_present": all(control in kernel for control in REQUIRED_KERNEL_CONTROLS),
        "aegis_improves_p99_vs_unthrottled": (
            tradeoff["aegis_vs_unthrottled_p99_ms"] is not None
            and tradeoff["aegis_vs_unthrottled_p99_ms"] < 0
        ),
        "aegis_spends_background_vs_unthrottled": (
            tradeoff["aegis_vs_unthrottled_background_mb_s"] is not None
            and tradeoff["aegis_vs_unthrottled_background_mb_s"] < 0
        ),
        "simple_is_lower_latency_lower_background": (
            tradeoff["aegis_vs_simple_p99_ms"] is not None
            and tradeoff["aegis_vs_simple_p99_ms"] > 0
            and tradeoff["aegis_vs_simple_background_mb_s"] is not None
            and tradeoff["aegis_vs_simple_background_mb_s"] > 0
        ),
        "kernel_controls_measured": all(
            kernel.get(control, {}).get("runs") and int(kernel[control]["runs"]) >= 3
            for control in REQUIRED_KERNEL_CONTROLS
        ),
    }
    return {
        "source": rel(COMPARISON),
        "overall_pass": bool(data.get("overall_pass")),
        "hero_modes": hero,
        "kernel_controls": kernel,
        "tradeoff": tradeoff,
        "checks": checks,
        "complete": all(checks.values()),
    }


def load_sensitivity(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by_case: dict[str, dict[str, Any]] = {}
    for row in rows:
        case = str(row.get("case"))
        by_case[case] = {
            "setting": row.get("setting"),
            "p99_ms": float(row["p99_ms"]),
            "deadline_misses": int(float(row["deadline_misses"])),
            "storage_mb_s": float(row["storage_mb_s"]),
            "policy_throttle_rows": int(float(row["policy_throttle_rows"])),
            "daemon_throttle_rows": int(float(row["daemon_throttle_rows"])),
            "transition_count": int(float(row["transition_count"])),
            "oscillation_count": int(float(row["oscillation_count"])),
            "hysteresis_enter_exit": row.get("hysteresis_enter_exit") == "True",
            "no_throttle_fallback": row.get("no_throttle_fallback") == "True",
        }

    checks = {
        "required_cases_present": all(case in by_case for case in REQUIRED_SENSITIVITY_CASES),
        "baseline_zero_miss": by_case.get("baseline", {}).get("deadline_misses") == 0,
        "slow_sampling_zero_miss": by_case.get("slow_sampling", {}).get("deadline_misses") == 0,
        "background_128k_zero_miss": by_case.get("background_128k", {}).get("deadline_misses") == 0,
        "high_threshold_less_invasive_but_higher_p99": (
            by_case.get("high_threshold", {}).get("policy_throttle_rows") == 0
            and by_case.get("baseline", {}).get("p99_ms") is not None
            and by_case.get("high_threshold", {}).get("p99_ms", 0.0) > by_case["baseline"]["p99_ms"]
        ),
        "queue_depth_2_fragile": by_case.get("queue_depth_2", {}).get("deadline_misses", 0) >= 1,
        "hysteresis_records_transitions": by_case.get("hysteresis_wave", {}).get("transition_count", 0) >= 1,
        "low_pressure_no_throttle_fallback": by_case.get("low_pressure_no_throttle", {}).get("no_throttle_fallback") is True,
    }
    return {
        "source": rel(path),
        "json_source": rel(SENSITIVITY_JSON),
        "artifact_overall_pass": bool(load_json(SENSITIVITY_JSON).get("overall_pass")),
        "cases": by_case,
        "checks": checks,
        "complete": all(checks.values()),
    }


def source_constants() -> dict[str, Any]:
    admission = read_text(ADMISSION_SOURCE)
    qos = read_text(QOS_SOURCE)
    patterns = {
        "gpu_min_batch_bytes": r"PQC_ADMISSION_DEFAULT_GPU_MIN_BATCH_BYTES\s+(\d+)ULL",
        "ai_qos_min_budget_ns": r"PQC_ADMISSION_DEFAULT_AI_QOS_MIN_BUDGET_NS\s+(\d+)ULL",
        "deadline_margin_ns": r"PQC_ADMISSION_DEFAULT_DEADLINE_MARGIN_NS\s+(\d+)ULL",
        "producer_slack_stale_ns": r"PQC_ADMISSION_DEFAULT_PRODUCER_SLACK_STALE_NS\s+(\d+)ULL",
        "queue_pressure_threshold": r"PQC_ADMISSION_DEFAULT_QUEUE_PRESSURE_THRESHOLD\s+([0-9.]+)",
        "qos_enter_util": r"PQC_QOS_MEM_ENTER_UTIL\",\s*([0-9.]+)\)",
        "qos_exit_util": r"PQC_QOS_MEM_EXIT_UTIL\",\s*([0-9.]+)\)",
        "qos_hold_samples": r"PQC_QOS_HOLD_SAMPLES\",\s*(\d+)\)",
    }
    text_by_key = {
        "gpu_min_batch_bytes": admission,
        "ai_qos_min_budget_ns": admission,
        "deadline_margin_ns": admission,
        "producer_slack_stale_ns": admission,
        "queue_pressure_threshold": admission,
        "qos_enter_util": qos,
        "qos_exit_util": qos,
        "qos_hold_samples": qos,
    }
    values: dict[str, Any] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text_by_key[key])
        if not match:
            values[key] = None
            continue
        raw = match.group(1)
        values[key] = float(raw) if "." in raw else int(raw)
    complete = all(value is not None for value in values.values())
    return {
        "sources": [rel(ADMISSION_SOURCE), rel(QOS_SOURCE)],
        "defaults": values,
        "complete": complete,
    }


def admission_guard() -> dict[str, Any]:
    data = load_json(ADMISSION_AUDIT)
    cases = data.get("cases")
    if not isinstance(cases, list):
        cases = []
    by_case = {str(case.get("case")): case for case in cases if isinstance(case, dict)}
    checks = {
        "audit_overall_pass": bool(data.get("overall_pass")),
        "slack_available_gpu": by_case.get("slack_available_gpu", {}).get("decision", {}).get("chosen_target") == "GPU",
        "no_slack_cpu": by_case.get("no_slack_cpu", {}).get("decision", {}).get("chosen_target") == "CPU",
        "deadline_elapsed_cpu": by_case.get("deadline_elapsed_cpu", {}).get("decision", {}).get("chosen_target") == "CPU",
    }
    return {
        "source": rel(ADMISSION_AUDIT),
        "cases": sorted(by_case),
        "checks": checks,
        "complete": all(checks.values()),
    }


def paper_guard() -> dict[str, Any]:
    files = [INTRO_TEX, EVAL_TEX, DISCUSSION_TEX]
    text_by_file = {rel(path): read_text(path) for path in files if path.exists()}
    combined = "\n".join(text_by_file.values())
    required = {phrase: phrase in combined for phrase in REQUIRED_PAPER_PHRASES}

    unguarded: list[dict[str, Any]] = []
    hits: list[dict[str, Any]] = []
    for path, text in text_by_file.items():
        lines = text.splitlines()
        for idx, line in enumerate(lines, start=1):
            for pattern in FORBIDDEN_UNGUARDED:
                if not pattern.search(line):
                    continue
                context = " ".join(lines[max(0, idx - 2):min(len(lines), idx + 2)]).lower()
                guarded = any(term in context for term in GUARD_TERMS)
                item = {"path": path, "line": idx, "text": line.strip(), "guarded": guarded}
                hits.append(item)
                if not guarded:
                    unguarded.append(item)
    return {
        "paper_files": sorted(text_by_file),
        "required_phrases": required,
        "forbidden_hits": hits,
        "unguarded_forbidden_hits": unguarded,
        "complete": all(required.values()) and not unguarded,
    }


def checklist_guard() -> dict[str, Any]:
    text = read_text(CHECKLIST)
    x9_closed = (
        "| X9 | DONE |" in text
        or "| X9 | SQLite QoS is a Pareto tradeoff" in text
    )
    x10_done_or_next = (
        "| X10 | DONE |" in text
        or "| X10 | NEXT |" in text
        or "| X10 | Generation robustness is tied to" in text
    )
    compressed_checklist = (
        "storage-visible QoS" in text
        and "Pareto tradeoff" in text
        and "foreground 4 KiB" in text
        and "default GPU byte gate is 128 KiB" in text
    )
    return {
        "source": rel(CHECKLIST),
        "x9_done": x9_closed,
        "closeout_artifact_named": (
            "x9_qos_admission_closeout.json" in text or compressed_checklist
        ),
        "x10_done_or_next": x10_done_or_next,
        "complete": (
            x9_closed
            and compressed_checklist
            and x10_done_or_next
        ),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    comp = report["comparison"]
    sens = report["sensitivity"]
    const = report["source_constants"]["defaults"]
    lines = [
        "# X9 QoS/admission closeout",
        "",
        f"Overall pass: `{report['overall_pass']}`",
        "",
        "## Pareto result",
    ]
    for mode in REQUIRED_HERO_MODES:
        row = comp["hero_modes"].get(mode, {})
        lines.append(
            f"- `{mode}`: p99={row.get('p99_ms'):.3f} ms, "
            f"misses={row.get('deadline_misses')}, background={row.get('background_mb_s'):.3f} MB/s"
        )
    lines.extend([
        "",
        "AEGIS-Q improves p99 over unthrottled storage while spending background throughput. "
        "The simple controller is the lower-latency/lower-background point, so the paper must "
        "present AEGIS-Q as a selected Pareto point, not a free throughput improvement.",
        "",
        "## Controller constants",
        "",
        f"- GPU min batch: `{const.get('gpu_min_batch_bytes')}` bytes",
        f"- Producer slack budget: `{const.get('ai_qos_min_budget_ns')}` ns",
        f"- Deadline margin: `{const.get('deadline_margin_ns')}` ns",
        f"- Producer slack stale: `{const.get('producer_slack_stale_ns')}` ns",
        f"- Queue pressure threshold: `{const.get('queue_pressure_threshold')}`",
        f"- Storage pressure hysteresis: enter `{const.get('qos_enter_util')}`, "
        f"exit `{const.get('qos_exit_util')}`, hold `{const.get('qos_hold_samples')}` samples",
        "",
        "## Sensitivity cases",
        "",
    ])
    for case in REQUIRED_SENSITIVITY_CASES:
        row = sens["cases"].get(case, {})
        lines.append(
            f"- `{case}`: p99={row.get('p99_ms'):.3f} ms, "
            f"misses={row.get('deadline_misses')}, storage={row.get('storage_mb_s'):.3f} MB/s"
        )
    lines.extend([
        "",
        "## Verdict",
        "",
        report["verdict"],
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report() -> dict[str, Any]:
    comparison = summarize_comparison(load_json(COMPARISON))
    sensitivity = load_sensitivity(SENSITIVITY_CSV)
    constants = source_constants()
    admission = admission_guard()
    paper = paper_guard()
    checklist = checklist_guard()
    checks = {
        "comparison_complete": comparison["complete"],
        "sensitivity_complete": sensitivity["complete"],
        "source_constants_complete": constants["complete"],
        "admission_complete": admission["complete"],
        "paper_complete": paper["complete"],
        "checklist_complete": checklist["complete"],
    }
    overall = all(checks.values())
    return {
        "artifact": "x9_qos_admission_closeout",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Retained evidence closeout for SQLite storage-visible QoS/admission design space; "
            "does not claim external application scheduling or broad workload coverage."
        ),
        "comparison": comparison,
        "sensitivity": sensitivity,
        "source_constants": constants,
        "admission_interface": admission,
        "paper_guard": paper,
        "checklist_guard": checklist,
        "checks": checks,
        "overall_pass": overall,
        "verdict": (
            "X9 is closed: the paper has enough retained evidence to answer the repeated "
            "QoS tradeoff review as a scoped Pareto/admission result, provided it keeps "
            "external application scheduling and free-throughput language out of claim scope."
            if overall else
            "X9 is not closed; inspect failed checks before claiming the QoS/admission design-space response."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report()
    json_path = args.out_dir / "x9_qos_admission_closeout.json"
    md_path = args.out_dir / "x9_qos_admission_closeout.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({
        "overall_pass": report["overall_pass"],
        "json": rel(json_path),
        "markdown": rel(md_path),
        "failed_checks": [name for name, ok in report["checks"].items() if not ok],
    }, indent=2, sort_keys=True))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
