#!/usr/bin/env python3
"""Build the E2 kernel-QoS hero integration closeout.

This is a narrow paper-facing integration guard.  It checks that the SQLite
hero claim is no longer compared only against internal modes: the retained table
must include app-only, unthrottled storage, simple-controller, AEGIS-Q policy,
and two Linux kernel QoS controls, while the paper keeps the comparison scoped.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "kernel_qos_hero_integration_closeout"

KERNEL_QOS = ROOT / "artifacts" / "validation" / "sqlite_kernel_qos_comparison" / "sqlite_kernel_qos_comparison.json"
E1_CLOSEOUT = ROOT / "artifacts" / "validation" / "sqlite_hero_validity_closeout" / "sqlite_hero_validity_closeout.json"
GENERATED_TABLE = ROOT / "Paper" / "generated_qos_recovery_table.tex"
EVAL_TEX = ROOT / "Paper" / "4_Evaluation.tex"
MAIN_TEX = ROOT / "Paper" / "main.tex"

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

REQUIRED_TABLE_LABELS = [
    "App only",
    "Unthrottled",
    "Simple ctrl.",
    "AEGIS-Q",
    "Kernel ionice",
    "Kernel IOWeight",
]

REQUIRED_PAPER_PHRASES = {
    "five_run_hero_one_run_kernel":
        "Table~\\ref{tab:qos_sqlite_recovery} reports five-run hero medians and three-run kernel QoS controls",
    "ionice_result":
        "\\texttt{ionice} reports 16.22~ms p99 with six misses",
    "ioweight_result":
        "\\texttt{IOWeight} reports 12.57~ms p99 with seven misses",
    "bounded_not_unique":
        "This is bounded storage-visible control, not a uniqueness or non-storage QoS claim",
    "separate_repeated_and_kernel":
        "five-run hero medians and three-run kernel QoS controls",
}

FORBIDDEN_UNGUARDED_PATTERNS = [
    ("unique_sqlite_recovery", re.compile(r"SQLite p99 recovery is unique to AEGIS-Q", re.I)),
    ("kernel_qos_cannot_recover", re.compile(r"Kernel QoS cannot recover SQLite p99", re.I)),
    ("beats_kernel_without_qualification", re.compile(r"AEGIS-Q beats standard kernel throttling", re.I)),
    ("always_beats_kernel", re.compile(r"always beats .*kernel", re.I)),
    ("all_kernel_controls", re.compile(r"all kernel (?:QoS )?controls", re.I)),
]

NEGATION_TERMS = (
    "not ",
    "no ",
    "without ",
    "unless ",
    "forbidden",
    "non-claim",
    "nonclaim",
    "bounded",
    "scoped",
    "one-run",
    "mismatch",
    "qualification",
)


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return data


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def path_meta(path: Path) -> dict[str, Any]:
    return {
        "path": relpath(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
    }


def median(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if not isinstance(value, dict):
        return None
    med = value.get("median")
    return float(med) if med is not None else None


def summarize_rows(comparison: dict[str, Any]) -> dict[str, Any]:
    rows = comparison.get("comparison_rows", [])
    if not isinstance(rows, list):
        rows = []

    hero_modes: dict[str, dict[str, Any]] = {}
    kernel_controls: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("row_type") == "hero_mode":
            mode = str(row.get("mode"))
            hero_modes[mode] = {
                "runs": row.get("runs"),
                "all_acceptable": row.get("all_acceptable"),
                "p99_ms": median(row, "foreground_p99_ms"),
                "deadline_misses": median(row, "deadline_misses"),
                "background_mb_s": median(row, "background_throughput_mb_s"),
            }
        elif row.get("row_type") == "kernel_baseline":
            control = str(row.get("kernel_control"))
            kernel_controls[control] = {
                "verdict": row.get("verdict"),
                "overall_pass": row.get("overall_pass"),
                "successful_rows": row.get("successful_rows"),
                "writer_start_rows": row.get("writer_start_rows"),
                "p99_ms": median(row, "foreground_p99_ms"),
                "deadline_misses": median(row, "deadline_misses"),
                "background_mb_s": median(row, "background_throughput_mb_s"),
                "source": row.get("source"),
            }
    return {
        "hero_modes": hero_modes,
        "kernel_controls": kernel_controls,
        "required_hero_modes_present": all(mode in hero_modes for mode in REQUIRED_HERO_MODES),
        "required_kernel_controls_present": all(control in kernel_controls for control in REQUIRED_KERNEL_CONTROLS),
        "hero_modes_five_run": all(int(hero_modes.get(mode, {}).get("runs") or 0) >= 5 for mode in REQUIRED_HERO_MODES),
        "kernel_controls_measured": all(
            kernel_controls.get(control, {}).get("verdict") == "measured"
            and kernel_controls.get(control, {}).get("overall_pass") is True
            for control in REQUIRED_KERNEL_CONTROLS
        ),
    }


def table_guard() -> dict[str, Any]:
    text = read_text(GENERATED_TABLE) if GENERATED_TABLE.exists() else ""
    required_labels = {label: label in text for label in REQUIRED_TABLE_LABELS}
    required_controls = {
        "ionice_class_3": "\\texttt{ionice -c 3}" in text,
        "ioweight_10": "\\texttt{IOWeight=10}" in text,
        "five_run_caption": "Hero rows are five-run medians" in text,
        "three_run_caption": "kernel rows are 3 retained runs each" in text,
    }
    return {
        "table": path_meta(GENERATED_TABLE),
        "required_labels": required_labels,
        "required_controls": required_controls,
        "complete": GENERATED_TABLE.exists() and all(required_labels.values()) and all(required_controls.values()),
    }


def paper_guard() -> dict[str, Any]:
    files = [path for path in [MAIN_TEX, EVAL_TEX] if path.exists()]
    text_by_file = {relpath(path): read_text(path) for path in files}
    combined = "\n".join(text_by_file.values())
    required = {name: phrase in combined for name, phrase in REQUIRED_PAPER_PHRASES.items()}

    hits: list[dict[str, Any]] = []
    unguarded: list[dict[str, Any]] = []
    for rel, text in text_by_file.items():
        lines = text.splitlines()
        for idx, line in enumerate(lines, start=1):
            for name, pattern in FORBIDDEN_UNGUARDED_PATTERNS:
                if not pattern.search(line):
                    continue
                context = " ".join(lines[max(0, idx - 3): min(len(lines), idx + 3)]).lower()
                guarded = any(term in context for term in NEGATION_TERMS)
                item = {
                    "kind": name,
                    "path": rel,
                    "line": idx,
                    "text": line.strip(),
                    "guarded": guarded,
                }
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


def source_row_paths_exist(rows: dict[str, Any]) -> bool:
    for control in REQUIRED_KERNEL_CONTROLS:
        raw = rows["kernel_controls"].get(control, {}).get("source")
        if not isinstance(raw, str) or not (ROOT / raw).exists():
            return False
    return True


def build_report() -> dict[str, Any]:
    inputs = {
        "kernel_qos_comparison": path_meta(KERNEL_QOS),
        "sqlite_hero_validity_closeout": path_meta(E1_CLOSEOUT),
        "generated_qos_recovery_table": path_meta(GENERATED_TABLE),
        "evaluation_tex": path_meta(EVAL_TEX),
        "main_tex": path_meta(MAIN_TEX),
    }
    kernel = read_json(KERNEL_QOS) if KERNEL_QOS.exists() else {}
    e1 = read_json(E1_CLOSEOUT) if E1_CLOSEOUT.exists() else {}
    rows = summarize_rows(kernel)
    table = table_guard()
    paper = paper_guard()

    close_conditions = {
        "required_inputs_present": all(meta["exists"] and meta["bytes"] > 0 for meta in inputs.values()),
        "e1_closeout_passes": bool(e1.get("overall_pass")),
        "b3_comparison_passes": bool(kernel.get("overall_pass")),
        "b3_parent_gate_closed": bool(kernel.get("parent_b3_gate_closed")),
        "two_kernel_baselines_measured": bool(kernel.get("two_kernel_baselines_measured")),
        "hero_modes_available": bool(kernel.get("hero_modes_available")) and rows["required_hero_modes_present"],
        "hero_rows_are_five_run": rows["hero_modes_five_run"],
        "kernel_rows_are_measured": rows["kernel_controls_measured"],
        "kernel_source_artifacts_exist": source_row_paths_exist(rows),
        "paper_compares_both_kernel_baselines": bool(kernel.get("paper_compares_both_kernel_baselines")),
        "generated_table_complete": bool(table.get("complete")),
        "paper_claim_guard_passes": bool(paper.get("complete")),
        "sqlite_uniqueness_claim_guard_allows_only_bounded_claim": bool(
            kernel.get("claim_guard", {}).get("sqlite_p99_uniqueness_claim_allowed")
        ),
    }

    return {
        "artifact": "kernel_qos_hero_integration_closeout",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifacts": {
            "kernel_qos_comparison": relpath(KERNEL_QOS),
            "sqlite_hero_validity_closeout": relpath(E1_CLOSEOUT),
            "generated_table": relpath(GENERATED_TABLE),
        },
        "inputs": inputs,
        "row_summary": rows,
        "table_guard": table,
        "paper_guard": paper,
        "claim_boundary": {
            "allowed": [
                "bounded SQLite foreground p99 comparison under the retained mounted secure-storage pressure workflow",
                "AEGIS-Q versus app-only, unthrottled storage, simple controller, ionice, and systemd IOWeight rows",
                "statement that kernel controls preserve more background throughput in the retained one-run controls",
                "statement that AEGIS-Q offers storage-visible policy/observability in this workload envelope",
            ],
            "forbidden": [
                "SQLite p99 recovery uniqueness without qualification",
                "claim that kernel QoS cannot recover SQLite p99 in general",
                "claim that AEGIS-Q beats standard kernel throttling without noting repetition mismatch",
                "claim that the result generalizes to all workloads or all kernel QoS controls",
            ],
        },
        "close_conditions": close_conditions,
        "overall_pass": all(close_conditions.values()),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Kernel QoS Hero Integration Closeout",
        "",
        f"- Overall pass: `{str(report['overall_pass']).lower()}`",
        f"- Generated: `{report['timestamp_utc']}`",
        "",
        "## Close Conditions",
        "",
    ]
    for key, value in report["close_conditions"].items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")
    lines.extend(["", "## Rows", ""])
    for mode, row in report["row_summary"]["hero_modes"].items():
        lines.append(
            f"- `{mode}`: runs={row['runs']}, p99_ms={row['p99_ms']}, "
            f"misses={row['deadline_misses']}, background_mb_s={row['background_mb_s']}"
        )
    for control, row in report["row_summary"]["kernel_controls"].items():
        lines.append(
            f"- `{control}`: verdict={row['verdict']}, p99_ms={row['p99_ms']}, "
            f"misses={row['deadline_misses']}, background_mb_s={row['background_mb_s']}"
        )
    lines.extend(["", "## Claim Boundary", ""])
    for item in report["claim_boundary"]["allowed"]:
        lines.append(f"- Allowed: {item}")
    for item in report["claim_boundary"]["forbidden"]:
        lines.append(f"- Forbidden: {item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    report = build_report()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "kernel_qos_hero_integration_closeout.json"
    md_path = args.out_dir / "kernel_qos_hero_integration_closeout.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)

    print(f"wrote {relpath(json_path)}")
    print(f"wrote {relpath(md_path)}")
    print(f"overall_pass={str(report['overall_pass']).lower()}")
    for key, value in report["close_conditions"].items():
        print(f"{key}={str(value).lower()}")
    if args.require_complete and not report["overall_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
