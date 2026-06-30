#!/usr/bin/env python3
"""Build the Gate B3 SQLite kernel-QoS comparison verdict.

This script does not rerun the mounted workload. It combines retained
four-mode SQLite hero artifacts with the two retained kernel-control baselines
and emits a fail-closed paper claim guard.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HERO = ROOT / "artifacts" / "validation" / "qos_sqlite_hero_methodology" / "qos_sqlite_hero_bundle.json"
DEFAULT_FALLBACK_HERO = ROOT / "artifacts" / "validation" / "qos_sqlite_hero_bundle" / "qos_sqlite_hero_bundle.json"
DEFAULT_IONICE = ROOT / "artifacts" / "validation" / "sqlite_kernel_qos_baseline" / "sqlite_kernel_qos_baseline.json"
DEFAULT_CGROUP = ROOT / "artifacts" / "validation" / "sqlite_kernel_qos_baseline_cgroup" / "sqlite_kernel_qos_baseline.json"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "sqlite_kernel_qos_comparison"
DEFAULT_PAPER_TABLE = ROOT / "Paper" / "generated_qos_recovery_table.tex"
REQUIRED_HERO_MODES = ["app_only", "unthrottled_storage", "simple_controller", "aegis_policy"]
REQUIRED_KERNEL_CONTROLS = ["ionice", "systemd_io_weight"]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def stat_value(obj: Any, field: str = "median") -> float | None:
    if isinstance(obj, dict):
        value = obj.get(field)
        return float(value) if value is not None else None
    if obj is None:
        return None
    return float(obj)


def metric_bundle(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        runs = obj.get("runs")
        if runs is None:
            runs = obj.get("samples")
        return {
            "median": stat_value(obj, "median"),
            "ci95_low": stat_value(obj, "ci95_low"),
            "ci95_high": stat_value(obj, "ci95_high"),
            "p05": stat_value(obj, "p05"),
            "p95": stat_value(obj, "p95"),
            "runs": int(runs) if runs is not None else None,
        }
    return {
        "median": stat_value(obj),
        "ci95_low": None,
        "ci95_high": None,
        "p05": None,
        "p95": None,
        "runs": 1 if obj is not None else None,
    }


def summarize_hero(path: Path) -> dict[str, Any]:
    data = load_json(path)
    rows: list[dict[str, Any]] = []

    if data.get("mode_summaries"):
        for row in data["mode_summaries"]:
            rows.append({
                "row_type": "hero_mode",
                "mode": row.get("mode"),
                "runs": row.get("runs"),
                "acceptable_runs": row.get("acceptable_runs"),
                "all_acceptable": row.get("all_acceptable"),
                "foreground_p99_ms": metric_bundle(row.get("p99_ms")),
                "foreground_p95_ms": metric_bundle(row.get("p95_ms")),
                "deadline_misses": metric_bundle(row.get("deadline_misses")),
                "background_throughput_mb_s": metric_bundle(row.get("storage_mb_s")),
                "policy_throttle_rows": metric_bundle(row.get("policy_throttle_rows")),
                "daemon_throttle_rows": metric_bundle(row.get("daemon_throttle_rows")),
                "telemetry_rows": metric_bundle(row.get("telemetry_rows")),
            })
    else:
        for mode in data.get("modes", []):
            foreground = mode.get("foreground") or {}
            background = mode.get("background") or {}
            policy = mode.get("policy") or {}
            daemon = mode.get("daemon_throttle") or {}
            telemetry = mode.get("telemetry") or {}
            rows.append({
                "row_type": "hero_mode",
                "mode": mode.get("mode"),
                "runs": 1,
                "acceptable_runs": 1 if mode.get("acceptable") else 0,
                "all_acceptable": bool(mode.get("acceptable")),
                "foreground_p99_ms": metric_bundle(foreground.get("p99_ms")),
                "foreground_p95_ms": metric_bundle(foreground.get("p95_ms")),
                "deadline_misses": metric_bundle(foreground.get("deadline_misses")),
                "background_throughput_mb_s": metric_bundle(background.get("throughput_mb_s")),
                "policy_throttle_rows": metric_bundle(policy.get("throttle_rows")),
                "daemon_throttle_rows": metric_bundle(daemon.get("throttled_rows")),
                "telemetry_rows": metric_bundle(telemetry.get("rows")),
            })

    found = {str(row.get("mode")) for row in rows}
    return {
        "source": rel(path),
        "overall_pass": bool(data.get("overall_pass")),
        "artifact_role": data.get("artifact_role", "single_bundle"),
        "repetitions_measured": data.get("repetitions_measured"),
        "rows": rows,
        "required_modes_present": all(mode in found for mode in REQUIRED_HERO_MODES),
        "missing_modes": [mode for mode in REQUIRED_HERO_MODES if mode not in found],
    }


def kernel_control_detail(control: dict[str, Any]) -> str:
    name = control.get("control")
    if name == "ionice":
        cls = control.get("class")
        level = control.get("level")
        if level is None:
            return f"ionice class {cls}"
        return f"ionice class {cls}, level {level}"
    if name == "systemd_io_weight":
        return f"systemd IOWeight={control.get('io_weight')}"
    return str(name)


def summarize_kernel(path: Path) -> dict[str, Any]:
    data = load_json(path)
    control = data.get("kernel_control") or {}
    summary = data.get("summary") or {}
    rows = data.get("rows") or []
    return {
        "row_type": "kernel_baseline",
        "source": rel(path),
        "verdict": data.get("verdict"),
        "overall_pass": bool(data.get("overall_pass")),
        "mode": rows[0].get("mode", data.get("modes", [{}])[0].get("mode")) if rows else None,
        "kernel_control": control.get("control"),
        "kernel_control_detail": kernel_control_detail(control),
        "successful_rows": control.get("successful_rows"),
        "writer_start_rows": control.get("writer_start_rows"),
        "foreground_p99_ms": metric_bundle(summary.get("foreground_p99_ms")),
        "deadline_misses": metric_bundle(summary.get("deadline_misses")),
        "background_throughput_mb_s": metric_bundle(summary.get("background_throughput_mb_s")),
        "raw_rows": rows,
    }


def median(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if not isinstance(value, dict):
        return None
    med = value.get("median")
    return float(med) if med is not None else None


def delta_rows(comparison_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_mode = {str(row.get("mode")): row for row in comparison_rows if row.get("row_type") == "hero_mode"}
    aegis = by_mode.get("aegis_policy")
    if not aegis:
        return []
    aegis_p99 = median(aegis, "foreground_p99_ms")
    aegis_bg = median(aegis, "background_throughput_mb_s")
    deltas = []
    for row in comparison_rows:
        p99 = median(row, "foreground_p99_ms")
        bg = median(row, "background_throughput_mb_s")
        if p99 is None or aegis_p99 is None:
            p99_delta = None
        else:
            p99_delta = aegis_p99 - p99
        if bg is None or aegis_bg is None:
            bg_delta = None
        else:
            bg_delta = aegis_bg - bg
        deltas.append({
            "against": row.get("mode") or row.get("kernel_control"),
            "row_type": row.get("row_type"),
            "aegis_minus_row_p99_ms": p99_delta,
            "aegis_minus_row_background_mb_s": bg_delta,
        })
    return deltas


def scan_paper_claims(paper_dir: Path) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    if not paper_dir.exists():
        return {
            "paper_dir": rel(paper_dir),
            "exists": False,
            "mentions_ionice": False,
            "mentions_systemd_io_weight": False,
            "mentions_kernel_qos": False,
            "mentions_repetition_mismatch": False,
            "mentions_no_uniqueness_boundary": False,
            "sqlite_p99_candidate_lines": candidates,
        }
    tex_files = sorted(paper_dir.rglob("*.tex"))
    sqlite_p99 = re.compile(r"sqlite.*p99|p99.*sqlite|recovers p99|recovery.*p99", re.IGNORECASE)
    for path in tex_files:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if sqlite_p99.search(line):
                candidates.append({
                    "path": rel(path),
                    "line": lineno,
                    "text": line.strip(),
                })
    joined = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in tex_files)
    return {
        "paper_dir": rel(paper_dir),
        "exists": True,
        "mentions_ionice": "ionice" in joined,
        "mentions_systemd_io_weight": ("systemd_io_weight" in joined or "IOWeight" in joined),
        "mentions_kernel_qos": "kernel QoS" in joined or "kernel-level" in joined,
        "mentions_repetition_mismatch": (
            ("five-run" in joined or "five retained runs" in joined)
            and (
                "one-run" in joined
                or "one retained run" in joined
                or "three-run" in joined
                or "3 retained runs" in joined
                or "retained proof rows" in joined
            )
        ),
        "mentions_no_uniqueness_boundary": (
            "not a uniqueness" in joined
            or "not a broad" in joined
            or "not a general" in joined
        ),
        "sqlite_p99_candidate_lines": candidates,
    }


def fmt_metric(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.2f}"


def latex_row_label(row: dict[str, Any]) -> str:
    labels = {
        "app_only": "App only",
        "unthrottled_storage": "Unthrottled",
        "simple_controller": "Simple ctrl.",
        "aegis_policy": "AEGIS-Q",
        "kernel_ionice_idle": "Kernel ionice",
        "kernel_systemd_io_weight": "Kernel IOWeight",
    }
    return labels.get(str(row.get("mode") or row.get("kernel_control")), str(row.get("mode") or row.get("kernel_control")))


def latex_control(row: dict[str, Any]) -> str:
    mode = str(row.get("mode") or row.get("kernel_control"))
    if mode == "app_only":
        return "none"
    if mode == "unthrottled_storage":
        return "none"
    if mode == "simple_controller":
        return "harness sleep"
    if mode == "aegis_policy":
        return "storage class"
    if mode == "kernel_ionice_idle":
        return "\\texttt{ionice -c 3}"
    if mode == "kernel_systemd_io_weight":
        return "\\texttt{IOWeight=10}"
    return "-"


def latex_scope(row: dict[str, Any]) -> str:
    runs = row.get("runs") or row.get("foreground_p99_ms", {}).get("runs")
    if row.get("row_type") == "hero_mode":
        return f"{runs}-run med."
    return f"{runs}-run proof"


def write_latex_table(report: dict[str, Any], path: Path) -> None:
    kernel_runs = sorted({
        int(row.get("foreground_p99_ms", {}).get("runs") or 0)
        for row in report["kernel_rows"]
    })
    kernel_scope = (
        f"{kernel_runs[0]} retained runs each"
        if len(kernel_runs) == 1 and kernel_runs[0] > 0
        else "retained proof rows"
    )
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{SQLite p99 under mounted secure-storage pressure.  Hero rows are five-run medians; kernel rows are "
        + kernel_scope
        + ".}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabularx}{\\columnwidth}{P{0.18\\columnwidth}|P{0.22\\columnwidth}|r|r|r|P{0.18\\columnwidth}}",
        "\\toprule",
        "\\textbf{Row} & \\textbf{Control} & \\textbf{p99} & \\textbf{Miss} & \\textbf{MB/s} & \\textbf{Evidence}\\\\",
        "\\midrule",
    ]
    for row in report["comparison_rows"]:
        lines.append(
            f"{latex_row_label(row)} & {latex_control(row)} & "
            f"{fmt_metric(median(row, 'foreground_p99_ms'))} & "
            f"{fmt_metric(median(row, 'deadline_misses'))} & "
            f"{fmt_metric(median(row, 'background_throughput_mb_s'))} & "
            f"{latex_scope(row)}\\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabularx}",
        "\\label{tab:qos_sqlite_recovery}",
        "\\end{table}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# SQLite Kernel QoS Comparison",
        "",
        f"- Overall pass: `{str(report['overall_pass']).lower()}`",
        f"- B3 code/artifact ready: `{str(report['b3_code_artifact_ready']).lower()}`",
        f"- Parent B3 gate closed: `{str(report['parent_b3_gate_closed']).lower()}`",
        f"- Kernel baselines measured: `{str(report['two_kernel_baselines_measured']).lower()}`",
        f"- Hero modes available: `{str(report['hero_modes_available']).lower()}`",
        "",
        "## Claim Guard",
        "",
        f"- SQLite p99 uniqueness claim allowed: `{str(report['claim_guard']['sqlite_p99_uniqueness_claim_allowed']).lower()}`",
        f"- Paper mentions ionice: `{str(report['paper_claim_scan']['mentions_ionice']).lower()}`",
        f"- Paper mentions systemd IOWeight: `{str(report['paper_claim_scan']['mentions_systemd_io_weight']).lower()}`",
        f"- Paper states repetition mismatch: `{str(report['paper_claim_scan']['mentions_repetition_mismatch']).lower()}`",
        f"- Paper states no-uniqueness boundary: `{str(report['paper_claim_scan']['mentions_no_uniqueness_boundary']).lower()}`",
        f"- Paper SQLite/p99 candidate lines: `{len(report['paper_claim_scan']['sqlite_p99_candidate_lines'])}`",
        "",
        "## Comparison Rows",
        "",
        "| row | type | p99 ms median | deadline misses median | bg MB/s median | runs |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in report["comparison_rows"]:
        label = row.get("mode") or row.get("kernel_control")
        lines.append(
            f"| `{label}` | `{row['row_type']}` | "
            f"{median(row, 'foreground_p99_ms')} | "
            f"{median(row, 'deadline_misses')} | "
            f"{median(row, 'background_throughput_mb_s')} | "
            f"{row.get('runs') or row.get('foreground_p99_ms', {}).get('runs')} |"
        )
    lines.extend([
        "",
        "## Conservative Boundary",
        "",
        "- The kernel-control rows currently have retained repetitions shown in the comparison table.",
        "- The repeated hero artifact and kernel baselines may still have different repetition counts and foreground sample counts, so paper text must avoid broad superiority wording.",
        "- The table includes both kernel controls, but no broad SQLite p99 uniqueness claim is allowed without broader platform and workload coverage.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hero", type=Path, default=DEFAULT_HERO)
    ap.add_argument("--fallback-hero", type=Path, default=DEFAULT_FALLBACK_HERO)
    ap.add_argument("--ionice", type=Path, default=DEFAULT_IONICE)
    ap.add_argument("--cgroup", type=Path, default=DEFAULT_CGROUP)
    ap.add_argument("--paper-dir", type=Path, default=ROOT / "Paper")
    ap.add_argument("--paper-table", type=Path, default=DEFAULT_PAPER_TABLE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    hero_path = args.hero if args.hero.exists() else args.fallback_hero
    args.out_dir.mkdir(parents=True, exist_ok=True)

    hero_summary = summarize_hero(hero_path)
    kernel_rows = [summarize_kernel(args.ionice), summarize_kernel(args.cgroup)]
    controls = {str(row.get("kernel_control")) for row in kernel_rows}
    comparison_rows = hero_summary["rows"] + kernel_rows

    two_kernel_baselines_measured = (
        all(row.get("overall_pass") and row.get("verdict") == "measured" for row in kernel_rows)
        and all(control in controls for control in REQUIRED_KERNEL_CONTROLS)
    )
    hero_modes_available = bool(hero_summary["required_modes_present"])
    b3_code_artifact_ready = two_kernel_baselines_measured and hero_modes_available

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": rel(args.out_dir),
        "overall_pass": b3_code_artifact_ready,
        "b3_code_artifact_ready": b3_code_artifact_ready,
        "parent_b3_gate_closed": False,
        "two_kernel_baselines_measured": two_kernel_baselines_measured,
        "hero_modes_available": hero_modes_available,
        "paper_compares_both_kernel_baselines": False,
        "source_artifacts": {
            "hero": hero_summary["source"],
            "ionice": rel(args.ionice),
            "systemd_io_weight": rel(args.cgroup),
        },
        "hero_summary": hero_summary,
        "kernel_rows": kernel_rows,
        "comparison_rows": comparison_rows,
        "aegis_policy_deltas": delta_rows(comparison_rows),
        "paper_claim_scan": {},
        "claim_guard": {
            "sqlite_p99_uniqueness_claim_allowed": False,
            "required_before_claim": [
                "Paper comparison table includes AEGIS-Q hero modes plus ionice and systemd_io_weight rows.",
                "Paper text states the current repetition mismatch between five-run hero summaries and one-run kernel-control baselines.",
                "Paper explains whether AEGIS-Q wins, loses, or offers different storage-visible control than kernel QoS.",
            ],
            "forbidden_until_closed": [
                "SQLite p99 recovery is unique to AEGIS-Q.",
                "Kernel QoS cannot recover SQLite p99.",
                "AEGIS-Q beats standard kernel throttling for SQLite p99 without qualification.",
            ],
        },
        "comparability_warnings": [
            "Kernel-control rows are currently one repetition each.",
            "The repeated hero summary and kernel baselines have different repetition counts and foreground sample counts.",
            "This artifact supports a paper comparison table, not a broad deployment or superiority claim.",
        ],
    }

    args.paper_table.parent.mkdir(parents=True, exist_ok=True)
    write_latex_table(report, args.paper_table)

    paper_scan = scan_paper_claims(args.paper_dir)
    paper_compares_both_kernel_baselines = (
        bool(paper_scan["mentions_ionice"])
        and bool(paper_scan["mentions_systemd_io_weight"])
        and bool(paper_scan["mentions_kernel_qos"])
        and bool(paper_scan["mentions_repetition_mismatch"])
        and bool(paper_scan["mentions_no_uniqueness_boundary"])
    )
    parent_b3_gate_closed = b3_code_artifact_ready and paper_compares_both_kernel_baselines
    report["paper_claim_scan"] = paper_scan
    report["paper_compares_both_kernel_baselines"] = paper_compares_both_kernel_baselines
    report["parent_b3_gate_closed"] = parent_b3_gate_closed
    report["claim_guard"]["sqlite_p99_uniqueness_claim_allowed"] = parent_b3_gate_closed
    report["paper_outputs"] = {
        "qos_recovery_table": rel(args.paper_table),
    }

    json_path = args.out_dir / "sqlite_kernel_qos_comparison.json"
    md_path = args.out_dir / "sqlite_kernel_qos_comparison.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({
        "out_dir": rel(args.out_dir),
        "overall_pass": report["overall_pass"],
        "parent_b3_gate_closed": report["parent_b3_gate_closed"],
        "two_kernel_baselines_measured": report["two_kernel_baselines_measured"],
        "hero_modes_available": report["hero_modes_available"],
        "paper_compares_both_kernel_baselines": report["paper_compares_both_kernel_baselines"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
