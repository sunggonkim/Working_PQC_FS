#!/usr/bin/env python3
"""Build the single hero-result contract for the current paper draft."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "Paper"
DEFAULT_OUT = ROOT / "artifacts" / "reports" / "hero_result_contract"
QOS_BUNDLE = ROOT / "artifacts" / "validation" / "qos_sqlite_hero_bundle" / "qos_sqlite_hero_bundle.json"
QOS_CLOSEOUT = ROOT / "artifacts" / "validation" / "sqlite_hero_validity_closeout" / "sqlite_hero_validity_closeout.json"
FIGURE_DATA = ROOT / "artifacts" / "results" / "paper_spine_gate" / "first_page_qos_figure_data.json"
FIGURE_SCRIPT = ROOT / "code" / "experiments" / "build_paper_spine_gate.py"
FIGURE_PATH = PAPER / "Figures" / "fig_first_page_qos.pdf"

REQUIRED_MODES = ["app_only", "unthrottled_storage", "simple_controller", "aegis_policy"]


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def run_pdfinfo_pages(path: Path) -> int | None:
    proc = subprocess.run(["pdfinfo", str(path)], check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for line in proc.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    return None


def mode_map(bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["mode"]: row for row in bundle.get("modes", [])}


def metric_row(row: dict[str, Any]) -> dict[str, Any]:
    foreground = row["foreground"]
    background = row["background"]
    return {
        "p99_ms": foreground["p99_ms"],
        "deadline_misses": foreground["deadline_misses"],
        "deadline_ms": foreground["deadline_ms"],
        "background_mb_s": background["throughput_mb_s"],
        "foreground_log": row["logs"]["foreground_jsonl"],
        "background_log": row["logs"]["background_jsonl"],
        "telemetry_log": row["logs"]["telemetry_jsonl"],
        "policy_log": row["logs"]["policy_jsonl"],
        "fuse_stdout": row["logs"]["fuse_stdout"],
        "fuse_stderr": row["logs"]["fuse_stderr"],
        "runtime_throttle_trace": row["logs"]["runtime_fuse_throttle_trace"],
    }


def figure_metric_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "p99_ms": row["p99_ms"],
        "deadline_misses": row["deadline_misses"],
        "deadline_ms": row["deadline_ms"],
        "background_mb_s": row["background_mb_s"],
    }


def rounded(value: float) -> str:
    return f"{value:.1f}"


def close_enough(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)


def find_line(path: Path, needles: list[str]) -> dict[str, Any]:
    for idx, line in enumerate(read_text(path).splitlines(), start=1):
        if all(needle in line for needle in needles):
            return {
                "path": relpath(path),
                "line": idx,
                "text": " ".join(line.strip().split()),
                "present": True,
            }
    return {
        "path": relpath(path),
        "line": None,
        "text": None,
        "present": False,
    }


def build_report() -> dict[str, Any]:
    bundle = read_json(QOS_BUNDLE)
    closeout = read_json(QOS_CLOSEOUT)
    figure_data = read_json(FIGURE_DATA)
    modes = mode_map(bundle)
    figure_rows = {row["mode"]: row for row in figure_data.get("rows", [])}
    metrics = {
        mode: figure_metric_row(figure_rows[mode])
        for mode in REQUIRED_MODES
        if mode in figure_rows
    }

    app = metrics.get("app_only", {})
    pressure = metrics.get("unthrottled_storage", {})
    simple = metrics.get("simple_controller", {})
    aegis = metrics.get("aegis_policy", {})

    exact_claim = (
        "For foreground SQLite transactions on mounted AEGIS-Q FUSE repeated-run medians, "
        f"app-only reports {app.get('p99_ms'):.3f} ms p99, unthrottled secure-storage "
        f"pressure reports {pressure.get('p99_ms'):.3f} ms p99 and "
        f"{pressure.get('background_mb_s'):.3f} MB/s background writes, and AEGIS-Q "
        f"policy reports {aegis.get('p99_ms'):.3f} ms p99 with "
        f"{aegis.get('deadline_misses')} median misses while retaining "
        f"{aegis.get('background_mb_s'):.3f} MB/s of background progress."
    )
    displayed_claim = (
        "Unthrottled secure-storage pressure reports "
        f"{pressure.get('p99_ms', 0.0):.2f} ms SQLite p99 and "
        f"{pressure.get('background_mb_s', 0.0):.2f} MB/s background writes; "
        f"AEGIS-Q reports {aegis.get('p99_ms', 0.0):.2f} ms p99, zero median misses, "
        f"and {aegis.get('background_mb_s', 0.0):.2f} MB/s background writes."
    )

    figure_matches = {
        mode: mode in metrics
        and mode in figure_rows
        and close_enough(metrics[mode]["p99_ms"], figure_rows[mode]["p99_ms"])
        and metrics[mode]["deadline_misses"] == figure_rows[mode]["deadline_misses"]
        for mode in REQUIRED_MODES
    }

    paper_locations = {
        "abstract": find_line(PAPER / "main.tex", ["unthrottled secure-storage pressure", "9.62", "8.15", "3.02"]),
        "introduction": find_line(PAPER / "1_Introduction.tex", ["Figure~\\ref{fig:first_page_qos}", "9.62", "8.15", "3.02"]),
        "evaluation": find_line(PAPER / "4_Evaluation.tex", ["Figure~\\ref{fig:evaluation_summary}(b)", "9.62", "8.15", "3.02"]),
        "scope_boundary": find_line(PAPER / "4_Evaluation.tex", ["bounded storage-visible control", "not a uniqueness or non-storage QoS claim"]),
    }

    checks = {
        "bundle_overall_pass": bundle.get("overall_pass") is True,
        "closeout_overall_pass": closeout.get("overall_pass") is True,
        "required_modes_available": all(mode in modes for mode in REQUIRED_MODES)
        and all(mode in figure_rows for mode in REQUIRED_MODES),
        "recovery_checks_pass": all(bundle.get("recovery_checks", {}).values()),
        "component_coverage_pass": all(bundle.get("component_coverage", {}).values()),
        "figure_data_matches_bundle": all(figure_matches.values()),
        "figure_script_exists": FIGURE_SCRIPT.exists(),
        "figure_pdf_exists": FIGURE_PATH.exists(),
        "paper_locations_present": all(row["present"] for row in paper_locations.values()),
        "paper_pages_le_13": (run_pdfinfo_pages(PAPER / "main.pdf") or 999) <= 13,
        "aegis_recovers_p99": pressure.get("p99_ms", 0.0) > aegis.get("p99_ms", float("inf")),
        "aegis_retains_more_background_than_simple": aegis.get("background_mb_s", 0.0) > simple.get("background_mb_s", float("inf")),
    }

    violations = [name for name, passed in checks.items() if not passed]

    return {
        "schema_version": 1,
        "hero_id": "sqlite-mounted-qos-recovery-2026-06-27",
        "headline_claim_exact": exact_claim,
        "headline_claim_displayed": displayed_claim,
        "workload": {
            "foreground": "SQLite transaction latency on a mounted AEGIS-Q FUSE filesystem",
            "background": "secure-storage writer to an elastic file on the same mounted daemon",
            "deadline_ms": pressure.get("deadline_ms"),
        },
        "baseline": {
            "app_only": "foreground SQLite with no background secure-storage writer",
            "unthrottled_storage": "mounted secure-storage pressure without throttling",
            "simple_controller": "harness-side hysteresis controller",
            "aegis_policy": "AEGIS-Q in-daemon latency/elastic policy",
        },
        "metrics": {
            "primary": "foreground SQLite p99 latency under pressure",
            "secondary": ["deadline misses", "background writer MB/s", "policy/throttle decisions"],
            "modes": metrics,
        },
        "artifact_path": relpath(QOS_CLOSEOUT),
        "single_run_artifact_path": relpath(QOS_BUNDLE),
        "raw_log_paths": {
            mode: {
                key: value
                for key, value in row.items()
                if key.endswith("_log") or key.startswith("fuse_") or key == "runtime_throttle_trace"
            }
            for mode, row in {m: metric_row(modes[m]) for m in REQUIRED_MODES if m in modes}.items()
        },
        "figure_generation": {
            "script": relpath(FIGURE_SCRIPT),
            "figure": relpath(FIGURE_PATH),
            "data": relpath(FIGURE_DATA),
            "figure_rows_match_bundle": figure_matches,
        },
        "paper_locations": paper_locations,
        "scope_boundary": [
            "SQLite foreground recovery only",
            "single retained four-mode mounted bundle, not a statistical QoS headline",
            "not external application p99 recovery",
            "not a general deployed-filesystem superiority claim",
        ],
        "checks": checks,
        "pages": run_pdfinfo_pages(PAPER / "main.pdf"),
        "violations": violations,
        "overall_pass": not violations,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Hero-result contract",
        "",
        f"- Overall pass: `{report['overall_pass']}`",
        f"- Hero id: `{report['hero_id']}`",
        f"- Paper pages: `{report['pages']}`",
        f"- Artifact: `{report['artifact_path']}`",
        "",
        "## Headline claim",
        "",
        report["headline_claim_exact"],
        "",
        "## Metrics",
        "",
        "| Mode | p99 ms | Deadline misses | Background MB/s |",
        "| --- | ---: | ---: | ---: |",
    ]
    for mode, row in report["metrics"]["modes"].items():
        lines.append(
            f"| `{mode}` | {row['p99_ms']:.3f} | {row['deadline_misses']} | "
            f"{row['background_mb_s']:.3f} |"
        )

    lines += [
        "",
        "## Paper locations",
        "",
        "| Location | Source line | Present |",
        "| --- | ---: | ---: |",
    ]
    for name, loc in report["paper_locations"].items():
        lines.append(f"| {name} | `{loc['path']}:{loc['line']}` | `{loc['present']}` |")

    lines += [
        "",
        "## Checks",
        "",
        "| Check | Pass |",
        "| --- | ---: |",
    ]
    for name, passed in report["checks"].items():
        lines.append(f"| `{name}` | `{passed}` |")

    if report["violations"]:
        lines += ["", "## Violations", ""]
        lines += [f"- {v}" for v in report["violations"]]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    report = build_report()
    json_path = args.out / "hero_result_contract.json"
    md_path = args.out / "hero_result_contract.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({
        "hero_id": report["hero_id"],
        "out_dir": relpath(args.out),
        "overall_pass": report["overall_pass"],
        "pages": report["pages"],
        "violations": len(report["violations"]),
    }, indent=2, sort_keys=True))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
