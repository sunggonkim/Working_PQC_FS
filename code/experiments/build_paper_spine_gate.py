#!/usr/bin/env python3
"""Generate and audit the AEGIS-Q first-page paper spine.

The gate is deliberately narrow.  It builds a quantitative first-page figure
from the retained mounted SQLite pressure bundle, then checks that the paper
contains the required first-page capability table, states the positive
contribution before defensive non-claim language, and maps each contribution to
both a design mechanism and an evaluation result.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "Paper"
FIGURES = PAPER / "Figures"
DEFAULT_REPORT_DIR = ROOT / "artifacts" / "reports" / "paper_spine_gate"
DEFAULT_RESULT_DIR = ROOT / "artifacts" / "results" / "paper_spine_gate"
QOS_CLOSEOUT = ROOT / "artifacts" / "validation" / "sqlite_hero_validity_closeout" / "sqlite_hero_validity_closeout.json"
QOS_BUNDLE = ROOT / "artifacts" / "validation" / "qos_sqlite_hero_bundle" / "qos_sqlite_hero_bundle.json"

MODE_LABELS = {
    "app_only": "App",
    "unthrottled_storage": "Pressure",
    "simple_controller": "Simple",
    "aegis_policy": "AEGIS-Q",
}

REQUIRED_TABLE_TERMS = [
    "Plaintext",
    "gocryptfs",
    "fscrypt",
    "dm-crypt",
    "fs-verity",
    "dm-integrity",
    "TPM/TEE",
    "GPU-storage",
    "AEGIS-Q",
]

DEFENSIVE_PREFIX_TERMS = [
    "does not claim",
    "do not claim",
    "not claimed",
    "not a replacement",
    "stopping short",
    "deliberately does not",
]

CONTRIBUTION_MAP = [
    {
        "id": "C1",
        "title": "Placement-safe storage format",
        "intro_needles": ["C1: Placement-safe storage format", "generation-bound AEAD", "persistent record format"],
        "design_needles": ["\\label{sec:design_pipeline}", "Data-before-mapping publication"],
        "evaluation_needles": ["\\label{sec:eval_workloads}", "generation fault matrix", "strict and epoch modes"],
    },
    {
        "id": "C2",
        "title": "CPU data lane, GPU/PQC maintenance lane",
        "intro_needles": ["C2: CPU data lane, GPU/PQC maintenance lane", "AES-GCM writes stay on the CPU", "ML-KEM envelope-refresh batches"],
        "design_needles": ["\\label{sec:design_uma}", "ML-KEM envelope refresh"],
        "evaluation_needles": ["\\label{sec:eval_performance}", "CPU/OpenSSL", "mounted ML-KEM-768 key-plane workflow"],
    },
    {
        "id": "C3",
        "title": "Recovery and replay boundary",
        "intro_needles": ["C3: Recovery and replay boundary", "replayable file witnesses", "replay-after-advance"],
        "design_needles": ["\\label{sec:design_security}", "TPM NV index", "not} rollback protection"],
        "evaluation_needles": ["replay-after-advance evidence", "Recovery/replay outcomes"],
    },
    {
        "id": "C4",
        "title": "Storage-visible QoS control",
        "intro_needles": ["C4: Storage-visible QoS control", "mounted SQLite result", "Figure~\\ref{fig:first_page_qos}"],
        "design_needles": ["user.pqc\\_qos\\_class", "mounted-FUSE throttling"],
        "evaluation_needles": ["\\label{sec:eval_qos}", "SQLite transaction latency", "bounded storage-visible control"],
    },
]


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_qos_rows() -> tuple[str, list[dict[str, Any]]]:
    if QOS_CLOSEOUT.exists():
        closeout = load_json(QOS_CLOSEOUT)
        summaries = closeout["repeated_methodology_summary"]["mode_summaries"]
        rows = []
        for mode, label in MODE_LABELS.items():
            summary = summaries[mode]
            rows.append({
                "mode": mode,
                "label": label,
                "p99_ms": float(summary["p99_ms"]["median"]),
                "deadline_misses": int(summary["deadline_misses"]["median"]),
                "deadline_ms": 10.0,
                "background_mb_s": float(summary["storage_mb_s"]["median"]),
                "telemetry_samples": 0,
                "acceptable": True,
            })
        return relpath(QOS_CLOSEOUT), rows

    bundle = load_json(QOS_BUNDLE)
    rows: list[dict[str, Any]] = []
    for mode in bundle["modes"]:
        name = mode["mode"]
        if name not in MODE_LABELS:
            continue
        fg = mode["foreground"]
        bg = mode.get("background") or {}
        rows.append({
            "mode": name,
            "label": MODE_LABELS[name],
            "p99_ms": float(fg["p99_ms"]),
            "deadline_misses": int(fg["deadline_misses"]),
            "deadline_ms": float(fg["deadline_ms"]),
            "background_mb_s": float(bg.get("throughput_mb_s", 0.0) or 0.0),
            "telemetry_samples": int((mode.get("telemetry") or {}).get("rows", 0) or 0),
            "acceptable": bool(mode.get("acceptable")),
        })
    return relpath(QOS_BUNDLE), rows


def write_figure_data(source: str, rows: list[dict[str, Any]], result_dir: Path) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    json_path = result_dir / "first_page_qos_figure_data.json"
    csv_path = result_dir / "first_page_qos_figure_data.csv"
    json_path.write_text(json.dumps({
        "source": source,
        "rows": rows,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def generate_figure(rows: list[dict[str, Any]], fig_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["label"] for row in rows]
    p99 = [row["p99_ms"] for row in rows]
    misses = [row["deadline_misses"] for row in rows]
    bg = [row["background_mb_s"] for row in rows]
    colors = ["#5f6b7a", "#b23a48", "#8a8f98", "#197278"]

    fig, ax = plt.subplots(figsize=(3.15, 1.55))
    bars = ax.bar(labels, p99, color=colors, width=0.66)
    ax.axhline(rows[0]["deadline_ms"], color="#222222", linewidth=0.8, linestyle="--")
    ax.text(3.1, rows[0]["deadline_ms"] + 0.22, "10 ms SLO", fontsize=6.4, ha="right")
    ax.set_ylabel("SQLite p99 (ms)", fontsize=7.2)
    ax.set_ylim(0, max(p99) + 2.2)
    ax.tick_params(axis="both", labelsize=6.8, length=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.45, alpha=0.55)
    for bar, value, miss, mb_s in zip(bars, p99, misses, bg):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.28,
                f"{value:.1f}", ha="center", va="bottom", fontsize=6.8)
        ax.text(bar.get_x() + bar.get_width() / 2, 0.42,
                f"{miss} miss\n{mb_s:.1f} MB/s", ha="center", va="bottom",
                fontsize=5.9, color="white" if value > 3.0 else "#222222")
    fig.tight_layout(pad=0.25)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path)
    plt.close(fig)


def run_pdftotext(path: Path, first_page: int | None = None,
                  last_page: int | None = None) -> str:
    cmd = ["pdftotext", "-layout"]
    if first_page is not None:
        cmd += ["-f", str(first_page)]
    if last_page is not None:
        cmd += ["-l", str(last_page)]
    cmd += [str(path), "-"]
    proc = subprocess.run(cmd, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.stdout


def run_pdfinfo_pages(path: Path) -> int | None:
    proc = subprocess.run(["pdfinfo", str(path)], check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for line in proc.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    return None


def all_present(text: str, needles: list[str]) -> bool:
    return all(needle in text for needle in needles)


def pdf_term_present(text: str, term: str) -> bool:
    pattern = r"\s*".join(re.escape(part) for part in term.split())
    pattern = pattern.replace(r"\-", r"-\s*")
    return re.search(pattern, text, re.IGNORECASE) is not None


def audit_paper(source: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    intro = (PAPER / "1_Introduction.tex").read_text(encoding="utf-8")
    design = (PAPER / "3_Design.tex").read_text(encoding="utf-8")
    eval_tex = (PAPER / "4_Evaluation.tex").read_text(encoding="utf-8")
    main = (PAPER / "main.tex").read_text(encoding="utf-8")
    first_page = run_pdftotext(PAPER / "main.pdf", 1, 1)
    full_pdf = run_pdftotext(PAPER / "main.pdf")
    abstract = main.split("\\begin{abstract}", 1)[1].split("\\end{abstract}", 1)[0]
    prefix = (abstract + "\n" + intro.split("This paper makes four contributions.", 1)[0]).lower()

    contribution_rows = []
    for item in CONTRIBUTION_MAP:
        intro_present = all_present(intro, item["intro_needles"])
        design_present = all_present(design, item["design_needles"])
        eval_present = all_present(eval_tex, item["evaluation_needles"])
        contribution_rows.append({
            "id": item["id"],
            "title": item["title"],
            "intro_map_present": intro_present,
            "design_present": design_present,
            "evaluation_present": eval_present,
            "mapped": intro_present and design_present and eval_present,
        })

    figure_values_present = all(
        (f"{row['p99_ms']:.1f}" in first_page or f"{row['p99_ms']:.2f}" in first_page)
        for row in rows
        if row["mode"] in {"app_only", "unthrottled_storage", "aegis_policy"}
    )
    table_terms_present = {
        term: pdf_term_present(first_page, term) or term in intro
        for term in REQUIRED_TABLE_TERMS
    }
    defensive_hits = [term for term in DEFENSIVE_PREFIX_TERMS if term in prefix]

    return {
        "pages": run_pdfinfo_pages(PAPER / "main.pdf"),
        "first_page_figure": {
            "present": "Figure 1:" in first_page and "SQLite p99" in first_page,
            "values_present": figure_values_present,
            "source": source,
            "figure": "Paper/Figures/fig_first_page_qos.pdf",
        },
        "first_page_table": {
            "present": "Table 1:" in first_page and "capability" in first_page.lower(),
            "required_terms": table_terms_present,
            "all_required_terms_present": all(table_terms_present.values()),
        },
        "positive_before_defensive": {
            "pass": not defensive_hits,
            "defensive_hits_before_contributions": defensive_hits,
        },
        "contribution_map": contribution_rows,
    }


def build_report(source: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    audit = audit_paper(source, rows)
    violations: list[str] = []
    if audit["pages"] != 12:
        violations.append("Paper/main.pdf is not exactly 12 pages")
    if not audit["first_page_figure"]["present"]:
        violations.append("first-page Figure 1 is missing")
    if not audit["first_page_figure"]["values_present"]:
        violations.append("first-page Figure 1 does not expose retained p99 values")
    if not audit["first_page_table"]["present"]:
        violations.append("first-page Table 1 is missing")
    if not audit["first_page_table"]["all_required_terms_present"]:
        violations.append("first-page Table 1 lacks one or more required comparison classes")
    if not audit["positive_before_defensive"]["pass"]:
        violations.append("defensive non-claim language appears before the contribution statement")
    if not all(row["mapped"] for row in audit["contribution_map"]):
        violations.append("one or more contributions lacks an intro/design/evaluation mapping")

    return {
        "schema_version": 1,
        "scope": [
            "first-page quantitative figure generated from retained SQLite QoS data",
            "first-page capability table source/PDF check",
            "positive-before-defensive ordering and contribution mapping",
        ],
        "figure_data": {
            "source": source,
            "rows": rows,
            "json": "artifacts/results/paper_spine_gate/first_page_qos_figure_data.json",
            "csv": "artifacts/results/paper_spine_gate/first_page_qos_figure_data.csv",
        },
        "paper_audit": audit,
        "violations": violations,
        "overall_pass": not violations,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Paper spine gate",
        "",
        f"- Overall pass: `{report['overall_pass']}`",
        f"- Page count: `{report['paper_audit']['pages']}`",
        f"- Figure source: `{report['figure_data']['source']}`",
        "",
        "## First-page figure data",
        "",
        "| Mode | p99 ms | Deadline misses | Background MB/s |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in report["figure_data"]["rows"]:
        lines.append(
            f"| {row['label']} | {row['p99_ms']:.3f} | "
            f"{row['deadline_misses']} | {row['background_mb_s']:.3f} |"
        )
    audit = report["paper_audit"]
    lines += [
        "",
        "## Gate checks",
        "",
        f"- First-page Figure 1 present: `{audit['first_page_figure']['present']}`",
        f"- First-page Figure 1 values present: `{audit['first_page_figure']['values_present']}`",
        f"- First-page Table 1 present: `{audit['first_page_table']['present']}`",
        f"- Required Table 1 terms present: `{audit['first_page_table']['all_required_terms_present']}`",
        f"- Positive-before-defensive ordering: `{audit['positive_before_defensive']['pass']}`",
        "",
        "## Contribution map",
        "",
        "| Contribution | Intro map | Design | Evaluation | Mapped |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in audit["contribution_map"]:
        lines.append(
            f"| {row['id']} {row['title']} | `{row['intro_map_present']}` | "
            f"`{row['design_present']}` | `{row['evaluation_present']}` | `{row['mapped']}` |"
        )
    if report["violations"]:
        lines += ["", "## Violations", ""]
        lines += [f"- {v}" for v in report["violations"]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    args = parser.parse_args()

    source, rows = load_qos_rows()
    write_figure_data(source, rows, args.result_dir)
    generate_figure(rows, FIGURES / "fig_first_page_qos.pdf")
    report = build_report(source, rows)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.report_dir / "paper_spine_gate.json"
    md_path = args.report_dir / "paper_spine_gate.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({
        "out_dir": relpath(args.report_dir),
        "overall_pass": report["overall_pass"],
        "pages": report["paper_audit"]["pages"],
        "violations": len(report["violations"]),
    }, indent=2, sort_keys=True))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
