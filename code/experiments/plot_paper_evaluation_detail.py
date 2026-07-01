#!/usr/bin/env python3
"""Render compact evaluation-detail figures from retained validation JSON."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_PDF = ROOT / "Paper" / "Figures" / "fig_recovery_qos_detail.pdf"
OUT_PNG = ROOT / "Paper" / "Figures" / "fig_recovery_qos_detail.png"

GENERATION = ROOT / "artifacts/validation/generation_fault_matrix/generation_fault_matrix.json"
DAEMON = ROOT / "artifacts/validation/daemon_power_fault_campaign/daemon_power_fault_campaign.json"
TPM = ROOT / "artifacts/validation/hardware_freshness_recovery_matrix/hardware_freshness_recovery_matrix.json"
QOS = ROOT / "artifacts/validation/qos_sensitivity_analysis/qos_sensitivity_analysis.json"
ADMISSION = ROOT / "artifacts/results/qos/m5_admission_sweep.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verdict_counts() -> tuple[list[str], dict[str, list[int]]]:
    sources = [
        ("Generation", load_json(GENERATION)["rows"], "oracle_verdict"),
        ("Daemon", load_json(DAEMON)["daemon_rows"], "verdict"),
        ("TPM/replay", load_json(TPM)["rows"], "verdict"),
    ]
    verdicts = [
        "latest_committed",
        "previous_committed",
        "fail_closed",
        "fail_closed_update_not_committed",
        "transient_probe_rejects_drift",
    ]
    labels: list[str] = []
    series = {verdict: [] for verdict in verdicts}
    for label, rows, key in sources:
        counts = Counter(row.get(key) or row.get("oracle_verdict") for row in rows)
        labels.append(label)
        for verdict in verdicts:
            series[verdict].append(counts.get(verdict, 0))
    return labels, series


def plot_recovery(ax) -> None:
    labels, series = verdict_counts()
    colors = {
        "latest_committed": "#197278",
        "previous_committed": "#4c78a8",
        "fail_closed": "#8a8f98",
        "fail_closed_update_not_committed": "#b36b00",
        "transient_probe_rejects_drift": "#7f3c8d",
    }
    names = {
        "latest_committed": "latest",
        "previous_committed": "previous",
        "fail_closed": "fail closed",
        "fail_closed_update_not_committed": "update refused",
        "transient_probe_rejects_drift": "PCR drift",
    }
    left = np.zeros(len(labels))
    y = np.arange(len(labels))
    for verdict, values in series.items():
        ax.barh(y, values, left=left, height=0.56, color=colors[verdict], label=names[verdict])
        for ypos, xpos, value in zip(y, left, values):
            if value:
                ax.text(xpos + value / 2, ypos, str(value), ha="center", va="center", fontsize=6.4, color="white")
        left += np.array(values)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Rows")
    ax.set_title("(a) Recovery oracle coverage")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, fontsize=5.7, ncol=3, loc="upper left", bbox_to_anchor=(-0.03, -0.25))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_qos(ax) -> None:
    rows = load_json(QOS)["cases"]
    keep = [
        "baseline",
        "tight_budget",
        "slow_sampling",
        "high_threshold",
        "queue_depth_2",
        "background_128k",
        "low_pressure_no_throttle",
        "hysteresis_wave",
    ]
    labels = {
        "baseline": "default",
        "tight_budget": "8 ms\nbudget",
        "slow_sampling": "80 ms\nsample",
        "high_threshold": "0.90\nthreshold",
        "queue_depth_2": "qd=2",
        "background_128k": "128 KiB\nchunks",
        "low_pressure_no_throttle": "low\npressure",
        "hysteresis_wave": "pressure\nwave",
    }
    by_case = {row["case"]: row for row in rows}
    selected = [by_case[name] for name in keep if name in by_case]
    x = np.arange(len(selected))
    p99 = [float(row["p99_ms"]) for row in selected]
    mbps = [float(row["storage_mb_s"]) for row in selected]
    misses = [int(row["deadline_misses"]) for row in selected]
    colors = ["#197278" if miss == 0 else "#b23a48" for miss in misses]
    bars = ax.bar(x, p99, color=colors, width=0.62)
    ax.axhline(10.0, color="#222222", linewidth=0.8, linestyle="--")
    ax.text(len(selected) - 0.25, 10.18, "10 ms", ha="right", va="bottom", fontsize=6.2)
    ax.set_xticks(x, [labels[row["case"]] for row in selected])
    ax.set_ylabel("SQLite p99 (ms)")
    ax.set_title("(b) QoS sensitivity envelope")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, max(12.8, max(p99) * 1.18))
    for bar, value, bg, miss in zip(bars, p99, mbps, misses):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.25, f"{value:.1f}", ha="center", va="bottom", fontsize=6.0)
        ax.text(bar.get_x() + bar.get_width() / 2, 0.35, f"{bg:.1f}", ha="center", va="bottom", fontsize=5.6, color="white")
        if miss:
            ax.text(bar.get_x() + bar.get_width() / 2, value * 0.50, f"{miss} miss", ha="center", va="center", fontsize=5.5, color="white")
    ax.text(0.01, 0.98, "white text: background MB/s", transform=ax.transAxes, ha="left", va="top", fontsize=5.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_admission(ax) -> None:
    rows = load_json(ADMISSION)
    budgets = np.array([row["budget_ns"] / 1_000_000 for row in rows], dtype=float)
    gpu_jobs = np.array([row["gpu_jobs"] for row in rows], dtype=float)
    cpu_jobs = np.array([row["cpu_jobs"] for row in rows], dtype=float)
    total_jobs = gpu_jobs + cpu_jobs
    gpu_ratio = np.divide(gpu_jobs, total_jobs, out=np.zeros_like(gpu_jobs), where=total_jobs > 0) * 100.0

    ax.plot(budgets, gpu_ratio, marker="o", color="#197278", linewidth=1.4, markersize=3.8)
    ax.fill_between(budgets, 0, gpu_ratio, color="#197278", alpha=0.12)
    ax.set_xlabel("Slack budget (ms)")
    ax.set_ylabel("GPU-admitted jobs (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(budgets)
    ax.set_title("(c) Elastic key-plane admission")
    ax.grid(axis="y", alpha=0.25)
    for x, y, g, c in zip(budgets, gpu_ratio, gpu_jobs.astype(int), cpu_jobs.astype(int)):
        ax.text(x, y + 4, f"{g}/{g + c}", ha="center", va="bottom", fontsize=5.6)
    ax.text(0.99, 0.08, "GPU/total jobs", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=5.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> int:
    plt.rcParams.update({"font.size": 7.2, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.25), gridspec_kw={"width_ratios": [0.9, 1.18, 0.92]})
    plot_recovery(axes[0])
    plot_qos(axes[1])
    plot_admission(axes[2])
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.35, top=0.82, wspace=0.43)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
