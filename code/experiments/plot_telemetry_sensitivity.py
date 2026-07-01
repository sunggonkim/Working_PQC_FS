#!/usr/bin/env python3
"""Render slack/telemetry admission sensitivity for the paper."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "artifacts" / "results" / "qos" / "m5_admission_sweep.json"
OUT_PDF = ROOT / "Paper" / "Figures" / "fig_telemetry_sensitivity.pdf"
OUT_PNG = ROOT / "Paper" / "Figures" / "fig_telemetry_sensitivity.png"


def main() -> int:
    data = json.loads(INPUT.read_text(encoding="utf-8"))
    budgets = np.array([row["budget_ns"] / 1_000_000 for row in data], dtype=float)
    gpu_jobs = np.array([row["gpu_jobs"] for row in data], dtype=float)
    cpu_jobs = np.array([row["cpu_jobs"] for row in data], dtype=float)
    total_jobs = gpu_jobs + cpu_jobs
    gpu_ratio = np.divide(gpu_jobs, total_jobs, out=np.zeros_like(gpu_jobs), where=total_jobs > 0) * 100.0

    plt.rcParams.update({"font.size": 7.4, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.35, 1.75))
    ax.plot(budgets, gpu_ratio, marker="o", color="#197278", linewidth=1.6, markersize=4.2)
    ax.fill_between(budgets, 0, gpu_ratio, color="#197278", alpha=0.12)
    ax.set_xlabel("Slack budget (ms)")
    ax.set_ylabel("GPU-admitted key jobs (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(budgets)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for x, y, g, c in zip(budgets, gpu_ratio, gpu_jobs.astype(int), cpu_jobs.astype(int)):
        ax.text(x, y + 4, f"{g}/{g + c}", ha="center", va="bottom", fontsize=6.2)
    ax.text(0.98, 0.08, "labels: GPU/total jobs", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=5.8)
    fig.tight_layout(pad=0.3)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
