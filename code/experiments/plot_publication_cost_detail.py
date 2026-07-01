#!/usr/bin/env python3
"""Render strict-publication cost detail from retained validation JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_PDF = ROOT / "Paper" / "Figures" / "fig_publication_cost_detail.pdf"

A4 = ROOT / "artifacts" / "validation" / "a4_hidden_overhead_accounting"
STRICT = ROOT / "artifacts" / "validation" / "strict_path_practicality" / "strict_path_practicality.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_fuse_latency(ax) -> None:
    rows = {row["op"]: row for row in load_json(A4 / "fuse_trace.json")["operations"]}
    ops = ["write", "read", "create", "release", "fsync"]
    labels = ["write", "read", "create", "release", "fsync"]
    values = []
    for op in ops:
        row = rows[op]
        calls = max(1, int(row["calls"]))
        values.append(float(row["total_ns"]) / calls / 1_000_000)

    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=["#4c78a8", "#4c78a8", "#8a8f98", "#8a8f98", "#c44e52"], width=0.64)
    ax.set_yscale("log")
    ax.set_xticks(x, labels, rotation=22, ha="right")
    ax.set_ylabel("Mean daemon latency (ms, log)")
    ax.set_title("(a) FUSE operation boundary")
    ax.grid(axis="y", which="both", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.18, f"{value:.2g}",
                ha="center", va="bottom", fontsize=6.1)


def plot_barrier_model(ax) -> None:
    payload = load_json(STRICT)["metrics"]
    strict = payload["strict_single_client_cost_boundary"]
    hybrid = payload["hybrid_epoch_grouped_current_100us"]
    loss = payload["hybrid_epoch_fault_matrix_loss_boundary"]

    labels = ["strict\nsingle", "strict\ngroup", "epoch\ngroup", "epoch\nloss row"]
    p99 = [
        strict["frozen_aegisq_p99_us_median"] / 1000.0,
        hybrid["strict_p99_ms"],
        hybrid["epoch_p99_ms"],
        loss["epoch_p99_ms"],
    ]
    syncs = [
        strict["strict_sync_family_ops_per_cycle_after_x6"],
        hybrid["strict_sync_count_total"],
        hybrid["epoch_sync_count_total"],
        loss["epoch_sync_count_total"],
    ]
    colors = ["#c44e52", "#5f6b7a", "#197278", "#b36b00"]
    x = np.arange(len(labels))
    bars = ax.bar(x, p99, color=colors, width=0.62)
    ax.set_xticks(x, labels)
    ax.set_ylabel("p99 latency (ms)")
    ax.set_title("(b) Strict vs. grouped publication")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, max(p99) * 1.22)
    for bar, value, sync in zip(bars, p99, syncs):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(p99) * 0.025,
                f"{value:.1f}", ha="center", va="bottom", fontsize=6.1)
        ax.text(bar.get_x() + bar.get_width() / 2, value * 0.18,
                f"{sync:g} sync", ha="center", va="center", fontsize=5.8, color="white")
    ax.text(0.02, 0.96, "labels: traced sync-family ops", transform=ax.transAxes,
            ha="left", va="top", fontsize=5.8)


def main() -> int:
    plt.rcParams.update({"font.size": 7.2, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.25), gridspec_kw={"width_ratios": [1.0, 1.22]})
    plot_fuse_latency(axes[0])
    plot_barrier_model(axes[1])
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=6.4, length=2)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.27, top=0.84, wspace=0.34)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
