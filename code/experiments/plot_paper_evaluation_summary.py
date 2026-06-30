#!/usr/bin/env python3
"""Render the paper-facing evaluation summary from retained validation JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF = ROOT / "Paper" / "Figures" / "fig_evaluation_summary.pdf"
DEFAULT_PNG = ROOT / "Paper" / "Figures" / "fig_evaluation_summary.png"


FROZEN_ROWS = [
    ("Plain", ROOT / "artifacts/validation/frozen_plaintext_contract/frozen_plaintext_contract.json"),
    ("gocryptfs", ROOT / "artifacts/validation/frozen_gocryptfs_contract/frozen_gocryptfs_contract.json"),
    ("AEGIS-Q", ROOT / "artifacts/validation/frozen_aegisq_contract/frozen_aegisq_contract.json"),
]
SQLITE_CLOSEOUT = ROOT / "artifacts/validation/sqlite_hero_validity_closeout/sqlite_hero_validity_closeout.json"
KERNEL_QOS_CLOSEOUT = (
    ROOT
    / "artifacts/validation/kernel_qos_hero_integration_closeout/kernel_qos_hero_integration_closeout.json"
)
KEYPLANE = ROOT / "artifacts/validation/keyplane_rekey_methodology/keyplane_rekey_workflow.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_triplet(metric: dict) -> tuple[float, float, float]:
    return float(metric["median"]), float(metric["ci95_low"]), float(metric["ci95_high"])


def asym_error(values: list[float], lows: list[float], highs: list[float]) -> np.ndarray:
    return np.array([
        [value - low for value, low in zip(values, lows)],
        [high - value for value, high in zip(values, highs)],
    ])


def frozen_panel(ax) -> None:
    labels: list[str] = []
    vals: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for label, path in FROZEN_ROWS:
        summary = load_json(path)["warm_cache_summary"]["metrics"]["throughput_mib_s"]
        median, low, high = metric_triplet(summary)
        labels.append(label)
        vals.append(median)
        lows.append(low)
        highs.append(high)

    colors = ["#5f6b7a", "#4c78a8", "#c44e52"]
    x = np.arange(len(labels))
    ax.bar(x, vals, color=colors, width=0.64, yerr=asym_error(vals, lows, highs), capsize=2.5)
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel("MiB/s, log")
    ax.set_title("(a) Frozen 4 KiB fdatasync cost")
    ax.grid(axis="y", which="both", alpha=0.25)
    for xpos, value in zip(x, vals):
        ax.text(xpos, value * 1.28, f"{value:.2g}", ha="center", va="bottom", fontsize=6.6)


def sqlite_panel(ax) -> None:
    closeout = load_json(SQLITE_CLOSEOUT)["repeated_methodology_summary"]["mode_summaries"]
    kernel = load_json(KERNEL_QOS_CLOSEOUT)["row_summary"]["kernel_controls"]
    ordered = [
        ("App", closeout["app_only"], "#5f6b7a", ""),
        ("Pressure", closeout["unthrottled_storage"], "#b23a48", ""),
        ("Simple", closeout["simple_controller"], "#8a8f98", ""),
        ("AEGIS-Q", closeout["aegis_policy"], "#197278", ""),
        ("ionice", kernel["ionice"], "#d89000", "//"),
        ("IOWeight", kernel["systemd_io_weight"], "#d89000", "\\\\"),
    ]

    labels: list[str] = []
    vals: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    colors: list[str] = []
    hatches: list[str] = []
    bg: list[float] = []
    for label, row, color, hatch in ordered:
        labels.append(label)
        colors.append(color)
        hatches.append(hatch)
        if "p99_ms" in row and isinstance(row["p99_ms"], dict):
            median, low, high = metric_triplet(row["p99_ms"])
            vals.append(median)
            lows.append(low)
            highs.append(high)
            bg.append(float(row["storage_mb_s"]["median"]))
        elif "p99_ms" in row:
            value = float(row["p99_ms"])
            vals.append(value)
            lows.append(value)
            highs.append(value)
            bg.append(float(row["background_mb_s"]))
        else:
            value = float(row["p99_ms"]["median"])
            vals.append(value)
            lows.append(float(row["p99_ms"]["ci95_low"]))
            highs.append(float(row["p99_ms"]["ci95_high"]))
            bg.append(float(row["storage_mb_s"]["median"]))

    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, width=0.68, yerr=asym_error(vals, lows, highs), capsize=2.3)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.axhline(10.0, color="#222222", linewidth=0.8, linestyle="--")
    ax.text(5.35, 10.35, "10 ms", fontsize=6.4, ha="right")
    ax.set_xticks(x, labels, rotation=24, ha="right")
    ax.set_ylabel("SQLite p99 (ms)")
    ax.set_title("(b) Mounted QoS and kernel controls")
    ax.grid(axis="y", alpha=0.25)
    for xpos, value, mb_s in zip(x, vals, bg):
        ax.text(xpos, value + 0.45, f"{value:.1f}", ha="center", va="bottom", fontsize=6.4)
        ax.text(xpos, 0.45, f"{mb_s:.1f}", ha="center", va="bottom", fontsize=5.9, color="white")


def keyplane_panel(ax) -> None:
    payload = load_json(KEYPLANE)
    rows = {row["mode"]: row for row in payload["mode_summaries"]}
    ordered = [
        ("CPU", rows["cpu_only"], "#3b6fb6"),
        ("GPU batch", rows["gpu_batch"], "#e07a36"),
        ("Fallback", rows["policy_fallback"], "#5f6b7a"),
    ]
    labels = [row[0] for row in ordered]
    vals = [float(row[1]["throughput_files_per_s_median"]) / 1000.0 for row in ordered]
    lows = [float(row[1]["throughput_files_per_s_ci95_low"]) / 1000.0 for row in ordered]
    highs = [float(row[1]["throughput_files_per_s_ci95_high"]) / 1000.0 for row in ordered]
    colors = [row[2] for row in ordered]

    x = np.arange(len(labels))
    ax.bar(x, vals, color=colors, width=0.64, yerr=asym_error(vals, lows, highs), capsize=2.5)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Files/s ($10^3$)")
    ax.set_title("(c) Mounted key-plane refresh")
    ax.grid(axis="y", alpha=0.25)
    speedup = payload["gpu_vs_cpu_speedup_summary"]["median"]
    ax.text(1, max(vals) * 1.04, f"{speedup:.2f}$\\times$", ha="center", va="bottom", fontsize=6.8)
    for xpos, value in zip(x, vals):
        ax.text(xpos, value * 0.08, f"{value:.1f}", ha="center", va="bottom", fontsize=6.3, color="white")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    args = parser.parse_args()

    plt.rcParams.update({"font.size": 7.4, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.25))
    frozen_panel(axes[0])
    sqlite_panel(axes[1])
    keyplane_panel(axes[2])
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=6.4, length=2)
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.28, top=0.83, wspace=0.42)
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.pdf, bbox_inches="tight")
    fig.savefig(args.png, dpi=300, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
