#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "motivation"


def load(path: str):
    return json.loads((ART / path).read_text())


def main() -> int:
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    series = [
        ("sqlite_contention_latency.json", "SQLite", "tier", ["plain", "full"]),
        ("inference_latency.json", "Inference", "mode", ["baseline", "yolov8_inference"]),
        ("spillover_latency.json", "Spillover", "mode", None),
        ("pipeline_latency.json", "Pipeline", "mode", None),
    ]
    colors = ["#4C78A8", "#F58518"]
    x = 0
    xticks = []
    labels = []
    for file, label, key, order in series:
        rows = load(file)
        if order is None:
            order = [r[key] for r in rows[:2]]
        for i, mode in enumerate(order):
            row = next(r for r in rows if r[key] == mode)
            median = row["median_ms"]
            hi = row.get("p95_ms") or row.get("p99_ms") or median
            ax.bar(x, median, width=0.35, color=colors[i % 2])
            ax.errorbar(x, median, yerr=[[0], [hi - median]], fmt="none", ecolor="black", capsize=3, lw=0.8)
            xticks.append(x)
            labels.append(f"{label}\n{mode}")
            x += 1
        x += 0.4
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("ms")
    ax.set_title("Median and tail latency across representative paths")
    fig.tight_layout()
    fig.savefig(ART / "tail_latency_summary.png", dpi=200)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
