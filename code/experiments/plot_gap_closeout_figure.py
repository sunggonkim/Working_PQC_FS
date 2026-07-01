#!/usr/bin/env python3
"""Render gap-closeout figure for SOSP review weaknesses.

Panels:
(a) Workload breadth (SQLite + model-cache + secure log)
(b) PQC lane utility across mounted file-batch sizes
(c) Kernel QoS breadth with measured and blocked controls
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "Paper" / "Figures"
FIG.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def workload_panel(ax):
    sqlite = load_json(ROOT / "artifacts" / "validation" / "qos_sqlite_hero_bundle" / "qos_sqlite_hero_bundle.json")
    model = load_json(ROOT / "artifacts" / "validation" / "model_cache_manifest_workload_ext" / "model_cache_manifest_workload.json")
    slog = load_json(ROOT / "artifacts" / "validation" / "secure_inference_log_macro_ext" / "secure_inference_log_macro_smoke.json")

    sqlite_p99_ms = None
    for m in sqlite.get("modes", []):
        if m.get("mode") == "aegis_policy":
            sqlite_p99_ms = float(m.get("foreground", {}).get("p99_ms", 0.0))
            break

    model_p99_ms = float(model["publish"]["summary"]["publish_latency"]["p99_us"]) / 1000.0
    slog_p99_ms = float(slog["append"]["summary"]["durable_append_latency"]["p99_us"]) / 1000.0

    labels = ["SQLite\n(txn)", "Cache-manifest\n(publish)", "Secure-log\n(append)"]
    vals = [sqlite_p99_ms or 0.0, model_p99_ms, slog_p99_ms]
    bars = ax.bar(labels, vals, color=["#365f9c", "#2e8b57", "#b5651d"])
    ax.set_ylabel("p99 latency (ms)")
    ax.set_title("(a) Workload breadth", fontsize=9)
    ax.set_ylim(0, max(vals) * 1.35)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.2f}", ha="center", va="bottom", fontsize=8)


def pqc_panel(ax):
    sweep_root = ROOT / "artifacts" / "validation" / "keyplane_rekey_sweep"
    points = []
    for p in sorted(sweep_root.glob("files_*/keyplane_rekey_workflow.json"), key=lambda x: int(x.parent.name.split("_")[1])):
        j = load_json(p)
        modes = {m["mode"]: m for m in j.get("modes", [])}
        cpu = float(modes.get("cpu_only", {}).get("total_rekey_usec", 0.0))
        gpu = float(modes.get("gpu_batch", {}).get("total_rekey_usec", 0.0))
        if cpu > 0 and gpu > 0:
            points.append((int(j.get("files_per_mode", 0)), cpu / gpu))

    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    ax.plot(xs, ys, marker="o", color="#7a3db8", linewidth=1.8)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("refreshed files per run")
    ax.set_ylabel("GPU/CPU speedup")
    ax.set_title("(b) PQC lane utility", fontsize=9)
    if ys:
        ax.set_ylim(min(ys) * 0.93, max(ys) * 1.08)


def kernel_panel(ax):
    ion = load_json(ROOT / "artifacts" / "validation" / "sqlite_kernel_qos_ionice" / "sqlite_kernel_qos_baseline.json")
    iow = load_json(ROOT / "artifacts" / "validation" / "sqlite_kernel_qos_ioweight" / "sqlite_kernel_qos_baseline.json")

    ion_p99 = float(ion.get("summary", {}).get("foreground_p99_ms", {}).get("median") or 0.0)
    iow_p99 = iow.get("summary", {}).get("foreground_p99_ms", {}).get("median")

    labels = ["ionice", "IOWeight", "BFQ", "io.latency", "io.cost", "blk-wbt"]
    vals = [ion_p99, 0.0, 0.0, 0.0, 0.0, 0.0]
    colors = ["#1f77b4", "#999999", "#bbbbbb", "#bbbbbb", "#bbbbbb", "#bbbbbb"]
    bars = ax.bar(labels, vals, color=colors)

    ax.set_ylabel("p99 latency (ms)")
    ax.set_title("(c) Kernel QoS breadth", fontsize=9)
    ax.set_ylim(0, max(ion_p99 * 1.3, 12))

    bars[1].set_hatch("//")
    for b in bars[2:]:
        b.set_hatch("xx")

    ax.text(bars[0].get_x() + bars[0].get_width() / 2, ion_p99 + 0.25, f"{ion_p99:.2f}", ha="center", va="bottom", fontsize=8)
    ax.text(bars[1].get_x() + bars[1].get_width() / 2, 0.45, "invalid", ha="center", va="bottom", fontsize=7)
    for i in range(2, len(bars)):
        ax.text(bars[i].get_x() + bars[i].get_width() / 2, 0.45, "blocked", ha="center", va="bottom", fontsize=7)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.5))
    workload_panel(axes[0])
    pqc_panel(axes[1])
    kernel_panel(axes[2])
    fig.tight_layout()
    pdf = FIG / "fig_gap_closeout.pdf"
    png = FIG / "fig_gap_closeout.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    print(str(pdf))


if __name__ == "__main__":
    main()
