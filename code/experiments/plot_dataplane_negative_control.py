#!/usr/bin/env python3
"""Render the AES-GCM data-plane placement negative-control figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "artifacts" / "validation" / "gpu_dataplane_negative_control" / "gpu_dataplane_negative_control.json"
FIG = ROOT / "Paper" / "Figures"


def load_cases() -> list[dict]:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    return payload["summary"]["cases"]


def size_label(total_bytes: int) -> str:
    if total_bytes < 1024 * 1024:
        return f"{total_bytes // 1024} KiB"
    return f"{total_bytes // (1024 * 1024)} MiB"


def main() -> int:
    cases = load_cases()
    x = [case["total_bytes"] for case in cases]
    cpu = [case["cpu_median_mib_s"] for case in cases]
    gpu = [case["gpu_median_mib_s"] for case in cases]
    ratio = [case["gpu_slower_ratio"] for case in cases]

    plt.rcParams.update({"font.size": 7.2, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.25, 1.85))
    ax.plot(x, cpu, marker="o", linewidth=1.5, markersize=3.8,
            color="#3b6fb6", label="CPU/OpenSSL")
    ax.plot(x, gpu, marker="s", linewidth=1.5, markersize=3.8,
            color="#e07a36", label="GPU managed")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(x, [size_label(v) for v in x], rotation=22, ha="right")
    ax.set_ylabel("AES-GCM throughput (MiB/s)")
    ax.set_xlabel("Authenticated data size")
    ax.grid(axis="both", which="both", linestyle=":", linewidth=0.5, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=6.3, length=2)
    ax.axvline(4096, color="#20242a", linewidth=0.8, linestyle="--")
    ax.text(4096 * 1.08, min(cpu + gpu) * 2.2, "4 KiB\nblock/page",
            fontsize=5.8, va="bottom", ha="left", color="#20242a")
    ax.legend(frameon=False, fontsize=6.3, loc="lower right")
    for xpos, ypos, slowdown in zip(x, gpu, ratio):
        ax.text(xpos, ypos * 1.45, f"{slowdown:.0f}x",
                fontsize=5.7, ha="center", va="bottom", color="#7a3e10")
    fig.tight_layout(pad=0.18)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_dataplane_negative_control.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_dataplane_negative_control.png", dpi=300, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
