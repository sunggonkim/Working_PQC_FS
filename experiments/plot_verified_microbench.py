#!/usr/bin/env python3
"""Render a paper figure exclusively from verified microbenchmark summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "artifacts" / "validation" / "microbench" / "summary.json"
DEFAULT_PDF = ROOT / "Paper" / "Figures" / "fig_verified_microbench.pdf"
DEFAULT_PNG = ROOT / "Paper" / "Figures" / "fig_verified_microbench.png"


def row(summary: dict, operation: str, target: str, batch: int, byte_count: int) -> dict:
    key = f"{operation}|{target}|{batch}|{byte_count}|measured"
    try:
        return summary["workload_map"][key]
    except KeyError as exc:
        raise KeyError(f"missing verified row {key}") from exc


def gpu_row(summary: dict, operation: str, batch: int) -> dict:
    key = f"{operation}|gpu|{batch}"
    try:
        return summary["gpu_mlkem"][key]
    except KeyError as exc:
        raise KeyError(f"missing verified GPU row {key}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))

    sizes = [4096, 65536, 1048576, 16777216]
    cpu_aes = [row(payload, "aes_gcm", "cpu", size // 4096, size)["throughput_per_s_median"] / 1e9 for size in sizes]
    gpu_aes = [row(payload, "aes_gcm", "gpu", size // 4096, size)["throughput_per_s_median"] / 1e9 for size in sizes]
    cpu_keygen = row(payload, "ml_kem_keygen", "cpu", 4096, 0)
    gpu_keygen = gpu_row(payload, "ml_kem_keygen", 4096)

    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, (aes_ax, kem_ax) = plt.subplots(1, 2, figsize=(7.0, 2.45))
    x = np.arange(len(sizes))
    width = 0.36
    aes_ax.bar(x - width / 2, cpu_aes, width, label="CPU/OpenSSL", color="#3b6fb6")
    aes_ax.bar(x + width / 2, gpu_aes, width, label="GPU managed-buffer", color="#e07a36")
    aes_ax.set_xticks(x, ["4 KiB", "64 KiB", "1 MiB", "16 MiB"])
    aes_ax.set_ylabel("Throughput (GB/s)")
    aes_ax.set_title("AES-256-GCM block path")
    aes_ax.grid(axis="y", alpha=0.25)
    aes_ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper center",
                  bbox_to_anchor=(0.5, 1.36))
    aes_ax.text(0.98, 0.82, "GPU includes staging + synchronization", ha="right", va="top",
                fontsize=6.2, transform=aes_ax.transAxes)

    kem_values = [cpu_keygen["throughput_per_s_median"] / 1e3,
                  gpu_keygen["throughput_per_s_median"] / 1e3]
    kem_low = [cpu_keygen["throughput_per_s_median"] - cpu_keygen["throughput_per_s_p05"],
               gpu_keygen["throughput_per_s_median"] - gpu_keygen["throughput_per_s_p05"]]
    kem_high = [cpu_keygen["throughput_per_s_p95"] - cpu_keygen["throughput_per_s_median"],
                gpu_keygen["throughput_per_s_p95"] - gpu_keygen["throughput_per_s_median"]]
    kem_ax.bar([0, 1], kem_values, color=["#3b6fb6", "#e07a36"], width=0.58,
               yerr=np.array([kem_low, kem_high]) / 1e3, capsize=3)
    kem_ax.set_xticks([0, 1], ["CPU/liboqs", "GPU/cuPQC"])
    kem_ax.set_ylabel("ML-KEM-768 keygens/s ($10^3$)")
    kem_ax.set_title("Batched ML-KEM-768 (4,096 ops)")
    kem_ax.grid(axis="y", alpha=0.25)
    speedup = gpu_keygen["throughput_per_s_median"] / cpu_keygen["throughput_per_s_median"]
    kem_ax.text(0.5, 0.93, f"{speedup:.1f}$\\times$ median throughput", ha="center", va="top",
                fontsize=7, transform=kem_ax.transAxes)

    # Reserve a separate legend band above the left title; a tight layout would
    # otherwise overlap the legend and title in the two-column paper figure.
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.20, top=0.73, wspace=0.30)
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.pdf, bbox_inches="tight")
    fig.savefig(args.png, dpi=300, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
