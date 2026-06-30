#!/usr/bin/env python3
"""Render paper figures with one visual grammar.

The figures are intentionally paper-facing: they encode the scoped runtime
claims and avoid mechanisms that are not implemented in the mounted path.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "Paper" / "Figures"
QOS_CLOSEOUT = ROOT / "artifacts/validation/sqlite_hero_validity_closeout/sqlite_hero_validity_closeout.json"

BLUE = "#3b6fb6"
TEAL = "#197278"
GREEN = "#3b8b6d"
ORANGE = "#e07a36"
RED = "#b23a48"
GRAY = "#5f6b7a"
LIGHT_GRAY = "#f3f5f7"
INK = "#20242a"


def save(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=6.8, length=2)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.45)


def draw_box(ax, x, y, w, h, title, body="", *, fc=LIGHT_GRAY, ec=GRAY,
             title_color=INK, lw=1.0, fs=6.5, body_fs=5.6, radius=0.035):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.015,rounding_size={radius}",
        linewidth=lw, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.72, title, ha="center", va="center",
            fontsize=fs, color=title_color, weight="bold")
    if body:
        ax.text(x + w / 2, y + h * 0.34, body, ha="center", va="center",
                fontsize=body_fs, color=INK, linespacing=1.0)
    return patch


def arrow(ax, start, end, *, color=GRAY, dashed=False, lw=1.2, rad=0.0):
    patch = FancyArrowPatch(
        start, end,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=lw,
        color=color,
        linestyle=(0, (3, 2)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)
    return patch


def first_page_qos() -> None:
    payload = json.loads(QOS_CLOSEOUT.read_text(encoding="utf-8"))
    summaries = payload["repeated_methodology_summary"]["mode_summaries"]
    modes = [
        ("App", "app_only", GRAY),
        ("Pressure", "unthrottled_storage", RED),
        ("Simple", "simple_controller", "#8a8f98"),
        ("AEGIS-Q", "aegis_policy", TEAL),
    ]
    labels = [m[0] for m in modes]
    p99 = [summaries[m[1]]["p99_ms"]["median"] for m in modes]
    lo = [summaries[m[1]]["p99_ms"]["ci95_low"] for m in modes]
    hi = [summaries[m[1]]["p99_ms"]["ci95_high"] for m in modes]
    bg = [summaries[m[1]]["storage_mb_s"]["median"] for m in modes]
    misses = [summaries[m[1]]["deadline_misses"]["median"] for m in modes]
    colors = [m[2] for m in modes]

    plt.rcParams.update({"font.size": 7.3, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.15, 1.6))
    x = range(len(labels))
    yerr = [[v - l for v, l in zip(p99, lo)], [h - v for v, h in zip(p99, hi)]]
    bars = ax.bar(x, p99, width=0.66, color=colors, yerr=yerr, capsize=2.4)
    ax.axhline(10.0, color=INK, linewidth=0.8, linestyle="--")
    ax.text(3.15, 10.25, "10 ms SLO", fontsize=6.2, ha="right")
    ax.set_ylabel("SQLite p99 (ms)")
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, max(hi) + 1.4)
    style_axes(ax)
    for bar, value, miss, mb_s in zip(bars, p99, misses, bg):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.25, f"{value:.1f}",
                ha="center", va="bottom", fontsize=6.5)
        ax.text(bar.get_x() + bar.get_width() / 2, 0.38,
                f"{miss:.0f} miss\n{mb_s:.1f} MB/s", ha="center", va="bottom",
                fontsize=5.7, color="white")
    fig.tight_layout(pad=0.25)
    save(fig, "fig_first_page_qos")


def problem_boundary() -> None:
    plt.rcParams.update({"font.size": 7.0, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.05, 1.72))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.93, "Why a storage-runtime boundary is needed",
            fontsize=8.2, weight="bold", color=INK, va="top")

    draw_box(ax, 0.03, 0.50, 0.20, 0.28, "Mature encryption",
             "fscrypt / dm-crypt /\ngocryptfs", fc="#eef4fb", ec=BLUE,
             fs=6.7, body_fs=5.9)
    draw_box(ax, 0.03, 0.15, 0.20, 0.24, "Missing policy",
             "No mounted CPU/GPU,\nQoS, replay boundary", fc="#ffffff", ec=BLUE, lw=0.9,
             fs=6.7, body_fs=5.9)

    draw_box(ax, 0.29, 0.50, 0.20, 0.28, "Naive GPU storage",
             "Fast primitive !=\nfast publication path", fc="#fff4ea", ec=ORANGE,
             fs=6.7, body_fs=5.9)
    draw_box(ax, 0.29, 0.15, 0.20, 0.24, "Hidden cost",
             "launch, staging,\nfsync, replay state", fc="#ffffff", ec=ORANGE, lw=0.9,
             fs=6.7, body_fs=5.9)

    draw_box(ax, 0.55, 0.50, 0.20, 0.28, "Edge pressure",
             "Foreground DB/logs\nshare DRAM + NVMe", fc="#f7edf0", ec=RED,
             fs=6.7, body_fs=5.9)
    draw_box(ax, 0.55, 0.15, 0.20, 0.24, "Failure mode",
             "Tail latency improves only\nif elastic work yields", fc="#ffffff", ec=RED, lw=0.9,
             fs=6.7, body_fs=5.9)

    draw_box(ax, 0.81, 0.36, 0.16, 0.34, "AEGIS-Q target",
             "One mounted path:\npublication + placement\n+ recovery boundary",
             fc="#eaf6f3", ec=TEAL, fs=6.7, body_fs=5.7)

    arrow(ax, (0.23, 0.64), (0.29, 0.64), color=GRAY)
    arrow(ax, (0.49, 0.64), (0.55, 0.64), color=GRAY)
    arrow(ax, (0.75, 0.64), (0.81, 0.58), color=GRAY)
    arrow(ax, (0.13, 0.50), (0.13, 0.39), color=BLUE)
    arrow(ax, (0.39, 0.50), (0.39, 0.39), color=ORANGE)
    arrow(ax, (0.65, 0.50), (0.65, 0.39), color=RED)

    ax.text(0.02, 0.03,
            "Design implication: accelerator placement is useful only after the storage format, publication order, and fallback rule are fixed.",
            fontsize=6.5, color=GRAY)
    save(fig, "fig_problem_boundary")


def architecture() -> None:
    plt.rcParams.update({"font.size": 7.0, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.15, 4.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    lanes = [
        (0.02, 0.75, 0.96, 0.18, "Application boundary", "#f7f8fa"),
        (0.02, 0.51, 0.96, 0.20, "Mounted runtime", "#f7fbff"),
        (0.02, 0.27, 0.96, 0.20, "Persistence and recovery", "#f7fbf8"),
        (0.02, 0.05, 0.96, 0.16, "Claim firewall", "#fff8f1"),
    ]
    for x, y, w, h, label, fc in lanes:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor="#d8dde3", linewidth=0.7))
        ax.text(x + 0.01, y + h - 0.025, label, fontsize=6.2, color=GRAY,
                va="top", weight="bold")

    draw_box(ax, 0.06, 0.79, 0.16, 0.105, "POSIX app",
             "read / write / fsync\nQoS class xattr", fc="#eef4fb", ec=BLUE)
    draw_box(ax, 0.28, 0.79, 0.16, 0.105, "Foreground work",
             "SQLite / append log\nlatency-sensitive", fc="#eef4fb", ec=BLUE)
    draw_box(ax, 0.52, 0.79, 0.16, 0.105, "Elastic work",
             "rekey / hash / batch\nslack-supplied", fc="#fff4ea", ec=ORANGE)

    draw_box(ax, 0.06, 0.56, 0.15, 0.115, "FUSE adapter",
             "namespace, fd state,\nper-file lock", fc="#ffffff", ec=BLUE)
    draw_box(ax, 0.25, 0.56, 0.15, 0.115, "Writeback",
             "coalesce, classify,\nstrict or epoch mode", fc="#ffffff", ec=TEAL)
    draw_box(ax, 0.44, 0.56, 0.15, 0.115, "CPU data lane",
             "AES-GCM block\nnonce = file, block, gen", fc="#ffffff", ec=GREEN)
    draw_box(ax, 0.63, 0.56, 0.15, 0.115, "Admission",
             "deadline, size,\nqueue, telemetry", fc="#ffffff", ec=ORANGE)
    draw_box(ax, 0.82, 0.56, 0.13, 0.115, "GPU lane",
             "ML-KEM batch\nmanaged buffers", fc="#ffffff", ec=ORANGE)

    draw_box(ax, 0.13, 0.32, 0.16, 0.115, "D/J/C publish",
             "ciphertext -> journal\n-> checkpoint", fc="#ffffff", ec=TEAL)
    draw_box(ax, 0.36, 0.32, 0.16, 0.115, "Key envelope",
             "scrypt mount key\nHMAC-wrapped DEK", fc="#ffffff", ec=GREEN)
    draw_box(ax, 0.59, 0.32, 0.16, 0.115, "Recovery oracle",
             "generation replay,\ntamper, remount", fc="#ffffff", ec=GREEN)
    draw_box(ax, 0.80, 0.32, 0.15, 0.115, "External anchor",
             "optional TPM NV\nreplay-after-advance", fc="#ffffff", ec=ORANGE)
    draw_box(ax, 0.30, 0.13, 0.38, 0.07, "Backing directory",
             ".pqcdata + .pqcmeta + marker + envelope/checkpoint xattrs",
             fc="#ffffff", ec=GRAY, fs=6.1, body_fs=5.1)

    arrow(ax, (0.22, 0.81), (0.28, 0.81), color=BLUE)
    arrow(ax, (0.14, 0.79), (0.14, 0.675), color=BLUE)
    arrow(ax, (0.21, 0.615), (0.25, 0.615), color=TEAL)
    arrow(ax, (0.40, 0.615), (0.44, 0.615), color=TEAL)
    arrow(ax, (0.52, 0.56), (0.22, 0.435), color=TEAL, rad=0.05)
    arrow(ax, (0.21, 0.32), (0.41, 0.20), color=TEAL, rad=-0.05)
    arrow(ax, (0.44, 0.79), (0.63, 0.675), color=ORANGE, dashed=True, rad=-0.08)
    arrow(ax, (0.68, 0.56), (0.82, 0.615), color=ORANGE, dashed=True)
    arrow(ax, (0.82, 0.56), (0.51, 0.435), color=ORANGE, dashed=True, rad=-0.12)
    arrow(ax, (0.68, 0.32), (0.68, 0.20), color=GREEN)
    arrow(ax, (0.87, 0.32), (0.69, 0.20), color=ORANGE, dashed=True, rad=-0.1)

    ax.text(0.04, 0.095,
            "Not claimed: NVMe-to-UVM DMA, eBPF/io_uring completion bypass, GPU side-channel defense,",
            fontsize=5.9, color="#8a4b18", va="center")
    ax.text(0.04, 0.065,
            "persistent PCR-bound freshness, or external app-scheduler recovery.",
            fontsize=5.9, color="#8a4b18", va="center")
    save(fig, "fig_architecture")


def publication_protocol() -> None:
    plt.rcParams.update({"font.size": 7.0, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.1, 1.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    states = [
        (0.03, "write buffer", "4 KiB block\nclassified"),
        (0.19, "D durable", "ciphertext sidecar\nfdatasync"),
        (0.35, "J durable", "mapping journal\nfdatasync"),
        (0.51, "C exposed", "checkpoint + size\nHMAC state"),
        (0.67, "anchor", "optional TPM NV\nattempt"),
        (0.83, "remount", "validate or\nfail closed"),
    ]
    for x, title, body in states:
        draw_box(ax, x, 0.55, 0.13, 0.24, title, body, fc="#ffffff", ec=TEAL,
                 fs=6.5, body_fs=5.6)
    for i in range(len(states) - 1):
        arrow(ax, (states[i][0] + 0.13, 0.67), (states[i + 1][0], 0.67), color=TEAL)

    draw_box(ax, 0.07, 0.15, 0.19, 0.18, "Tail without J",
             "ciphertext exists but\nmapping is unreachable", fc="#f7f8fa", ec=GRAY,
             fs=6.3, body_fs=5.5)
    draw_box(ax, 0.30, 0.15, 0.19, 0.18, "Stale generation",
             "older record cannot\nsupersede newer map", fc="#f7f8fa", ec=GRAY,
             fs=6.3, body_fs=5.5)
    draw_box(ax, 0.53, 0.15, 0.19, 0.18, "Replay-after-advance",
             "TPM-backed stale disk\nis rejected", fc="#fff8f1", ec=ORANGE,
             fs=6.3, body_fs=5.5)
    draw_box(ax, 0.76, 0.15, 0.18, 0.18, "Open boundary",
             "no physical power-loss\ncertification claim", fc="#fff8f1", ec=RED,
             fs=6.3, body_fs=5.5)
    arrow(ax, (0.25, 0.55), (0.17, 0.33), color=GRAY, dashed=True)
    arrow(ax, (0.42, 0.55), (0.40, 0.33), color=GRAY, dashed=True)
    arrow(ax, (0.73, 0.55), (0.62, 0.33), color=ORANGE, dashed=True)
    arrow(ax, (0.89, 0.55), (0.85, 0.33), color=RED, dashed=True)

    ax.text(0.02, 0.94, "Publication protocol: data before mapping, mapping before exposure",
            fontsize=8.0, weight="bold", color=INK, va="top")
    save(fig, "fig_publication_protocol")


def main() -> int:
    first_page_qos()
    problem_boundary()
    architecture()
    publication_protocol()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
