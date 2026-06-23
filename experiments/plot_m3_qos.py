#!/usr/bin/env python3
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Setup directories
ROOT = Path('/home/thor/skim/pqc_encrpyted_fs')
CSV_PATH = ROOT / 'artifacts' / 'm3_qos_results.csv'
FIGS = ROOT / 'Paper' / 'Figures'
FIGS.mkdir(parents=True, exist_ok=True)

# Styling configuration
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'axes.linewidth': 0.8,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'legend.frameon': False,
    'legend.fontsize': 8.5,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
})

# Harmonious Color Palette
C_BASE  = '#BFD4F2'  # Light Blue for Baseline
C_STAT  = '#E07A36'  # Orange for Static FUSE
C_PHASE = '#3CA374'  # Green for Phase-Aware AEGIS-Q

# Read CSV
configs = []
ttft = []
tps = []
yolo = []
sqlite = []

with open(CSV_PATH, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        configs.append(row['Configuration'])
        ttft.append(float(row['LLM_TTFT_ms']))
        tps.append(float(row['LLM_TPS']))
        yolo.append(float(row['YOLO_p99_ms']))
        sqlite.append(float(row['SQLite_latency_ms']))

# 2x2 Subplots for detailed comparison
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7.2, 5.0))
x = np.arange(len(configs))
width = 0.45
colors = [C_BASE, C_STAT, C_PHASE]

# Subplot 1: LLM Time to First Token (TTFT)
bars1 = ax1.bar(x, ttft, width, color=colors, edgecolor='black', linewidth=0.5)
ax1.set_ylabel('TTFT (ms)')
ax1.set_title('(a) LLM Time to First Token')
ax1.set_xticks(x)
ax1.set_xticklabels(configs, rotation=15)
ax1.grid(axis='y', linestyle=':', alpha=0.5)
ax1.set_ylim(0, 1500)
for bar in bars1:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 30, f'{h:.0f} ms', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Subplot 2: LLM Generation Throughput (TPS)
bars2 = ax2.bar(x, tps, width, color=colors, edgecolor='black', linewidth=0.5)
ax2.set_ylabel('Tokens / Second')
ax2.set_title('(b) LLM Token Generation Rate')
ax2.set_xticks(x)
ax2.set_xticklabels(configs, rotation=15)
ax2.grid(axis='y', linestyle=':', alpha=0.5)
ax2.set_ylim(0, 60)
for bar in bars2:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Subplot 3: YOLOv8 p99 Tail Latency
bars3 = ax3.bar(x, yolo, width, color=colors, edgecolor='black', linewidth=0.5)
ax3.set_ylabel('p99 Latency (ms)')
ax3.set_title('(c) YOLOv8 p99 Inference Latency')
ax3.set_xticks(x)
ax3.set_xticklabels(configs, rotation=15)
ax3.grid(axis='y', linestyle=':', alpha=0.5)
ax3.set_ylim(0, 1.5)
for bar in bars3:
    h = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, h + 0.03, f'{h:.2f} ms', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Subplot 4: SQLite WAL Commit Latency
bars4 = ax4.bar(x, sqlite, width, color=colors, edgecolor='black', linewidth=0.5)
ax4.set_ylabel('Commit Latency (ms)')
ax4.set_title('(d) SQLite WAL Commit Latency')
ax4.set_xticks(x)
ax4.set_xticklabels(configs, rotation=15)
ax4.grid(axis='y', linestyle=':', alpha=0.5)
ax4.set_ylim(0, 11)
for bar in bars4:
    h = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2, h + 0.2, f'{h:.1f} ms', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(FIGS / 'fig_m3_qos.pdf', bbox_inches='tight')
plt.savefig(FIGS / 'fig_m3_qos.png', dpi=180, bbox_inches='tight')
plt.close()
print("Generated fig_m3_qos.pdf and png successfully!")
