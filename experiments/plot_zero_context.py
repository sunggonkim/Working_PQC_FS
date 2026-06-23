#!/usr/bin/env python3
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Setup directories
ROOT = Path('/home/thor/skim/pqc_encrpyted_fs')
CSV_PATH = ROOT / 'artifacts' / 'zero_context' / 'latency_breakdown.csv'
FIGS = ROOT / 'Paper' / 'Figures'
FIGS.mkdir(parents=True, exist_ok=True)

# Styling configuration
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'axes.linewidth': 0.8,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'legend.frameon': False,
    'legend.fontsize': 9,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
})

# Harmonious Color Palette
C_SWITCH = '#C8324A'  # Red for context switch
C_ATTACH = '#9974D1'  # Purple for UVM attach
C_IO     = '#E07A36'  # Orange for NVMe disk I/O
C_CRYPT  = '#5B8DEF'  # Blue for Crypto execution (GPU/CPU)

# Read CSV
schemes = []
t_switch = []
t_attach = []
t_io = []
t_crypt = []

with open(CSV_PATH, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        schemes.append(row['Scheme'])
        t_switch.append(float(row['T_switch_us']))
        t_attach.append(float(row['T_attach_us']))
        t_io.append(float(row['T_io_us']))
        t_crypt.append(float(row['T_crypt_us']))

# Stack data
x = np.arange(len(schemes))
width = 0.45

fig, ax = plt.subplots(figsize=(6.2, 3.4))

# Stacked bar plot
bar_switch = ax.bar(x, t_switch, width, label='Context Switch ($T_{switch}$)', color=C_SWITCH, edgecolor='black', linewidth=0.5)
bar_attach = ax.bar(x, t_attach, width, bottom=t_switch, label='UVM Mappings ($T_{attach}$)', color=C_ATTACH, edgecolor='black', linewidth=0.5)

bottom_io = np.array(t_switch) + np.array(t_attach)
bar_io = ax.bar(x, t_io, width, bottom=bottom_io, label='NVMe I/O DMA ($T_{io}$)', color=C_IO, edgecolor='black', linewidth=0.5)

bottom_crypt = bottom_io + np.array(t_io)
bar_crypt = ax.bar(x, t_crypt, width, bottom=bottom_crypt, label='Cryptographic Exec ($T_{crypt}$)', color=C_CRYPT, edgecolor='black', linewidth=0.5)

# Add values above bars
totals = np.array(t_switch) + np.array(t_attach) + np.array(t_io) + np.array(t_crypt)
for idx, total in enumerate(totals):
    ax.text(idx, total + 50, f'{total:.0f} μs', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(schemes)
ax.set_ylabel('Latency per 256KB block (μs)')
ax.set_title('Write Latency Breakdown: CPU baseline vs. FUSE vs. eBPF+io_uring')
ax.grid(axis='y', linestyle=':', alpha=0.5)
ax.set_ylim(0, 3800)

# Legend (reversed order for better alignment with stack)
handles = [bar_crypt, bar_io, bar_attach, bar_switch]
labels = [h.get_label() for h in handles]
ax.legend(handles, labels, loc='upper right', ncol=1, fontsize=8.5)

plt.tight_layout()
plt.savefig(FIGS / 'fig_latency_breakdown.pdf', bbox_inches='tight')
plt.savefig(FIGS / 'fig_latency_breakdown.png', dpi=180, bbox_inches='tight')
plt.close()
print("Generated fig_latency_breakdown.pdf and png successfully!")
