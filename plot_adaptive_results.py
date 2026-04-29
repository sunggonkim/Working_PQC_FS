import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.autolayout': True,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

out_dir = "/home/thor/skim/pqc_encrpyted_fs/figures"
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------
# Figure 1: Adaptive Resource Orchestration (Crypto Workload Route)
# ---------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(8, 5))
scenarios = ['Scenario A\n(AI-Heavy / YOLO Active)', 'Scenario B\n(I/O-Heavy / YOLO Idle)']

# Crypto-related resource usage
cpu_route = [7.5, 0.0]  # In AI-heavy, crypto routed to CPU (100% of crypto is on CPU)
gpu_route = [0.0, 11.1] # In I/O-heavy, crypto routed to GPU (driver CPU overhead shown)

x = np.arange(len(scenarios))
width = 0.35

ax1.bar(x - width/2, cpu_route, width, label='Crypto on CPU (Fallback)', color='#ff9999', edgecolor='black')
ax1.bar(x + width/2, gpu_route, width, label='Crypto on GPU (16-Streams)', color='#66b3ff', edgecolor='black')

ax1.set_ylabel('CPU Utilization for Crypto (%)')
ax1.set_title('Adaptive Heterogeneous Resource Routing')
ax1.set_xticks(x)
ax1.set_xticklabels(scenarios)
ax1.legend()

ax1.text(-0.17, 8.5, 'Zero GPU Interference\n(YOLO Protected)', ha='center', va='bottom', color='red', fontsize=12)
ax1.text(1.17, 12.0, 'Zero CPU Lock Thrashing\n(I/O Scaled)', ha='center', va='bottom', color='blue', fontsize=12)

plt.savefig(os.path.join(out_dir, 'fig1_adaptive_routing.png'), dpi=300)
plt.close(fig1)

# ---------------------------------------------------------
# Figure 2: Seamless Throughput Maintenance
# ---------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(8, 5))

# Data from empirical benchmark
throughputs = [6.97, 27.87]
targets = [7.0, 28.0] # 1 camera at 30fps = ~7MB/s, 4 cameras = ~28MB/s

bars = ax2.bar(x, throughputs, width=0.5, color='#abdda4', edgecolor='black', label='Measured Throughput')
ax2.plot(x, targets, 'kX', markersize=12, label='Target Required Bandwidth')

ax2.set_ylabel('Aggregate Throughput (MB/s)')
ax2.set_title('Seamless QoS Across Routing Paths')
ax2.set_xticks(x)
ax2.set_xticklabels(scenarios)
ax2.legend(loc='upper left')
ax2.set_ylim(0, 35)

for bar in bars:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f} MB/s', ha='center', va='bottom', fontweight='bold')

plt.savefig(os.path.join(out_dir, 'fig2_adaptive_throughput.png'), dpi=300)
plt.close(fig2)

print("Adaptive graphs successfully generated at:", out_dir)
