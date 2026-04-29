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
# Figure 1: QoS (Target 60 FPS Single Camera Benchmark)
# ---------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(8, 5))
methods = ['NVMe Raw', 'CPU-PQC v3', 'GPU-PQC v3']
fps_real = [57.8, 56.6, 55.5]

x = np.arange(len(methods))
bars = ax1.bar(x, fps_real, width=0.5, color=['#cccccc', '#ff9999', '#66b3ff'], edgecolor='black')
ax1.axhline(y=60, color='r', linestyle='--', label='Target (60 FPS)')

ax1.set_ylabel('Measured FPS')
ax1.set_title('Camera QoS (Target 60 FPS)')
ax1.set_xticks(x)
ax1.set_xticklabels(methods)
ax1.set_ylim(0, 65)
ax1.legend()

for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}', ha='center', va='bottom', fontweight='bold')

plt.savefig(os.path.join(out_dir, 'fig1_qos_fps.png'), dpi=300)
plt.close(fig1)

# ---------------------------------------------------------
# Figure 2: Realistic Camera Scalability (1, 2, 4 cameras)
# ---------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(8, 5))
concurrency = ['1 Cam', '2 Cams', '4 Cams']

cpu_throughput = [6.97, 13.94, 27.86]  
gpu_throughput = [6.97, 13.94, 27.87] 

x = np.arange(len(concurrency))
width = 0.35

rects1 = ax2.bar(x - width/2, cpu_throughput, width, label='CPU-PQC', color='#ff9999', hatch='//', edgecolor='black')
rects2 = ax2.bar(x + width/2, gpu_throughput, width, label='GPU-PQC', color='#66b3ff', edgecolor='black')

ax2.set_ylabel('Aggregate Throughput (MB/s)')
ax2.set_title('Realistic Scalability (30 FPS Cameras)')
ax2.set_xticks(x)
ax2.set_xticklabels(concurrency)
ax2.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax2.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.savefig(os.path.join(out_dir, 'fig2_scalability.png'), dpi=300)
plt.close(fig2)

# ---------------------------------------------------------
# Figure 3: CPU Utilization under 4-Camera Load
# ---------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(6, 5))

categories = ['CPU-PQC', 'GPU-PQC']
cpu_usage = [4.47, 11.18]  

x = np.arange(len(categories))
bars = ax3.bar(x, cpu_usage, width=0.5, color=['#ffcc99', '#99ff99'], edgecolor='black')

ax3.set_ylabel('CPU Utilization (%)')
ax3.set_title('CPU Overhead (4-Camera Load)')
ax3.set_xticks(x)
ax3.set_xticklabels(categories)
ax3.set_ylim(0, 20)

for bar in bars:
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold')

plt.savefig(os.path.join(out_dir, 'fig3_utilization.png'), dpi=300)
plt.close(fig3)

print("Graphs successfully generated at:", out_dir)
