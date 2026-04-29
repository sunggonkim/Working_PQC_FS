import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'figure.autolayout': True,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

out_dir = "/home/thor/skim/pqc_encrpyted_fs/figures"
os.makedirs(out_dir, exist_ok=True)

try:
    telemetry = pd.read_csv("citadel_telemetry_log.csv")
    io_log = pd.read_csv("citadel_io_log.csv")
except FileNotFoundError:
    print("Logs not found yet. Run the benchmark first.")
    exit(1)

# ---------------------------------------------------------
# Figure 10: Dynamic Disruptions (Timeline)
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Top plot: System State Space
ax1.step(telemetry["time_s"], telemetry["c_cpu"], label="CPU Contention (C_cpu)", color="#3498db", linewidth=2.5, where='post')
ax1.step(telemetry["time_s"], telemetry["c_gpu"], label="GPU Contention (C_gpu)", color="#e74c3c", linewidth=2.5, where='post')
ax1.step(telemetry["time_s"], telemetry["thermal"], label="Thermal Guardrail (T)", color="#f39c12", linewidth=2.5, linestyle="--", where='post')
ax1.set_ylabel("Quantized State Level\n(0:Idle → 3:Critical)")
ax1.set_title("CITADEL Fig 10 Homage: Dynamic Multi-tenant Disruptions")
ax1.legend(loc="upper left")
ax1.set_yticks([0, 1, 2, 3])
ax1.set_ylim(-0.5, 4)

# Phase annotations
phase_changes = telemetry.drop_duplicates(subset=['phase'], keep='first')
for idx, row in phase_changes.iterrows():
    ax1.axvline(x=row['time_s'], color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(x=row['time_s'], color='gray', linestyle=':', alpha=0.5)
    ax1.text(row['time_s'] + 1, 3.5, row['phase'].split(':')[0], rotation=0, fontsize=10, fontweight='bold')

# Bottom plot: Q-Learning Throughput Maintenance
# Smooth the throughput a bit for readability
window = max(1, len(io_log) // 50)
smoothed_mbps = io_log["throughput_mbps"].rolling(window, min_periods=1).mean()

ax2.plot(io_log["time_s"], io_log["throughput_mbps"], color="#2ecc71", alpha=0.3, label="Raw I/O (MB/s)")
ax2.plot(io_log["time_s"], smoothed_mbps, color="#27ae60", linewidth=3, label="Q-Learning Maintained QoS")
ax2.set_xlabel("Time (Seconds)")
ax2.set_ylabel("Throughput (MB/s)")
ax2.set_title("QoS Resilience via Q-Learning Action Switching")
ax2.legend(loc="upper right")
ax2.set_ylim(0, max(io_log["throughput_mbps"].max()*1.2, 50))

plt.savefig(os.path.join(out_dir, "fig10_citadel_disruptions.png"), dpi=300)
plt.close()

# ---------------------------------------------------------
# Figure 8: Stress Analysis Bar Chart
# ---------------------------------------------------------
fig2, ax = plt.subplots(figsize=(8, 5))

# Group by state to find average throughput
io_log['state'] = 'Unknown'
for i, row in io_log.iterrows():
    t = row['time_s']
    # Find active phase
    active = telemetry[telemetry['time_s'] <= t].iloc[-1]
    io_log.at[i, 'state'] = active['phase']

avg_throughput = io_log.groupby('state')['throughput_mbps'].mean().to_dict()

phases_to_plot = [
    "Normal Autonomous Driving",
    "Disruption: YOLO Burst (GPU=Max)",
    "Disruption: SLAM Burst (CPU=Max)",
    "Disruption: Thermal Warning"
]

labels = ["Normal", "GPU Busy\n(YOLO)", "CPU Busy\n(SLAM)", "Thermal\nWarning"]
values = [avg_throughput.get(p, 0) for p in phases_to_plot]

colors = ['#bdc3c7', '#e74c3c', '#3498db', '#f39c12']
bars = ax.bar(labels, values, color=colors, edgecolor='black')

ax.set_ylabel('Average Throughput (MB/s)')
ax.set_title('CITADEL Fig 8 Homage: Stress Resilience Analysis')
ax.set_ylim(0, max(values) * 1.3 if values else 50)

for bar in bars:
    yval = bar.get_height()
    if yval > 0:
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f} MB/s', ha='center', va='bottom', fontweight='bold')

plt.savefig(os.path.join(out_dir, "fig8_stress_analysis.png"), dpi=300)
plt.close()

print("CITADEL plotting complete. Check the 'figures' directory.")
