import pandas as pd
import matplotlib.pyplot as plt
import os


df = pd.read_csv("artifacts/results/freshness/m4_freshness/m4_tradeoff.csv")

fig, ax1 = plt.subplots(figsize=(8, 6))

color = 'tab:red'
ax1.set_xlabel('Rollback Window N (Blocks)', fontsize=12)
ax1.set_ylabel('Write Throughput (MB/s)', color=color, fontsize=12)
ax1.plot(df['Window_N'], df['Seq_Throughput_MBps'], marker='o', color='tab:red', linewidth=2, label='Sequential Write')
ax1.plot(df['Window_N'], df['Rand_Throughput_MBps'], marker='^', color='tab:orange', linewidth=2, label='Random Write')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xscale('log')
ax1.set_yscale('log')

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Data Loss Vulnerability (Bytes)', color=color, fontsize=12)
ax2.plot(df['Window_N'], df['Data_Loss_Bytes'], marker='s', linestyle='--', color=color, linewidth=2, label='Data Loss')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_yscale('log')

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.title('Security-Performance Trade-off (TPM Anchor Freshness Window)', fontsize=14)
fig.tight_layout()

plt.savefig("Paper/Figures/m4_freshness_tradeoff.png", dpi=300)
print("Plot saved to Paper/Figures/m4_freshness_tradeoff.png")
