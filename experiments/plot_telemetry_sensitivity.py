#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt

def plot_admission_sensitivity():
    print("Plotting admission control sensitivity...")
    with open("artifacts/m5_admission_sweep.json", "r") as f:
        data = json.load(f)
    
    budgets = [d["budget_ns"] / 1000000 for d in data] # ms
    gpu_jobs = [d["gpu_jobs"] for d in data]
    cpu_jobs = [d["cpu_jobs"] for d in data]
    
    # Calculate % routed to GPU vs CPU
    total_jobs = [g + c for g, c in zip(gpu_jobs, cpu_jobs)]
    gpu_ratio = [g / t * 100 if t > 0 else 0 for g, t in zip(gpu_jobs, total_jobs)]
    
    fig, ax1 = plt.subplots(figsize=(6, 4))
    
    color = '#1f77b4'
    ax1.set_xlabel('AI Strictness Budget (ms)', fontsize=10)
    ax1.set_ylabel('PQC Jobs admitted to GPU (%)', color=color, fontsize=10)
    ax1.plot(budgets, gpu_ratio, marker='o', color=color, linewidth=2, label="GPU Admission Rate")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(-5, 105)
    
    fig.tight_layout()
    plt.title("Admission Control Sensitivity Analysis", fontsize=12)
    plt.savefig("Paper/Figures/fig_telemetry_sensitivity.pdf")
    print("Saved to Paper/Figures/fig_telemetry_sensitivity.pdf")

if __name__ == "__main__":
    plot_admission_sensitivity()
