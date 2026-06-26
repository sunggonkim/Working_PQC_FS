import csv
import os

# TPM latency (ms) measured in hardware round-trip
TPM_LATENCY_MS = 9.42
TPM_LATENCY_S = TPM_LATENCY_MS / 1000.0

BLOCK_SIZE = 4096

windows = [1, 10, 100, 1000]

os.makedirs("artifacts/m4_freshness", exist_ok=True)
with open("artifacts/m4_freshness/m4_tradeoff.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Window_N", "Sustainable_Throughput_MBps", "Data_Loss_Bytes"])
    
    for N in windows:
        # Data loss if attacker rolls back the disk to the last anchored state
        data_loss_bytes = N * BLOCK_SIZE
        
        # Sustainable throughput = Data anchored per TPM commit / TPM commit latency
        # Because the background queue acts as a shock absorber, short bursts can be higher,
        # but sustained throughput is strictly bounded by this rate.
        bytes_per_commit = N * BLOCK_SIZE
        throughput_bps = bytes_per_commit / TPM_LATENCY_S
        throughput_mbps = throughput_bps / (1024 * 1024)
        
        writer.writerow([N, round(throughput_mbps, 2), data_loss_bytes])

print("Generated artifacts/m4_freshness/m4_tradeoff.csv")
