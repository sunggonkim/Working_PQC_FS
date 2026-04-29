#!/usr/bin/env python3
"""
Standard I/O Workload Benchmark for PQC-FUSE Q-Learning Verification
=====================================================================
Runs 3 standard workload profiles used in systems papers (FAST, OSDI, USENIX ATC):
  1. Sequential Write (1MB chunks) - video/lidar streaming
  2. Random Write (4KB chunks)     - metadata/DB logging  
  3. Mixed Concurrent              - seq + rand simultaneously

Measures: Throughput (MB/s), IOPS, Latency (avg/p95/p99)
"""

import os
import sys
import time
import random
import threading
import statistics
import csv
import json

MOUNT_DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pqc_mnt"
RESULTS_FILE = "standard_bench_results.json"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def write_sequential(mount_dir, duration_s=15, chunk_size=1024*1024):
    """Sequential 1MB writes - simulates camera/lidar streaming"""
    results = {"latencies_ms": [], "bytes_written": 0, "ops": 0}
    data = os.urandom(chunk_size)
    bench_dir = os.path.join(mount_dir, "seq_bench")
    ensure_dir(bench_dir)
    
    start = time.time()
    idx = 0
    while time.time() - start < duration_s:
        fpath = os.path.join(bench_dir, f"seq_{idx}.bin")
        t0 = time.monotonic()
        try:
            with open(fpath, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            lat = (time.monotonic() - t0) * 1000
            results["latencies_ms"].append(lat)
            results["bytes_written"] += chunk_size
            results["ops"] += 1
        except Exception as e:
            pass
        # cleanup to avoid filling disk
        try:
            os.remove(fpath)
        except:
            pass
        idx += 1
    
    results["elapsed_s"] = time.time() - start
    return results

def write_random(mount_dir, duration_s=15, chunk_size=4096):
    """Random 4KB writes - simulates DB/metadata logging"""
    results = {"latencies_ms": [], "bytes_written": 0, "ops": 0}
    data = os.urandom(chunk_size)
    bench_dir = os.path.join(mount_dir, "rand_bench")
    ensure_dir(bench_dir)
    
    # Pre-create a file and write at random offsets
    fpath = os.path.join(bench_dir, "rand_target.bin")
    # Create a 10MB file first
    try:
        with open(fpath, "wb") as f:
            f.write(os.urandom(10 * 1024 * 1024))
    except:
        pass
    
    start = time.time()
    while time.time() - start < duration_s:
        offset = random.randint(0, (10*1024*1024) - chunk_size)
        t0 = time.monotonic()
        try:
            fd = os.open(fpath, os.O_WRONLY)
            os.lseek(fd, offset, os.SEEK_SET)
            os.write(fd, data)
            os.fsync(fd)
            os.close(fd)
            lat = (time.monotonic() - t0) * 1000
            results["latencies_ms"].append(lat)
            results["bytes_written"] += chunk_size
            results["ops"] += 1
        except Exception as e:
            pass
    
    try:
        os.remove(fpath)
    except:
        pass
    results["elapsed_s"] = time.time() - start
    return results

def compute_stats(results, label):
    lats = results["latencies_ms"]
    if not lats:
        return {"label": label, "error": "no data"}
    
    elapsed = results["elapsed_s"]
    total_mb = results["bytes_written"] / (1024*1024)
    throughput = total_mb / elapsed if elapsed > 0 else 0
    iops = results["ops"] / elapsed if elapsed > 0 else 0
    
    lats_sorted = sorted(lats)
    p95_idx = int(len(lats_sorted) * 0.95)
    p99_idx = int(len(lats_sorted) * 0.99)
    
    return {
        "label": label,
        "total_ops": results["ops"],
        "total_mb": round(total_mb, 2),
        "elapsed_s": round(elapsed, 2),
        "throughput_mbps": round(throughput, 2),
        "iops": round(iops, 1),
        "lat_avg_ms": round(statistics.mean(lats), 3),
        "lat_p95_ms": round(lats_sorted[p95_idx] if p95_idx < len(lats_sorted) else lats[-1], 3),
        "lat_p99_ms": round(lats_sorted[p99_idx] if p99_idx < len(lats_sorted) else lats[-1], 3),
    }

def run_mixed(mount_dir, duration_s=15):
    """Run sequential and random simultaneously"""
    seq_result = [None]
    rand_result = [None]
    
    def seq_worker():
        seq_result[0] = write_sequential(mount_dir, duration_s, 1024*1024)
    def rand_worker():
        rand_result[0] = write_random(mount_dir, duration_s, 4096)
    
    t1 = threading.Thread(target=seq_worker)
    t2 = threading.Thread(target=rand_worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    return seq_result[0], rand_result[0]

# ── Also run telemetry scenarios during tests ──
def set_telemetry(c, g, t):
    try:
        with open("/tmp/telemetry.csv.tmp", "w") as f:
            f.write(f"{c},{g},{t}\n")
        os.rename("/tmp/telemetry.csv.tmp", "/tmp/telemetry.csv")
    except:
        pass

def clear_telemetry():
    try:
        os.remove("/tmp/telemetry.csv")
    except:
        pass

if __name__ == "__main__":
    print("=" * 70)
    print("  PQC-FUSE Standard I/O Workload Benchmark")
    print("  Mount point:", MOUNT_DIR)
    print("=" * 70)
    
    all_results = {}
    
    # ── Test 1: Independent Sequential Write (GPU idle scenario) ──
    print("\n[1/5] Sequential Write (1MB x 15s) — GPU idle, expect GPU routing...")
    set_telemetry(0, 0, 0)  # idle state
    time.sleep(0.5)
    r = write_sequential(MOUNT_DIR, duration_s=15)
    s = compute_stats(r, "Sequential 1MB (GPU Idle)")
    all_results["seq_idle"] = s
    print(f"  → {s['throughput_mbps']} MB/s | {s['iops']} IOPS | Avg Lat: {s['lat_avg_ms']}ms | P95: {s['lat_p95_ms']}ms")
    
    # ── Test 2: Independent Random Write (GPU idle scenario) ──
    print("\n[2/5] Random Write (4KB x 15s) — GPU idle, expect CPU routing for small I/O...")
    r = write_random(MOUNT_DIR, duration_s=15)
    s = compute_stats(r, "Random 4KB (GPU Idle)")
    all_results["rand_idle"] = s
    print(f"  → {s['throughput_mbps']} MB/s | {s['iops']} IOPS | Avg Lat: {s['lat_avg_ms']}ms | P95: {s['lat_p95_ms']}ms")
    
    # ── Test 3: Sequential Write with GPU busy (AI-heavy) ──
    print("\n[3/5] Sequential Write (1MB x 15s) — GPU BUSY (YOLO sim), expect CPU fallback...")
    set_telemetry(1, 3, 1)  # GPU critical
    time.sleep(0.5)
    r = write_sequential(MOUNT_DIR, duration_s=15)
    s = compute_stats(r, "Sequential 1MB (GPU Busy)")
    all_results["seq_gpu_busy"] = s
    print(f"  → {s['throughput_mbps']} MB/s | {s['iops']} IOPS | Avg Lat: {s['lat_avg_ms']}ms | P95: {s['lat_p95_ms']}ms")
    
    # ── Test 4: Random Write with CPU busy ──
    print("\n[4/5] Random Write (4KB x 15s) — CPU BUSY (SLAM sim), expect GPU routing...")
    set_telemetry(3, 0, 1)  # CPU critical
    time.sleep(0.5)
    r = write_random(MOUNT_DIR, duration_s=15)
    s = compute_stats(r, "Random 4KB (CPU Busy)")
    all_results["rand_cpu_busy"] = s
    print(f"  → {s['throughput_mbps']} MB/s | {s['iops']} IOPS | Avg Lat: {s['lat_avg_ms']}ms | P95: {s['lat_p95_ms']}ms")
    
    # ── Test 5: Mixed Concurrent (Seq + Rand, dynamic state changes) ──
    print("\n[5/5] Mixed Workload (Seq 1MB + Rand 4KB concurrent, 15s, idle state)...")
    set_telemetry(0, 0, 0)
    time.sleep(0.5)
    seq_r, rand_r = run_mixed(MOUNT_DIR, duration_s=15)
    s_seq = compute_stats(seq_r, "Mixed-Seq 1MB")
    s_rand = compute_stats(rand_r, "Mixed-Rand 4KB")
    all_results["mixed_seq"] = s_seq
    all_results["mixed_rand"] = s_rand
    print(f"  Seq  → {s_seq['throughput_mbps']} MB/s | {s_seq['iops']} IOPS | P95: {s_seq['lat_p95_ms']}ms")
    print(f"  Rand → {s_rand['throughput_mbps']} MB/s | {s_rand['iops']} IOPS | P95: {s_rand['lat_p95_ms']}ms")
    
    clear_telemetry()
    
    # Save raw JSON
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Workload':<35} {'MB/s':>8} {'IOPS':>8} {'AvgLat':>8} {'P95':>8} {'P99':>8}")
    print("-" * 70)
    for k, v in all_results.items():
        if "error" in v:
            continue
        print(f"  {v['label']:<35} {v['throughput_mbps']:>8} {v['iops']:>8} {v['lat_avg_ms']:>8} {v['lat_p95_ms']:>8} {v['lat_p99_ms']:>8}")
    print("=" * 70)
    print(f"\nRaw data saved to {RESULTS_FILE}")
