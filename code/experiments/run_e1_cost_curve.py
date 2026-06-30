#!/usr/bin/env python3
"""
experiments/run_e1_cost_curve.py — AEGIS-Q E1: Encryption cost curve
=====================================================================
Measures p50/p99 latency at 4KiB..4MiB block sizes for:
  - CPU-only    (PQC_EXECUTION_MODE=cpu)
  - GPU-only    (PQC_EXECUTION_MODE=gpu)
  - Adaptive    (PQC_EXECUTION_MODE=adaptive)

Usage:
  python3 code/experiments/run_e1_cost_curve.py [--binary ./build/pqc_fuse] [--iters 200] [--out e1_results.json]

Output:
  JSON with shape { mode: { block_size_kb: { p50_ms, p99_ms, mean_ms } } }

Requires:
  - fusermount in PATH
  - $PQC_MASTER_PASSWORD set (or hardcoded below for benchmarking)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import shutil
import signal
import statistics

MODES = ["cpu", "gpu", "adaptive"]
BLOCK_SIZES_KB = [4, 16, 64, 128, 256, 512, 1024, 2048, 4096]
DEFAULT_ITERS = 200
WARMUP_ITERS = 20


def mount_fuse(binary, storage_dir, mount_dir, mode, env):
    """Start pqc_fuse and return (process, logfile)."""
    logfile = tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False)
    e = dict(env)
    e["PQC_EXECUTION_MODE"] = mode
    proc = subprocess.Popen(
        [binary, storage_dir, mount_dir, "-f"],
        env=e,
        stdout=subprocess.DEVNULL,
        stderr=logfile,
    )
    # Wait for FUSE to be ready
    for _ in range(40):
        time.sleep(0.1)
        r = subprocess.run(["mountpoint", "-q", mount_dir], capture_output=True)
        if r.returncode == 0:
            return proc, logfile.name
    proc.kill()
    with open(logfile.name) as f:
        sys.stderr.write(f"FUSE mount failed ({mode}):\n" + f.read() + "\n")
    raise RuntimeError(f"FUSE did not become ready for mode={mode}")


def umount_fuse(proc, mount_dir):
    subprocess.run(["fusermount", "-u", mount_dir],
                   capture_output=True, timeout=10)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def measure_latencies(mount_dir, block_size_bytes, iters, warmup):
    """Write+read a file of block_size_bytes, return list of round-trip ms."""
    path = os.path.join(mount_dir, "bench.bin")
    payload = os.urandom(block_size_bytes)
    latencies = []

    for i in range(warmup + iters):
        t_w0 = time.perf_counter()
        with open(path, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        t_w1 = time.perf_counter()
        
        t_r0 = time.perf_counter()
        with open(path, "rb") as f:
            _ = f.read()
        t_r1 = time.perf_counter()
        
        if i >= warmup:
            latencies.append({
                'write': (t_w1 - t_w0) * 1000.0,
                'read': (t_r1 - t_r0) * 1000.0,
                'total': (t_r1 - t_w0) * 1000.0
            })

    # cleanup
    try:
        os.unlink(path)
    except OSError:
        pass
    return latencies


def run_one_mode(binary, mode, block_sizes_kb, iters, warmup, env):
    results = {}
    for kb in block_sizes_kb:
        storage = tempfile.mkdtemp(prefix="pqc_s_")
        mount = tempfile.mkdtemp(prefix="pqc_m_")
        try:
            proc, logfile = mount_fuse(binary, storage, mount, mode, env)
            try:
                lats = measure_latencies(mount, kb * 1024, iters, warmup)
                lats_w = sorted([L['write'] for L in lats])
                lats_r = sorted([L['read'] for L in lats])
                lats_t = sorted([L['total'] for L in lats])
                results[kb] = {
                    "p50_ms": lats_t[len(lats_t) // 2],
                    "p99_ms": lats_t[int(len(lats_t) * 0.99)],
                    "mean_ms": statistics.mean(lats_t),
                    "samples": len(lats),
                }
                print(f"  [{mode:8s}] {kb:5d} KiB: "
                      f"WRITE p50={lats_w[len(lats_w)//2]:.2f}ms  "
                      f"READ p50={lats_r[len(lats_r)//2]:.2f}ms  "
                      f"TOTAL p50={lats_t[len(lats_t)//2]:.2f}ms", flush=True)
            finally:
                umount_fuse(proc, mount)
                try:
                    os.unlink(logfile)
                except OSError:
                    pass
        finally:
            shutil.rmtree(storage, ignore_errors=True)
            shutil.rmtree(mount, ignore_errors=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="AEGIS-Q E1 cost curve")
    parser.add_argument("--binary", default="./build/pqc_fuse")
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERS)
    parser.add_argument("--sizes-kb", nargs="+", type=int, default=[16, 64, 128, 256, 512],
                        help="Block sizes in KiB to benchmark")
    parser.add_argument("--modes", nargs="+", type=str, default=["cpu", "gpu"],
                        help="Execution modes to benchmark")
    parser.add_argument("--out", default="e1_results.json")
    parser.add_argument("--password", default=None)
    args = parser.parse_args()

    env = dict(os.environ)
    if args.password:
        env["PQC_MASTER_PASSWORD"] = args.password
    elif "PQC_MASTER_PASSWORD" not in env:
        env["PQC_MASTER_PASSWORD"] = "benchmark_password"

    print(f"AEGIS-Q E1 Cost Curve — iters={args.iters} warmup={args.warmup}")
    print(f"  sizes_kb = {args.sizes_kb}")
    print(f"  modes    = {args.modes}")

    all_results = {}
    for mode in args.modes:
        print(f"\n── Mode: {mode} ──")
        all_results[mode] = run_one_mode(
            args.binary, mode, args.sizes_kb, args.iters, args.warmup, env
        )

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults written to {args.out}")

    # Print comparison table
    print("\n── Summary: p50 (ms) ──")
    header = f"{'Size':>8}" + "".join(f"{m:>12}" for m in args.modes)
    print(header)
    for kb in args.sizes_kb:
        row = f"{kb:>7}K"
        for mode in args.modes:
            v = all_results.get(mode, {}).get(kb, {}).get("p50_ms", float("nan"))
            row += f"{v:>11.2f}ms"
        print(row)

    # Key finding
    if "cpu" in all_results and "gpu" in all_results:
        wins = 0
        total = 0
        for kb in args.sizes_kb:
            cpu_p50 = all_results["cpu"].get(kb, {}).get("p50_ms", None)
            gpu_p50 = all_results["gpu"].get(kb, {}).get("p50_ms", None)
            if cpu_p50 and gpu_p50:
                total += 1
                if gpu_p50 < cpu_p50:
                    wins += 1
        print(f"\nGPU < CPU: {wins}/{total} block sizes")
        if wins == 0:
            print("⚠  GPU still slower — check pool hit rate (pool_hits field in --um-smoke)")


if __name__ == "__main__":
    main()
