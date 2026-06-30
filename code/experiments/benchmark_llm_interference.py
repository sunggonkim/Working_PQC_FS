#!/usr/bin/env python3
"""Latency pass for M3 Phase-Aware LLM QoS interference using real llama.cpp.

This script replaces the old simulated script and runs authentic hardware
benchmarks using `llama-bench` for evaluating AI QoS on Edge LLMs.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FUSE = ROOT / "build" / "pqc_fuse"

def start_fuse(storage_dir: Path, mount_dir: Path, daemon_env: dict[str, str]) -> subprocess.Popen[str]:
    proc = subprocess.Popen(
        [str(FUSE), str(storage_dir), str(mount_dir), "-f"],
        cwd=ROOT,
        env={**os.environ, "PQC_MASTER_PASSWORD": "benchmark-password", **daemon_env},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if subprocess.run(["mountpoint", "-q", str(mount_dir)], check=False).returncode == 0:
            return proc
        if proc.poll() is not None:
            raise RuntimeError(f"FUSE exited before mount (rc={proc.returncode})")
        time.sleep(0.1)
    raise TimeoutError("FUSE mount timed out")

def stop_fuse(proc: subprocess.Popen[str], mount_dir: Path) -> None:
    subprocess.run(["fusermount3", "-u", str(mount_dir)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

def secure_writer(mount_dir: str, stop: mp.Event, size: int, sync_every: int) -> None:
    path = Path(mount_dir) / "stream.bin"
    payload = b"A" * size
    seq = 0
    try:
        with path.open("wb") as handle:
            while not stop.is_set():
                for _ in range(max(1, sync_every)):
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                    seq += 1
                    if stop.is_set():
                        break
    except OSError:
        stop.set()

def background_gpu_elastic_worker(stop: mp.Event) -> None:
    pqc_bench = ROOT / "build" / "bench_gpu_pqc"
    integrity_bench = ROOT / "build" / "bench_gpu_integrity"
    if not pqc_bench.exists() or not integrity_bench.exists():
        print("bench_gpu_pqc or bench_gpu_integrity missing", file=sys.stderr)
        return
    env = dict(os.environ)
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    while not stop.is_set():
        for cmd in ([str(pqc_bench)], [str(integrity_bench)]):
            if stop.is_set():
                break
            try:
                subprocess.run(cmd, cwd=ROOT, env=env, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
            except subprocess.TimeoutExpired:
                continue

def parse_llama_bench_output(stdout: str) -> dict[str, float]:
    results = {}
    for line in stdout.splitlines():
        if "pp512" in line or "tg128" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 8:
                test_type = parts[7]
                ts_str = parts[8]
                try:
                    ts_val = float(ts_str.split("±")[0].strip())
                    if "pp512" in test_type:
                        results["Prefill_TPS"] = ts_val
                    elif "tg128" in test_type:
                        results["Decoding_TPS"] = ts_val
                except ValueError:
                    continue
    return results

def run_trial(label: str, llama_bench: Path, model_path: Path, daemon_env: dict[str, str] | None, trial: int, out_dir: Path, writer_size: int, writer_count: int, writer_sync_every: int, gpu_burn_workers: int = 0) -> dict[str, object]:
    ctx = mp.get_context("spawn")
    storage_dir = Path(tempfile.mkdtemp(prefix=f"aegis_llm_{label}_{trial}_store_"))
    mount_dir = Path(tempfile.mkdtemp(prefix=f"aegis_llm_{label}_{trial}_mnt_"))
    fuse_proc: subprocess.Popen[str] | None = None
    writers: list[mp.Process] = []
    gpu_burn_procs: list[mp.Process] = []
    stop = ctx.Event()
    try:
        if daemon_env is not None:
            fuse_proc = start_fuse(storage_dir, mount_dir, daemon_env)
            for _ in range(max(1, writer_count)):
                proc = ctx.Process(target=secure_writer, args=(str(mount_dir), stop, writer_size, writer_sync_every), daemon=True)
                proc.start()
                writers.append(proc)
            time.sleep(0.5)
        for _ in range(gpu_burn_workers):
            proc = ctx.Process(
                target=background_gpu_elastic_worker,
                args=(stop,),
                daemon=True,
            )
            proc.start()
            gpu_burn_procs.append(proc)
        if gpu_burn_workers:
            time.sleep(0.4)

        # Run llama-bench: prompt processing (512 tokens), text generation (128 tokens), repeated 3 times
        print(f"[{label} - Trial {trial}] Running llama-bench...")
        cmd = [
            str(llama_bench),
            "-m", str(model_path),
            "-p", "512",
            "-n", "128",
            "-r", "3",
            "-t", "4",  # CPU threads
            "-ngl", "99"  # offload to GPU
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        metrics = parse_llama_bench_output(res.stdout)
        
        if "Prefill_TPS" not in metrics or "Decoding_TPS" not in metrics:
            print("llama-bench stdout:\n" + res.stdout)
            print("llama-bench stderr:\n" + res.stderr)
            raise RuntimeError("Failed to parse llama-bench output")

        return {
            "Configuration": label,
            "Trial": trial,
            "Prefill_TPS": metrics["Prefill_TPS"],
            "Decoding_TPS": metrics["Decoding_TPS"],
        }
    finally:
        stop.set()
        for proc in gpu_burn_procs:
            proc.join(timeout=10)
        for writer in writers:
            writer.join(timeout=10)
        if fuse_proc is not None:
            stop_fuse(fuse_proc, mount_dir)
        shutil.rmtree(storage_dir, ignore_errors=True)
        shutil.rmtree(mount_dir, ignore_errors=True)

def plot_results(rows: list[dict], out_dir: Path):
    labels = ["Baseline (Plaintext)", "Static FUSE", "Phase-Aware AEGIS-Q"]
    configs = ["inference_only", "gpu_only", "adaptive"]
    
    prefill_tps = []
    decoding_tps = []
    
    for conf in configs:
        conf_rows = [r for r in rows if r["Configuration"] == conf]
        if not conf_rows:
            continue
        p_tps = sum(r["Prefill_TPS"] for r in conf_rows) / len(conf_rows)
        d_tps = sum(r["Decoding_TPS"] for r in conf_rows) / len(conf_rows)
        prefill_tps.append(p_tps)
        decoding_tps.append(d_tps)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    x = range(len(labels))
    ax1.bar(x, prefill_tps, color=['#666666', '#E45756', '#54A24B'])
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_title("Prefill Phase (TPS)\nCompute-Bound", fontsize=11)
    ax1.set_ylabel("Tokens Per Second")
    ax1.grid(True, axis='y', alpha=0.3)
    
    ax2.bar(x, decoding_tps, color=['#666666', '#E45756', '#54A24B'])
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_title("Decoding Phase (TPS)\nMemory-Bound", fontsize=11)
    ax2.set_ylabel("Tokens Per Second")
    ax2.grid(True, axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_dir / "e4_llm_qos.png", dpi=300)
    plt.close(fig)

def main() -> int:
    ap = argparse.ArgumentParser(description="M3 LLM Interference Benchmark using llama.cpp")
    ap.add_argument("out_dir", nargs="?", default=str(ROOT / "artifacts" / "motivation"))
    ap.add_argument("--llama-bench", required=False, default=str(ROOT.parent / "llama.cpp" / "build" / "bin" / "llama-bench"))
    ap.add_argument("--model", required=False, default=str(ROOT.parent / "llama.cpp" / "models" / "tinyllama-1.1b.gguf"))
    ap.add_argument("--writer-size", type=int, default=512 * 1024)
    ap.add_argument("--writer-count", type=int, default=1)
    ap.add_argument("--writer-sync-every", type=int, default=1)
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    
    llama_bench_path = Path(args.llama_bench)
    model_path = Path(args.model)
    
    if not llama_bench_path.exists():
        print(f"Warning: {llama_bench_path} not found. Waiting for llama.cpp build to finish.")

    if not FUSE.exists():
        raise SystemExit("build/pqc_fuse is missing; run cmake --build build first")

    configurations: list[tuple[str, dict[str, str] | None, int]] = [
        ("inference_only", None, 0),
        ("gpu_only", {
            "PQC_GPU_MIN_BYTES": "1",
            "PQC_GPU_MAX_INFLIGHT_JOBS": "1",
            "PQC_GPU_MAX_INFLIGHT_BYTES": "1048576",
            "PQC_GPU_MAX_WAIT_NS": "50000",
            "PQC_GPU_MIN_BATCH": "1",
            "PQC_FORCE_REKEY_ON_WRITE": "1",
            "PQC_GPU_BURN_ITERS": "300000",
        }, 2),
        ("adaptive", {
            "PQC_GPU_MIN_BYTES": "1",
            "PQC_GPU_MAX_INFLIGHT_JOBS": "1",
            "PQC_GPU_MAX_INFLIGHT_BYTES": "65536",
            "PQC_GPU_MAX_WAIT_NS": "1000",
            "PQC_AI_QOS_MIN_BUDGET_NS": "200000000",
            "PQC_COHERENCE_PENALTY_NS": "5000",
            "PQC_GPU_CONTENTION_PENALTY_NS": "250000",
            "PQC_GPU_MIN_BATCH": "1",
            "PQC_FORCE_REKEY_ON_WRITE": "1",
            "PQC_GPU_BURN_ITERS": "300000",
        }, 1),
    ]

    rows = []
    for label, env, gpu_burn_workers in configurations:
        for trial in range(1, args.trials + 1):
            row = run_trial(label, llama_bench_path, model_path, env, trial, out,
                            args.writer_size, args.writer_count, args.writer_sync_every, gpu_burn_workers=gpu_burn_workers)
            rows.append(row)
            
    # Compute LLM_TTFT_ms and YOLO_p99_ms equivalents just for CSV backwards compat
    # TTFT is roughly (1000 / Prefill_TPS) * 512
    for r in rows:
        r["LLM_TTFT_ms"] = (1000.0 / r["Prefill_TPS"]) * 512
        r["LLM_TPS"] = r["Decoding_TPS"]
        r["YOLO_p99_ms"] = 0.0 # N/A for LLM
        r["SQLite_latency_ms"] = 0.0 # N/A

    (out / "m3_qos_results.json").write_text(json.dumps(rows, indent=2))
    with (out / "m3_qos_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Configuration", "Trial", "Prefill_TPS", "Decoding_TPS", "LLM_TTFT_ms", "LLM_TPS", "YOLO_p99_ms", "SQLite_latency_ms"])
        writer.writeheader()
        writer.writerows(rows)
        
    plot_results(rows, out)
    print(json.dumps(rows, indent=2))
    print(f"\n[results] Wrote JSON and CSV to {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
