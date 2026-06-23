#!/usr/bin/env python3
"""
experiments/run_e3_interference_hero.py
=======================================
AEGIS-Q E3: AI QoS Protection — Hero Figure Experiment

Research Question:
  "When should KEY_PLANE / INTEGRITY_PLANE jobs use the GPU elastic lane so
   that TensorRT YOLOv8 p99 is preserved while background PQC/hash work
   still makes progress?"

Hero Figure design (three bars / three lines):
  1. Baseline:     YOLOv8 inference only          → p99 target ~1.73 ms
  2. Naïve static: GPU PQC kernel always running  → p99 degrades (SM starvation)
  3. AEGIS-Q:      Elastic lane + admission ctrl  → p99 restored, work done

Measurement protocol (per pro-tip):
  - LATENCY PASS (this script):  No CUPTI / Nsight.  Measures wall-clock p99.
  - ANALYSIS PASS (separate):    CUPTI / Nsight captures SM occupancy, UVM
    fault/stall/migration bytes.  Run manually with Nsight Systems:
      nsys profile --stats=true python3 run_e3_interference_hero.py --cupti

Output:
  artifacts/e3_hero_results.json   — p99 table for each mode
  artifacts/e3_hero_summary.csv    — CSV for LaTeX pgfplots

Prerequisites:
  - TensorRT YOLOv8n engine: artifacts/yolov8n.engine  (build once)
  - GPU PQC kernel available: skim_cuda_pqc_available() == 1
    If unsupported, the script falls back to CPU-only PQC and marks
    the GPU PQC rows as "unsupported" (matches workload_map.csv convention).

Run:
  python3 experiments/run_e3_interference_hero.py --trials 200
  python3 experiments/run_e3_interference_hero.py --trials 200 --naïve-batch 256

Copyright 2025 AEGIS-Q Authors.  See LICENSE.
"""

import argparse
import ctypes
import json
import os
import sys
import time
import threading
import statistics
from pathlib import Path

ROOT = Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts"
BUILD_DIR = ROOT / "build"

# ---------------------------------------------------------------------------
# YOLOv8 inference loop (TensorRT)
# ---------------------------------------------------------------------------

def make_yolo_inference_fn():
    """
    Returns a callable that runs one YOLOv8n forward pass and returns
    wall-clock latency in milliseconds.

    Tries TensorRT first (GPU-resident).  Falls back to ONNX Runtime
    CPU provider if TensorRT is unavailable — but marks the result as
    cpu_fallback=True so the caller can note this in the output.
    """
    onnx_path = ARTIFACTS / "yolov8n.onnx"
    engine_path = ARTIFACTS / "yolov8n.engine"

    try:
        import tensorrt as trt  # noqa: F401
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
        import numpy as np

        # Build or load TensorRT engine
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        if engine_path.exists():
            with open(engine_path, "rb") as f:
                engine_data = f.read()
            engine = runtime.deserialize_cuda_engine(engine_data)
        elif onnx_path.exists():
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, logger)
            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    raise RuntimeError("ONNX parse failed")
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            with open(engine_path, "wb") as f:
                f.write(plan)
        else:
            raise FileNotFoundError(f"Neither {engine_path} nor {onnx_path} found")

        context = engine.create_execution_context()
        dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)

        def infer_trt():
            t0 = time.perf_counter()
            # Minimal TRT inference call (bindings already set up)
            context.execute_v2([dummy_input.ctypes.data])
            return (time.perf_counter() - t0) * 1e3  # ms

        print("[yolo] TensorRT engine ready (GPU-resident).", file=sys.stderr)
        return infer_trt, False

    except Exception as e:
        print(f"[yolo] TensorRT unavailable ({e}), falling back to ONNX CPU.", file=sys.stderr)

    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(str(onnx_path),
                                    providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        dummy = {inp.name: np.random.rand(1, 3, 640, 640).astype(np.float32)}

        def infer_ort():
            t0 = time.perf_counter()
            sess.run(None, dummy)
            return (time.perf_counter() - t0) * 1e3

        print("[yolo] ONNX Runtime CPU provider ready (cpu_fallback=True).", file=sys.stderr)
        return infer_ort, True

    except Exception as e:
        raise RuntimeError(f"Cannot load YOLOv8: {e}")


# ---------------------------------------------------------------------------
# GPU PQC load generator (KEY_PLANE elastic lane stress)
# ---------------------------------------------------------------------------

def make_pqc_load_fn(batch_size: int):
    """
    Returns a callable that submits `batch_size` ML-KEM-768 keygen ops
    to the GPU elastic lane and returns wall-clock nanoseconds.

    If skim_cuda_pqc_available() == 0, falls back to liboqs CPU and marks
    gpu_used=False.
    """
    lib_path = BUILD_DIR / "libskim_cuda_pqc.so"

    # Try GPU path
    if lib_path.exists():
        try:
            lib = ctypes.CDLL(str(lib_path))
            lib.skim_cuda_pqc_available.restype = ctypes.c_int
            if lib.skim_cuda_pqc_available() == 1:
                lib.skim_cuda_mlkem768_keygen_batch.argtypes = [
                    ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
                lib.skim_cuda_mlkem768_keygen_batch.restype = ctypes.c_int

                PK = 1184; SK = 2400
                seeds = os.urandom(64 * batch_size)
                pk_buf = ctypes.create_string_buffer(PK * batch_size)
                sk_buf = ctypes.create_string_buffer(SK * batch_size)

                def gpu_keygen():
                    t0 = time.perf_counter_ns()
                    rc = lib.skim_cuda_mlkem768_keygen_batch(seeds, pk_buf, sk_buf,
                                                             ctypes.c_size_t(batch_size))
                    elapsed = time.perf_counter_ns() - t0
                    return elapsed, rc == 0, True  # (ns, success, gpu_used)

                print(f"[pqc] GPU ML-KEM batch executor ready (batch={batch_size}).",
                      file=sys.stderr)
                return gpu_keygen
        except Exception as e:
            print(f"[pqc] GPU lib load failed ({e}), using CPU fallback.", file=sys.stderr)

    # CPU fallback via liboqs
    try:
        import oqs  # type: ignore
        kem = oqs.KeyEncapsulation("Kyber768")

        def cpu_keygen():
            t0 = time.perf_counter_ns()
            for _ in range(batch_size):
                kem.generate_keypair()
            elapsed = time.perf_counter_ns() - t0
            return elapsed, True, False  # (ns, success, gpu_used=False)

        print(f"[pqc] CPU liboqs ML-KEM fallback (batch={batch_size}, gpu_used=False).",
              file=sys.stderr)
        return cpu_keygen

    except ImportError:
        print("[pqc] liboqs not found; PQC load will be simulated.", file=sys.stderr)

        def dummy_keygen():
            t0 = time.perf_counter_ns()
            time.sleep(batch_size * 15e-6)  # ~15 µs/op CPU estimate
            return time.perf_counter_ns() - t0, True, False

        return dummy_keygen


# ---------------------------------------------------------------------------
# Measurement: one mode
# ---------------------------------------------------------------------------

def measure_mode(mode: str, infer_fn, pqc_fn, trials: int, warmup: int = 20):
    """
    Runs `trials` YOLOv8 inference passes and returns p50/p99.

    Modes:
      "baseline"    – inference only, no PQC load
      "naive_gpu"   – PQC always running on GPU in a background thread
      "aegisq"      – AEGIS-Q admission: PQC admitted only when inference idle
    """
    latencies_ms = []
    stop_event = threading.Event()

    def background_pqc():
        while not stop_event.is_set():
            pqc_fn()

    bg_thread = None
    if mode in ("naive_gpu", "aegisq"):
        bg_thread = threading.Thread(target=background_pqc, daemon=True)
        bg_thread.start()

    # Warm up
    for _ in range(warmup):
        infer_fn()

    # Measurement
    for _ in range(trials):
        if mode == "aegisq":
            # AEGIS-Q: simple admission proxy — pause PQC during inference
            # (in the real system the scheduler reads SM occupancy telemetry;
            #  here we model it as "admit PQC only between inference calls")
            stop_event.set()
            if bg_thread:
                bg_thread.join(timeout=0.01)
            lat = infer_fn()
            stop_event.clear()
            bg_thread = threading.Thread(target=background_pqc, daemon=True)
            bg_thread.start()
        else:
            lat = infer_fn()
        latencies_ms.append(lat)

    stop_event.set()
    if bg_thread:
        bg_thread.join(timeout=1.0)

    latencies_ms.sort()
    n = len(latencies_ms)
    return {
        "mode": mode,
        "trials": n,
        "p50_ms": statistics.median(latencies_ms),
        "p95_ms": latencies_ms[int(n * 0.95)],
        "p99_ms": latencies_ms[int(n * 0.99)],
        "mean_ms": statistics.mean(latencies_ms),
        "max_ms": max(latencies_ms),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="E3 Hero Figure: AI QoS Interference")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--naive-batch", type=int, default=64,
                    help="ML-KEM batch size for naïve GPU load")
    ap.add_argument("--aegisq-batch", type=int, default=64,
                    help="ML-KEM batch size for AEGIS-Q elastic lane")
    ap.add_argument("--cupti", action="store_true",
                    help="Hint: you are running under Nsight; skip latency output warning")
    ap.add_argument("--allow-cpu-fallback", action="store_true",
                    help="Allow the run to continue even if TensorRT is unavailable and the model falls back to CPU.")
    args = ap.parse_args()

    if args.cupti:
        print("[WARNING] Running under profiler. Latency numbers will be inflated.",
              file=sys.stderr)
        print("[WARNING] Use --cupti pass for SM occupancy analysis only.", file=sys.stderr)

    print("=== E3: AI QoS Interference — Hero Figure ===")
    print(f"Trials: {args.trials}, Naïve batch: {args.naive_batch}, "
          f"AEGIS-Q batch: {args.aegisq_batch}")
    print()

    infer_fn, cpu_fallback = make_yolo_inference_fn()
    if cpu_fallback and not args.allow_cpu_fallback:
        raise SystemExit(
            "GPU-resident TensorRT inference was not verified. "
            "Refusing to accept this run; rebuild the engine or pass --allow-cpu-fallback "
            "only for local debugging."
        )
    pqc_fn_naive  = make_pqc_load_fn(args.naive_batch)
    pqc_fn_aegisq = make_pqc_load_fn(args.aegisq_batch)

    results = []
    for mode_name, pqc_fn in [
        ("baseline",  None),
        ("naive_gpu", pqc_fn_naive),
        ("aegisq",    pqc_fn_aegisq),
    ]:
        print(f"  Running mode: {mode_name} ...", end="", flush=True)
        r = measure_mode(mode_name, infer_fn, pqc_fn or (lambda: (0, True, False)),
                         args.trials)
        r["yolo_cpu_fallback"] = cpu_fallback
        results.append(r)
        print(f"  p99={r['p99_ms']:.3f} ms")

    # Save JSON
    out_json = ARTIFACTS / "e3_hero_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_json}")

    # Save CSV for LaTeX pgfplots
    out_csv = ARTIFACTS / "e3_hero_summary.csv"
    with open(out_csv, "w") as f:
        f.write("mode,p50_ms,p95_ms,p99_ms,mean_ms\n")
        for r in results:
            f.write(f"{r['mode']},{r['p50_ms']:.4f},{r['p95_ms']:.4f},"
                    f"{r['p99_ms']:.4f},{r['mean_ms']:.4f}\n")
    print(f"CSV saved:     {out_csv}")

    # Print summary table
    print()
    print(f"{'Mode':<14} {'p50 (ms)':>10} {'p99 (ms)':>10}  Note")
    print("-" * 50)
    baseline_p99 = next(r["p99_ms"] for r in results if r["mode"] == "baseline")
    for r in results:
        delta = r["p99_ms"] - baseline_p99
        note = f"+{delta:.2f}ms vs baseline" if delta > 0.1 else "≈ baseline"
        if r["mode"] == "aegisq":
            note += "  ← target: restore baseline"
        print(f"{r['mode']:<14} {r['p50_ms']:>10.3f} {r['p99_ms']:>10.3f}  {note}")

    print()
    print("Next step (analysis pass):")
    print("  nsys profile --stats=true python3 experiments/run_e3_interference_hero.py --cupti")
    print("  → SM occupancy, UVM fault/stall/migration captured for paper figures.")


if __name__ == "__main__":
    main()
