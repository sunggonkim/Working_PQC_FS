#!/usr/bin/env python3
"""Latency pass for M6 TensorRT interference.

The script is intentionally self-contained:
  - It can build a TensorRT engine from an ONNX file if no engine exists.
  - It validates that the inference path is GPU-resident by refusing CPU fallbacks.
  - It runs baseline, CPU-only, GPU-only, and adaptive secure-I/O policies.

The profiling pass is separate (`experiments/profile_e3_nsys.py`).
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import multiprocessing as mp
import os
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorrt as trt


ROOT = Path(__file__).resolve().parent.parent
FUSE = ROOT / "build" / "pqc_fuse"
CUDA = ctypes.CDLL("libcudart.so")
CUDA.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
CUDA.cudaFree.argtypes = [ctypes.c_void_p]
CUDA.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
CUDA.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
CUDA.cudaDeviceSynchronize.argtypes = []
CUDA.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
CUDA.cudaStreamCreateWithPriority.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
CUDA.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
CUDA.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
CUDA.cudaDeviceGetStreamPriorityRange.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
CUDA.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
CUDA.cudaFreeHost.argtypes = [ctypes.c_void_p]


def _cuda_ok(rc: int, msg: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{msg} (cuda rc={rc})")


def cuda_malloc(nbytes: int) -> int:
    ptr = ctypes.c_void_p()
    _cuda_ok(CUDA.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)), "cudaMalloc failed")
    return int(ptr.value)


def cuda_free(ptr: int) -> None:
    if ptr:
        CUDA.cudaFree(ctypes.c_void_p(ptr))


def cuda_memcpy_htod(dst: int, src: np.ndarray) -> None:
    _cuda_ok(
        CUDA.cudaMemcpy(
            ctypes.c_void_p(dst),
            src.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(src.nbytes),
            1,  # cudaMemcpyHostToDevice
        ),
        "cudaMemcpy H2D failed",
    )


def cuda_memcpy_dtoh(dst: np.ndarray, src: int) -> None:
    _cuda_ok(
        CUDA.cudaMemcpy(
            dst.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_void_p(src),
            ctypes.c_size_t(dst.nbytes),
            2,  # cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy D2H failed",
    )


def cuda_sync() -> None:
    _cuda_ok(CUDA.cudaDeviceSynchronize(), "cudaDeviceSynchronize failed")


def cuda_init() -> None:
    _cuda_ok(CUDA.cudaFree(ctypes.c_void_p(0)), "cuda runtime init failed")


def cuda_stream_create(priority: int = 0) -> int:
    stream = ctypes.c_void_p()
    flags = 1  # cudaStreamNonBlocking
    rc = CUDA.cudaStreamCreateWithPriority(ctypes.byref(stream), ctypes.c_uint(flags), ctypes.c_int(priority))
    if rc != 0:
        _cuda_ok(CUDA.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate failed")
    return int(stream.value)


def cuda_stream_destroy(stream: int) -> None:
    if stream:
        CUDA.cudaStreamDestroy(ctypes.c_void_p(stream))


def cuda_stream_priority_range() -> tuple[int, int]:
    least = ctypes.c_int()
    greatest = ctypes.c_int()
    _cuda_ok(
        CUDA.cudaDeviceGetStreamPriorityRange(ctypes.byref(least), ctypes.byref(greatest)),
        "cudaDeviceGetStreamPriorityRange failed",
    )
    return greatest.value, least.value


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


def percentile(samples: list[float], q: int) -> float:
    return statistics.quantiles(samples, n=100, method="inclusive")[q - 1]


def summary(samples: list[float]) -> dict[str, float]:
    return {
        "count": len(samples),
        "p50_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 95),
        "p99_ms": percentile(samples, 99),
        "mean_ms": statistics.fmean(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def parse_shape(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.lower().replace("x", ",").replace(" ", "").split(",") if x)


@dataclass
class TRTArtifacts:
    engine_path: Path
    model_name: str
    input_shape: tuple[int, ...]
    output_dir: Path


def build_engine_from_onnx(onnx_path: Path, engine_path: Path, input_shape: tuple[int, ...]) -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    onnx_bytes = onnx_path.read_bytes()
    if not parser.parse(onnx_bytes):
        msgs = "\n".join(parser.get_error(i).desc() for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT ONNX parse failed for {onnx_path}:\n{msgs}")
    config = builder.create_builder_config()
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
    except Exception:
        pass
    if builder.platform_has_fast_fp16:
        try:
            config.set_flag(trt.BuilderFlag.FP16)
        except Exception:
            pass
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(input_name, input_shape, input_shape, input_shape)
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build an engine from {onnx_path}")
    engine_path.write_bytes(bytes(serialized))


def load_or_build_engine(engine_path: Path, onnx_path: Path, input_shape: tuple[int, ...]) -> Path:
    if engine_path.exists():
        return engine_path
    if not onnx_path.exists():
        raise FileNotFoundError(f"Neither engine nor ONNX file exists: {engine_path}, {onnx_path}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    build_engine_from_onnx(onnx_path, engine_path, input_shape)
    return engine_path


def make_infer_fn(engine_path: Path, input_shape: tuple[int, ...], stream_handle: int | None = None):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Could not deserialize engine: {engine_path}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Could not create execution context")

    input_name = None
    output_names = []
    for idx in range(engine.num_io_tensors):
        name = engine.get_tensor_name(idx)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_name = name
        else:
            output_names.append(name)
    if input_name is None:
        raise RuntimeError("Engine has no input bindings")
    context.set_input_shape(input_name, input_shape)
    if not context.all_binding_shapes_specified:
        raise RuntimeError("TensorRT binding shapes not fully specified")

    host_input = np.random.random_sample(input_shape).astype(np.float32)
    host_outputs = []
    device_ptrs: list[int] = []
    stream = stream_handle or cuda_stream_create()
    pinned_input = ctypes.c_void_p()
    pinned_outputs: list[ctypes.c_void_p] = []
    try:
        _cuda_ok(CUDA.cudaHostAlloc(ctypes.byref(pinned_input), ctypes.c_size_t(host_input.nbytes), ctypes.c_uint(0)), "cudaHostAlloc failed")
        ctypes.memmove(pinned_input.value, host_input.ctypes.data, host_input.nbytes)
        input_ptr = cuda_malloc(host_input.nbytes)
        device_ptrs.append(input_ptr)
        _cuda_ok(
            CUDA.cudaMemcpyAsync(
                ctypes.c_void_p(input_ptr),
                pinned_input,
                ctypes.c_size_t(host_input.nbytes),
                1,
                ctypes.c_void_p(stream),
            ),
            "cudaMemcpyAsync H2D failed",
        )
        for name in output_names:
            out_shape = tuple(context.get_tensor_shape(name))
            if any(dim < 0 for dim in out_shape):
                out_shape = tuple(dim if dim > 0 else 1 for dim in out_shape)
            dtype = engine.get_tensor_dtype(name)
            np_dtype = trt.nptype(dtype)
            host_out = np.empty(out_shape, dtype=np_dtype)
            host_outputs.append((name, host_out))
            device_ptrs.append(cuda_malloc(host_out.nbytes))
            pinned = ctypes.c_void_p()
            _cuda_ok(CUDA.cudaHostAlloc(ctypes.byref(pinned), ctypes.c_size_t(host_out.nbytes), ctypes.c_uint(0)), "cudaHostAlloc failed")
            pinned_outputs.append(pinned)
        context.set_tensor_address(input_name, device_ptrs[0])
        for j, (name, host_out) in enumerate(host_outputs, start=1):
            context.set_tensor_address(name, device_ptrs[j])

        def infer():
            ctypes.memmove(pinned_input.value, host_input.ctypes.data, host_input.nbytes)
            _cuda_ok(
                CUDA.cudaMemcpyAsync(
                    ctypes.c_void_p(device_ptrs[0]),
                    pinned_input,
                    ctypes.c_size_t(host_input.nbytes),
                    1,
                    ctypes.c_void_p(stream),
                ),
                "cudaMemcpyAsync H2D failed",
            )
            t0 = time.perf_counter()
            if not context.execute_async_v3(stream):
                raise RuntimeError("TensorRT execute_async_v3 returned failure")
            _cuda_ok(CUDA.cudaStreamSynchronize(ctypes.c_void_p(stream)), "cudaStreamSynchronize failed")
            elapsed_ms = (time.perf_counter() - t0) * 1e3
            for j, (name, host_out) in enumerate(host_outputs, start=1):
                _cuda_ok(
                    CUDA.cudaMemcpyAsync(
                        pinned_outputs[j - 1],
                        ctypes.c_void_p(device_ptrs[j]),
                        ctypes.c_size_t(host_out.nbytes),
                        2,
                        ctypes.c_void_p(stream),
                    ),
                    "cudaMemcpyAsync D2H failed",
                )
                _cuda_ok(CUDA.cudaStreamSynchronize(ctypes.c_void_p(stream)), "cudaStreamSynchronize failed")
                ctypes.memmove(host_out.ctypes.data, pinned_outputs[j - 1].value, host_out.nbytes)
            return elapsed_ms

        return infer, False, host_input.shape
    except Exception:
        for ptr in device_ptrs:
            cuda_free(ptr)
        if pinned_input.value:
            CUDA.cudaFreeHost(pinned_input)
        for pinned in pinned_outputs:
            if pinned.value:
                CUDA.cudaFreeHost(pinned)
        if stream_handle is None:
            cuda_stream_destroy(stream)
        raise
    finally:
        pass


def background_gpu_worker(stop: mp.Event, engine_path: str, input_shape: str, priority: int) -> None:
    try:
        cuda_init()
        infer_fn, _, _ = make_infer_fn(Path(engine_path), parse_shape(input_shape), stream_handle=cuda_stream_create(priority))
    except Exception as exc:
        print(json.dumps({"gpu_burn_error": str(exc)}), file=sys.stderr, flush=True)
        return
    while not stop.is_set():
        try:
            infer_fn()
        except Exception:
            break


def background_gpu_elastic_worker(stop: mp.Event, repeat: int = 1) -> None:
    pqc_bench = ROOT / "build" / "bench_gpu_pqc"
    integrity_bench = ROOT / "build" / "bench_gpu_integrity"
    if not pqc_bench.exists() or not integrity_bench.exists():
        print(json.dumps({
            "gpu_elastic_burn_error": "bench_gpu_pqc or bench_gpu_integrity missing"
        }), file=sys.stderr, flush=True)
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


def run_trial(label: str, engine_path: Path, input_shape: tuple[int, ...], daemon_env: dict[str, str] | None, trial: int, duration_s: int, out_dir: Path, writer_size: int, writer_count: int, writer_sync_every: int, gpu_burn_workers: int = 0) -> dict[str, object]:
    ctx = mp.get_context("spawn")
    storage_dir = Path(tempfile.mkdtemp(prefix=f"aegis_trt_{label}_{trial}_store_"))
    mount_dir = Path(tempfile.mkdtemp(prefix=f"aegis_trt_{label}_{trial}_mnt_"))
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
        infer_fn, cpu_fallback, _ = make_infer_fn(engine_path, input_shape)
        if cpu_fallback:
            raise RuntimeError("GPU-resident TensorRT inference was not verified")
        samples = []
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            samples.append(infer_fn())
        if len(samples) < 20:
            raise RuntimeError(f"insufficient TensorRT samples: {len(samples)}")
        return {
            "mode": label,
            "trial": trial,
            "duration_s": duration_s,
            "engine": str(engine_path),
            "count": len(samples),
            "samples_ms": samples,
            **summary(samples),
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


def plot(rows: list[dict[str, object]], out_dir: Path) -> None:
    colors = {
        "inference_only": "#666666",
        "cpu_only": "#4C78A8",
        "gpu_only": "#E45756",
        "adaptive": "#54A24B",
    }
    labels = {
        "inference_only": "Inference only",
        "cpu_only": "CPU-only secure I/O",
        "gpu_only": "GPU-only secure I/O",
        "adaptive": "AEGIS-Q adaptive",
    }
    fig, (cdf_ax, tail_ax) = plt.subplots(1, 2, figsize=(7.1, 2.65))
    modes = ["inference_only", "cpu_only", "gpu_only", "adaptive"]
    for mode in modes:
        samples = sorted(
            value
            for row in rows
            if str(row["mode"]).endswith(f":{mode}")
            for value in row["samples_ms"]
        )
        if not samples:
            continue
        cdf_ax.plot(samples, [(i + 1) / len(samples) for i in range(len(samples))], lw=1.3, color=colors[mode], label=labels[mode])
    tail_rows = []
    for mode in modes:
        mode_rows = [row for row in rows if str(row["mode"]).endswith(f":{mode}")]
        if not mode_rows:
            continue
        tail_rows.append({
            "mode": mode,
            "p50_ms": statistics.median(float(row["p50_ms"]) for row in mode_rows),
            "p99_ms": statistics.median(float(row["p99_ms"]) for row in mode_rows),
        })
    x = range(len(tail_rows))
    p50 = [float(row["p50_ms"]) for row in tail_rows]
    p99 = [float(row["p99_ms"]) for row in tail_rows]
    tail_ax.bar([i - 0.17 for i in x], p50, width=0.34, label="p50", color="#4C78A8")
    tail_ax.bar([i + 0.17 for i in x], p99, width=0.34, label="p99", color="#E45756")
    cdf_ax.set_xlabel("TensorRT GPU latency (ms)")
    cdf_ax.set_ylabel("CDF")
    cdf_ax.set_title("Inference latency distribution", fontsize=9)
    cdf_ax.grid(True, alpha=0.22)
    tail_ax.set_xticks(list(x))
    tail_ax.set_xticklabels(["only", "CPU", "GPU", "adaptive"], fontsize=8)
    tail_ax.set_ylabel("TensorRT GPU latency (ms)")
    tail_ax.set_title("Tail-latency summary", fontsize=9)
    tail_ax.grid(True, axis="y", alpha=0.22)
    tail_ax.legend(fontsize=7, frameon=False)
    handles, labels_ = cdf_ax.get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(top=0.72, wspace=0.36)
    fig.savefig(out_dir / "e3_tensorrt_qos.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="E3/M6 latency pass for TensorRT interference")
    ap.add_argument("out_dir", nargs="?", default=str(ROOT / "artifacts" / "motivation"))
    ap.add_argument("--engine", required=False, default=str(ROOT / "artifacts" / "yolov8n.plan"))
    ap.add_argument("--onnx", required=False, default=str(ROOT / "artifacts" / "yolov8n.onnx"))
    ap.add_argument("--model-name", default=os.environ.get("AEGIS_TRT_MODEL", "yolov8"))
    ap.add_argument("--duration", type=int, default=5)
    ap.add_argument("--input-shape", default="1x3x640x640")
    ap.add_argument("--writer-size", type=int, default=512 * 1024, help="Bytes written per fsync cycle by each background writer")
    ap.add_argument("--writer-count", type=int, default=1, help="Number of concurrent background writers")
    ap.add_argument("--writer-sync-every", type=int, default=1, help="How many write+fsync iterations each writer performs per loop")
    ap.add_argument("--trials", type=int, default=3, help="Independent repetitions per policy")
    args = ap.parse_args()

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    engine_path = load_or_build_engine(Path(args.engine).resolve(), Path(args.onnx).resolve(), parse_shape(args.input_shape))

    if not FUSE.exists():
        raise SystemExit("build/pqc_fuse is missing; run cmake --build build first")

    configurations: list[tuple[str, dict[str, str] | None, int]] = [
        (f"{args.model_name}:inference_only", None, 0),
        (f"{args.model_name}:cpu_only", {"PQC_GPU_MIN_BYTES": "1073741824"}, 0),
        (f"{args.model_name}:gpu_only", {
            "PQC_GPU_MIN_BYTES": "1",
            "PQC_GPU_MAX_INFLIGHT_JOBS": "1",
            "PQC_GPU_MAX_INFLIGHT_BYTES": "1048576",
            "PQC_GPU_MAX_WAIT_NS": "50000",
            "PQC_GPU_MIN_BATCH": "1",
            "PQC_FORCE_REKEY_ON_WRITE": "1",
            "PQC_GPU_BURN_ITERS": "300000",
        }, 2),
        (f"{args.model_name}:adaptive", {
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
        for trial in range(args.trials):
            rows.append(
                run_trial(label, engine_path, parse_shape(args.input_shape), env, trial, args.duration, out,
                          args.writer_size, args.writer_count, args.writer_sync_every, gpu_burn_workers=gpu_burn_workers)
            )
    (out / "tensorrt_interference.json").write_text(json.dumps(rows, indent=2))
    with (out / "tensorrt_interference.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "trial", "duration_s", "engine", "count", "p50_ms", "p95_ms", "p99_ms", "mean_ms", "stdev_ms"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    plot(rows, out)
    print(json.dumps([{key: row[key] for key in ("mode", "trial", "count", "p50_ms", "p95_ms", "p99_ms")} for row in rows], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
