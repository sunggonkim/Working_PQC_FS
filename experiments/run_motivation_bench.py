#!/usr/bin/env python3
import csv
import json
import os
import sqlite3
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import multiprocessing
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build" / "pqc_fuse"
ARTIFACTS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "artifacts" / "motivation"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd, env=None, capture_stderr=False):
    merged = os.environ.copy()
    if env:
        merged.update(env)
    res = subprocess.run(
        cmd,
        cwd=ROOT,
        env=merged,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE if capture_stderr else subprocess.STDOUT,
        text=True,
        check=True,
    )
    if capture_stderr:
        return res.stdout, res.stderr
    return res.stdout


def parse_json_lines(text):
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        rows.append(json.loads(line))
    return rows


def run_smoke(policy_name, env):
    out = run_cmd([str(BUILD), "--scheduler-smoke"], env=env, capture_stderr=True)
    rows = parse_json_lines("\n".join(out))
    path = ARTIFACTS / f"scheduler_smoke_{policy_name}.jsonl"
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return rows


def start_fuse(storage_dir, mount_dir, daemon_env=None):
    env = {**os.environ, "PQC_MASTER_PASSWORD": "test-password"}
    if daemon_env:
        env.update(daemon_env)
    proc = subprocess.Popen(
        [str(BUILD), str(storage_dir), str(mount_dir), "-f", "-o", "writeback_cache"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.time() + 20
    while time.time() < deadline:
        probe = subprocess.run(["mountpoint", "-q", str(mount_dir)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if probe.returncode == 0:
            break
        time.sleep(0.5)
    time.sleep(0.5)
    return proc


def stop_fuse(proc, mount_dir):
    try:
        subprocess.run(["fusermount3", "-u", str(mount_dir)], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def write_latency(mount_dir, name, size, tier=None):
    path = mount_dir / name
    data = b"A" * size
    path.touch(exist_ok=True)
    if tier is not None:
        os.setxattr(path, b"user.pqc_tier", str(tier).encode("utf-8"))
    t0 = time.perf_counter_ns()
    with open(path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    t1 = time.perf_counter_ns()
    with open(path, "rb") as f:
        readback = f.read()
    assert readback == data, f"round-trip mismatch for {path}"
    return (t1 - t0) / 1e6


def summarize(samples):
    ordered = sorted(samples)
    p95 = statistics.quantiles(ordered, n=20, method="inclusive")[18] if len(ordered) > 1 else ordered[0]
    p99 = statistics.quantiles(ordered, n=100, method="inclusive")[98] if len(ordered) > 1 else ordered[0]
    return {
        "median_ms": statistics.median(ordered),
        "p95_ms": p95,
        "p99_ms": p99,
        "mean_ms": statistics.fmean(ordered),
        "stdev_ms": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
    }


def run_fuse_bench(samples_per_point=12):
    """Measure static CPU/GPU and adaptive placement in the daemon itself.

    The old harness set scheduler variables in writer processes after the FUSE
    daemon had started, so the supposed policy ablations were not real.  Each
    mode below starts a fresh daemon with its policy environment fixed before
    mounting the filesystem.
    """
    modes = [
        ("plaintext_fuse", {"PQC_EXECUTION_MODE": "adaptive"}, 2),
        ("cpu_only", {"PQC_EXECUTION_MODE": "cpu"}, None),
        ("gpu_only", {"PQC_EXECUTION_MODE": "gpu", "PQC_GPU_MIN_BYTES": "4096"}, None),
        # Adaptive must retain the production batch threshold.  Lowering this
        # to 4 KiB turns the policy into a disguised GPU-only baseline and
        # cannot test the intended CPU-fast/GPU-batch crossover.
        ("adaptive", {"PQC_EXECUTION_MODE": "adaptive"}, None),
    ]
    sizes = [4096, 16384, 131072, 524288]
    rows = []
    for mode, daemon_env, tier in modes:
        storage_dir = Path(tempfile.mkdtemp(prefix=f"aegis_{mode}_store_"))
        mount_dir = Path(tempfile.mkdtemp(prefix=f"aegis_{mode}_mnt_"))
        proc = start_fuse(storage_dir, mount_dir, daemon_env)
        try:
            for size in sizes:
                samples = [
                    write_latency(mount_dir, f"{mode}_{size}_{i}.bin", size, tier=tier)
                    for i in range(samples_per_point)
                ]
                rows.append({
                    "tier": mode,
                    "size": size,
                    "samples_ms": samples,
                    **summarize(samples),
                    "daemon_env": daemon_env,
                })
        finally:
            stop_fuse(proc, mount_dir)
            shutil.rmtree(storage_dir, ignore_errors=True)
            shutil.rmtree(mount_dir, ignore_errors=True)
    path = ARTIFACTS / "fuse_latency.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["tier", "size", "samples_ms", "median_ms", "p95_ms", "p99_ms", "mean_ms", "stdev_ms", "daemon_env"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    with (ARTIFACTS / "fuse_latency.json").open("w") as f:
        json.dump(rows, f, indent=2)
    return rows


def run_gocryptfs_bench():
    storage_dir = Path(tempfile.mkdtemp(prefix="gocryptfs_store_"))
    mount_dir = Path(tempfile.mkdtemp(prefix="gocryptfs_mnt_"))
    password_cmd = "printf %s test-password"
    try:
        run_cmd([
            "gocryptfs",
            "-q",
            "-init",
            "-extpass",
            password_cmd,
            str(storage_dir),
        ])
        proc = subprocess.Popen(
            [
                "gocryptfs",
                "-q",
                "-allow_other",
                "-extpass",
                password_cmd,
                str(storage_dir),
                str(mount_dir),
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.time() + 20
        while time.time() < deadline:
            probe = subprocess.run(["mountpoint", "-q", str(mount_dir)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if probe.returncode == 0:
                break
            time.sleep(0.5)
        time.sleep(0.5)
        sizes = [4096, 16384, 131072, 524288]
        rows = []
        for size in sizes:
            samples = []
            for i in range(3):
                lat = write_latency(mount_dir, f"gocryptfs_{size}_{i}.bin", size, tier=None)
                samples.append(lat)
            rows.append({
                "tier": "gocryptfs",
                "size": size,
                "samples_ms": samples,
                "median_ms": statistics.median(samples),
            })
        path = ARTIFACTS / "gocryptfs_latency.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["tier", "size", "samples_ms", "median_ms"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return rows
    finally:
        subprocess.run(["fusermount3", "-u", str(mount_dir)], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if 'proc' in locals() and proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        shutil.rmtree(storage_dir, ignore_errors=True)
        shutil.rmtree(mount_dir, ignore_errors=True)


def cpu_burn(stop_event):
    try:
        os.sched_setaffinity(0, {0})
    except Exception:
        pass
    x = 0
    while not stop_event.is_set():
        x = (x * 1103515245 + 12345) & 0xFFFFFFFF


def mnist_infer(stop_event, model_path):
    import numpy as np
    import onnxruntime as ort

    try:
        os.sched_setaffinity(0, {0})
    except Exception:
        pass
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    rng = np.random.default_rng(0)
    sample = rng.random((1, 1, 28, 28), dtype=np.float32)
    while not stop_event.is_set():
        sess.run(None, {input_name: sample})


def yolo_infer(stop_event, model_path):
    import numpy as np
    import onnxruntime as ort

    try:
        os.sched_setaffinity(0, {0})
    except Exception:
        pass
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    rng = np.random.default_rng(1)
    sample = rng.random((1, 3, 640, 640), dtype=np.float32)
    for _ in range(3):
        sess.run(None, {input_name: sample})
    while not stop_event.is_set():
        sess.run(None, {input_name: sample})


def yolo_timed_infer(model_path, iterations=24):
    import numpy as np
    import onnxruntime as ort

    try:
        os.sched_setaffinity(0, {0})
    except Exception:
        pass
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    rng = np.random.default_rng(7)
    sample = rng.random((1, 3, 640, 640), dtype=np.float32)
    samples = []
    for _ in range(3):
        sess.run(None, {input_name: sample})
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        sess.run(None, {input_name: sample})
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1e6)
    return samples


def squeeze_infer(stop_event, model_path):
    import numpy as np
    import onnxruntime as ort

    try:
        os.sched_setaffinity(0, {0})
    except Exception:
        pass
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    rng = np.random.default_rng(3)
    sample = rng.random((1, 3, 224, 224), dtype=np.float32)
    for _ in range(3):
        sess.run(None, {input_name: sample})
    while not stop_event.is_set():
        sess.run(None, {input_name: sample})


def squeeze_timed_infer(model_path, iterations=24):
    import numpy as np
    import onnxruntime as ort

    try:
        os.sched_setaffinity(0, {0})
    except Exception:
        pass
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    rng = np.random.default_rng(11)
    sample = rng.random((1, 3, 224, 224), dtype=np.float32)
    samples = []
    for _ in range(3):
        sess.run(None, {input_name: sample})
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        sess.run(None, {input_name: sample})
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1e6)
    return samples


def run_contention_bench():
    storage_dir = Path(tempfile.mkdtemp(prefix="skim_cont_store_"))
    mount_dir = Path(tempfile.mkdtemp(prefix="skim_cont_mnt_"))
    proc = start_fuse(storage_dir, mount_dir)
    stop_event = multiprocessing.Event()
    burner = None
    try:
        try:
            os.sched_setaffinity(0, {0})
        except Exception:
            pass
        rows = []
        for mode in ["baseline", "cpu_contention"]:
            if mode == "cpu_contention":
                burner = multiprocessing.Process(target=cpu_burn, args=(stop_event,), daemon=True)
                burner.start()
                time.sleep(0.5)
            samples = []
            for i in range(20):
                samples.append(write_latency(mount_dir, f"{mode}_{i}.bin", 131072, tier=None))
            rows.append({
                "mode": mode,
                "size": 131072,
                "samples_ms": samples,
                "median_ms": statistics.median(samples),
                "p95_ms": statistics.quantiles(samples, n=20, method="inclusive")[18],
            })
            if mode == "cpu_contention":
                stop_event.set()
                burner.join(timeout=5)
        with (ARTIFACTS / "contention_latency.json").open("w") as f:
            json.dump(rows, f, indent=2)
        with (ARTIFACTS / "contention_latency.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["mode", "size", "samples_ms", "median_ms", "p95_ms"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return rows
    finally:
        stop_event.set()
        if burner is not None:
            burner.join(timeout=5)
        stop_fuse(proc, mount_dir)
        shutil.rmtree(storage_dir, ignore_errors=True)
        shutil.rmtree(mount_dir, ignore_errors=True)


def run_inference_bench():
    model_specs = [
        ("yolov8", ROOT / "artifacts" / "models" / "yolov8n.onnx", [
            "https://huggingface.co/cabelo/yolov8/resolve/main/yolov8n.onnx?download=true",
            "https://dl.opencv.org/models/yolov8/yolov8n.onnx",
        ], (1, 3, 640, 640), yolo_infer, yolo_timed_infer),
        ("squeezenet", ROOT / "artifacts" / "models" / "squeezenet1.1.onnx", [
            "https://huggingface.co/qualcomm/SqueezeNet-1.1/resolve/main/SqueezeNet.onnx?download=true",
            "https://huggingface.co/onnxmodelzoo/squeezenet1.0-12/resolve/main/squeezenet1.0-12.onnx?download=true",
        ], (1, 3, 224, 224), squeeze_infer, squeeze_timed_infer),
        ("mnist", ROOT / "artifacts" / "mnist-8.onnx", [
            "https://github.com/ankitshah009/mnist-onnx/raw/main/mnist-8.onnx",
        ], (1, 1, 28, 28), mnist_infer, None),
    ]
    storage_dir = Path(tempfile.mkdtemp(prefix="skim_ai_store_"))
    mount_dir = Path(tempfile.mkdtemp(prefix="skim_ai_mnt_"))
    proc = start_fuse(storage_dir, mount_dir)
    stop_event = multiprocessing.Event()
    worker = None
    try:
        try:
            os.sched_setaffinity(0, {0})
        except Exception:
            pass
        rows = []
        for workload, model_path, urls, shape, infer_fn, timed_fn in model_specs:
            if not model_path.exists():
                model_path.parent.mkdir(parents=True, exist_ok=True)
                last_error = None
                for url in urls:
                    try:
                        with urllib.request.urlopen(url, timeout=120) as r:
                            model_path.write_bytes(r.read())
                        break
                    except Exception as exc:
                        last_error = exc
                if not model_path.exists():
                    rows.append({
                        "workload": workload,
                        "mode": "unavailable",
                        "size": 131072,
                        "samples_ms": [],
                        "median_ms": None,
                        "p99_ms": None,
                        "error": str(last_error),
                    })
                    continue
            for mode in ["baseline", f"{workload}_inference"]:
                if mode.endswith("_inference"):
                    worker = multiprocessing.Process(target=infer_fn, args=(stop_event, model_path), daemon=True)
                    worker.start()
                    time.sleep(1.5)
                else:
                    write_latency(mount_dir, "baseline_warmup.bin", 131072, tier=None)
                samples = [write_latency(mount_dir, f"{mode}_{i}.bin", 131072, tier=None) for i in range(12)]
                rows.append({
                    "workload": workload,
                    "mode": mode,
                    "size": 131072,
                    "samples_ms": samples,
                    "median_ms": statistics.median(samples),
                    "p99_ms": statistics.quantiles(samples, n=100, method="inclusive")[98],
                })
                if mode.endswith("_inference"):
                    stop_event.set()
                    worker.join(timeout=10)
                    stop_event.clear()
        with (ARTIFACTS / "inference_latency.json").open("w") as f:
            json.dump(rows, f, indent=2)
        with (ARTIFACTS / "inference_latency.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["workload", "mode", "size", "samples_ms", "median_ms", "p99_ms", "error"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return rows
    finally:
        stop_event.set()
        if worker is not None:
            worker.join(timeout=10)
        stop_fuse(proc, mount_dir)
        shutil.rmtree(storage_dir, ignore_errors=True)
        shutil.rmtree(mount_dir, ignore_errors=True)


def writer_worker(mount_dir, prefix, size, count, q):
    try:
        os.sched_setaffinity(0, {0})
    except Exception:
        pass
    samples = []
    for i in range(count):
        samples.append(write_latency(mount_dir, f"{prefix}_{i}.bin", size, tier=None))
    q.put(samples)


def run_pressure_spill_bench():
    storage_dir = Path(tempfile.mkdtemp(prefix="skim_spill_store_"))
    mount_dir = Path(tempfile.mkdtemp(prefix="skim_spill_mnt_"))
    env = {
        "PQC_EXECUTION_MODE": "adaptive",
        "PQC_GPU_MAX_INFLIGHT_JOBS": "1",
        "PQC_GPU_MAX_INFLIGHT_BYTES": str(128 * 1024),
        "PQC_GPU_MIN_BYTES": "4096",
    }
    proc = start_fuse(storage_dir, mount_dir, env)
    try:
        ctx = multiprocessing.get_context("spawn")
        q1 = ctx.Queue()
        q2 = ctx.Queue()
        p1 = ctx.Process(target=writer_worker, args=(mount_dir, "a", 131072, 4, q1))
        p2 = ctx.Process(target=writer_worker, args=(mount_dir, "b", 131072, 4, q2))
        p1.start(); p2.start()
        s1 = q1.get()
        s2 = q2.get()
        p1.join(); p2.join()
        rows = [
            {"mode": "writer_a", "samples_ms": s1, "median_ms": statistics.median(s1), "p95_ms": statistics.quantiles(s1, n=20, method="inclusive")[18]},
            {"mode": "writer_b", "samples_ms": s2, "median_ms": statistics.median(s2), "p95_ms": statistics.quantiles(s2, n=20, method="inclusive")[18]},
        ]
        with (ARTIFACTS / "spillover_latency.json").open("w") as f:
            json.dump(rows, f, indent=2)
        with (ARTIFACTS / "spillover_latency.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["mode", "samples_ms", "median_ms", "p95_ms"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        plt.figure(figsize=(6, 3.5))
        labels = [r["mode"] for r in rows]
        medians = [r["median_ms"] for r in rows]
        p95s = [r["p95_ms"] for r in rows]
        x = range(len(labels))
        plt.bar([i - 0.15 for i in x], medians, width=0.3, label="median")
        plt.bar([i + 0.15 for i in x], p95s, width=0.3, label="p95")
        plt.xticks(list(x), labels)
        plt.ylabel("Write latency (ms)")
        plt.title("Dynamic spill-over under GPU inflight cap")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(ARTIFACTS / "spillover_latency.png", dpi=200)
        plt.close()
        return rows
    finally:
        stop_fuse(proc, mount_dir)
        shutil.rmtree(storage_dir, ignore_errors=True)
        shutil.rmtree(mount_dir, ignore_errors=True)


def run_edge_pipeline_bench():
    model_path = ROOT / "artifacts" / "models" / "yolov8n.onnx"
    model_error = None
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urls = [
            "https://huggingface.co/cabelo/yolov8/resolve/main/yolov8n.onnx?download=true",
            "https://dl.opencv.org/models/yolov8/yolov8n.onnx",
        ]
        for url in urls:
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    model_path.write_bytes(r.read())
                model_error = None
                break
            except Exception as exc:
                model_error = exc
    storage_dir = Path(tempfile.mkdtemp(prefix="skim_pipe_store_"))
    mount_dir = Path(tempfile.mkdtemp(prefix="skim_pipe_mnt_"))
    proc = start_fuse(storage_dir, mount_dir)
    ctx = multiprocessing.get_context("spawn")
    try:
        if model_error is not None and not model_path.exists():
            rows = [{
                "mode": "unavailable",
                "samples_ms": [],
                "median_ms": None,
                "p99_ms": None,
                "error": str(model_error),
            }]
            with (ARTIFACTS / "pipeline_latency.json").open("w") as f:
                json.dump(rows, f, indent=2)
            with (ARTIFACTS / "pipeline_latency.csv").open("w", newline="") as f:
                writer_csv = csv.DictWriter(f, fieldnames=["mode", "samples_ms", "median_ms", "p99_ms", "error"])
                writer_csv.writeheader()
                for row in rows:
                    writer_csv.writerow(row)
            return rows
        rows = []
        for mode in ["baseline", "with_storage"]:
            writer = None
            if mode == "with_storage":
                q = ctx.Queue()
                writer = ctx.Process(target=writer_worker, args=(mount_dir, "pipe", 131072, 30, q), daemon=True)
                writer.start()
                time.sleep(0.8)
            samples = yolo_timed_infer(model_path, iterations=24)
            rows.append({
                "mode": mode,
                "samples_ms": samples,
                "median_ms": statistics.median(samples),
                "p99_ms": statistics.quantiles(samples, n=100, method="inclusive")[98],
            })
            if writer is not None:
                writer.join(timeout=10)
        with (ARTIFACTS / "pipeline_latency.json").open("w") as f:
            json.dump(rows, f, indent=2)
        with (ARTIFACTS / "pipeline_latency.csv").open("w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=["mode", "samples_ms", "median_ms", "p99_ms"])
            writer_csv.writeheader()
            for row in rows:
                writer_csv.writerow(row)
        plt.figure(figsize=(6, 3.5))
        labels = [r["mode"] for r in rows]
        medians = [r["median_ms"] for r in rows]
        p99s = [r["p99_ms"] for r in rows]
        x = range(len(labels))
        plt.bar([i - 0.15 for i in x], medians, width=0.3, label="median")
        plt.bar([i + 0.15 for i in x], p99s, width=0.3, label="p99")
        plt.xticks(list(x), labels)
        plt.ylabel("YOLO inference latency (ms)")
        plt.title("Edge pipeline inference under concurrent secure writes")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(ARTIFACTS / "pipeline_latency.png", dpi=200)
        plt.close()
        return rows
    finally:
        stop_fuse(proc, mount_dir)
        shutil.rmtree(storage_dir, ignore_errors=True)
        shutil.rmtree(mount_dir, ignore_errors=True)


def run_crash_regression_bench(trials=3):
    rows = []
    for trial in range(trials):
        storage_dir = Path(tempfile.mkdtemp(prefix=f"skim_crash_store_{trial}_"))
        mount_dir = Path(tempfile.mkdtemp(prefix=f"skim_crash_mnt_{trial}_"))
        proc = start_fuse(storage_dir, mount_dir)
        proc2 = None
        try:
            p = mount_dir / "case.bin"
            baseline = b"A" * (1024 * 1024)
            with open(p, "wb") as f:
                f.write(baseline)
                f.flush()
                os.fsync(f.fileno())

            import threading
            def writer():
                try:
                    with open(p, "wb") as f:
                        f.write(b"B" * (256 * 1024))
                        f.flush()
                        time.sleep(0.2)
                        f.write(b"B" * (768 * 1024))
                        f.flush()
                except Exception:
                    pass

            t = threading.Thread(target=writer, daemon=True)
            t.start()
            time.sleep(0.05)
            proc.send_signal(signal.SIGKILL)
            t.join(timeout=2)
            time.sleep(0.5)
            subprocess.run(["fusermount3", "-u", str(mount_dir)], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            proc2 = start_fuse(storage_dir, mount_dir)
            try:
                with open(p, "rb") as f:
                    first16 = f.read(16)
                rows.append({
                    "trial": trial,
                    "first16": first16.decode("ascii", errors="replace"),
                    "preserved_baseline": first16 == b"A" * 16,
                    "result": "read_ok" if first16 == b"A" * 16 else "read_new",
                })
            except OSError as exc:
                rows.append({
                    "trial": trial,
                    "first16": "",
                    "preserved_baseline": False,
                    "result": f"fail_closed_errno_{exc.errno}",
                })
        finally:
            if proc2 is not None and proc2.poll() is None:
                proc2.send_signal(signal.SIGINT)
                try:
                    proc2.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc2.kill()
            if proc.poll() is None:
                proc.kill()
            subprocess.run(["fusermount3", "-u", str(mount_dir)], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            shutil.rmtree(storage_dir, ignore_errors=True)
            shutil.rmtree(mount_dir, ignore_errors=True)
    with (ARTIFACTS / "crash_regression.json").open("w") as f:
        json.dump(rows, f, indent=2)
    with (ARTIFACTS / "crash_regression.txt").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return rows


def run_crash_replay_matrix_bench(trials=5):
    rows = []
    cutpoints = [0.0, 0.02, 0.05, 0.10]
    for cut in cutpoints:
        for trial in range(trials):
            storage_dir = Path(tempfile.mkdtemp(prefix=f"skim_replay_store_{int(cut*100)}_{trial}_"))
            mount_dir = Path(tempfile.mkdtemp(prefix=f"skim_replay_mnt_{int(cut*100)}_{trial}_"))
            proc = start_fuse(storage_dir, mount_dir)
            try:
                p = mount_dir / "replay.bin"
                baseline = b"A" * (1024 * 1024)
                with open(p, "wb") as f:
                    f.write(baseline)
                    f.flush()
                    os.fsync(f.fileno())
                import threading
                write_plan = [b"B" * (128 * 1024), b"C" * (128 * 1024), b"D" * (128 * 1024)]
                def writer():
                    try:
                        with open(p, "wb") as f:
                            for chunk in write_plan:
                                f.write(chunk)
                                f.flush()
                                time.sleep(0.03)
                    except Exception:
                        pass
                t = threading.Thread(target=writer, daemon=True)
                t.start()
                time.sleep(cut)
                proc.send_signal(signal.SIGKILL)
                t.join(timeout=2)
                subprocess.run(["fusermount3", "-u", str(mount_dir)], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                proc = start_fuse(storage_dir, mount_dir)
                try:
                    with open(p, "rb") as f:
                        head = f.read(16)
                        f.seek(1024 * 1024 - 16)
                        tail = f.read(16)
                        f.seek(0, os.SEEK_END)
                        size = f.tell()
                    rows.append({
                        "cutpoint_s": cut,
                        "trial": trial,
                        "size": size,
                        "head": head.decode("ascii", errors="replace"),
                        "tail": tail.decode("ascii", errors="replace"),
                        "baseline_head": head == b"A" * 16,
                        "result": "read_ok" if head == b"A" * 16 and tail == b"A" * 16 else "read_new",
                    })
                except OSError as exc:
                    rows.append({
                        "cutpoint_s": cut,
                        "trial": trial,
                        "size": 0,
                        "head": "",
                        "tail": "",
                        "baseline_head": False,
                        "result": f"fail_closed_errno_{exc.errno}",
                    })
            finally:
                stop_fuse(proc, mount_dir)
                shutil.rmtree(storage_dir, ignore_errors=True)
                shutil.rmtree(mount_dir, ignore_errors=True)
    with (ARTIFACTS / "crash_replay_matrix.json").open("w") as f:
        json.dump(rows, f, indent=2)
    with (ARTIFACTS / "crash_replay_matrix.txt").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    summary = {}
    for row in rows:
        key = f"{row['cutpoint_s']:.2f}"
        bucket = summary.setdefault(key, {"cutpoint_s": row["cutpoint_s"], "trials": 0, "preserved": 0, "sizes": []})
        bucket["trials"] += 1
        bucket["preserved"] += 1 if row["baseline_head"] and row["head"] == "AAAAAAAAAAAAAAAA" and row["tail"] == "AAAAAAAAAAAAAAAA" else 0
        bucket["sizes"].append(row["size"])
    summary_rows = []
    for key in sorted(summary.keys(), key=lambda x: float(x)):
        bucket = summary[key]
        summary_rows.append({
            "cutpoint_s": bucket["cutpoint_s"],
            "trials": bucket["trials"],
            "preserved": bucket["preserved"],
            "success_rate": bucket["preserved"] / bucket["trials"] if bucket["trials"] else 0.0,
            "median_size": statistics.median(bucket["sizes"]) if bucket["sizes"] else 0,
        })
    with (ARTIFACTS / "crash_replay_summary.json").open("w") as f:
        json.dump(summary_rows, f, indent=2)
    with (ARTIFACTS / "crash_replay_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cutpoint_s", "trials", "preserved", "success_rate", "median_size"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    plt.figure(figsize=(6, 3.5))
    xs = [r["cutpoint_s"] for r in summary_rows]
    ys = [r["success_rate"] for r in summary_rows]
    plt.plot(xs, ys, marker="o")
    plt.ylim(0, 1.05)
    plt.xlabel("Crash cut-point (s)")
    plt.ylabel("Replay preservation rate")
    plt.title("Crash/replay resistance across cut-points")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "crash_replay_summary.png", dpi=200)
    plt.close()
    return rows


def run_um_bench():
    um_rows = []
    try:
        out = run_cmd([str(BUILD), "--um-smoke"])
        um_rows.append(json.loads(out.strip()))
    except Exception as exc:
        um_rows.append({"error": str(exc)})
    with (ARTIFACTS / "um_counters.json").open("w") as f:
        json.dump(um_rows, f, indent=2)
    return um_rows


def run_sqlite_bench():
    storage_dir = Path(tempfile.mkdtemp(prefix="skim_sqlite_store_"))
    mount_dir = Path(tempfile.mkdtemp(prefix="skim_sqlite_mnt_"))
    proc = start_fuse(storage_dir, mount_dir, {"PQC_ALLOW_SQLITE_MMAP": "1"})
    try:
        rows = []
        for tier_name, tier in [("full", None), ("plain", 2)]:
            db_path = mount_dir / f"{tier_name}.db"
            if db_path.exists():
                db_path.unlink()
            db_path.touch(exist_ok=True)
            if tier is not None:
                os.setxattr(db_path, b"user.pqc_tier", str(tier).encode("utf-8"))
            conn = sqlite3.connect(str(db_path), timeout=1.0, check_same_thread=False)
            try:
                # WAL + FULL forces SQLite through the fsync ordering path.
                # Some FUSE backends still reject SQLite's WAL mmap path; in
                # that case we record the failure and fall back to DELETE so
                # the benchmark still emits a usable artifact bundle.
                conn.execute("PRAGMA mmap_size=0")
                requested_mode = "WAL"
                actual_mode = "WAL"
                sync_mode = "FULL"
                fallback_error = None
                try:
                    conn.execute("PRAGMA locking_mode=EXCLUSIVE")
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=FULL")
                    conn.execute("PRAGMA temp_store=MEMORY")
                    conn.execute("CREATE TABLE IF NOT EXISTS t(id INTEGER PRIMARY KEY, v TEXT)")
                    conn.commit()
                except sqlite3.OperationalError as exc:
                    fallback_error = str(exc)
                    conn.close()
                    if db_path.exists():
                        db_path.unlink()
                    db_path.touch(exist_ok=True)
                    conn = sqlite3.connect(str(db_path), timeout=1.0, check_same_thread=False)
                    conn.execute("PRAGMA mmap_size=0")
                    conn.execute("PRAGMA locking_mode=EXCLUSIVE")
                    conn.execute("PRAGMA journal_mode=DELETE")
                    conn.execute("PRAGMA synchronous=EXTRA")
                    conn.execute("PRAGMA temp_store=MEMORY")
                    conn.execute("CREATE TABLE IF NOT EXISTS t(id INTEGER PRIMARY KEY, v TEXT)")
                    conn.commit()
                    actual_mode = "DELETE"
                    sync_mode = "EXTRA"
                samples = []
                for batch in range(20):
                    payload = [(batch * 1000 + i, "x" * 128) for i in range(100)]
                    t0 = time.perf_counter_ns()
                    conn.executemany("INSERT OR REPLACE INTO t(id, v) VALUES (?, ?)", payload)
                    conn.commit()
                    t1 = time.perf_counter_ns()
                    samples.append((t1 - t0) / 1e6)
                integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
                rows.append({
                    "tier": tier_name,
                    "requested_mode": requested_mode,
                    "actual_mode": actual_mode,
                    "sync_mode": sync_mode,
                    "fallback_error": fallback_error,
                    "integrity_check": integrity,
                    "samples_ms": samples,
                    "median_ms": statistics.median(samples),
                    "p95_ms": statistics.quantiles(samples, n=20, method="inclusive")[18],
                })
            except sqlite3.Error as exc:
                rows.append({
                    "tier": tier_name,
                    "requested_mode": requested_mode,
                    "actual_mode": actual_mode,
                    "sync_mode": sync_mode,
                    "fallback_error": fallback_error or str(exc),
                    "integrity_check": "error",
                    "samples_ms": [],
                    "median_ms": None,
                    "p95_ms": None,
                    "error": str(exc),
                })
            finally:
                conn.close()
        with (ARTIFACTS / "sqlite_latency.json").open("w") as f:
            json.dump(rows, f, indent=2)
        with (ARTIFACTS / "sqlite_latency.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["tier", "requested_mode", "actual_mode", "sync_mode", "fallback_error", "integrity_check", "samples_ms", "median_ms", "p95_ms", "error"], lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        labels = [row["tier"] for row in rows if row.get("median_ms") is not None]
        medians = [row["median_ms"] for row in rows if row.get("median_ms") is not None]
        p95s = [row["p95_ms"] for row in rows if row.get("p95_ms") is not None]
        x = range(len(labels))
        plt.figure(figsize=(6, 3.5))
        plt.bar([i - 0.15 for i in x], medians, width=0.3, label="median")
        plt.bar([i + 0.15 for i in x], p95s, width=0.3, label="p95")
        plt.xticks(list(x), labels)
        plt.ylabel("Commit latency (ms)")
        plt.title("SQLite WAL/FULL commit latency on AEGIS-Q tiers")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(ARTIFACTS / "sqlite_latency.png", dpi=200)
        plt.close()
        return rows
    finally:
        stop_fuse(proc, mount_dir)
        shutil.rmtree(storage_dir, ignore_errors=True)
        shutil.rmtree(mount_dir, ignore_errors=True)


def sqlite_reader(stop_event, db_path, journal_mode="DELETE", sync_mode="EXTRA"):
    conn = sqlite3.connect(str(db_path), timeout=1.0, check_same_thread=False)
    try:
        conn.execute("PRAGMA mmap_size=0")
        try:
            conn.execute(f"PRAGMA journal_mode={journal_mode}")
            conn.execute(f"PRAGMA synchronous={sync_mode}")
        except sqlite3.OperationalError:
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.execute("PRAGMA synchronous=EXTRA")
        while not stop_event.is_set():
            try:
                conn.execute("SELECT COUNT(*) FROM t").fetchone()
            except sqlite3.OperationalError:
                pass
            except sqlite3.DatabaseError:
                break
    finally:
        conn.close()


def sqlite_exec_with_retry(conn, sql, params=None, attempts=5, delay=0.02):
    last_exc = None
    for attempt in range(attempts):
        try:
            if params is None:
                return conn.execute(sql)
            return conn.execute(sql, params)
        except sqlite3.OperationalError as exc:
            last_exc = exc
            time.sleep(delay * (attempt + 1))
    raise last_exc


def run_sqlite_contention_bench(trials_per_tier=1, sessions=1, batches=2, seed_rows=50, timeout_s=20):
    def _contention_run(storage_dir, mount_dir, requested_mode):
        proc = start_fuse(storage_dir, mount_dir, {"PQC_ALLOW_SQLITE_MMAP": "1"})
        try:
            rows = []
            errors = []
            start_ns = time.monotonic_ns()
            deadline_ns = start_ns + int(timeout_s * 1e9)

            def timed_out() -> bool:
                return time.monotonic_ns() >= deadline_ns

            for tier_name, tier in [("full", None), ("plain", 2)]:
                if timed_out():
                    rows.append({
                        "tier": tier_name,
                        "requested_mode": requested_mode,
                        "actual_mode": "UNKNOWN",
                        "sync_mode": "UNKNOWN",
                        "fallback_error": "timeout",
                        "integrity_check": "error",
                        "samples_ms": [],
                        "median_ms": None,
                        "p95_ms": None,
                        "error": "timeout",
                    })
                    continue

                db_path = mount_dir / f"{tier_name}_cont.db"
                db_path.touch(exist_ok=True)
                if tier is not None:
                    os.setxattr(db_path, b"user.pqc_tier", str(tier).encode("utf-8"))

                tier_error = None
                tier_integrity = "error"
                fallback_error = None
                actual_mode = requested_mode
                sync_mode = "FULL" if requested_mode == "WAL" else "EXTRA"

                # Bootstrap schema and seed rows.
                for bootstrap in range(2):
                    if timed_out():
                        tier_error = "timeout"
                        break
                    conn = sqlite3.connect(str(db_path), timeout=1.0, check_same_thread=False)
                    try:
                        sqlite_exec_with_retry(conn, "PRAGMA mmap_size=0")
                        if requested_mode == "WAL":
                            sqlite_exec_with_retry(conn, "PRAGMA locking_mode=NORMAL")
                            sqlite_exec_with_retry(conn, "PRAGMA journal_mode=WAL")
                            sqlite_exec_with_retry(conn, "PRAGMA synchronous=FULL")
                        else:
                            sqlite_exec_with_retry(conn, "PRAGMA locking_mode=EXCLUSIVE")
                            sqlite_exec_with_retry(conn, "PRAGMA journal_mode=DELETE")
                            sqlite_exec_with_retry(conn, "PRAGMA synchronous=EXTRA")
                        sqlite_exec_with_retry(conn, "PRAGMA temp_store=MEMORY")
                        sqlite_exec_with_retry(conn, "CREATE TABLE IF NOT EXISTS t(id INTEGER PRIMARY KEY, v TEXT)")
                        for i in range(seed_rows):
                            sqlite_exec_with_retry(conn, "INSERT OR REPLACE INTO t(id, v) VALUES (?, ?)", (i, "seed"))
                        conn.commit()
                        tier_integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
                        tier_error = None
                        break
                    except sqlite3.Error as exc:
                        tier_error = str(exc)
                        time.sleep(0.05 * (bootstrap + 1))
                    finally:
                        conn.close()

                if tier_error and requested_mode == "WAL":
                    fallback_error = tier_error
                    tier_error = None
                    actual_mode = "DELETE"
                    sync_mode = "EXTRA"
                    # Re-bootstrap in rollback journal mode if WAL cannot be sustained.
                    for bootstrap in range(2):
                        if timed_out():
                            tier_error = "timeout"
                            break
                        conn = sqlite3.connect(str(db_path), timeout=1.0, check_same_thread=False)
                        try:
                            sqlite_exec_with_retry(conn, "PRAGMA mmap_size=0")
                            sqlite_exec_with_retry(conn, "PRAGMA locking_mode=EXCLUSIVE")
                            sqlite_exec_with_retry(conn, "PRAGMA journal_mode=DELETE")
                            sqlite_exec_with_retry(conn, "PRAGMA synchronous=EXTRA")
                            sqlite_exec_with_retry(conn, "PRAGMA temp_store=MEMORY")
                            sqlite_exec_with_retry(conn, "CREATE TABLE IF NOT EXISTS t(id INTEGER PRIMARY KEY, v TEXT)")
                            for i in range(seed_rows):
                                sqlite_exec_with_retry(conn, "INSERT OR REPLACE INTO t(id, v) VALUES (?, ?)", (i, "seed"))
                            conn.commit()
                            tier_integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
                            tier_error = None
                            break
                        except sqlite3.Error as exc:
                            tier_error = str(exc)
                            time.sleep(0.05 * (bootstrap + 1))
                        finally:
                            conn.close()

                if tier_error:
                    rows.append({
                        "tier": tier_name,
                        "requested_mode": requested_mode,
                        "actual_mode": actual_mode,
                        "sync_mode": sync_mode,
                        "fallback_error": fallback_error,
                        "integrity_check": tier_integrity,
                        "samples_ms": [],
                        "median_ms": None,
                        "p95_ms": None,
                        "error": tier_error,
                    })
                    errors.append({"tier": tier_name, "error": tier_error})
                    continue

                stop_event = multiprocessing.Event()
                reader = multiprocessing.Process(
                    target=sqlite_reader,
                    args=(stop_event, db_path, actual_mode, sync_mode),
                    daemon=True,
                )
                reader.start()

                samples = []
                tier_error = None
                for session in range(sessions):
                    if timed_out():
                        tier_error = "timeout"
                        break
                    conn = sqlite3.connect(str(db_path), timeout=1.0, check_same_thread=False)
                    try:
                        sqlite_exec_with_retry(conn, "PRAGMA mmap_size=0")
                        if actual_mode == "WAL":
                            sqlite_exec_with_retry(conn, "PRAGMA locking_mode=NORMAL")
                            sqlite_exec_with_retry(conn, "PRAGMA journal_mode=WAL")
                            sqlite_exec_with_retry(conn, "PRAGMA synchronous=FULL")
                        else:
                            sqlite_exec_with_retry(conn, "PRAGMA locking_mode=EXCLUSIVE")
                            sqlite_exec_with_retry(conn, "PRAGMA journal_mode=DELETE")
                            sqlite_exec_with_retry(conn, "PRAGMA synchronous=EXTRA")
                        sqlite_exec_with_retry(conn, "PRAGMA temp_store=MEMORY")
                        for batch in range(batches):
                            if timed_out():
                                tier_error = "timeout"
                                break
                            payload = [(batch * 1000 + i, "y" * 128) for i in range(50)]
                            for attempt in range(3):
                                try:
                                    t0 = time.perf_counter_ns()
                                    conn.executemany("INSERT OR REPLACE INTO t(id, v) VALUES (?, ?)", payload)
                                    conn.commit()
                                    t1 = time.perf_counter_ns()
                                    samples.append((t1 - t0) / 1e6)
                                    tier_error = None
                                    break
                                except sqlite3.OperationalError as exc:
                                    tier_error = str(exc)
                                    time.sleep(0.01 * (attempt + 1))
                                except sqlite3.Error as exc:
                                    tier_error = str(exc)
                                    break
                        if samples:
                            tier_integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
                            break
                    except sqlite3.Error as exc:
                        tier_error = str(exc)
                        time.sleep(0.05 * (session + 1))
                    finally:
                        conn.close()

                stop_event.set()
                reader.join(timeout=5)

                if samples:
                    rows.append({
                        "tier": tier_name,
                        "requested_mode": requested_mode,
                        "actual_mode": actual_mode,
                        "sync_mode": sync_mode,
                        "fallback_error": fallback_error,
                        "integrity_check": tier_integrity,
                        "samples_ms": samples,
                        "median_ms": statistics.median(samples),
                        "p95_ms": summarize(samples)["p95_ms"],
                    })
                else:
                    rows.append({
                        "tier": tier_name,
                        "requested_mode": requested_mode,
                        "actual_mode": actual_mode,
                        "sync_mode": sync_mode,
                        "fallback_error": fallback_error,
                        "integrity_check": "error",
                        "samples_ms": [],
                        "median_ms": None,
                        "p95_ms": None,
                        "error": tier_error or "unknown sqlite contention failure",
                    })
                    if tier_error:
                        errors.append({"tier": tier_name, "error": tier_error})

            with (ARTIFACTS / "sqlite_contention_latency.json").open("w") as f:
                json.dump(rows, f, indent=2)
            with (ARTIFACTS / "sqlite_contention_latency.csv").open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["tier", "requested_mode", "actual_mode", "sync_mode", "fallback_error", "integrity_check", "samples_ms", "median_ms", "p95_ms", "error"],
                    lineterminator="\n",
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            if any(row.get("median_ms") is not None for row in rows):
                labels = [row["tier"] for row in rows if row.get("median_ms") is not None]
                medians = [row["median_ms"] for row in rows if row.get("median_ms") is not None]
                p95s = [row["p95_ms"] for row in rows if row.get("p95_ms") is not None]
                x = range(len(labels))
                plt.figure(figsize=(6, 3.5))
                plt.bar([i - 0.15 for i in x], medians, width=0.3, label="median")
                plt.bar([i + 0.15 for i in x], p95s, width=0.3, label="p95")
                plt.xticks(list(x), labels)
                plt.ylabel("Commit latency (ms)")
                plt.title(f"SQLite commit latency under read contention ({requested_mode})")
                plt.grid(True, axis="y", alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(ARTIFACTS / "sqlite_contention_latency.png", dpi=200)
                plt.close()
            else:
                with (ARTIFACTS / "sqlite_contention_latency.txt").open("w") as f:
                    for err in errors:
                        f.write(f"{err['tier']}: {err['error']}\n")
            return rows
        finally:
            stop_fuse(proc, mount_dir)
            shutil.rmtree(storage_dir, ignore_errors=True)
            shutil.rmtree(mount_dir, ignore_errors=True)

    if os.environ.get("PQC_SQLITE_CONTENTION_FAST") == "1":
        storage_dir = Path(tempfile.mkdtemp(prefix="skim_sqlite_cont_fast_store_"))
        mount_dir = Path(tempfile.mkdtemp(prefix="skim_sqlite_cont_fast_mnt_"))
        return _contention_run(storage_dir, mount_dir, "WAL")

    storage_dir = Path(tempfile.mkdtemp(prefix="skim_sqlite_cont_store_"))
    mount_dir = Path(tempfile.mkdtemp(prefix="skim_sqlite_cont_mnt_"))
    return _contention_run(storage_dir, mount_dir, "WAL")


def plot_smoke(smoke_rows):
    xs = [r["bytes"] for r in smoke_rows if r["event"] == "scheduler_smoke_job"]
    ys = [0 if r["target"] == "CPU" else 1 for r in smoke_rows if r["event"] == "scheduler_smoke_job"]
    labels = ["CPU", "GPU"]
    plt.figure(figsize=(6, 3))
    plt.scatter(xs, ys, s=120)
    plt.yticks([0, 1], labels)
    plt.xlabel("Job size (bytes)")
    plt.ylabel("Chosen target")
    plt.title("Scheduler smoke decisions")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "scheduler_smoke_decisions.png", dpi=200)
    plt.close()


def plot_secure_io_cost(rows):
    by_tier = {}
    for row in rows:
        by_tier.setdefault(row["tier"], []).append(row)
    styles = {
        "plaintext_fuse": ("#666666", "Plaintext FUSE"),
        "cpu_only": ("#4C78A8", "CPU-only"),
        "gpu_only": ("#E45756", "GPU-only"),
        "adaptive": ("#54A24B", "AEGIS-Q adaptive"),
    }
    fig, (lat_ax, tail_ax) = plt.subplots(1, 2, figsize=(7.1, 2.65))
    for tier, vals in sorted(by_tier.items()):
        vals = sorted(vals, key=lambda r: r["size"])
        xs = [r["size"] for r in vals]
        p50 = [r["median_ms"] for r in vals]
        p99 = [r["p99_ms"] for r in vals]
        color, label = styles.get(tier, (None, tier))
        lat_ax.plot(xs, p50, marker="o", ms=3.5, lw=1.7, color=color, label=label)
        lat_ax.fill_between(xs, p50, p99, color=color, alpha=0.13)
        tail_ax.plot(xs, [hi / lo for lo, hi in zip(p50, p99)], marker="o", ms=3.5, lw=1.7, color=color, label=label)
    for ax in (lat_ax, tail_ax):
        ax.set_xscale("log", base=2)
        ax.grid(True, which="both", alpha=0.22)
        ax.set_xlabel("Write size (bytes)")
    lat_ax.set_ylabel("Write + fsync latency (ms)")
    lat_ax.set_title("p50 with p99 envelope", fontsize=9)
    tail_ax.set_ylabel("p99 / p50")
    tail_ax.set_title("Tail amplification", fontsize=9)
    handles, labels = lat_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(top=0.73, wspace=0.34)
    fig.savefig(ARTIFACTS / "e1_secure_io_cost.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_baseline(rows):
    by_tier = {}
    for row in rows:
        by_tier.setdefault(row["tier"], []).append(row)
    plt.figure(figsize=(7, 4))
    for tier, vals in sorted(by_tier.items()):
        vals = sorted(vals, key=lambda r: r["size"])
        xs = [r["size"] for r in vals]
        ys = [r["median_ms"] for r in vals]
        plt.plot(xs, ys, marker="o", label=tier)
    plt.xscale("log", base=2)
    plt.xlabel("Write size (bytes)")
    plt.ylabel("Median write latency (ms)")
    plt.title("Baseline comparison: plaintext vs AEGIS-Q: Adaptive Edge Guard for PQC-Backed Secure Storage vs gocryptfs")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "baseline_write_latency.png", dpi=200)
    plt.close()


def plot_contention(rows):
    labels = [row["mode"] for row in rows]
    medians = [row["median_ms"] for row in rows]
    p95s = [row["p95_ms"] for row in rows]
    x = range(len(labels))
    plt.figure(figsize=(6, 3.5))
    plt.bar([i - 0.15 for i in x], medians, width=0.3, label="median")
    plt.bar([i + 0.15 for i in x], p95s, width=0.3, label="p95")
    plt.xticks(list(x), labels)
    plt.ylabel("Write latency (ms)")
    plt.title("Latency under CPU contention proxy")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "contention_latency.png", dpi=200)
    plt.close()


def plot_inference(rows):
    labels = [row["mode"] for row in rows]
    medians = [row["median_ms"] for row in rows]
    p99s = [row["p99_ms"] for row in rows]
    x = range(len(labels))
    plt.figure(figsize=(6, 3.5))
    plt.bar([i - 0.15 for i in x], medians, width=0.3, label="median")
    plt.bar([i + 0.15 for i in x], p99s, width=0.3, label="p99")
    plt.xticks(list(x), labels)
    plt.ylabel("Write latency (ms)")
    plt.title("Latency under YOLOv8 inference interference")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "inference_latency.png", dpi=200)
    plt.close()


def main():
    if not BUILD.exists():
        raise SystemExit("build/pqc_fuse missing; run cmake --build build first")

    smoke_runs = {
        "cpu_only": run_smoke("cpu_only", {"PQC_GPU_MIN_BYTES": "999999999"}),
        "default": run_smoke("default", {}),
        "coherence_strict": run_smoke(
            "coherence_strict",
            {"PQC_GPU_MIN_BYTES": "1", "PQC_COHERENCE_PENALTY_NS": "500"},
        ),
    }
    with (ARTIFACTS / "smoke_summary.json").open("w") as f:
        json.dump(smoke_runs, f, indent=2)

    latency_rows = run_fuse_bench()
    baseline_rows = []
    try:
        baseline_rows = run_gocryptfs_bench()
    except Exception as exc:
        baseline_rows = [{"tier": "gocryptfs_unavailable", "size": 0, "samples_ms": [0], "median_ms": 0, "error": str(exc)}]
    with (ARTIFACTS / "fuse_latency.json").open("w") as f:
        json.dump(latency_rows, f, indent=2)
    with (ARTIFACTS / "baseline_latency.json").open("w") as f:
        json.dump(baseline_rows, f, indent=2)

    plot_smoke(smoke_runs["default"])
    plot_secure_io_cost(latency_rows)
    if baseline_rows and baseline_rows[0].get("tier") != "gocryptfs_unavailable":
        plot_baseline(
            [{"tier": "skim_full", "size": row["size"], "median_ms": row["median_ms"]} for row in latency_rows if row["tier"] == "full"] +
            [{"tier": "skim_plain", "size": row["size"], "median_ms": row["median_ms"]} for row in latency_rows if row["tier"] == "plain"] +
            baseline_rows
        )
    contention_rows = run_contention_bench()
    plot_contention(contention_rows)
    inference_rows = run_inference_bench()
    plot_inference(inference_rows)
    run_pressure_spill_bench()
    run_edge_pipeline_bench()
    run_crash_regression_bench()
    run_crash_replay_matrix_bench()
    run_um_bench()
    run_sqlite_bench()
    run_sqlite_contention_bench()


if __name__ == "__main__":
    main()
