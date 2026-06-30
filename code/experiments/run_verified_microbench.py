#!/usr/bin/env python3
"""Run and retain evidence for the AEGIS-Q GPU/CPU microbenchmarks.

This harness intentionally accepts results only from the built binaries.  It
does not synthesize samples, estimate missing configurations, or relabel an
unsupported executor as measured.  Each raw stdout capture is retained so a
plot can be regenerated without relying on a transcribed paper number.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "microbench"
GPU_ROW = re.compile(r"^(ml_kem_(?:keygen|encaps|decaps)),gpu,(\d+),([0-9.]+),([0-9.]+)$")


def command_output(command: list[str]) -> str:
    return subprocess.run(command, cwd=ROOT, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def command_capture(command: list[str], timeout_s: float = 10.0) -> dict[str, object]:
    if shutil.which(command[0]) is None and not Path(command[0]).exists():
        return {"argv": command, "available": False, "returncode": None, "stdout": "", "stderr": ""}
    try:
        proc = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "argv": command,
            "available": True,
            "timeout": True,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }
    return {
        "argv": command,
        "available": True,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def quantile(samples: list[float], q: float) -> float:
    ordered = sorted(samples)
    index = round((len(ordered) - 1) * q)
    return ordered[index]


def bootstrap_ci(samples: list[float], seed_text: str, trials: int = 10000,
                 alpha: float = 0.05) -> tuple[float, float]:
    if not samples:
        raise ValueError("empty samples")
    if len(samples) == 1:
        return samples[0], samples[0]
    seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    values: list[float] = []
    n = len(samples)
    for _ in range(trials):
        values.append(statistics.median(samples[rng.randrange(n)] for _ in range(n)))
    values.sort()
    lo = max(0, min(trials - 1, int((alpha / 2.0) * trials)))
    hi = max(0, min(trials - 1, int((1.0 - alpha / 2.0) * trials) - 1))
    return values[lo], values[hi]


def read_cpu_governors() -> dict[str, object]:
    governors: dict[str, int] = {}
    paths = sorted(Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_governor"))
    for path in paths:
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            value = "unreadable"
        governors[value] = governors.get(value, 0) + 1
    return {"paths_observed": len(paths), "governor_counts": governors}


def start_thermal_log(out_dir: Path, interval_ms: int) -> tuple[subprocess.Popen[str] | None, object | None, dict[str, object]]:
    path = out_dir / "thermal_tegrastats.log"
    if shutil.which("tegrastats") is None:
        path.write_text("tegrastats unavailable\n", encoding="utf-8")
        return None, None, {
            "available": False,
            "path": str(path.relative_to(ROOT)),
            "command": ["tegrastats", "--interval", str(interval_ms)],
        }
    fp = path.open("w", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(
        ["tegrastats", "--interval", str(interval_ms)],
        cwd=ROOT,
        text=True,
        stdout=fp,
        stderr=subprocess.STDOUT,
    )
    time.sleep(0.25)
    return proc, fp, {
        "available": True,
        "path": str(path.relative_to(ROOT)),
        "command": ["tegrastats", "--interval", str(interval_ms)],
        "started": proc.poll() is None,
    }


def stop_thermal_log(proc: subprocess.Popen[str] | None, fp: object | None,
                     status: dict[str, object]) -> dict[str, object]:
    if proc is not None:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
        status["returncode"] = proc.returncode
    if fp is not None:
        fp.close()
    path_text = status.get("path")
    if isinstance(path_text, str):
        path = ROOT / path_text
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            lines = []
        status["line_count"] = len(lines)
        status["nonempty"] = any(line.strip() for line in lines)
    return status


def process_snapshot() -> dict[str, object]:
    captured = command_capture(["ps", "-eo", "pid,ppid,comm,pcpu,pmem,args", "--sort=-pcpu"], timeout_s=5.0)
    stdout = str(captured.get("stdout", ""))
    captured["stdout"] = "\n".join(stdout.splitlines()[:80])
    captured["truncated_to_lines"] = 80
    return captured


def methodology_manifest(args: argparse.Namespace, thermal_status: dict[str, object]) -> dict[str, object]:
    governors = read_cpu_governors()
    governor_counts = governors.get("governor_counts", {})
    governor_ready = bool(governor_counts) and set(governor_counts) == {"performance"}
    return {
        "methodology_id": "aegisq-primitive-microbench-methodology-v1",
        "warmup": {
            "warmup_runs": args.warmup_runs,
            "artifacts": [
                f"warmup_integrity_{index:02d}.txt" for index in range(args.warmup_runs)
            ] + [
                f"warmup_workload_map_{index:02d}.csv" for index in range(args.warmup_runs)
            ] + [
                f"warmup_gpu_mlkem_{index:02d}.csv" for index in range(args.warmup_runs)
            ],
        },
        "run_count": {
            "measured_runs": args.runs,
            "headline_minimum_repetitions": 5,
            "meets_headline_minimum": args.runs >= 5,
        },
        "confidence_interval_method": {
            "name": "nonparametric bootstrap",
            "confidence_level": 0.95,
            "resamples": 10000,
            "unit": "independent measured repetitions per primitive row",
        },
        "outlier_policy": {
            "policy": "retain_all_completed_repetitions",
            "infrastructure_failure_policy": "fail the harness instead of substituting values",
            "winsorization": "disabled",
        },
        "cpu_gpu_clocks_or_power_mode": {
            "required_cpu_governor": "performance",
            "observed_cpu_governors": governors,
            "cpu_governor_ready": governor_ready,
            "nvpmodel_q": command_capture(["nvpmodel", "-q"], timeout_s=10.0),
            "jetson_clocks_show": command_capture(["jetson_clocks", "--show"], timeout_s=10.0),
        },
        "thermal_logging": thermal_status,
        "background_process_control": {
            "policy": "no unrelated foreground GPU/CPU/storage jobs during measurement",
            "process_snapshot": process_snapshot(),
        },
        "cache_dropping_policy": {
            "scope": "primitive CPU/GPU microbenchmarks only; no filesystem cache state is claimed",
            "warm_cache": "not_applicable",
            "cold_cache": "not_applicable",
        },
        "failure_handling": {
            "missing_binary": "fatal",
            "integrity_failure": "fatal",
            "command_failure": "fatal",
            "unsupported_configuration": "not emitted as zero or success",
        },
    }


def platform_manifest() -> dict[str, object]:
    model_path = Path("/proc/device-tree/model")
    model = model_path.read_bytes().rstrip(b"\0").decode(errors="replace") if model_path.exists() else "unknown"
    nvcc = subprocess.run(["nvcc", "--version"], text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False).stdout.strip()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "kernel": platform.platform(),
        "machine": platform.machine(),
        "device_model": model,
        "cpu_count": os.cpu_count(),
        "nvcc": nvcc,
        "commands": {
            "workload_map": [str(BUILD / "workload_map_bench")],
            "gpu_mlkem": [str(BUILD / "bench_gpu_pqc")],
            "gpu_integrity": [str(BUILD / "bench_gpu_integrity"), "--only-tests"],
        },
    }


def summarize_workload(raw_paths: list[Path]) -> dict[str, dict[str, float | int | str]]:
    values: dict[tuple[str, str, int, int, str], list[dict[str, float]]] = defaultdict(list)
    for path in raw_paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["status"] != "measured":
                    continue
                key = (row["operation"], row["target"], int(row["batch"]),
                       int(row["bytes"]), row["status"])
                values[key].append({
                    "p50_us": float(row["p50_us"]),
                    "p99_us": float(row["p99_us"]),
                    "throughput_per_s": float(row["throughput_per_s"]),
                })
    summary: dict[str, dict[str, float | int | str]] = {}
    for key, rows in values.items():
        operation, target, batch, byte_count, status = key
        throughput = [row["throughput_per_s"] for row in rows]
        ci_low, ci_high = bootstrap_ci(throughput, "|".join(map(str, key)))
        summary["|".join(map(str, key))] = {
            "operation": operation,
            "target": target,
            "batch": batch,
            "bytes": byte_count,
            "status": status,
            "runs": len(rows),
            "p50_us_median": statistics.median(row["p50_us"] for row in rows),
            "p99_us_median": statistics.median(row["p99_us"] for row in rows),
            "throughput_per_s_median": statistics.median(throughput),
            "throughput_per_s_p05": quantile(throughput, 0.05),
            "throughput_per_s_p95": quantile(throughput, 0.95),
            "throughput_per_s_ci95_low": ci_low,
            "throughput_per_s_ci95_high": ci_high,
        }
    return summary


def summarize_gpu_mlkem(raw_paths: list[Path]) -> dict[str, dict[str, float | int | str]]:
    values: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    for path in raw_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            match = GPU_ROW.match(line.strip())
            if match:
                operation, batch, ops_per_s, wall_ms = match.groups()
                values[(operation, int(batch))].append((float(ops_per_s), float(wall_ms)))
    summary: dict[str, dict[str, float | int | str]] = {}
    for (operation, batch), rows in values.items():
        throughputs = [row[0] for row in rows]
        wall_times = [row[1] for row in rows]
        ci_low, ci_high = bootstrap_ci(throughputs, f"{operation}|gpu|{batch}")
        summary[f"{operation}|gpu|{batch}"] = {
            "operation": operation,
            "target": "gpu",
            "batch": batch,
            "runs": len(rows),
            "throughput_per_s_median": statistics.median(throughputs),
            "throughput_per_s_p05": quantile(throughputs, 0.05),
            "throughput_per_s_p95": quantile(throughputs, 0.95),
            "throughput_per_s_ci95_low": ci_low,
            "throughput_per_s_ci95_high": ci_high,
            "wall_ms_median": statistics.median(wall_times),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--thermal-interval-ms", type=int, default=100)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.runs < 2:
        raise SystemExit("--runs must be at least 2")
    if args.warmup_runs < 0:
        raise SystemExit("--warmup-runs must be non-negative")
    args.out = args.out if args.out.is_absolute() else ROOT / args.out
    for binary in ("workload_map_bench", "bench_gpu_pqc", "bench_gpu_integrity"):
        if not (BUILD / binary).is_file():
            raise SystemExit(f"missing {BUILD / binary}; build the project first")

    args.out.mkdir(parents=True, exist_ok=True)
    workload_paths: list[Path] = []
    gpu_paths: list[Path] = []
    thermal_proc, thermal_fp, thermal_status = start_thermal_log(args.out, args.thermal_interval_ms)
    try:
        for run in range(args.warmup_runs):
            integrity = command_output([str(BUILD / "bench_gpu_integrity"), "--only-tests"])
            if "[SUCCESS] All correctness tests passed." not in integrity:
                raise RuntimeError(f"GPU integrity warmup failed in run {run}")
            (args.out / f"warmup_integrity_{run:02d}.txt").write_text(integrity, encoding="utf-8")
            (args.out / f"warmup_workload_map_{run:02d}.csv").write_text(
                command_output([str(BUILD / "workload_map_bench")]),
                encoding="utf-8",
            )
            (args.out / f"warmup_gpu_mlkem_{run:02d}.csv").write_text(
                command_output([str(BUILD / "bench_gpu_pqc")]),
                encoding="utf-8",
            )
        for run in range(args.runs):
            integrity = command_output([str(BUILD / "bench_gpu_integrity"), "--only-tests"])
            if "[SUCCESS] All correctness tests passed." not in integrity:
                raise RuntimeError(f"GPU integrity correctness failed in run {run}")
            integrity_path = args.out / f"integrity_{run:02d}.txt"
            integrity_path.write_text(integrity, encoding="utf-8")

            workload_path = args.out / f"workload_map_{run:02d}.csv"
            workload_path.write_text(command_output([str(BUILD / "workload_map_bench")]), encoding="utf-8")
            workload_paths.append(workload_path)

            gpu_path = args.out / f"gpu_mlkem_{run:02d}.csv"
            gpu_path.write_text(command_output([str(BUILD / "bench_gpu_pqc")]), encoding="utf-8")
            gpu_paths.append(gpu_path)
    finally:
        thermal_status = stop_thermal_log(thermal_proc, thermal_fp, thermal_status)

    result = {
        "manifest": platform_manifest(),
        "methodology": methodology_manifest(args, thermal_status),
        "runs": args.runs,
        "warmup_runs": args.warmup_runs,
        "workload_map": summarize_workload(workload_paths),
        "gpu_mlkem": summarize_gpu_mlkem(gpu_paths),
    }
    (args.out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "runs": args.runs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
