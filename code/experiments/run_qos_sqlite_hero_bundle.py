#!/usr/bin/env python3
"""Build a synchronized mounted-FUSE SQLite QoS recovery bundle.

The checklist items closed by this harness are the experiment construction and
the SQLite foreground recovery table.  Each mode runs the final
``build/pqc_fuse`` binary, records foreground SQLite transaction latency,
optionally runs a background secure-storage writer through the same mount,
emits a workload-pressure telemetry stream, and retains the corresponding
policy traces.

Modes:
  * app_only: foreground SQLite transactions on mounted AEGIS-Q.
  * unthrottled_storage: SQLite plus background secure writes, no controller.
  * simple_controller: SQLite plus background writes throttled by the harness.
  * aegis_policy: SQLite plus background writes; telemetry drives the mounted
    daemon's in-FUSE throttle path through ``PQC_TELEMETRY_FILE``.

The telemetry source is deliberately recorded as workload pressure derived from
foreground slack plus background-storage activity.  This is a SQLite recovery
result; it is not a hardware-PMU-backed TensorRT or foreground-AI result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import platform
import random
import shutil
import signal
import sqlite3
import statistics
import subprocess
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FUSE_BIN = ROOT / "build" / "pqc_fuse"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "qos_sqlite_hero_bundle"

REQUIRED_MODES = [
    "app_only",
    "unthrottled_storage",
    "simple_controller",
    "aegis_policy",
]


@dataclass(frozen=True)
class ModeConfig:
    name: str
    background_writer: bool
    daemon_throttle: bool
    harness_throttle: bool
    controller_label: str


MODE_CONFIGS = {
    "app_only": ModeConfig(
        name="app_only",
        background_writer=False,
        daemon_throttle=False,
        harness_throttle=False,
        controller_label="none",
    ),
    "unthrottled_storage": ModeConfig(
        name="unthrottled_storage",
        background_writer=True,
        daemon_throttle=False,
        harness_throttle=False,
        controller_label="none",
    ),
    "simple_controller": ModeConfig(
        name="simple_controller",
        background_writer=True,
        daemon_throttle=False,
        harness_throttle=True,
        controller_label="harness_hysteresis",
    ),
    "aegis_policy": ModeConfig(
        name="aegis_policy",
        background_writer=True,
        daemon_throttle=True,
        harness_throttle=False,
        controller_label="mounted_fuse_hysteresis",
    ),
}


def realtime_ns() -> int:
    return time.time_ns()


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def command_capture(command: list[str], timeout_s: float = 10.0) -> dict[str, Any]:
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


def read_cpu_governors() -> dict[str, Any]:
    governors: dict[str, int] = {}
    paths = sorted(Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_governor"))
    for path in paths:
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            value = "unreadable"
        governors[value] = governors.get(value, 0) + 1
    return {"paths_observed": len(paths), "governor_counts": governors}


def start_thermal_log(out_dir: Path, interval_ms: int) -> tuple[subprocess.Popen[str] | None, Any, dict[str, Any]]:
    path = out_dir / "thermal_tegrastats.log"
    if shutil.which("tegrastats") is None:
        path.write_text("tegrastats unavailable\n", encoding="utf-8")
        return None, None, {
            "available": False,
            "path": relpath(path),
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
        "path": relpath(path),
        "command": ["tegrastats", "--interval", str(interval_ms)],
        "started": proc.poll() is None,
    }


def stop_thermal_log(proc: subprocess.Popen[str] | None, fp: Any,
                     status: dict[str, Any]) -> dict[str, Any]:
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


def process_snapshot() -> dict[str, Any]:
    captured = command_capture(["ps", "-eo", "pid,ppid,comm,pcpu,pmem,args", "--sort=-pcpu"], timeout_s=5.0)
    stdout = str(captured.get("stdout", ""))
    captured["stdout"] = "\n".join(stdout.splitlines()[:80])
    captured["truncated_to_lines"] = 80
    return captured


def platform_manifest() -> dict[str, Any]:
    model_path = Path("/proc/device-tree/model")
    model = model_path.read_bytes().rstrip(b"\0").decode(errors="replace") if model_path.exists() else "unknown"
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "system": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "device_model": model,
        "cpu_count": os.cpu_count(),
        "kernel": command_output(["uname", "-a"]),
        "git_head": command_output(["git", "rev-parse", "HEAD"]),
        "git_dirty_short": command_output(["git", "status", "--short"]),
        "fuse_binary": {
            "path": relpath(FUSE_BIN),
            "exists": FUSE_BIN.exists(),
            "sha256": sha256_bytes(FUSE_BIN.read_bytes()) if FUSE_BIN.exists() else None,
        },
        "nvcc": command_capture(["nvcc", "--version"], timeout_s=10.0),
    }


def methodology_manifest(args: argparse.Namespace, thermal_status: dict[str, Any]) -> dict[str, Any]:
    governors = read_cpu_governors()
    governor_counts = governors.get("governor_counts", {})
    governor_ready = bool(governor_counts) and set(governor_counts) == {"performance"}
    return {
        "methodology_id": "aegisq-sqlite-qos-methodology-v1",
        "warmup": {
            "warmup_runs": args.warmup_runs,
            "full_workload_warmup": args.warmup_runs > 0,
            "artifacts": [
                relpath(args.out_dir / f"warmup_{index:02d}") for index in range(args.warmup_runs)
            ],
        },
        "run_count": {
            "measured_repetitions": args.repetitions,
            "headline_minimum_repetitions": 5,
            "meets_headline_minimum": args.repetitions >= 5,
        },
        "confidence_interval_method": {
            "name": "nonparametric bootstrap",
            "confidence_level": 0.95,
            "resamples": 10000,
            "unit": "independent mounted SQLite QoS workflow repetitions per mode",
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
        "cache_state_policy": {
            "scope": "mounted SQLite QoS workflow, not frozen filesystem baseline",
            "warm_cache": "full workload warmup is retained when --warmup-runs is nonzero",
            "cold_cache": "not claimed; each mode run uses a fresh temporary lower directory and mount",
        },
        "failure_handling": {
            "missing_binary": "fatal",
            "mount_failure": "fatal",
            "mode_failure": "fatal",
            "component_coverage_failure": "fatal",
            "unsupported_configuration": "not emitted as zero or success",
        },
    }


def json_dump_line(fp: Any, row: dict[str, Any]) -> None:
    fp.write(json.dumps(row, sort_keys=True) + "\n")
    fp.flush()


def write_runtime_telemetry(path: Path, mem_util: float, tensor_util: float,
                            budget_ns: int, queue_depth: int) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        f"{mem_util:.6f} {tensor_util:.6f} {int(budget_ns)} {int(queue_depth)}\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def parse_pressure_pattern(pattern: str) -> list[float]:
    values: list[float] = []
    for token in pattern.split(","):
        token = token.strip()
        if not token:
            continue
        count = 1
        if ":" in token:
            value_text, count_text = token.split(":", 1)
            count = max(1, int(count_text))
        else:
            value_text = token
        value = max(0.0, min(1.0, float(value_text)))
        values.extend([value] * count)
    return values


def background_pressure_for_sample(args: argparse.Namespace, sample_index: int,
                                   active: bool) -> float:
    if not active:
        return 0.10
    pattern = getattr(args, "_background_pressure_pattern_values", [])
    if pattern:
        return pattern[(sample_index - 1) % len(pattern)]
    return args.background_pressure_util


def set_qos_class(path: Path, qos_class: str) -> None:
    os.setxattr(path, b"user.pqc_qos_class", qos_class.encode("ascii"))


def read_cpu_snapshot() -> tuple[int, int] | None:
    try:
        parts = Path("/proc/stat").read_text(encoding="ascii").splitlines()[0].split()
    except (OSError, IndexError):
        return None
    if not parts or parts[0] != "cpu":
        return None
    values = [int(part) for part in parts[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    total = sum(values)
    return idle, total


def cpu_utilization(prev: tuple[int, int] | None,
                    curr: tuple[int, int] | None) -> float | None:
    if prev is None or curr is None:
        return None
    idle_delta = curr[0] - prev[0]
    total_delta = curr[1] - prev[1]
    if total_delta <= 0:
        return None
    busy_delta = total_delta - idle_delta
    return max(0.0, min(1.0, busy_delta / total_delta))


def read_gpu_utilization() -> tuple[float | None, str]:
    candidates = [
        Path("/sys/devices/gpu.0/load"),
        Path("/sys/devices/platform/gpu.0/load"),
    ]
    for path in candidates:
        try:
            raw = path.read_text(encoding="ascii").strip()
        except OSError:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if value > 100.0:
            value = value / 1000.0
        else:
            value = value / 100.0
        return max(0.0, min(1.0, value)), str(path)
    return None, "unavailable"


def fusermount_command() -> str:
    for name in ("fusermount3", "fusermount"):
        if shutil.which(name):
            return name
    return "fusermount3"


@dataclass
class FuseHandle:
    proc: subprocess.Popen[bytes]
    stdout: Any
    stderr: Any


def start_fuse(storage_dir: Path, mount_dir: Path, mode_dir: Path,
               telemetry_file: Path, admission_trace: Path,
               throttle_trace: Path, config: ModeConfig,
               args: argparse.Namespace) -> FuseHandle:
    log_dir = mode_dir / "mount_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout = (log_dir / "pqc_fuse.stdout.txt").open("wb")
    stderr = (log_dir / "pqc_fuse.stderr.txt").open("wb")

    env = os.environ.copy()
    env.update(
        {
            "PQC_MASTER_PASSWORD": "sqlite-hero-password",
            "PQC_FRESHNESS_ANCHOR_BACKEND": "file",
            "PQC_FRESHNESS_ANCHOR_PATH": str(mode_dir / "anchor.bin"),
            "PQC_ALLOW_SQLITE_MMAP": "1",
            "PQC_KEY_ROTATION_INTERVAL_S": "0",
            "PQC_ENABLE_ADMISSION_ON_WRITE": "1",
            "PQC_ADMISSION_TRACE_PATH": str(admission_trace),
            "PQC_ADMISSION_INITIAL_BUDGET_NS": str(int(args.deadline_ms * 1_000_000)),
            "PQC_ADMISSION_WRITE_DEADLINE_NS": str(int(args.deadline_ms * 1_000_000)),
            "PQC_TELEMETRY_FILE": str(telemetry_file),
            "PQC_TELEMETRY_POLL_MS": str(args.telemetry_poll_ms),
            "PQC_ENABLE_QOS_THROTTLE_ON_WRITE": "1" if config.daemon_throttle else "0",
            "PQC_QOS_THROTTLE_TRACE_PATH": str(throttle_trace),
            "PQC_QOS_THROTTLE_SLEEP_US": str(args.daemon_throttle_sleep_us),
            "PQC_QOS_MEM_ENTER_UTIL": str(args.enter_util),
            "PQC_QOS_MEM_EXIT_UTIL": str(args.exit_util),
            "PQC_QOS_HOLD_SAMPLES": str(args.hold_samples),
        }
    )
    env.pop("PQC_FORCE_REKEY_ON_WRITE", None)
    proc = subprocess.Popen(
        [str(FUSE_BIN), str(storage_dir), str(mount_dir), "-f"],
        cwd=ROOT,
        env=env,
        stdout=stdout,
        stderr=stderr,
    )
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if subprocess.run(["mountpoint", "-q", str(mount_dir)], check=False).returncode == 0:
            return FuseHandle(proc=proc, stdout=stdout, stderr=stderr)
        if proc.poll() is not None:
            stdout.close()
            stderr.close()
            raise RuntimeError(f"FUSE exited before mount for {config.name}: rc={proc.returncode}")
        time.sleep(0.05)
    stdout.close()
    stderr.close()
    raise TimeoutError(f"timed out waiting for FUSE mount for {config.name}")


def stop_fuse(handle: FuseHandle | None, mount_dir: Path) -> None:
    subprocess.run(
        [fusermount_command(), "-u", str(mount_dir)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if handle is None:
        return
    if handle.proc.poll() is None:
        handle.proc.send_signal(signal.SIGINT)
        try:
            handle.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            handle.proc.kill()
            handle.proc.wait(timeout=5)
    handle.stdout.close()
    handle.stderr.close()


def apply_background_ionice(ionice_class: str | None,
                            ionice_level: int | None) -> dict[str, Any]:
    if not ionice_class:
        return {"enabled": False, "reason": "not_configured"}
    if shutil.which("ionice") is None:
        return {"enabled": False, "reason": "ionice_missing"}
    argv = ["ionice", "-c", str(ionice_class)]
    if ionice_level is not None and str(ionice_class) != "3":
        argv.extend(["-n", str(ionice_level)])
    argv.extend(["-p", str(os.getpid())])
    proc = subprocess.run(
        argv,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "enabled": True,
        "argv": argv,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def background_writer_worker(mount_dir: str, writer_id: int, stop_flag: mp.Event,
                             throttle_flag: mp.Value, log_path: str,
                             chunk_bytes: int, fsync_every: int,
                             harness_sleep_us: int,
                             ionice_class: str | None = None,
                             ionice_level: int | None = None) -> None:
    path = Path(mount_dir) / f"background_{writer_id}.bin"
    path.touch(exist_ok=True)
    set_qos_class(path, "elastic")
    payload = bytes([65 + (writer_id % 26)]) * chunk_bytes
    chunks = 0
    bytes_written = 0
    sleep_events = 0
    sleep_us_total = 0
    start_ns = realtime_ns()
    ionice_result = apply_background_ionice(ionice_class, ionice_level)
    with open(log_path, "a", encoding="utf-8", buffering=1) as log, path.open("wb", buffering=0) as fp:
        json_dump_line(log, {
            "event": "writer_start",
            "timestamp_ns": start_ns,
            "writer_id": writer_id,
            "chunk_bytes": chunk_bytes,
            "fsync_every": fsync_every,
            "kernel_qos": {
                "control": "ionice",
                "class": ionice_class,
                "level": ionice_level,
                "apply_result": ionice_result,
            },
        })
        while not stop_flag.is_set():
            if int(throttle_flag.value) != 0:
                sleep_events += 1
                sleep_us_total += harness_sleep_us
                json_dump_line(log, {
                    "event": "harness_throttle_sleep",
                    "timestamp_ns": realtime_ns(),
                    "writer_id": writer_id,
                    "sleep_us": harness_sleep_us,
                    "sleep_events": sleep_events,
                })
                time.sleep(harness_sleep_us / 1_000_000.0)
                continue
            t0 = realtime_ns()
            fp.write(payload)
            chunks += 1
            bytes_written += chunk_bytes
            fsync_done = False
            if fsync_every > 0 and chunks % fsync_every == 0:
                fp.flush()
                os.fsync(fp.fileno())
                fsync_done = True
            json_dump_line(log, {
                "event": "write",
                "timestamp_ns": t0,
                "writer_id": writer_id,
                "chunk_index": chunks,
                "bytes": chunk_bytes,
                "bytes_written": bytes_written,
                "fsync": fsync_done,
            })
        fp.flush()
        os.fsync(fp.fileno())
        elapsed_ns = max(1, realtime_ns() - start_ns)
        json_dump_line(log, {
            "event": "writer_stop",
            "timestamp_ns": realtime_ns(),
            "writer_id": writer_id,
            "chunks_written": chunks,
            "bytes_written": bytes_written,
            "harness_throttle_sleeps": sleep_events,
            "harness_throttle_sleep_us_total": sleep_us_total,
            "elapsed_ns": elapsed_ns,
            "throughput_mb_s": bytes_written / (1024.0 * 1024.0) / (elapsed_ns / 1_000_000_000.0),
        })


class Hysteresis:
    def __init__(self, enter: float, exit_: float, hold_samples: int):
        self.enter = enter
        self.exit = exit_
        self.hold_samples = max(1, hold_samples)
        self.state = "open"
        self.below_exit_count = 0

    def update(self, pressure: float) -> dict[str, Any]:
        event = "hold"
        if self.state == "open":
            self.below_exit_count = 0
            if pressure >= self.enter:
                self.state = "throttled"
                event = "enter"
        else:
            if pressure <= self.exit:
                self.below_exit_count += 1
                if self.below_exit_count >= self.hold_samples:
                    self.state = "open"
                    self.below_exit_count = 0
                    event = "exit"
            else:
                self.below_exit_count = 0
        return {
            "state": self.state,
            "event": event,
            "throttle": 1 if self.state == "throttled" else 0,
            "below_exit_count": self.below_exit_count,
        }


def telemetry_sampler(config: ModeConfig, args: argparse.Namespace,
                      telemetry_file: Path, telemetry_log: Path,
                      policy_log: Path, latencies: deque[float],
                      latency_lock: threading.Lock, stop_event: threading.Event,
                      throttle_flag: mp.Value) -> None:
    interval_s = args.telemetry_interval_ms / 1000.0
    deadline_ms = float(args.deadline_ms)
    controller = Hysteresis(args.enter_util, args.exit_util, args.hold_samples)
    sample_index = 0
    prev_cpu = read_cpu_snapshot()
    with telemetry_log.open("w", encoding="utf-8", buffering=1) as telemetry_fp, \
            policy_log.open("w", encoding="utf-8", buffering=1) as policy_fp:
        while not stop_event.is_set():
            with latency_lock:
                recent = list(latencies)[-args.telemetry_window:]
                total_latency_samples = len(latencies)
            p50 = percentile(recent, 50.0)
            p95 = percentile(recent, 95.0)
            p99 = percentile(recent, 99.0)
            last_ms = recent[-1] if recent else None
            misses = sum(1 for value in recent if value > deadline_ms)
            if p95 is None:
                slack_ms = deadline_ms
                latency_pressure = 0.10
            else:
                slack_ms = deadline_ms - p95
                latency_pressure = 0.10 if p95 <= deadline_ms else min(1.0, 0.70 + (p95 - deadline_ms) / max(deadline_ms, 0.001))
            next_sample_index = sample_index + 1
            background_pressure = background_pressure_for_sample(
                args, next_sample_index, config.background_writer
            )
            pressure = max(background_pressure, latency_pressure)
            controller_pressure = pressure
            if total_latency_samples < args.controller_warmup_transactions:
                controller_pressure = min(controller_pressure, args.exit_util)
            tensor_util = 0.10
            budget_ns = max(0, int(slack_ms * 1_000_000))
            queue_depth = misses
            decision = controller.update(controller_pressure)
            curr_cpu = read_cpu_snapshot()
            cpu_util = cpu_utilization(prev_cpu, curr_cpu)
            if curr_cpu is not None:
                prev_cpu = curr_cpu
            gpu_util, gpu_util_source = read_gpu_utilization()
            if gpu_util is None:
                gpu_util = tensor_util
                gpu_util_source = "telemetry_tensor_proxy"
            if config.harness_throttle:
                throttle_flag.value = decision["throttle"]
            else:
                throttle_flag.value = 0
            write_runtime_telemetry(telemetry_file, controller_pressure, tensor_util, budget_ns, queue_depth)
            sample_index = next_sample_index
            base = {
                "timestamp_ns": realtime_ns(),
                "sample_index": sample_index,
                "mode": config.name,
                "source": "foreground_sqlite_slack_and_background_storage_pressure",
                "deadline_ms": deadline_ms,
                "total_latency_samples": total_latency_samples,
                "window_samples": len(recent),
                "last_latency_ms": last_ms,
                "p50_window_ms": p50,
                "p95_window_ms": p95,
                "p99_window_ms": p99,
                "deadline_misses_window": misses,
                "slack_ms": slack_ms,
                "raw_pressure_util": pressure,
                "mem_bandwidth_util": controller_pressure,
                "tensor_core_util": tensor_util,
                "ai_qos_budget_remaining_ns": budget_ns,
                "ai_queue_depth": queue_depth,
                "background_writer_active": config.background_writer,
                "cpu_utilization": cpu_util,
                "cpu_utilization_source": "/proc/stat",
                "gpu_utilization": gpu_util,
                "gpu_utilization_source": gpu_util_source,
            }
            json_dump_line(telemetry_fp, base)
            policy_row = {
                **base,
                "controller": config.controller_label,
                "daemon_throttle_enabled": config.daemon_throttle,
                "harness_throttle_enabled": config.harness_throttle,
                "policy_state": decision["state"] if config.controller_label != "none" else "disabled",
                "policy_event": decision["event"] if config.controller_label != "none" else "none",
                "policy_throttle": decision["throttle"] if config.controller_label != "none" else 0,
                "below_exit_count": decision["below_exit_count"] if config.controller_label != "none" else 0,
                "writer_throttle_flag": int(throttle_flag.value),
                "controller_warmup_transactions": args.controller_warmup_transactions,
                "enter_util": args.enter_util,
                "exit_util": args.exit_util,
                "hold_samples": args.hold_samples,
            }
            json_dump_line(policy_fp, policy_row)
            stop_event.wait(interval_s)


def run_sqlite_foreground(mount_dir: Path, out_path: Path,
                          latencies: deque[float], latency_lock: threading.Lock,
                          args: argparse.Namespace) -> dict[str, Any]:
    db_path = mount_dir / "foreground.db"
    db_path.touch(exist_ok=True)
    set_qos_class(db_path, "latency")
    conn = sqlite3.connect(str(db_path), timeout=5.0, check_same_thread=False)
    rows_written = 0
    samples: list[float] = []
    try:
        conn.execute("PRAGMA mmap_size=0")
        conn.execute("PRAGMA locking_mode=EXCLUSIVE")
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, payload TEXT)")
        conn.commit()
        with out_path.open("w", encoding="utf-8", buffering=1) as fp:
            for txn in range(args.transactions):
                payload = "x" * args.sqlite_payload_bytes
                batch = [
                    (txn * args.rows_per_txn + row, f"{txn:06d}:{row:03d}:{payload}")
                    for row in range(args.rows_per_txn)
                ]
                start_ns = realtime_ns()
                start_perf = time.perf_counter_ns()
                conn.executemany("INSERT OR REPLACE INTO events(id, payload) VALUES (?, ?)", batch)
                conn.commit()
                end_perf = time.perf_counter_ns()
                latency_ms = (end_perf - start_perf) / 1_000_000.0
                rows_written += len(batch)
                samples.append(latency_ms)
                with latency_lock:
                    latencies.append(latency_ms)
                json_dump_line(fp, {
                    "event": "sqlite_transaction",
                    "timestamp_ns": start_ns,
                    "transaction": txn,
                    "rows": len(batch),
                    "rows_written_total": rows_written,
                    "latency_ms": latency_ms,
                    "deadline_ms": args.deadline_ms,
                    "deadline_miss": latency_ms > args.deadline_ms,
                })
                if args.inter_transaction_sleep_ms > 0:
                    time.sleep(args.inter_transaction_sleep_ms / 1000.0)
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        row_count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    finally:
        conn.close()
    return {
        "samples": len(samples),
        "rows_written": rows_written,
        "row_count": row_count,
        "integrity_check": integrity,
        "p50_ms": percentile(samples, 50.0),
        "p95_ms": percentile(samples, 95.0),
        "p99_ms": percentile(samples, 99.0),
        "mean_ms": sum(samples) / len(samples) if samples else None,
        "max_ms": max(samples) if samples else None,
        "deadline_ms": args.deadline_ms,
        "deadline_misses": sum(1 for value in samples if value > args.deadline_ms),
        "latency_log": relpath(out_path),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "missing"))
        counts[value] = counts.get(value, 0) + 1
    return counts


def summarize_background(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stop_rows = [row for row in rows if row.get("event") == "writer_stop"]
    write_rows = [row for row in rows if row.get("event") == "write"]
    sleep_rows = [row for row in rows if row.get("event") == "harness_throttle_sleep"]
    bytes_written = sum(int(row.get("bytes", 0)) for row in write_rows)
    elapsed_ns = max((int(row.get("elapsed_ns", 0)) for row in stop_rows), default=0)
    throughput_mb_s = 0.0
    if bytes_written > 0 and elapsed_ns > 0:
        throughput_mb_s = bytes_written / (1024.0 * 1024.0) / (elapsed_ns / 1_000_000_000.0)
    return {
        "writer_started": bool(stop_rows or write_rows),
        "write_events": len(write_rows),
        "bytes_written": bytes_written,
        "throughput_mb_s": throughput_mb_s,
        "harness_throttle_sleeps": len(sleep_rows),
        "harness_throttle_sleep_us_total": sum(int(row.get("sleep_us", 0)) for row in sleep_rows),
        "writer_stop_rows": stop_rows,
    }


def average_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return sum(values) / len(values) if values else None


def run_mode(config: ModeConfig, out_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    mode_dir = out_root / config.name
    if mode_dir.exists():
        shutil.rmtree(mode_dir)
    mode_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f"aegis_qos_{config.name}_"))
    storage_dir = tmp / "store"
    mount_dir = tmp / "mnt"
    storage_dir.mkdir()
    mount_dir.mkdir()

    telemetry_file = mode_dir / "runtime_telemetry.txt"
    telemetry_log = mode_dir / "telemetry_trace.jsonl"
    policy_log = mode_dir / "policy_trace.jsonl"
    sqlite_log = mode_dir / "foreground_sqlite_latency.jsonl"
    sqlite_csv = mode_dir / "foreground_sqlite_latency.csv"
    writer_log = mode_dir / "background_writer.jsonl"
    admission_trace = mode_dir / "runtime_fuse_admission_trace.jsonl"
    throttle_trace = mode_dir / "runtime_fuse_throttle_trace.jsonl"
    write_runtime_telemetry(telemetry_file, 0.10, 0.10, int(args.deadline_ms * 1_000_000), 0)

    ctx = mp.get_context("spawn")
    writer_stop = ctx.Event()
    writer_throttle = ctx.Value("i", 0)
    writer_procs: list[mp.Process] = []
    sampler_stop = threading.Event()
    latencies: deque[float] = deque(maxlen=max(args.telemetry_window * 4, args.transactions * 2))
    latency_lock = threading.Lock()
    sampler_thread: threading.Thread | None = None
    fuse: FuseHandle | None = None
    error: str | None = None

    try:
        fuse = start_fuse(storage_dir, mount_dir, mode_dir, telemetry_file,
                          admission_trace, throttle_trace, config, args)
        if config.background_writer:
            for writer_id in range(args.background_writers):
                proc = ctx.Process(
                    target=background_writer_worker,
                    args=(
                        str(mount_dir),
                        writer_id,
                        writer_stop,
                        writer_throttle,
                        str(writer_log),
                        args.background_chunk_bytes,
                        args.background_fsync_every,
                        args.harness_throttle_sleep_us,
                        getattr(args, "background_ionice_class", None),
                        getattr(args, "background_ionice_level", None),
                    ),
                    daemon=True,
                )
                proc.start()
                writer_procs.append(proc)
            if args.background_warmup_ms > 0:
                time.sleep(args.background_warmup_ms / 1000.0)
        else:
            writer_log.write_text("", encoding="utf-8")

        sampler_thread = threading.Thread(
            target=telemetry_sampler,
            args=(
                config,
                args,
                telemetry_file,
                telemetry_log,
                policy_log,
                latencies,
                latency_lock,
                sampler_stop,
                writer_throttle,
            ),
            daemon=True,
        )
        sampler_thread.start()
        foreground = run_sqlite_foreground(mount_dir, sqlite_log, latencies, latency_lock, args)
        time.sleep(args.post_foreground_drain_ms / 1000.0)
    except Exception as exc:
        error = str(exc)
        foreground = {
            "samples": 0,
            "rows_written": 0,
            "row_count": 0,
            "integrity_check": "error",
            "deadline_misses": 0,
            "latency_log": relpath(sqlite_log),
            "error": error,
        }
    finally:
        sampler_stop.set()
        if sampler_thread is not None:
            sampler_thread.join(timeout=5)
        writer_stop.set()
        for proc in writer_procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)
        stop_fuse(fuse, mount_dir)
        shutil.rmtree(tmp, ignore_errors=True)

    telemetry_rows = load_jsonl(telemetry_log)
    policy_rows = load_jsonl(policy_log)
    writer_rows = load_jsonl(writer_log)
    throttle_rows = load_jsonl(throttle_trace)
    admission_rows = load_jsonl(admission_trace)
    background = summarize_background(writer_rows)
    throttle_summary = {
        "rows": len(throttle_rows),
        "throttled_rows": sum(1 for row in throttle_rows if int(row.get("throttled", 0)) != 0),
        "open_rows": sum(1 for row in throttle_rows if int(row.get("throttled", 0)) == 0),
        "sleep_us_total": sum(int(row.get("sleep_us", 0)) for row in throttle_rows),
    }
    policy_summary = {
        "rows": len(policy_rows),
        "states": count_by(policy_rows, "policy_state"),
        "events": count_by(policy_rows, "policy_event"),
        "throttle_rows": sum(1 for row in policy_rows if int(row.get("policy_throttle", 0)) != 0),
    }
    telemetry_summary = {
        "rows": len(telemetry_rows),
        "high_pressure_rows": sum(1 for row in telemetry_rows if float(row.get("mem_bandwidth_util", 0.0)) >= args.enter_util),
        "zero_budget_rows": sum(1 for row in telemetry_rows if int(row.get("ai_qos_budget_remaining_ns", 0)) == 0),
        "avg_cpu_utilization": average_field(telemetry_rows, "cpu_utilization"),
        "avg_gpu_utilization": average_field(telemetry_rows, "gpu_utilization"),
        "gpu_utilization_sources": sorted({
            str(row.get("gpu_utilization_source", "missing")) for row in telemetry_rows
        }),
    }
    admission_summary = {
        "rows": len(admission_rows),
        "targets": count_by(admission_rows, "chosen_target"),
        "deferrals": count_by(admission_rows, "deferral_reason"),
    }

    if sqlite_log.exists():
        rows = load_jsonl(sqlite_log)
        with sqlite_csv.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=[
                    "timestamp_ns",
                    "transaction",
                    "rows",
                    "latency_ms",
                    "deadline_ms",
                    "deadline_miss",
                ],
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    "timestamp_ns": row.get("timestamp_ns"),
                    "transaction": row.get("transaction"),
                    "rows": row.get("rows"),
                    "latency_ms": row.get("latency_ms"),
                    "deadline_ms": row.get("deadline_ms"),
                    "deadline_miss": row.get("deadline_miss"),
                })

    acceptable = (
        error is None
        and foreground.get("samples") == args.transactions
        and foreground.get("integrity_check") == "ok"
        and telemetry_summary["rows"] > 0
        and policy_summary["rows"] > 0
    )
    if config.background_writer:
        acceptable = acceptable and background["bytes_written"] > 0
    else:
        acceptable = acceptable and background["bytes_written"] == 0
    if config.harness_throttle:
        acceptable = acceptable and background["harness_throttle_sleeps"] > 0
    if config.daemon_throttle:
        acceptable = acceptable and throttle_summary["rows"] > 0
        if getattr(args, "require_daemon_throttle", True):
            acceptable = acceptable and throttle_summary["throttled_rows"] > 0

    mode_summary = {
        "mode": config.name,
        "acceptable": acceptable,
        "error": error,
        "config": {
            "background_writer": config.background_writer,
            "daemon_throttle": config.daemon_throttle,
            "harness_throttle": config.harness_throttle,
            "controller_label": config.controller_label,
        },
        "foreground": foreground,
        "background": background,
        "telemetry": telemetry_summary,
        "policy": policy_summary,
        "daemon_throttle": throttle_summary,
        "admission": admission_summary,
        "logs": {
            "foreground_jsonl": relpath(sqlite_log),
            "foreground_csv": relpath(sqlite_csv),
            "background_jsonl": relpath(writer_log),
            "telemetry_jsonl": relpath(telemetry_log),
            "policy_jsonl": relpath(policy_log),
            "runtime_telemetry": relpath(telemetry_file),
            "runtime_fuse_admission_trace": relpath(admission_trace),
            "runtime_fuse_throttle_trace": relpath(throttle_trace),
            "fuse_stdout": relpath(mode_dir / "mount_logs" / "pqc_fuse.stdout.txt"),
            "fuse_stderr": relpath(mode_dir / "mount_logs" / "pqc_fuse.stderr.txt"),
        },
    }
    (mode_dir / "mode_summary.json").write_text(
        json.dumps(mode_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return mode_summary


def command_output(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def run_bundle(out_dir: Path, args: argparse.Namespace, write_tables: bool,
               write_paper_table: bool) -> dict[str, Any]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    mode_summaries = [run_mode(MODE_CONFIGS[name], out_dir, args) for name in REQUIRED_MODES]
    mode_names = {mode["mode"] for mode in mode_summaries}
    component_coverage = {
        "required_modes_present": all(name in mode_names for name in REQUIRED_MODES),
        "foreground_sqlite_logs": all(mode["foreground"].get("samples", 0) > 0 for mode in mode_summaries),
        "mounted_fuse_logs": all(Path(ROOT / mode["logs"]["fuse_stderr"]).exists() for mode in mode_summaries),
        "background_secure_writer_logs": all(
            (not mode["config"]["background_writer"]) or mode["background"]["bytes_written"] > 0
            for mode in mode_summaries
        ),
        "telemetry_sampler_logs": all(mode["telemetry"]["rows"] > 0 for mode in mode_summaries),
        "policy_trace_logs": all(mode["policy"]["rows"] > 0 for mode in mode_summaries),
        "aegis_daemon_throttle_trace": any(
            mode["mode"] == "aegis_policy" and mode["daemon_throttle"]["throttled_rows"] > 0
            for mode in mode_summaries
        ),
        "simple_controller_trace": any(
            mode["mode"] == "simple_controller" and mode["background"]["harness_throttle_sleeps"] > 0
            for mode in mode_summaries
        ),
    }
    recovery_checks = compute_recovery_checks(mode_summaries)
    overall_pass = (
        all(mode["acceptable"] for mode in mode_summaries)
        and all(component_coverage.values())
        and all(recovery_checks.values())
    )
    report = {
        "artifact": "qos_sqlite_hero_bundle",
        "overall_pass": overall_pass,
        "scope": [
            "Reports synchronized four-mode mounted-FUSE SQLite foreground QoS recovery.",
            "Does not claim AI inference or TensorRT p99 restoration.",
            "Uses workload-pressure telemetry derived from foreground SQLite slack plus background storage activity.",
        ],
        "command": ["python3", "code/experiments/run_qos_sqlite_hero_bundle.py"],
        "args": serializable_args,
        "platform": platform_manifest(),
        "component_coverage": component_coverage,
        "recovery_checks": recovery_checks,
        "modes": mode_summaries,
    }
    json_path = out_dir / "qos_sqlite_hero_bundle.json"
    md_path = out_dir / "qos_sqlite_hero_bundle.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    if write_tables:
        write_recovery_tables(report, out_dir, write_paper_table=write_paper_table)
    return report


def write_methodology_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# SQLite QoS Methodology Bundle",
        "",
        f"- Overall pass: `{str(report['overall_pass']).lower()}`",
        f"- Measured repetitions: `{report['repetitions_measured']}`",
        f"- Warmup runs: `{report['warmup_runs']}`",
        "- Scope: repeated mounted SQLite/FUSE QoS workflow methodology evidence.",
        "- Non-claim: this is not foreground AI/TensorRT p99 recovery.",
        "",
        "## Mode Summaries",
        "",
    ]
    for mode in report["mode_summaries"]:
        lines.append(
            f"- `{mode['mode']}` runs=`{mode['runs']}`, "
            f"acceptable_runs=`{mode['acceptable_runs']}`, "
            f"p99_median_ms=`{mode['p99_ms']['median']:.3f}`, "
            f"p99_ci95_ms=`[{mode['p99_ms']['ci95_low']:.3f}, {mode['p99_ms']['ci95_high']:.3f}]`, "
            f"storage_median_mb_s=`{mode['storage_mb_s']['median']:.3f}`"
        )
    lines.extend(["", "## Recovery Checks", ""])
    for key, value in report["recovery_check_summary"].items():
        lines.append(
            f"- `{key}`: true_runs=`{value['true_runs']}/{value['runs']}`, "
            f"all_true=`{str(value['all_true']).lower()}`"
        )
    lines.extend(["", "## Component Coverage", ""])
    for key, value in report["component_coverage_summary"].items():
        lines.append(
            f"- `{key}`: true_runs=`{value['true_runs']}/{value['runs']}`, "
            f"all_true=`{str(value['all_true']).lower()}`"
        )
    methodology = report.get("methodology") or {}
    clocks = methodology.get("cpu_gpu_clocks_or_power_mode") or {}
    thermal = methodology.get("thermal_logging") or {}
    lines.extend([
        "",
        "## Methodology Metadata",
        "",
        f"- Run count meets headline minimum: `{str((methodology.get('run_count') or {}).get('meets_headline_minimum')).lower()}`",
        f"- Full workload warmup retained: `{str((methodology.get('warmup') or {}).get('full_workload_warmup')).lower()}`",
        f"- CPU governor ready: `{str(clocks.get('cpu_governor_ready')).lower()}`",
        f"- Thermal log nonempty: `{str(thermal.get('nonempty')).lower()}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# SQLite QoS Recovery Bundle",
        "",
        f"- Overall pass: `{str(report['overall_pass']).lower()}`",
        "- Scope: synchronized SQLite foreground QoS recovery under mounted secure-storage pressure.",
        "- Foreground: SQLite DELETE/FULL transactions on mounted AEGIS-Q FUSE.",
        "- Telemetry source: foreground SQLite slack plus background-storage pressure, not hardware PMU/CUPTI/TensorRT.",
        "- Non-claim: this is not foreground AI/TensorRT p99 recovery.",
        "",
        "## Recovery checks",
        "",
    ]
    for key, value in report.get("recovery_checks", {}).items():
        lines.append(f"- {key}: `{str(value).lower()}`")
    lines.extend([
        "",
        "## Modes",
        "",
        "| mode | acceptable | fg p99 ms | misses | bg MB | telemetry rows | policy throttled rows | daemon throttled rows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for mode in report["modes"]:
        fg = mode["foreground"]
        bg = mode["background"]
        lines.append(
            f"| `{mode['mode']}` | `{str(mode['acceptable']).lower()}` | "
            f"{fg.get('p99_ms') if fg.get('p99_ms') is not None else 'n/a'} | "
            f"{fg.get('deadline_misses', 'n/a')} | "
            f"{bg.get('bytes_written', 0) / (1024.0 * 1024.0):.3f} | "
            f"{mode['telemetry']['rows']} | "
            f"{mode['policy']['throttle_rows']} | "
            f"{mode['daemon_throttle']['throttled_rows']} |"
        )
    lines.extend([
        "",
        "## Required components",
        "",
    ])
    for item, present in report["component_coverage"].items():
        lines.append(f"- {item}: `{str(present).lower()}`")
    lines.extend([
        "",
        "## Raw logs",
        "",
    ])
    for mode in report["modes"]:
        lines.append(f"- `{mode['mode']}`")
        for label, rel in mode["logs"].items():
            lines.append(f"  - {label}: `{rel}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def storage_mb_s(mode: dict[str, Any]) -> float:
    if mode["background"].get("throughput_mb_s") is not None:
        return float(mode["background"].get("throughput_mb_s", 0.0))
    stops = mode["background"].get("writer_stop_rows", [])
    bytes_written = float(mode["background"].get("bytes_written", 0))
    elapsed_ns = max((int(row.get("elapsed_ns", 0)) for row in stops), default=0)
    if bytes_written <= 0.0 or elapsed_ns <= 0:
        return 0.0
    return bytes_written / (1024.0 * 1024.0) / (elapsed_ns / 1_000_000_000.0)


def mode_display_name(mode: str) -> str:
    return {
        "app_only": "App",
        "unthrottled_storage": "Unthrottled",
        "simple_controller": "Simple",
        "aegis_policy": "AEGIS-Q",
    }.get(mode, mode)


def compute_recovery_checks(modes: list[dict[str, Any]]) -> dict[str, bool]:
    by_name = {mode["mode"]: mode for mode in modes}
    required = {"app_only", "unthrottled_storage", "simple_controller", "aegis_policy"}
    if set(by_name) != required:
        return {
            "required_modes_available": False,
            "pressure_raises_p99": False,
            "pressure_causes_deadline_miss": False,
            "simple_recovers_p99": False,
            "simple_removes_deadline_misses": False,
            "aegis_recovers_p99": False,
            "aegis_removes_deadline_misses": False,
            "aegis_keeps_more_storage_than_simple": False,
            "aegis_records_throttle_decisions": False,
        }

    app = by_name["app_only"]
    unthrottled = by_name["unthrottled_storage"]
    simple = by_name["simple_controller"]
    aegis = by_name["aegis_policy"]

    app_p99 = float(app["foreground"].get("p99_ms", float("inf")))
    unthrottled_p99 = float(unthrottled["foreground"].get("p99_ms", float("inf")))
    simple_p99 = float(simple["foreground"].get("p99_ms", float("inf")))
    aegis_p99 = float(aegis["foreground"].get("p99_ms", float("inf")))
    simple_mb_s = storage_mb_s(simple)
    aegis_mb_s = storage_mb_s(aegis)

    return {
        "required_modes_available": True,
        "pressure_raises_p99": unthrottled_p99 > app_p99,
        "pressure_causes_deadline_miss": int(unthrottled["foreground"].get("deadline_misses", 0)) > 0,
        "simple_recovers_p99": simple_p99 < unthrottled_p99,
        "simple_removes_deadline_misses": int(simple["foreground"].get("deadline_misses", 0)) == 0,
        "aegis_recovers_p99": aegis_p99 < unthrottled_p99,
        "aegis_removes_deadline_misses": int(aegis["foreground"].get("deadline_misses", 0)) == 0,
        "aegis_keeps_more_storage_than_simple": aegis_mb_s > simple_mb_s,
        "aegis_records_throttle_decisions": (
            int(aegis["policy"].get("throttle_rows", 0)) > 0
            and int(aegis["daemon_throttle"].get("throttled_rows", 0)) > 0
        ),
    }


def metric_summary(values: list[float], seed_text: str) -> dict[str, Any]:
    ci_low, ci_high = bootstrap_ci(values, seed_text)
    return {
        "runs": len(values),
        "median": statistics.median(values),
        "p05": quantile(values, 0.05),
        "p95": quantile(values, 0.95),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def summarize_repetitions(reports: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    mode_summaries: list[dict[str, Any]] = []
    for mode_name in REQUIRED_MODES:
        rows = [
            next(mode for mode in report["modes"] if mode["mode"] == mode_name)
            for report in reports
        ]
        p99 = [float(row["foreground"].get("p99_ms", 0.0)) for row in rows]
        p95 = [float(row["foreground"].get("p95_ms", 0.0)) for row in rows]
        misses = [float(row["foreground"].get("deadline_misses", 0)) for row in rows]
        storage = [storage_mb_s(row) for row in rows]
        telemetry_rows = [float(row["telemetry"].get("rows", 0)) for row in rows]
        policy_rows = [float(row["policy"].get("throttle_rows", 0)) for row in rows]
        daemon_rows = [float(row["daemon_throttle"].get("throttled_rows", 0)) for row in rows]
        mode_summaries.append(
            {
                "mode": mode_name,
                "runs": len(rows),
                "acceptable_runs": sum(1 for row in rows if row.get("acceptable")),
                "all_acceptable": all(row.get("acceptable") for row in rows),
                "p99_ms": metric_summary(p99, f"{mode_name}|p99_ms"),
                "p95_ms": metric_summary(p95, f"{mode_name}|p95_ms"),
                "deadline_misses": metric_summary(misses, f"{mode_name}|deadline_misses"),
                "storage_mb_s": metric_summary(storage, f"{mode_name}|storage_mb_s"),
                "telemetry_rows": metric_summary(telemetry_rows, f"{mode_name}|telemetry_rows"),
                "policy_throttle_rows": metric_summary(policy_rows, f"{mode_name}|policy_rows"),
                "daemon_throttle_rows": metric_summary(daemon_rows, f"{mode_name}|daemon_rows"),
            }
        )

    recovery_keys = sorted({key for report in reports for key in report.get("recovery_checks", {})})
    recovery_summary = {
        key: {
            "runs": len(reports),
            "true_runs": sum(1 for report in reports if report.get("recovery_checks", {}).get(key)),
            "all_true": all(report.get("recovery_checks", {}).get(key) for report in reports),
        }
        for key in recovery_keys
    }
    component_keys = sorted({key for report in reports for key in report.get("component_coverage", {})})
    component_summary = {
        key: {
            "runs": len(reports),
            "true_runs": sum(1 for report in reports if report.get("component_coverage", {}).get(key)),
            "all_true": all(report.get("component_coverage", {}).get(key) for report in reports),
        }
        for key in component_keys
    }
    return mode_summaries, recovery_summary, component_summary


def write_recovery_tables(report: dict[str, Any], out_dir: Path,
                          write_paper_table: bool = True) -> None:
    rows: list[dict[str, Any]] = []
    for mode in report["modes"]:
        telemetry = mode["telemetry"]
        policy_rows = int(mode["policy"].get("throttle_rows", 0))
        daemon_rows = int(mode["daemon_throttle"].get("throttled_rows", 0))
        rows.append({
            "mode": mode["mode"],
            "label": mode_display_name(mode["mode"]),
            "p50_ms": mode["foreground"].get("p50_ms"),
            "p95_ms": mode["foreground"].get("p95_ms"),
            "p99_ms": mode["foreground"].get("p99_ms"),
            "deadline_misses": mode["foreground"].get("deadline_misses"),
            "storage_mb_s": storage_mb_s(mode),
            "avg_cpu_utilization": telemetry.get("avg_cpu_utilization"),
            "avg_gpu_utilization": telemetry.get("avg_gpu_utilization"),
            "telemetry_samples": telemetry.get("rows"),
            "policy_throttle_rows": policy_rows,
            "daemon_throttle_rows": daemon_rows,
            "throttle_decisions": policy_rows + daemon_rows,
        })

    csv_path = out_dir / "qos_recovery_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    def fmt(value: Any, digits: int = 1) -> str:
        if value is None:
            return "--"
        return f"{float(value):.{digits}f}"

    def util(value: Any) -> str:
        if value is None:
            return "--"
        return f"{float(value) * 100.0:.0f}"

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{SQLite foreground recovery under mounted secure-storage pressure; "
        "latencies are p50/p95/p99 in ms and CPU/GPU are utilization percent.}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabularx}{\\columnwidth}{P{0.18\\columnwidth}|P{0.20\\columnwidth}|r|r|r|r|r}",
        "\\toprule",
        "\\textbf{Mode} & \\textbf{p50/p95/p99} & \\textbf{Miss} & \\textbf{MB/s} & \\textbf{CPU/GPU} & \\textbf{Samp.} & \\textbf{Thr.}\\\\",
        "\\midrule",
    ]
    for row in rows:
        lat = f"{fmt(row['p50_ms'])}/{fmt(row['p95_ms'])}/{fmt(row['p99_ms'])}"
        cpu_gpu = f"{util(row['avg_cpu_utilization'])}/{util(row['avg_gpu_utilization'])}"
        lines.append(
            f"{row['label']} & {lat} & {int(row['deadline_misses'])} & "
            f"{fmt(row['storage_mb_s'])} & {cpu_gpu} & "
            f"{int(row['telemetry_samples'])} & {int(row['throttle_decisions'])}\\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabularx}",
        "\\label{tab:qos_sqlite_recovery}",
        "\\end{table}",
        "",
    ])
    table_text = "\n".join(lines)
    (out_dir / "qos_recovery_table.tex").write_text(table_text, encoding="utf-8")
    if write_paper_table:
        (ROOT / "Paper" / "generated_qos_recovery_table.tex").write_text(table_text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--thermal-interval-ms", type=int, default=100)
    parser.add_argument("--skip-paper-table", action="store_true")
    parser.add_argument("--transactions", type=int, default=48)
    parser.add_argument("--rows-per-txn", type=int, default=8)
    parser.add_argument("--sqlite-payload-bytes", type=int, default=256)
    parser.add_argument("--deadline-ms", type=float, default=10.0)
    parser.add_argument("--inter-transaction-sleep-ms", type=float, default=1.0)
    parser.add_argument("--post-foreground-drain-ms", type=int, default=250)
    parser.add_argument("--background-writers", type=int, default=1)
    parser.add_argument("--background-chunk-bytes", type=int, default=65536)
    parser.add_argument("--background-fsync-every", type=int, default=1)
    parser.add_argument("--background-warmup-ms", type=int, default=100)
    parser.add_argument("--telemetry-interval-ms", type=int, default=20)
    parser.add_argument("--telemetry-poll-ms", type=int, default=10)
    parser.add_argument("--telemetry-window", type=int, default=12)
    parser.add_argument("--controller-warmup-transactions", type=int, default=2)
    parser.add_argument("--background-pressure-util", type=float, default=0.85)
    parser.add_argument(
        "--background-pressure-pattern",
        default="",
        help="Optional comma-separated pressure pattern, with value:count tokens.",
    )
    parser.add_argument("--enter-util", type=float, default=0.70)
    parser.add_argument("--exit-util", type=float, default=0.60)
    parser.add_argument("--hold-samples", type=int, default=1)
    parser.add_argument("--harness-throttle-sleep-us", type=int, default=5000)
    parser.add_argument("--daemon-throttle-sleep-us", type=int, default=30000)
    args = parser.parse_args()
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be at least 1")
    if args.warmup_runs < 0:
        raise SystemExit("--warmup-runs must be non-negative")
    args.out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    args._background_pressure_pattern_values = parse_pressure_pattern(args.background_pressure_pattern)
    args.require_daemon_throttle = True

    if not FUSE_BIN.exists():
        raise SystemExit("missing build/pqc_fuse; run cmake --build build first")

    methodology_mode = args.warmup_runs > 0 or args.repetitions > 1
    if not methodology_mode:
        report = run_bundle(
            args.out_dir,
            args,
            write_tables=True,
            write_paper_table=not args.skip_paper_table,
        )
        print(json.dumps({
            "overall_pass": report["overall_pass"],
            "out": str(args.out_dir / "qos_sqlite_hero_bundle.json"),
            "modes": {mode["mode"]: mode["acceptable"] for mode in report["modes"]},
        }, sort_keys=True))
        return 0 if report["overall_pass"] else 1

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    thermal_proc, thermal_fp, thermal_status = start_thermal_log(args.out_dir, args.thermal_interval_ms)
    try:
        warmups = [
            run_bundle(
                args.out_dir / f"warmup_{index:02d}",
                args,
                write_tables=False,
                write_paper_table=False,
            )
            for index in range(args.warmup_runs)
        ]
        repetitions = [
            run_bundle(
                args.out_dir / f"rep_{index:02d}",
                args,
                write_tables=False,
                write_paper_table=False,
            )
            for index in range(args.repetitions)
        ]
    finally:
        thermal_status = stop_thermal_log(thermal_proc, thermal_fp, thermal_status)

    mode_summaries, recovery_summary, component_summary = summarize_repetitions(repetitions)
    overall_pass = (
        bool(repetitions)
        and all(report["overall_pass"] for report in repetitions)
        and all(report["overall_pass"] for report in warmups)
        and all(summary["all_acceptable"] for summary in mode_summaries)
        and all(summary["all_true"] for summary in recovery_summary.values())
        and all(summary["all_true"] for summary in component_summary.values())
    )
    report = {
        "artifact": "qos_sqlite_hero_bundle",
        "artifact_role": "methodology-strengthened repeated SQLite QoS workflow run",
        "overall_pass": overall_pass,
        "warmup_runs": args.warmup_runs,
        "repetitions_measured": args.repetitions,
        "warmups": warmups,
        "repetitions": repetitions,
        "modes": repetitions[0]["modes"] if repetitions else [],
        "modes_schema": "first_measured_repetition",
        "mode_summaries": mode_summaries,
        "recovery_check_summary": recovery_summary,
        "component_coverage_summary": component_summary,
        "platform": platform_manifest(),
        "methodology": methodology_manifest(args, thermal_status),
        "scope": [
            "Repeated mounted-FUSE SQLite foreground QoS recovery methodology evidence.",
            "Does not claim AI inference or TensorRT p99 restoration.",
            "Current paper table source remains the single retained QoS bundle unless explicitly regenerated.",
        ],
    }
    json_path = args.out_dir / "qos_sqlite_hero_bundle.json"
    md_path = args.out_dir / "qos_sqlite_hero_bundle.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_methodology_markdown(report, md_path)
    print(json.dumps({
        "overall_pass": overall_pass,
        "out": str(json_path),
        "repetitions": args.repetitions,
        "warmup_runs": args.warmup_runs,
    }, sort_keys=True))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
