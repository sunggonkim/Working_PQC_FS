#!/usr/bin/env python3
"""Exercise the mounted key-plane rekey workflow.

This is not a primitive-only ML-KEM benchmark.  The harness mounts the final
``build/pqc_fuse`` binary, opens a batch of encrypted files, forces the
background rekey worker to refresh their authenticated envelopes, and retains
the FUSE/admission logs for three modes:

* CPU-only key-plane refresh.
* GPU-batched key-plane refresh when explicit slack is available.
* Policy fallback to CPU when the producer-facing slack budget is absent.

The current workflow is intentionally scoped: it refreshes open-file DEKs and
rewrites AEGIS-Q's HMAC-authenticated envelope.  It does not claim a deployed
hardware-backed credential lifecycle or a persistent ML-KEM ciphertext
hierarchy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FUSE_BIN = ROOT / "build" / "pqc_fuse"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "keyplane_rekey_workflow"

REKEY_RE = re.compile(
    r"REKEY WORKER: batched (?P<files>\d+) files (?P<usec>[0-9.]+).*"
    r"\(target=(?P<target>CPU|GPU), run=(?P<run>CPU|GPU)\)"
)
DETAIL_RE = re.compile(
    r"REKEY WORKER DETAIL: work_bytes=(?P<work_bytes>\d+) "
    r"budget_ns=(?P<budget_ns>\d+) decision_reason=(?P<decision>\d+) "
    r"deferral_reason=(?P<deferral>\d+)"
)


@dataclass
class FuseProc:
    proc: subprocess.Popen[bytes]
    stdout: Any
    stderr: Any


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
    fuse_sha = sha256_bytes(FUSE_BIN.read_bytes()) if FUSE_BIN.exists() else None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "kernel": platform.platform(),
        "machine": platform.machine(),
        "device_model": model,
        "cpu_count": os.cpu_count(),
        "fuse_binary": {
            "path": relpath(FUSE_BIN),
            "exists": FUSE_BIN.exists(),
            "sha256": fuse_sha,
        },
        "nvcc": command_capture(["nvcc", "--version"], timeout_s=10.0),
        "uname": command_capture(["uname", "-a"], timeout_s=5.0),
        "git_head": command_capture(["git", "rev-parse", "HEAD"], timeout_s=5.0),
    }


def methodology_manifest(args: argparse.Namespace, thermal_status: dict[str, Any]) -> dict[str, Any]:
    governors = read_cpu_governors()
    governor_counts = governors.get("governor_counts", {})
    governor_ready = bool(governor_counts) and set(governor_counts) == {"performance"}
    return {
        "methodology_id": "aegisq-keyplane-rekey-methodology-v1",
        "warmup": {
            "warmup_runs": args.warmup_runs,
            "full_workload_warmup": args.warmup_runs > 0,
            "artifacts": [
                relpath(args.out / f"warmup_{index:02d}") for index in range(args.warmup_runs)
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
            "unit": "independent mounted rekey workflow repetitions per mode",
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
            "scope": "mounted open-file envelope-refresh workflow, not filesystem throughput comparison",
            "warm_cache": "full workload warmup is retained when --warmup-runs is nonzero",
            "cold_cache": "not claimed; each repetition uses a fresh temporary lower directory and mount",
        },
        "failure_handling": {
            "missing_binary": "fatal",
            "mount_failure": "fatal",
            "mode_failure": "fatal",
            "unsupported_configuration": "not emitted as zero or success",
        },
    }


def start_fuse(storage_dir: Path, mount_dir: Path, out_dir: Path, env: dict[str, str]) -> FuseProc:
    log_dir = out_dir / "mount_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout = (log_dir / "pqc_fuse.stdout.txt").open("wb")
    stderr = (log_dir / "pqc_fuse.stderr.txt").open("wb")
    proc = subprocess.Popen(
        [str(FUSE_BIN), str(storage_dir), str(mount_dir), "-f"],
        cwd=ROOT,
        env=env,
        stdout=stdout,
        stderr=stderr,
    )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if subprocess.run(["mountpoint", "-q", str(mount_dir)], check=False).returncode == 0:
            return FuseProc(proc=proc, stdout=stdout, stderr=stderr)
        if proc.poll() is not None:
            stdout.close()
            stderr.close()
            raise RuntimeError(f"FUSE exited before mount: rc={proc.returncode}")
        time.sleep(0.05)
    stdout.close()
    stderr.close()
    raise TimeoutError("timed out waiting for FUSE mount")


def stop_fuse(handle: FuseProc | None, mount_dir: Path) -> None:
    subprocess.run(
        ["fusermount3", "-u", str(mount_dir)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if handle is not None:
        if handle.proc.poll() is None:
            handle.proc.send_signal(signal.SIGINT)
            try:
                handle.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                handle.proc.kill()
                handle.proc.wait(timeout=3)
        handle.stdout.close()
        handle.stderr.close()


def parse_admission_trace(path: Path) -> dict[str, Any]:
    records = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return {
        "path": relpath(path),
        "records": records,
        "targets": {
            "CPU": sum(1 for row in records if row.get("chosen_target") == "CPU"),
            "GPU": sum(1 for row in records if row.get("chosen_target") == "GPU"),
        },
        "ai_qos_exhausted_records": [
            row for row in records if int(row.get("deferral_reason", 0)) & 0x02
        ],
    }


def parse_rekey_log(stderr_path: Path) -> dict[str, Any]:
    text = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
    events = []
    pending: dict[str, Any] | None = None
    for line in text.splitlines():
        match = REKEY_RE.search(line)
        if match:
            pending = {
                "files": int(match.group("files")),
                "usec": float(match.group("usec")),
                "target": match.group("target"),
                "run": match.group("run"),
                "line": line,
            }
            events.append(pending)
            continue
        detail = DETAIL_RE.search(line)
        if detail and pending is not None:
            pending.update(
                {
                    "work_bytes": int(detail.group("work_bytes")),
                    "budget_ns": int(detail.group("budget_ns")),
                    "decision_reason": int(detail.group("decision")),
                    "deferral_reason": int(detail.group("deferral")),
                    "detail_line": line,
                }
            )
    return {
        "path": relpath(stderr_path),
        "sha256": sha256_bytes(stderr_path.read_bytes()) if stderr_path.exists() else None,
        "events": events,
        "event_count": len(events),
        "files_refreshed": sum(event["files"] for event in events),
        "run_counts": {
            "CPU": sum(1 for event in events if event["run"] == "CPU"),
            "GPU": sum(1 for event in events if event["run"] == "GPU"),
        },
        "target_counts": {
            "CPU": sum(1 for event in events if event["target"] == "CPU"),
            "GPU": sum(1 for event in events if event["target"] == "GPU"),
        },
    }


def mode_env(base: dict[str, str], out_dir: Path, mode: str, files: int, collect_ms: int) -> dict[str, str]:
    env = base.copy()
    env.update(
        {
            "PQC_MASTER_PASSWORD": "keyplane-workflow-password",
            "PQC_FRESHNESS_ANCHOR_BACKEND": "file",
            "PQC_FRESHNESS_ANCHOR_PATH": str(out_dir / "anchor.bin"),
            "PQC_ADMISSION_TRACE_PATH": str(out_dir / "admission_trace.jsonl"),
            "PQC_FORCE_REKEY_ON_WRITE": "1",
            "PQC_KEY_ROTATION_INTERVAL_S": "0",
            "PQC_REKEY_BATCH_COLLECT_MS": str(collect_ms),
            "PQC_REKEY_BATCH_MAX": str(files),
            "PQC_GPU_MIN_BATCH_BYTES": "4096",
            "PQC_REKEY_GPU_KERNEL_EST_NS": "250000",
            "PQC_REKEY_DEADLINE_NS": "5000000000",
            "PQC_PRODUCER_SLACK_STALE_NS": "10000000000",
        }
    )
    if mode == "cpu_only":
        env["PQC_GPU_MIN_BATCH"] = str(files + 1)
        env["PQC_ADMISSION_INITIAL_BUDGET_NS"] = "100000000"
    elif mode == "gpu_batch":
        env["PQC_GPU_MIN_BATCH"] = "4"
        env["PQC_ADMISSION_INITIAL_BUDGET_NS"] = "100000000"
        env["PQC_TELEMETRY_MEM_BANDWIDTH"] = "0.10"
        env["PQC_TELEMETRY_TENSOR_CORE"] = "0.10"
    elif mode == "policy_fallback":
        env["PQC_GPU_MIN_BATCH"] = "4"
        env["PQC_ADMISSION_INITIAL_BUDGET_NS"] = "0"
        env["PQC_TELEMETRY_MEM_BANDWIDTH"] = "0.90"
        env["PQC_TELEMETRY_TENSOR_CORE"] = "0.10"
    else:
        raise ValueError(mode)
    return env


def run_mode(mode: str, out_root: Path, files: int, wait_s: float, collect_ms: int) -> dict[str, Any]:
    out_dir = out_root / mode
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f"aegis_keyplane_{mode}_"))
    storage_dir = tmp / "store"
    mount_dir = tmp / "mnt"
    storage_dir.mkdir()
    mount_dir.mkdir()
    handle: FuseProc | None = None
    fds: list[int] = []
    try:
        env = mode_env(os.environ.copy(), out_dir, mode, files, collect_ms)
        handle = start_fuse(storage_dir, mount_dir, out_dir, env)
        payload = (mode.encode("utf-8") + b":") * 1024
        for index in range(files):
            fd = os.open(mount_dir / f"file_{index:03d}.bin", os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
            fds.append(fd)
            os.write(fd, payload[:4096])
        time.sleep(wait_s)
        for fd in fds:
            try:
                os.fsync(fd)
            except OSError:
                pass
        time.sleep(0.5)
    finally:
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass
        stop_fuse(handle, mount_dir)
        shutil.rmtree(tmp, ignore_errors=True)

    stderr_path = out_dir / "mount_logs" / "pqc_fuse.stderr.txt"
    admission_path = out_dir / "admission_trace.jsonl"
    rekey = parse_rekey_log(stderr_path)
    admission = parse_admission_trace(admission_path)
    total_usec = sum(float(event["usec"]) for event in rekey["events"])
    throughput_files_per_s = (
        rekey["files_refreshed"] / (total_usec / 1_000_000.0)
        if total_usec > 0
        else 0.0
    )
    if mode == "cpu_only":
        acceptable = rekey["event_count"] > 0 and rekey["run_counts"]["GPU"] == 0
    elif mode == "gpu_batch":
        acceptable = rekey["event_count"] > 0 and rekey["run_counts"]["GPU"] > 0
    else:
        acceptable = (
            rekey["event_count"] > 0
            and rekey["run_counts"]["GPU"] == 0
            and len(admission["ai_qos_exhausted_records"]) > 0
        )
    result = {
        "mode": mode,
        "files_requested": files,
        "acceptable": acceptable,
        "rekey": rekey,
        "admission": admission,
        "total_rekey_usec": total_usec,
        "throughput_files_per_s": throughput_files_per_s,
        "scope": (
            "Mounted open-file envelope refresh; not deployed key lifecycle, "
            "persistent KEM hierarchy, or foreground QoS recovery."
        ),
    }
    (out_dir / f"{mode}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def run_repetition(index: int, out_root: Path, args: argparse.Namespace,
                   prefix: str, legacy_layout: bool = False) -> dict[str, Any]:
    rep_dir = out_root if legacy_layout else out_root / f"{prefix}_{index:02d}"
    rep_dir.mkdir(parents=True, exist_ok=True)
    modes = [
        run_mode(mode, rep_dir, args.files, args.wait_s, args.collect_ms)
        for mode in ("cpu_only", "gpu_batch", "policy_fallback")
    ]
    by_mode = {mode["mode"]: mode for mode in modes}
    cpu_total = by_mode["cpu_only"]["total_rekey_usec"]
    gpu_total = by_mode["gpu_batch"]["total_rekey_usec"]
    speedup = cpu_total / gpu_total if gpu_total > 0 else 0.0
    all_modes_acceptable = all(mode["acceptable"] for mode in modes)
    return {
        "repetition": index,
        "path": relpath(rep_dir),
        "modes": modes,
        "gpu_vs_cpu_speedup": speedup,
        "maintenance_visible_benefit": speedup > 1.0,
        "all_modes_acceptable": all_modes_acceptable,
        "overall_pass": all_modes_acceptable and speedup > 1.0,
    }


def summarize_mode_repetitions(repetitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for mode_name in ("cpu_only", "gpu_batch", "policy_fallback"):
        rows = [
            next(mode for mode in repetition["modes"] if mode["mode"] == mode_name)
            for repetition in repetitions
        ]
        total_usec = [float(row["total_rekey_usec"]) for row in rows]
        throughput = [float(row["throughput_files_per_s"]) for row in rows]
        total_ci = bootstrap_ci(total_usec, f"{mode_name}|total_rekey_usec")
        throughput_ci = bootstrap_ci(throughput, f"{mode_name}|throughput")
        summaries.append(
            {
                "mode": mode_name,
                "runs": len(rows),
                "acceptable_runs": sum(1 for row in rows if row["acceptable"]),
                "all_acceptable": all(row["acceptable"] for row in rows),
                "files_refreshed_per_run": [row["rekey"]["files_refreshed"] for row in rows],
                "rekey_events_per_run": [row["rekey"]["event_count"] for row in rows],
                "admission_targets_per_run": [row["admission"]["targets"] for row in rows],
                "total_rekey_usec_median": statistics.median(total_usec),
                "total_rekey_usec_p05": quantile(total_usec, 0.05),
                "total_rekey_usec_p95": quantile(total_usec, 0.95),
                "total_rekey_usec_ci95_low": total_ci[0],
                "total_rekey_usec_ci95_high": total_ci[1],
                "throughput_files_per_s_median": statistics.median(throughput),
                "throughput_files_per_s_p05": quantile(throughput, 0.05),
                "throughput_files_per_s_p95": quantile(throughput, 0.95),
                "throughput_files_per_s_ci95_low": throughput_ci[0],
                "throughput_files_per_s_ci95_high": throughput_ci[1],
            }
        )
    return summaries


def summarize_speedups(repetitions: list[dict[str, Any]]) -> dict[str, Any]:
    speedups = [float(repetition["gpu_vs_cpu_speedup"]) for repetition in repetitions]
    ci = bootstrap_ci(speedups, "gpu_vs_cpu_speedup")
    return {
        "runs": len(speedups),
        "median": statistics.median(speedups),
        "p05": quantile(speedups, 0.05),
        "p95": quantile(speedups, 0.95),
        "ci95_low": ci[0],
        "ci95_high": ci[1],
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Key-Plane Rekey Workflow",
        "",
        f"- Overall pass: `{str(result['overall_pass']).lower()}`",
        f"- Files per mode: `{result['files_per_mode']}`",
        f"- GPU-vs-CPU speedup: `{result['gpu_vs_cpu_speedup']:.3f}`",
        f"- Measured repetitions: `{result['repetitions_measured']}`",
        f"- Warmup runs: `{result['warmup_runs']}`",
        "",
        "## Mode Summaries",
    ]
    for mode in result["mode_summaries"]:
        lines.append(
            f"- `{mode['mode']}` runs=`{mode['runs']}`, "
            f"acceptable_runs=`{mode['acceptable_runs']}`, "
            f"median_total_ms=`{mode['total_rekey_usec_median'] / 1000.0:.3f}`, "
            f"ci95_total_ms=`[{mode['total_rekey_usec_ci95_low'] / 1000.0:.3f}, "
            f"{mode['total_rekey_usec_ci95_high'] / 1000.0:.3f}]`, "
            f"median_throughput_files_s=`{mode['throughput_files_per_s_median']:.1f}`"
        )
    lines.extend(["", "## Representative Modes"])
    for mode in result["modes"]:
        lines.append(
            f"- `{mode['mode']}` acceptable=`{str(mode['acceptable']).lower()}`, "
            f"events=`{mode['rekey']['event_count']}`, "
            f"files_refreshed=`{mode['rekey']['files_refreshed']}`, "
            f"total_ms=`{mode['total_rekey_usec'] / 1000.0:.3f}`, "
            f"throughput_files_s=`{mode['throughput_files_per_s']:.1f}`, "
            f"run_counts=`{mode['rekey']['run_counts']}`, "
            f"admission_targets=`{mode['admission']['targets']}`"
        )
    lines.extend(["", "## Scope"])
    lines.append(
        "This artifact supports only a mounted open-file envelope-refresh workflow. "
        "It does not prove hardware-backed credential release, persistent PCR binding, "
        "or foreground QoS recovery."
    )
    methodology = result.get("methodology") or {}
    if methodology:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--files", type=int, default=16)
    parser.add_argument("--wait-s", type=float, default=3.0)
    parser.add_argument("--collect-ms", type=int, default=1000)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--thermal-interval-ms", type=int, default=100)
    args = parser.parse_args()
    if args.files < 4:
        raise SystemExit("--files must be at least 4")
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be at least 1")
    if args.warmup_runs < 0:
        raise SystemExit("--warmup-runs must be non-negative")
    if not FUSE_BIN.exists():
        raise SystemExit(f"missing {FUSE_BIN}; build first")

    args.out = args.out if args.out.is_absolute() else ROOT / args.out
    args.out.mkdir(parents=True, exist_ok=True)
    thermal_proc, thermal_fp, thermal_status = start_thermal_log(args.out, args.thermal_interval_ms)
    try:
        warmups = [
            run_repetition(index, args.out, args, prefix="warmup")
            for index in range(args.warmup_runs)
        ]
        legacy_layout = args.repetitions == 1 and args.warmup_runs == 0
        repetitions = [
            run_repetition(index, args.out, args, prefix="rep", legacy_layout=legacy_layout)
            for index in range(args.repetitions)
        ]
    finally:
        thermal_status = stop_thermal_log(thermal_proc, thermal_fp, thermal_status)

    modes = repetitions[0]["modes"]
    mode_summaries = summarize_mode_repetitions(repetitions)
    speedup_summary = summarize_speedups(repetitions)
    speedup = float(speedup_summary["median"])
    result = {
        "artifact": "keyplane_rekey_workflow",
        "files_per_mode": args.files,
        "collect_ms": args.collect_ms,
        "wait_s": args.wait_s,
        "warmup_runs": args.warmup_runs,
        "repetitions_measured": args.repetitions,
        "warmups": warmups,
        "repetitions": repetitions,
        "modes": modes,
        "modes_schema": "first_measured_repetition",
        "mode_summaries": mode_summaries,
        "gpu_vs_cpu_speedup_summary": speedup_summary,
        "gpu_vs_cpu_speedup": speedup,
        "maintenance_visible_benefit": speedup > 1.0,
        "platform": platform_manifest(),
        "methodology": methodology_manifest(args, thermal_status),
        "overall_pass": (
            all(summary["all_acceptable"] for summary in mode_summaries)
            and speedup > 1.0
        ),
    }
    json_path = args.out / "keyplane_rekey_workflow.json"
    md_path = args.out / "keyplane_rekey_workflow.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(result, md_path)
    print(json.dumps({"overall_pass": result["overall_pass"], "out": str(json_path)}, sort_keys=True))
    return 0 if result["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
