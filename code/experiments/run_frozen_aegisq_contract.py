#!/usr/bin/env python3
"""Run AEGIS-Q under the frozen filesystem workload contract.

The harness executes only the AEGIS-Q mode from
``artifacts/validation/frozen_workload_contract/frozen_workload_contract.json``.
It records a contract-valid warm-cache fio row, retains raw logs, and marks the
cold-cache row invalid when privileged cache dropping is unavailable.  It does
not produce a cross-system comparison by itself.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import signal
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FUSE_BIN = ROOT / "build" / "pqc_fuse"
DEFAULT_CONTRACT = (
    ROOT
    / "artifacts"
    / "validation"
    / "frozen_workload_contract"
    / "frozen_workload_contract.json"
)
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "frozen_aegisq_contract"


@dataclass
class FuseHandle:
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
    executable = command[0]
    if shutil.which(executable) is None and not Path(executable).exists():
        return {
            "argv": command,
            "available": False,
            "returncode": None,
            "stdout": "",
            "stderr": "",
        }
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


def process_snapshot() -> dict[str, Any]:
    captured = command_capture(
        ["ps", "-eo", "pid,ppid,comm,pcpu,pmem,args", "--sort=-pcpu"],
        timeout_s=5.0,
    )
    stdout = str(captured.get("stdout", ""))
    captured["stdout"] = "\n".join(stdout.splitlines()[:100])
    captured["truncated_to_lines"] = 100
    return captured


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


def stop_thermal_log(
    proc: subprocess.Popen[str] | None,
    fp: Any,
    status: dict[str, Any],
) -> dict[str, Any]:
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


def platform_manifest(contract_path: Path) -> dict[str, Any]:
    model_path = Path("/proc/device-tree/model")
    model = (
        model_path.read_bytes().rstrip(b"\0").decode(errors="replace")
        if model_path.exists()
        else "unknown"
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "system": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "device_model": model,
        "cpu_count": os.cpu_count(),
        "cpu_governors": read_cpu_governors(),
        "uname": command_capture(["uname", "-a"], timeout_s=5.0),
        "findmnt_root": command_capture(
            ["findmnt", "-T", str(ROOT), "-no", "SOURCE,FSTYPE,TARGET,OPTIONS"],
            timeout_s=5.0,
        ),
        "df_root": command_capture(["df", "-PT", str(ROOT)], timeout_s=5.0),
        "fio_version": command_capture(["fio", "--version"], timeout_s=5.0),
        "fusermount": command_capture([fusermount_command(), "--version"], timeout_s=5.0),
        "nvpmodel_q": command_capture(["nvpmodel", "-q"], timeout_s=10.0),
        "jetson_clocks_show": command_capture(["jetson_clocks", "--show"], timeout_s=10.0),
        "git_head": command_capture(["git", "rev-parse", "HEAD"], timeout_s=5.0),
        "git_dirty_short": command_capture(["git", "status", "--short"], timeout_s=5.0),
        "contract": {
            "path": relpath(contract_path),
            "sha256": sha256_bytes(contract_path.read_bytes()) if contract_path.exists() else None,
        },
        "fuse_binary": {
            "path": relpath(FUSE_BIN),
            "exists": FUSE_BIN.exists(),
            "sha256": sha256_bytes(FUSE_BIN.read_bytes()) if FUSE_BIN.exists() else None,
        },
        "process_snapshot": process_snapshot(),
    }


def quantile(samples: list[float], q: float) -> float:
    ordered = sorted(samples)
    index = round((len(ordered) - 1) * q)
    return ordered[index]


def bootstrap_ci(
    samples: list[float],
    seed_text: str,
    trials: int = 10000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not samples:
        raise ValueError("empty bootstrap sample")
    if len(samples) == 1:
        return samples[0], samples[0]
    seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    n = len(samples)
    values = [
        statistics.median(samples[rng.randrange(n)] for _ in range(n))
        for _ in range(trials)
    ]
    values.sort()
    lo = max(0, min(trials - 1, int((alpha / 2.0) * trials)))
    hi = max(0, min(trials - 1, int((1.0 - alpha / 2.0) * trials) - 1))
    return values[lo], values[hi]


def fusermount_command() -> str:
    for name in ("fusermount3", "fusermount"):
        if shutil.which(name):
            return name
    return "fusermount3"


def start_fuse(storage_dir: Path, mount_dir: Path, out_dir: Path,
               password: str, extra_env: dict[str, str] | None = None) -> FuseHandle:
    log_dir = out_dir / "mount_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout = (log_dir / "pqc_fuse.stdout.txt").open("wb")
    stderr = (log_dir / "pqc_fuse.stderr.txt").open("wb")
    env = os.environ.copy()
    env.update(
        {
            "PQC_MASTER_PASSWORD": password,
            "PQC_FRESHNESS_ANCHOR_BACKEND": "file",
            "PQC_FRESHNESS_ANCHOR_PATH": str(storage_dir / ".anchor"),
            "PQC_ENABLE_QOS_THROTTLE_ON_WRITE": "0",
            "PQC_KEY_ROTATION_INTERVAL_S": "0",
        }
    )
    if extra_env:
        env.update(extra_env)
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
            raise RuntimeError(f"FUSE exited before mount: rc={proc.returncode}")
        time.sleep(0.05)
    stdout.close()
    stderr.close()
    raise TimeoutError("timed out waiting for AEGIS-Q FUSE mount")


def stop_fuse(handle: FuseHandle | None, mount_dir: Path, out_dir: Path) -> dict[str, Any]:
    log_dir = out_dir / "mount_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    unmount = subprocess.run(
        [fusermount_command(), "-u", str(mount_dir)],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    (log_dir / "unmount.stdout.txt").write_text(unmount.stdout, encoding="utf-8")
    (log_dir / "unmount.stderr.txt").write_text(unmount.stderr, encoding="utf-8")
    if handle is None:
        return {"argv": [fusermount_command(), "-u", str(mount_dir)], "returncode": unmount.returncode}
    if handle.proc.poll() is None:
        handle.proc.send_signal(signal.SIGINT)
        try:
            handle.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            handle.proc.kill()
            handle.proc.wait(timeout=5)
    handle.stdout.close()
    handle.stderr.close()
    return {
        "argv": [fusermount_command(), "-u", str(mount_dir)],
        "returncode": unmount.returncode,
        "fuse_returncode": handle.proc.returncode,
    }


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ValueError(f"missing contract object in {path}")
    return payload


def workload_profile(contract_payload: dict[str, Any]) -> dict[str, Any]:
    profiles = contract_payload["contract"].get("workload_profiles", [])
    if not profiles:
        raise ValueError("contract has no workload_profiles")
    return profiles[0]


def fio_command(profile: dict[str, Any], bench_dir: Path) -> list[str]:
    fio_common = profile["mount_options"]["fio_common"]
    runtime = str(fio_common["runtime"]).removesuffix("s")
    ramp_time = str(fio_common["ramp_time"]).removesuffix("s")
    argv = [
        "fio",
        "--name=frozen_randrw_4k_fdatasync",
        f"--ioengine={fio_common['ioengine']}",
        f"--rw={fio_common['rw']}",
        f"--rwmixread={fio_common['rwmixread']}",
        f"--bs={fio_common['bs']}",
        f"--direct={fio_common['direct']}",
        f"--fdatasync={fio_common['fdatasync']}",
        f"--iodepth={fio_common['iodepth']}",
        f"--numjobs={fio_common['numjobs']}",
        f"--size={fio_common['size']}",
    ]
    if "fallocate" in fio_common:
        argv.append(f"--fallocate={fio_common['fallocate']}")
    if "allow_file_create" in fio_common:
        argv.append(f"--allow_file_create={fio_common['allow_file_create']}")
    if "overwrite" in fio_common:
        argv.append(f"--overwrite={fio_common['overwrite']}")
    argv.extend(
        [
            "--time_based",
            f"--runtime={runtime}",
            f"--ramp_time={ramp_time}",
            f"--directory={bench_dir}",
            "--output-format=json",
        ]
    )
    return argv


def prepared_fio_filename(profile: dict[str, Any], bench_dir: Path) -> Path:
    prep = profile["mount_options"].get("file_preparation", {})
    filename = prep.get("filename", "frozen_randrw_4k_fdatasync.0.0")
    return bench_dir / filename


def precreate_fio_file(profile: dict[str, Any], bench_dir: Path, out_dir: Path) -> dict[str, Any]:
    path = prepared_fio_filename(profile, bench_dir)
    size = int(profile["file_size_bytes"])
    out_path = out_dir / "file_preparation.json"
    start_ns = time.time_ns()
    method = "posix_fallocate"
    rc = 0
    error: str | None = None
    fd = -1
    try:
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.posix_fallocate(fd, 0, size)
        except AttributeError:
            method = "ftruncate_no_posix_fallocate"
            os.ftruncate(fd, size)
        except OSError as exc:
            method = "ftruncate_after_posix_fallocate_failure"
            error = str(exc)
            os.ftruncate(fd, size)
    except OSError as exc:
        rc = exc.errno or 1
        error = str(exc)
    finally:
        if fd >= 0:
            os.close(fd)
    end_ns = time.time_ns()
    try:
        stat = path.stat()
        observed_size: int | None = stat.st_size
    except OSError:
        observed_size = None
    payload = {
        "path": relpath(path),
        "requested_size_bytes": size,
        "observed_size_bytes": observed_size,
        "method": method,
        "returncode": rc,
        "error": error,
        "start_realtime_ns": start_ns,
        "end_realtime_ns": end_ns,
        "wall_seconds": (end_ns - start_ns) / 1_000_000_000,
        "valid": rc == 0 and observed_size == size,
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    payload["artifact"] = relpath(out_path)
    return payload


def run_fio(label: str, argv: list[str], out_dir: Path) -> dict[str, Any]:
    return run_fio_with_timeout(label, argv, out_dir, timeout_s=None)


def run_fio_with_timeout(
    label: str,
    argv: list[str],
    out_dir: Path,
    timeout_s: float | None,
) -> dict[str, Any]:
    run_dir = out_dir / "fio_raw"
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / f"{label}.json"
    stderr_path = run_dir / f"{label}.stderr.txt"
    command_path = run_dir / f"{label}.command.json"
    start_ns = time.time_ns()
    proc = subprocess.Popen(
        argv,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        proc.kill()
        stdout, stderr = proc.communicate()
        stdout = stdout or exc.stdout or ""
        stderr = stderr or exc.stderr or ""
    end_ns = time.time_ns()
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    command = {
        "label": label,
        "argv": argv,
        "cwd": str(ROOT),
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "timeout_s": timeout_s,
        "start_realtime_ns": start_ns,
        "end_realtime_ns": end_ns,
        "wall_seconds": (end_ns - start_ns) / 1_000_000_000,
        "stdout_json": relpath(stdout_path),
        "stderr": relpath(stderr_path),
    }
    command_path.write_text(json.dumps(command, indent=2, sort_keys=True), encoding="utf-8")
    return command


def percentile_ns(section: dict[str, Any], pct: str) -> float | None:
    try:
        return float(section["clat_ns"]["percentile"][pct])
    except (KeyError, TypeError, ValueError):
        return None


def parse_fio_result(command: dict[str, Any], cache_state: str, repetition: int | None) -> dict[str, Any]:
    path = ROOT / command["stdout_json"]
    row: dict[str, Any] = {
        "label": command["label"],
        "cache_state": cache_state,
        "repetition": repetition,
        "returncode": command["returncode"],
        "timed_out": command.get("timed_out", False),
        "timeout_s": command.get("timeout_s"),
        "fio_json": command["stdout_json"],
        "fio_stderr": command["stderr"],
        "wall_seconds": command["wall_seconds"],
        "valid": False,
    }
    if command.get("timed_out"):
        row["error"] = "fio timed out before emitting a complete JSON result"
        return row
    if command["returncode"] != 0:
        row["error"] = "fio returned non-zero"
        return row
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        job = payload["jobs"][0]
    except (OSError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        row["error"] = f"fio JSON parse failed: {exc}"
        return row
    read = job.get("read", {})
    write = job.get("write", {})
    read_bw = float(read.get("bw_bytes", 0.0) or 0.0)
    write_bw = float(write.get("bw_bytes", 0.0) or 0.0)
    read_ios = int(read.get("total_ios", 0) or 0)
    write_ios = int(write.get("total_ios", 0) or 0)
    row.update(
        {
            "fio_version": payload.get("fio version"),
            "job_name": job.get("jobname"),
            "read_ios": read_ios,
            "write_ios": write_ios,
            "total_ios": read_ios + write_ios,
            "read_io_bytes": int(read.get("io_bytes", 0) or 0),
            "write_io_bytes": int(write.get("io_bytes", 0) or 0),
            "throughput_mib_s": (read_bw + write_bw) / (1024.0 * 1024.0),
            "read_iops": float(read.get("iops", 0.0) or 0.0),
            "write_iops": float(write.get("iops", 0.0) or 0.0),
            "read_clat_p50_us": (percentile_ns(read, "50.000000") or 0.0) / 1000.0,
            "read_clat_p95_us": (percentile_ns(read, "95.000000") or 0.0) / 1000.0,
            "read_clat_p99_us": (percentile_ns(read, "99.000000") or 0.0) / 1000.0,
            "read_clat_p99_9_us": (percentile_ns(read, "99.900000") or 0.0) / 1000.0,
            "write_clat_p50_us": (percentile_ns(write, "50.000000") or 0.0) / 1000.0,
            "write_clat_p95_us": (percentile_ns(write, "95.000000") or 0.0) / 1000.0,
            "write_clat_p99_us": (percentile_ns(write, "99.000000") or 0.0) / 1000.0,
            "write_clat_p99_9_us": (percentile_ns(write, "99.900000") or 0.0) / 1000.0,
        }
    )
    for pct in ("p50", "p95", "p99", "p99_9"):
        row[f"latency_{pct}_us"] = max(
            float(row[f"read_clat_{pct}_us"]),
            float(row[f"write_clat_{pct}_us"]),
        )
    row["valid"] = row["total_ios"] > 0 and row["throughput_mib_s"] > 0.0
    return row


def summarize_rows(rows: list[dict[str, Any]], profile: dict[str, Any]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("valid") and row.get("cache_state") == "warm"]
    metrics = [
        "throughput_mib_s",
        "latency_p50_us",
        "latency_p95_us",
        "latency_p99_us",
        "latency_p99_9_us",
        "read_clat_p99_us",
        "write_clat_p99_us",
    ]
    ci_contract = profile["confidence_interval_method"]
    summary: dict[str, Any] = {
        "cache_state": "warm",
        "valid_repetitions": len(valid),
        "expected_repetitions": profile["repetition_count"],
        "metrics": {},
        "latency_definition": (
            "latency_pXX_us is the conservative maximum of read and write fio "
            "clat_ns percentile values for the mixed 70/30 workload; read/write "
            "direction percentiles are retained separately."
        ),
    }
    for metric in metrics:
        samples = [float(row[metric]) for row in valid if row.get(metric) is not None]
        if not samples:
            continue
        ci_low, ci_high = bootstrap_ci(
            samples,
            f"aegisq|warm|{metric}",
            trials=int(ci_contract["resamples"]),
            alpha=1.0 - float(ci_contract["confidence_level"]),
        )
        summary["metrics"][metric] = {
            "samples": samples,
            "median": statistics.median(samples),
            "p05": quantile(samples, 0.05),
            "p95": quantile(samples, 0.95),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
        }
    return summary


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "label",
        "cache_state",
        "repetition",
        "valid",
        "returncode",
        "throughput_mib_s",
        "latency_p50_us",
        "latency_p95_us",
        "latency_p99_us",
        "latency_p99_9_us",
        "read_iops",
        "write_iops",
        "read_clat_p99_us",
        "write_clat_p99_us",
        "fio_json",
        "fio_stderr",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def probe_cold_cache_capability(out_dir: Path) -> dict[str, Any]:
    probe_dir = out_dir / "cold_cache_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    sudo_probe = command_capture(["sudo", "-n", "true"], timeout_s=5.0)
    (probe_dir / "sudo_probe.json").write_text(
        json.dumps(sudo_probe, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if os.geteuid() == 0:
        sync_result = command_capture(["sync"], timeout_s=30.0)
        drop_path = Path("/proc/sys/vm/drop_caches")
        try:
            drop_path.write_text("3\n", encoding="ascii")
            drop_result = {"available": True, "returncode": 0, "stderr": "", "stdout": ""}
        except OSError as exc:
            drop_result = {
                "available": True,
                "returncode": 1,
                "stderr": str(exc),
                "stdout": "",
            }
        return {
            "status": "available" if drop_result["returncode"] == 0 else "invalid",
            "sync": sync_result,
            "drop_caches": drop_result,
            "sudo_probe": sudo_probe,
        }
    return {
        "status": "invalid_not_run",
        "reason": (
            "cold-cache contract requires privileged cache dropping after "
            "unmount/remount; this run had no noninteractive sudo/root access"
        ),
        "sudo_probe": sudo_probe,
    }


def storage_snapshot(storage_dir: Path, out_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_bytes = 0
    if storage_dir.exists():
        for path in sorted(storage_dir.rglob("*")):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            rel = str(path.relative_to(storage_dir))
            rows.append({"path": rel, "bytes": stat.st_size})
            total_bytes += stat.st_size
    snapshot = {
        "storage_dir": str(storage_dir),
        "file_count": len(rows),
        "total_file_bytes": total_bytes,
        "files": rows[:200],
        "truncated_to_files": 200,
    }
    snapshot_path = out_dir / "storage_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    snapshot["path"] = relpath(snapshot_path)
    return snapshot


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload["warm_cache_summary"]
    metrics = summary["metrics"]
    lines = [
        "# AEGIS-Q Frozen Workload Contract Run",
        "",
        f"- Overall pass: `{str(payload['overall_pass']).lower()}`",
        f"- Contract ID: `{payload['contract_id']}`",
        f"- Scope: {payload['scope']}",
        f"- File preparation valid: `{str((payload.get('file_preparation') or {}).get('valid')).lower()}`",
        f"- Warm-cache valid repetitions: `{summary['valid_repetitions']}`",
        f"- Cold-cache status: `{payload['cold_cache']['status']}`",
        f"- Comparison ready: `{str(payload['comparison_ready']).lower()}`",
        "",
        "## Warm-Cache Summary",
        "",
    ]
    for metric in (
        "throughput_mib_s",
        "latency_p50_us",
        "latency_p95_us",
        "latency_p99_us",
        "latency_p99_9_us",
    ):
        if metric in metrics:
            row = metrics[metric]
            lines.append(
                f"- `{metric}` median `{row['median']:.6g}`, "
                f"95% CI [`{row['ci95_low']:.6g}`, `{row['ci95_high']:.6g}`]"
            )
    lines.extend(
        [
            "",
            "## Retained Artifacts",
            "",
            f"- JSON summary: `{payload['artifacts']['json']}`",
            f"- CSV repetitions: `{payload['artifacts']['csv']}`",
            f"- File preparation: `{payload['artifacts']['file_preparation']}`",
            f"- Raw fio directory: `{payload['artifacts']['fio_raw_dir']}`",
            f"- Mount logs: `{payload['artifacts']['mount_logs']}`",
            f"- Platform manifest: `{payload['artifacts']['platform_manifest']}`",
            f"- Thermal log: `{payload['artifacts']['thermal_log']}`",
            "",
            "## Non-Claims",
            "",
            "- This is not a gocryptfs/fscrypt/dm-crypt/plaintext comparison.",
            "- The cold-cache row is not reported as a result unless privileged cache dropping is available.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--password", default="aegisq-frozen-contract-password")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--thermal-interval-ms", type=int, default=500)
    parser.add_argument("--fio-timeout-s", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    args.contract = args.contract if args.contract.is_absolute() else ROOT / args.contract
    args.out = args.out if args.out.is_absolute() else ROOT / args.out
    if not args.contract.exists():
        raise SystemExit(f"missing frozen workload contract: {args.contract}")
    if not FUSE_BIN.exists():
        raise SystemExit(f"missing final FUSE binary: {FUSE_BIN}")
    if shutil.which("fio") is None:
        raise SystemExit("fio is required for the frozen workload contract")
    if args.warmup_runs < 1:
        raise SystemExit("--warmup-runs must be at least 1 for the frozen contract")

    contract_payload = load_contract(args.contract)
    profile = workload_profile(contract_payload)
    contract_reps = int(profile["repetition_count"])
    repetitions = contract_reps if args.repetitions is None else args.repetitions
    if repetitions < 1:
        raise SystemExit("--repetitions must be positive")
    if args.out.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.out} exists; pass --overwrite to replace this harness output")
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    platform_path = args.out / "platform_manifest.json"
    platform_data = platform_manifest(args.contract)
    platform_path.write_text(json.dumps(platform_data, indent=2, sort_keys=True), encoding="utf-8")
    self_test = command_capture([str(FUSE_BIN), "--self-test"], timeout_s=60.0)
    (args.out / "pqc_fuse_self_test.json").write_text(
        json.dumps(self_test, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if self_test.get("returncode") != 0:
        raise SystemExit("pqc_fuse --self-test failed; refusing to run benchmark")

    rows: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    abort_reason: str | None = None
    storage_dir = Path(tempfile.mkdtemp(prefix="aegisq_frozen_storage_"))
    mount_dir = Path(tempfile.mkdtemp(prefix="aegisq_frozen_mnt_"))
    fuse: FuseHandle | None = None
    cleanup: dict[str, Any] = {
        "storage_dir": str(storage_dir),
        "mount_dir": str(mount_dir),
        "removed": False,
    }
    thermal_proc, thermal_fp, thermal_status = start_thermal_log(args.out, args.thermal_interval_ms)
    unmount_result: dict[str, Any] | None = None
    backing_snapshot: dict[str, Any] | None = None
    file_preparation: dict[str, Any] | None = None
    try:
        fuse = start_fuse(storage_dir, mount_dir, args.out, args.password)
        bench_dir = mount_dir / "contract"
        bench_dir.mkdir(parents=True, exist_ok=True)
        file_preparation = precreate_fio_file(profile, bench_dir, args.out)
        if not file_preparation.get("valid"):
            abort_reason = "file preparation did not create the contract file"
        argv = fio_command(profile, bench_dir)
        command_template = profile["mount_options"]["command_template"]
        (args.out / "fio_command_template.txt").write_text(command_template + "\n", encoding="utf-8")

        if abort_reason is None:
            for index in range(args.warmup_runs):
                command = run_fio_with_timeout(f"warmup_{index:02d}", argv, args.out, args.fio_timeout_s)
                commands.append(command)
                row = parse_fio_result(command, "warmup", None)
                rows.append(row)
                if not row.get("valid"):
                    abort_reason = f"warmup {index} did not produce a valid fio result"
                    break
        if abort_reason is None:
            for rep in range(repetitions):
                command = run_fio_with_timeout(f"warm_rep_{rep:02d}", argv, args.out, args.fio_timeout_s)
                commands.append(command)
                row = parse_fio_result(command, "warm", rep)
                rows.append(row)
                if not row.get("valid"):
                    abort_reason = f"measured repetition {rep} did not produce a valid fio result"
                    break
    finally:
        unmount_result = stop_fuse(fuse, mount_dir, args.out)
        backing_snapshot = storage_snapshot(storage_dir, args.out)
        thermal_status = stop_thermal_log(thermal_proc, thermal_fp, thermal_status)
        if not args.keep_temp:
            shutil.rmtree(storage_dir, ignore_errors=True)
            shutil.rmtree(mount_dir, ignore_errors=True)
            cleanup["removed"] = True

    cold_cache = probe_cold_cache_capability(args.out)
    csv_path = args.out / "frozen_aegisq_repetitions.csv"
    write_csv(rows, csv_path)
    summary = summarize_rows(rows, profile)
    valid_warm = [
        row for row in rows
        if row.get("cache_state") == "warm" and row.get("valid")
    ]
    contract_repetition_match = repetitions == contract_reps
    warm_cache_pass = len(valid_warm) == contract_reps and contract_repetition_match
    manifest = {
        "methodology_id": "aegisq-frozen-workload-contract-v2",
        "warmup": {
            "warmup_runs": args.warmup_runs,
            "full_fio_profile_warmup": args.warmup_runs >= 1,
        },
        "file_preparation": {
            "required": profile["mount_options"].get("file_preparation"),
            "observed": file_preparation,
        },
        "run_count": {
            "measured_repetitions": repetitions,
            "contract_repetitions": contract_reps,
            "matches_contract": contract_repetition_match,
        },
        "confidence_interval_method": profile["confidence_interval_method"],
        "outlier_policy": {
            "policy": "retain_all_completed_repetitions",
            "infrastructure_failure_policy": "fail/mark invalid with raw log and return code",
            "winsorization": "disabled",
        },
        "cache_state_policy": {
            "warm": "one full fio warmup pass followed by measured repetitions without cache dropping",
            "cold": "invalid unless privileged drop_caches after unmount/remount is available",
        },
        "failure_handling": {
            "missing_binary": "fatal",
            "fio_failure": "invalid row with raw stdout/stderr and return code",
            "mount_failure": "fatal",
            "unsupported_cold_cache": "reported as invalid_not_run, not folded into warm-cache results",
            "fio_timeout": "invalid row with retained command log and backing-store snapshot",
        },
    }
    payload = {
        "overall_pass": warm_cache_pass,
        "contract_compliant_warm_cache": warm_cache_pass,
        "comparison_ready": False,
        "scope": (
            "AEGIS-Q-only warm-cache execution of the frozen filesystem "
            "workload contract.  Baseline modes and valid cold-cache rows are "
            "separate checklist items."
        ),
        "non_claims": [
            "not a plaintext/gocryptfs/fscrypt/dm-crypt comparison",
            "not a cold-cache result when drop_caches is unavailable",
            "not a QoS or SQLite hero result",
        ],
        "contract_id": contract_payload["contract"]["contract_id"],
        "contract_sha256_recorded": contract_payload.get("contract_sha256"),
        "contract_path_sha256": platform_data["contract"]["sha256"],
        "filesystem_mode": "aegis_q",
        "workload_profile": profile["profile_id"],
        "fio_command": fio_command(profile, Path("${AEGIS_MOUNT}") / "contract"),
        "methodology": manifest,
        "file_preparation": file_preparation,
        "platform_manifest": relpath(platform_path),
        "thermal_logging": thermal_status,
        "cold_cache": cold_cache,
        "warm_cache_summary": summary,
        "abort_reason": abort_reason,
        "repetitions": rows,
        "commands": commands,
        "unmount": unmount_result,
        "cleanup": cleanup,
        "storage_snapshot": backing_snapshot,
        "artifacts": {
            "json": relpath(args.out / "frozen_aegisq_contract.json"),
            "csv": relpath(csv_path),
            "markdown": relpath(args.out / "frozen_aegisq_contract.md"),
            "fio_raw_dir": relpath(args.out / "fio_raw"),
            "file_preparation": relpath(args.out / "file_preparation.json"),
            "mount_logs": relpath(args.out / "mount_logs"),
            "platform_manifest": relpath(platform_path),
            "storage_snapshot": relpath(args.out / "storage_snapshot.json"),
            "thermal_log": thermal_status.get("path"),
        },
    }
    json_path = args.out / "frozen_aegisq_contract.json"
    md_path = args.out / "frozen_aegisq_contract.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(build_markdown(payload), encoding="utf-8")

    print(
        json.dumps(
            {
                "overall_pass": payload["overall_pass"],
                "json": relpath(json_path),
                "markdown": relpath(md_path),
                "csv": relpath(csv_path),
                "warm_valid_repetitions": summary["valid_repetitions"],
                "cold_cache_status": cold_cache["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
