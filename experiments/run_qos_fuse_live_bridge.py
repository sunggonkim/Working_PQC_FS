#!/usr/bin/env python3
"""Run a mounted-FUSE writer under live telemetry and a retained admission path.

This harness is intentionally conservative:

  * It mounts the real FUSE filesystem and writes through it.
  * It samples live `tegrastats` while optional GPU pressure is active.
  * The mounted FUSE process reads the same live samples through
    `PQC_TELEMETRY_FILE`, so the real write flush path calls `pqc_admit()` with
    workload-time pressure rather than a separate smoke-process state.
  * The same live samples are also replayed through
    `pqc_fuse --admission-telemetry-smoke` to retain per-sample decision
    diagnostics.

The result is stronger than a pure smoke test because real FUSE routing changes
inside the mounted daemon in the same execution. It is still not a full
PMU/CUPTI/Nsight-backed controller proof; the live source is `tegrastats`.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
FUSE_BIN = ROOT / "build" / "pqc_fuse"
GPU_BURNER = ROOT / "experiments" / "gpu_burner"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "qos_fuse_live_bridge"

GPU_POWER_RE = re.compile(r"VDD_GPU\s+(\d+)mW")
GR3D_RE = re.compile(r"GR3D_FREQ\s+(?:(\d+)%|@)")


class HysteresisController:
    def __init__(self, enter_threshold: float, exit_threshold: float, hold_samples: int = 2):
        if exit_threshold > enter_threshold:
            raise ValueError("exit_threshold must be <= enter_threshold")
        self.enter_threshold = float(enter_threshold)
        self.exit_threshold = float(exit_threshold)
        self.hold_samples = max(1, int(hold_samples))
        self.state = "open"
        self._below_exit_count = 0

    def update(self, value: float) -> dict[str, Any]:
        value = float(value)
        event = "hold"
        if self.state == "open":
            self._below_exit_count = 0
            if value >= self.enter_threshold:
                self.state = "throttled"
                event = "enter"
        else:
            if value <= self.exit_threshold:
                self._below_exit_count += 1
                if self._below_exit_count >= self.hold_samples:
                    self.state = "open"
                    self._below_exit_count = 0
                    event = "exit"
            else:
                self._below_exit_count = 0
        return {
            "state": self.state,
            "event": event,
            "throttle": 1 if self.state == "throttled" else 0,
            "pressure_value": value,
            "enter_threshold": self.enter_threshold,
            "exit_threshold": self.exit_threshold,
            "hold_samples": self.hold_samples,
            "below_exit_count": self._below_exit_count,
        }


def parse_tegrastats(line: str) -> dict[str, Any]:
    gpu_power = None
    gr3d = None
    m = GPU_POWER_RE.search(line)
    if m:
        gpu_power = int(m.group(1))
    m = GR3D_RE.search(line)
    if m and m.group(1) is not None:
        gr3d = int(m.group(1))
    return {
        "raw": line.rstrip("\n"),
        "gpu_power_mw": gpu_power,
        "gr3d_percent": gr3d,
        "mem_bandwidth_util": min(1.0, max(0.0, (gpu_power or 0) / 35000.0)),
        "tensor_core_util": min(1.0, max(0.0, (gr3d or 0) / 100.0)),
    }


def write_runtime_telemetry(path: Path,
                            telemetry: dict[str, Any],
                            budget_ns: int = 2_000_000,
                            queue_depth: int = 0) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        f"{telemetry['mem_bandwidth_util']:.6f} "
        f"{telemetry['tensor_core_util']:.6f} "
        f"{budget_ns:d} {queue_depth:d}\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def start_fuse(storage_dir: Path,
               mount_dir: Path,
               log_dir: Path,
               telemetry_file: Path,
               runtime_trace: Path,
               throttle_trace: Path) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["PQC_MASTER_PASSWORD"] = env.get("PQC_MASTER_PASSWORD", "test-password")
    env["PQC_ENABLE_ADMISSION_ON_WRITE"] = "1"
    env["PQC_ADMISSION_TRACE_PATH"] = str(runtime_trace)
    env["PQC_TELEMETRY_FILE"] = str(telemetry_file)
    env["PQC_TELEMETRY_POLL_MS"] = env.get("PQC_TELEMETRY_POLL_MS", "25")
    env["PQC_ADMISSION_INITIAL_BUDGET_NS"] = env.get("PQC_ADMISSION_INITIAL_BUDGET_NS", "2000000")
    env["PQC_ADMISSION_WRITE_DEADLINE_NS"] = env.get("PQC_ADMISSION_WRITE_DEADLINE_NS", "10000000")
    env["PQC_ENABLE_QOS_THROTTLE_ON_WRITE"] = "1"
    env["PQC_QOS_THROTTLE_TRACE_PATH"] = str(throttle_trace)
    env["PQC_QOS_THROTTLE_SLEEP_US"] = env.get("PQC_QOS_THROTTLE_SLEEP_US", "50000")
    stdout_fp = (log_dir / "pqc_fuse.stdout.txt").open("w", encoding="utf-8")
    stderr_fp = (log_dir / "pqc_fuse.stderr.txt").open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [str(FUSE_BIN), str(storage_dir), str(mount_dir), "-f"],
        cwd=ROOT,
        env=env,
        stdout=stdout_fp,
        stderr=stderr_fp,
        text=True,
    )
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if subprocess.run(["mountpoint", "-q", str(mount_dir)], check=False).returncode == 0:
            return proc
        if proc.poll() is not None:
            raise RuntimeError(f"FUSE exited before mount: rc={proc.returncode}")
        time.sleep(0.05)
    raise TimeoutError("timed out waiting for FUSE mount")


def stop_fuse(proc: subprocess.Popen[str] | None, mount_dir: Path) -> None:
    subprocess.run(
        ["fusermount3", "-u", str(mount_dir)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if proc is not None and proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def writer_worker(mount_dir: str,
                  stop_flag: mp.Event,
                  throttle_flag: mp.Value,
                  result_queue: mp.Queue,
                  chunk_bytes: int,
                  fsync_every: int) -> None:
    path = Path(mount_dir) / "stream.bin"
    payload = b"W" * chunk_bytes
    chunks_written = 0
    throttle_sleeps = 0
    sleep_time_s = 0.0
    started = time.monotonic()
    with path.open("wb", buffering=0) as f:
        while not stop_flag.is_set():
            if throttle_flag.value:
                throttle_sleeps += 1
                time.sleep(0.05)
                sleep_time_s += 0.05
                continue
            f.write(payload)
            chunks_written += 1
            if fsync_every > 0 and (chunks_written % fsync_every) == 0:
                f.flush()
                os.fsync(f.fileno())
        f.flush()
        os.fsync(f.fileno())
    elapsed = max(1e-9, time.monotonic() - started)
    result_queue.put({
        "chunks_written": chunks_written,
        "bytes_written": chunks_written * chunk_bytes,
        "throttle_sleeps": throttle_sleeps,
        "sleep_time_s": sleep_time_s,
        "elapsed_s": elapsed,
        "throughput_mb_s": (chunks_written * chunk_bytes) / (1024.0 * 1024.0 * elapsed),
    })


def run_admission_sample(out_dir: Path, idx: int, telemetry: dict[str, Any]) -> dict[str, Any]:
    trace = out_dir / f"sample_{idx:02d}.jsonl"
    stdout = out_dir / f"sample_{idx:02d}.stdout"
    stderr = out_dir / f"sample_{idx:02d}.stderr"
    env = os.environ.copy()
    env.update({
        "PQC_ADMISSION_TRACE_PATH": str(trace),
        "PQC_TELEMETRY_MEM_BANDWIDTH": f"{telemetry['mem_bandwidth_util']:.6f}",
        "PQC_TELEMETRY_TENSOR_CORE": f"{telemetry['tensor_core_util']:.6f}",
        "PQC_ADMISSION_SMOKE_AI_BUDGET_NS": "2000000",
        "PQC_ADMISSION_SMOKE_CPU_QUEUE_DEPTH": "1",
        "PQC_ADMISSION_SMOKE_GPU_QUEUE_DEPTH": "1",
        "PQC_ADMISSION_SMOKE_BYTES": "131072",
        "PQC_ADMISSION_SMOKE_GPU_KERNEL_NS": "100000",
        "PQC_ADMISSION_SMOKE_H2D_NS": "100000",
        "PQC_ADMISSION_SMOKE_D2H_NS": "100000",
    })
    proc = subprocess.run(
        [str(FUSE_BIN), "--admission-telemetry-smoke"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout.write_text(proc.stdout, encoding="utf-8")
    stderr.write_text(proc.stderr, encoding="utf-8")
    summary = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            summary = json.loads(line)
            break
    trace_rows = []
    if trace.exists():
        for line in trace.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                trace_rows.append(json.loads(line))
    return {
        "sample": idx,
        "returncode": proc.returncode,
        "telemetry": telemetry,
        "summary": summary,
        "trace": str(trace.relative_to(ROOT)),
        "stdout": str(stdout.relative_to(ROOT)),
        "stderr": str(stderr.relative_to(ROOT)),
        "trace_rows": trace_rows,
    }


def load_runtime_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def terminate(proc: subprocess.Popen[Any] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--interval-ms", type=int, default=250)
    ap.add_argument("--chunk-bytes", type=int, default=131072)
    ap.add_argument("--fsync-every", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--start-gpu-burner", action="store_true", default=True)
    ap.add_argument("--no-gpu-burner", dest="start_gpu_burner", action="store_false")
    args = ap.parse_args()

    if not FUSE_BIN.exists():
        raise SystemExit("missing build/pqc_fuse; run cmake --build build first")

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    storage_dir = Path(tempfile.mkdtemp(prefix="aegis_qos_store_"))
    mount_dir = Path(tempfile.mkdtemp(prefix="aegis_qos_mnt_"))
    mount_log_dir = out_dir / "mount_logs"
    mount_log_dir.mkdir(parents=True, exist_ok=True)
    telemetry_file = out_dir / "runtime_telemetry.txt"
    runtime_trace = out_dir / "runtime_fuse_admission_trace.jsonl"
    throttle_trace = out_dir / "runtime_fuse_throttle_trace.jsonl"
    write_runtime_telemetry(
        telemetry_file,
        {
            "mem_bandwidth_util": 0.0,
            "tensor_core_util": 0.0,
        },
    )

    ctx = mp.get_context("spawn")
    stop_flag = ctx.Event()
    throttle_flag = ctx.Value("i", 0)
    result_queue: mp.Queue = ctx.Queue()

    fuse_proc: subprocess.Popen[str] | None = None
    burner: subprocess.Popen[Any] | None = None
    tegra: subprocess.Popen[str] | None = None
    writer_proc: mp.Process | None = None
    samples: list[dict[str, Any]] = []
    try:
        fuse_proc = start_fuse(storage_dir, mount_dir, mount_log_dir,
                               telemetry_file, runtime_trace, throttle_trace)
        writer_proc = ctx.Process(
            target=writer_worker,
            args=(str(mount_dir), stop_flag, throttle_flag, result_queue,
                  args.chunk_bytes, args.fsync_every),
            daemon=True,
        )
        writer_proc.start()

        if args.start_gpu_burner and GPU_BURNER.exists():
            burner = subprocess.Popen(
                [str(GPU_BURNER), str(max(4, args.samples))],
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(0.5)

        tegra = subprocess.Popen(
            ["sudo", "-S", "tegrastats", "--interval", str(args.interval_ms)],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert tegra.stdin is not None
        sudo_password = os.environ.get("PQC_SUDO_PASSWORD")
        if not sudo_password:
            raise RuntimeError("PQC_SUDO_PASSWORD must be set for sudo tegrastats collection")
        tegra.stdin.write(sudo_password + "\n")
        tegra.stdin.flush()
        assert tegra.stdout is not None

        controller = HysteresisController(
            enter_threshold=float(os.environ.get("PQC_QOS_GPU_POWER_ENTER_MW", "15000")),
            exit_threshold=float(os.environ.get("PQC_QOS_GPU_POWER_EXIT_MW", "9000")),
            hold_samples=int(os.environ.get("PQC_QOS_HOLD_SAMPLES", "2")),
        )

        for idx in range(args.samples):
            line = tegra.stdout.readline()
            if not line:
                break
            telemetry = parse_tegrastats(line)
            write_runtime_telemetry(telemetry_file, telemetry)
            admission = run_admission_sample(out_dir, idx + 1, telemetry)
            decision = controller.update(telemetry["gpu_power_mw"] or 0)
            # Do not pause the writer in the harness.  The retained claim is
            # now the in-daemon FUSE throttle path, so the mounted writer must
            # continue issuing writes while the daemon applies any delay.
            throttle_flag.value = 0
            admission["throttle_decision"] = decision
            samples.append(admission)
    finally:
        stop_flag.set()
        if writer_proc is not None:
            writer_proc.join(timeout=10)
        terminate(tegra)
        terminate(burner)
        stop_fuse(fuse_proc, mount_dir)

    writer_stats: dict[str, Any] = {}
    if not result_queue.empty():
        writer_stats = result_queue.get()

    decision_counts: dict[str, int] = {}
    throttle_counts: dict[str, int] = {"open": 0, "throttled": 0}
    for sample in samples:
        target = (sample.get("summary") or {}).get("chosen_target", "missing")
        decision_counts[target] = decision_counts.get(target, 0) + 1
        state = sample["throttle_decision"]["state"]
        throttle_counts[state] = throttle_counts.get(state, 0) + 1

    runtime_rows = load_runtime_trace(runtime_trace)
    throttle_rows = load_runtime_trace(throttle_trace)
    runtime_decision_counts: dict[str, int] = {}
    runtime_low_pressure_targets: dict[str, int] = {}
    runtime_high_pressure_targets: dict[str, int] = {}
    for row in runtime_rows:
        target = row.get("chosen_target", "missing")
        runtime_decision_counts[target] = runtime_decision_counts.get(target, 0) + 1
        mem_util = float(row.get("telemetry_mem_bandwidth_util", 0.0))
        if mem_util >= 0.70:
            runtime_high_pressure_targets[target] = runtime_high_pressure_targets.get(target, 0) + 1
        else:
            runtime_low_pressure_targets[target] = runtime_low_pressure_targets.get(target, 0) + 1
    throttle_decision_counts = {
        "open": sum(1 for row in throttle_rows if int(row.get("throttled", 0)) == 0),
        "throttled": sum(1 for row in throttle_rows if int(row.get("throttled", 0)) != 0),
    }
    throttle_sleep_us_total = sum(int(row.get("sleep_us", 0)) for row in throttle_rows)

    report = {
        "note": "Mounted-FUSE workload plus in-daemon live tegrastats telemetry. This is same-run FUSE throttle evidence, not a PMU/CUPTI/Nsight-backed controller proof.",
        "samples_requested": args.samples,
        "samples_recorded": len(samples),
        "gpu_burner_started": bool(args.start_gpu_burner and GPU_BURNER.exists()),
        "smoke_decision_counts": decision_counts,
        "throttle_state_counts": throttle_counts,
        "runtime_decision_counts": runtime_decision_counts,
        "runtime_low_pressure_targets": runtime_low_pressure_targets,
        "runtime_high_pressure_targets": runtime_high_pressure_targets,
        "runtime_trace_rows": len(runtime_rows),
        "runtime_throttle_trace_rows": len(throttle_rows),
        "runtime_throttle_counts": throttle_decision_counts,
        "runtime_throttle_sleep_us_total": throttle_sleep_us_total,
        "writer_stats": writer_stats,
        "runtime_telemetry_file": str(telemetry_file.relative_to(ROOT)),
        "runtime_trace": str(runtime_trace.relative_to(ROOT)),
        "runtime_throttle_trace": str(throttle_trace.relative_to(ROOT)),
        "mount_logs": {
            "stdout": str((mount_log_dir / "pqc_fuse.stdout.txt").relative_to(ROOT)),
            "stderr": str((mount_log_dir / "pqc_fuse.stderr.txt").relative_to(ROOT)),
        },
        "samples": samples,
    }
    json_path = out_dir / "qos_fuse_live_bridge.json"
    md_path = out_dir / "qos_fuse_live_bridge.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# QoS mounted-FUSE live bridge",
        "",
        "This bundle writes through a real FUSE mount while sampling live `tegrastats`.",
        "The mounted daemon reads the live sample stream through `PQC_TELEMETRY_FILE`; the real write flush path applies telemetry-derived throttle and writes `runtime_fuse_throttle_trace.jsonl`.",
        "The DATA plane is structurally CPU-only in this codebase, so `runtime_fuse_admission_trace.jsonl` can remain empty without invalidating the throttle result.",
        "The per-sample smoke process is retained only as a diagnostic replay of the same telemetry values.",
        "This is stronger than a pure smoke path because real FUSE flushes are delayed inside the mounted daemon, but it is still not a PMU/CUPTI/Nsight-backed controller proof.",
        "",
        f"- Samples recorded: `{len(samples)}` / requested `{args.samples}`",
        f"- GPU burner started: `{report['gpu_burner_started']}`",
        f"- Smoke admission decision counts: `{decision_counts}`",
        f"- Runtime FUSE admission decision counts: `{runtime_decision_counts}`",
        f"- Runtime low-pressure targets: `{runtime_low_pressure_targets}`",
        f"- Runtime high-pressure targets: `{runtime_high_pressure_targets}`",
        f"- Runtime trace rows: `{len(runtime_rows)}`",
        f"- Runtime throttle counts: `{throttle_decision_counts}`",
        f"- Runtime throttle trace rows: `{len(throttle_rows)}`",
        f"- Runtime throttle sleep total: `{throttle_sleep_us_total}` us",
        f"- Throttle state counts: `{throttle_counts}`",
        f"- Writer stats: `{writer_stats}`",
        f"- Runtime trace: `{report['runtime_trace']}`",
        f"- Runtime throttle trace: `{report['runtime_throttle_trace']}`",
        f"- FUSE stderr: `{report['mount_logs']['stderr']}`",
        "",
        "| sample | gpu_power_mw | gr3d_percent | mem_util | tensor_util | admission_target | throttle_state | trace |",
        "|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for sample in samples:
        telemetry = sample["telemetry"]
        summary = sample.get("summary") or {}
        throttle = sample["throttle_decision"]
        lines.append(
            f"| {sample['sample']} | {telemetry.get('gpu_power_mw')} | {telemetry.get('gr3d_percent')} | "
            f"{telemetry['mem_bandwidth_util']:.4f} | {telemetry['tensor_core_util']:.4f} | "
            f"{summary.get('chosen_target')} | {throttle.get('state')} | `{sample['trace']}` |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "out_dir": str(out_dir),
        "samples": len(samples),
        "writer_bytes": writer_stats.get("bytes_written", 0),
        "throttle_counts": throttle_counts,
    }, indent=2))

    shutil.rmtree(storage_dir, ignore_errors=True)
    shutil.rmtree(mount_dir, ignore_errors=True)
    return 0 if samples else 1


if __name__ == "__main__":
    raise SystemExit(main())
