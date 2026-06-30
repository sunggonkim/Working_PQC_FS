#!/usr/bin/env python3
"""Run a kernel-control SQLite QoS baseline on the mounted AEGIS-Q path.

This is the first Gate B3 baseline runner.  It keeps the SQLite foreground and
secure-storage background writer shape from ``run_qos_sqlite_hero_bundle.py``,
but replaces the AEGIS-Q/harness controller with a standard Linux control:
``ionice`` applied only to the background writer process.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import run_qos_sqlite_hero_bundle as hero


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "sqlite_kernel_qos_baseline"


def metric_summary(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return {"samples": 0, "median": None, "min": None, "max": None}
    return {
        "samples": len(samples),
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
    }


def rel(path: Path) -> str:
    return hero.relpath(path)


def read_kernel_control_rows(mode: dict[str, Any]) -> list[dict[str, Any]]:
    log_path = ROOT / mode["logs"]["background_jsonl"]
    rows = hero.load_jsonl(log_path)
    return [row for row in rows if row.get("event") == "writer_start"]


def summarize_kernel_control(modes: list[dict[str, Any]]) -> dict[str, Any]:
    start_rows: list[dict[str, Any]] = []
    for mode in modes:
        start_rows.extend(read_kernel_control_rows(mode))
    apply_results = [
        ((row.get("kernel_qos") or {}).get("apply_result") or {})
        for row in start_rows
    ]
    return {
        "control": "ionice",
        "writer_start_rows": len(start_rows),
        "applied_rows": sum(1 for result in apply_results if result.get("enabled") is True),
        "successful_rows": sum(1 for result in apply_results if result.get("returncode") == 0),
        "failed_rows": sum(1 for result in apply_results if result.get("enabled") is True and result.get("returncode") != 0),
        "missing_rows": sum(1 for result in apply_results if result.get("enabled") is not True),
        "details": apply_results,
    }


def read_current_cgroup() -> dict[str, Any]:
    try:
        text = Path("/proc/self/cgroup").read_text(encoding="ascii").strip()
    except OSError as exc:
        return {"available": False, "error": str(exc)}
    path = ""
    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0":
            path = parts[2]
            break
    full = Path("/sys/fs/cgroup") / path.lstrip("/")
    io_weight = None
    if full.exists():
        try:
            io_weight = (full / "io.weight").read_text(encoding="ascii").strip()
        except OSError:
            io_weight = None
    return {
        "available": True,
        "proc_self_cgroup": text,
        "cgroup_v2_path": path,
        "cgroup_fs_path": str(full),
        "io_weight": io_weight,
    }


def parse_systemd_scope_unit(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Running as unit:"):
            rest = line.split(":", 1)[1].strip()
            return rest.split(";", 1)[0].strip()
    return None


def systemctl_user_show(unit: str | None) -> dict[str, Any]:
    if not unit:
        return {"available": False, "reason": "unit_not_observed"}
    if shutil.which("systemctl") is None:
        return {"available": False, "reason": "systemctl_missing", "unit": unit}
    argv = [
        "systemctl",
        "--user",
        "show",
        unit,
        "-p",
        "IOWeight",
        "-p",
        "ControlGroup",
        "-p",
        "Slice",
        "-p",
        "Delegate",
    ]
    proc = subprocess.run(
        argv,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    values: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return {
        "available": True,
        "unit": unit,
        "argv": argv,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "properties": values,
    }


def systemd_worker_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mount-dir", type=Path, required=True)
    parser.add_argument("--writer-id", type=int, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--chunk-bytes", type=int, required=True)
    parser.add_argument("--fsync-every", type=int, required=True)
    parser.add_argument("--io-weight", type=int, required=True)
    args = parser.parse_args(argv)

    path = args.mount_dir / f"background_{args.writer_id}.bin"
    path.touch(exist_ok=True)
    hero.set_qos_class(path, "elastic")
    payload = bytes([65 + (args.writer_id % 26)]) * args.chunk_bytes
    chunks = 0
    bytes_written = 0
    start_ns = hero.realtime_ns()
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    with args.log_path.open("a", encoding="utf-8", buffering=1) as log, \
            path.open("wb", buffering=0) as fp:
        hero.json_dump_line(log, {
            "event": "writer_start",
            "timestamp_ns": start_ns,
            "writer_id": args.writer_id,
            "chunk_bytes": args.chunk_bytes,
            "fsync_every": args.fsync_every,
            "kernel_qos": {
                "control": "systemd_io_weight",
                "io_weight": args.io_weight,
                "apply_result": {
                    "enabled": True,
                    "method": "systemd-run --user --scope -p IOWeight",
                    "runtime_cgroup": read_current_cgroup(),
                },
            },
        })
        while not args.stop_file.exists():
            t0 = hero.realtime_ns()
            fp.write(payload)
            chunks += 1
            bytes_written += args.chunk_bytes
            fsync_done = False
            if args.fsync_every > 0 and chunks % args.fsync_every == 0:
                fp.flush()
                os.fsync(fp.fileno())
                fsync_done = True
            hero.json_dump_line(log, {
                "event": "write",
                "timestamp_ns": t0,
                "writer_id": args.writer_id,
                "chunk_index": chunks,
                "bytes": args.chunk_bytes,
                "bytes_written": bytes_written,
                "fsync": fsync_done,
            })
        fp.flush()
        os.fsync(fp.fileno())
        elapsed_ns = max(1, hero.realtime_ns() - start_ns)
        hero.json_dump_line(log, {
            "event": "writer_stop",
            "timestamp_ns": hero.realtime_ns(),
            "writer_id": args.writer_id,
            "chunks_written": chunks,
            "bytes_written": bytes_written,
            "elapsed_ns": elapsed_ns,
            "throughput_mb_s": bytes_written / (1024.0 * 1024.0) / (elapsed_ns / 1_000_000_000.0),
            "kernel_qos": {
                "control": "systemd_io_weight",
                "io_weight": args.io_weight,
                "runtime_cgroup": read_current_cgroup(),
            },
        })
    return 0


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# SQLite Kernel QoS Baseline",
        "",
        f"- Overall pass: `{str(report['overall_pass']).lower()}`",
        f"- Verdict: `{report['verdict']}`",
        f"- Control: `{report['kernel_control']['control']}`",
        f"- Repetitions: `{report['repetitions']}`",
        "- Scope: first kernel-control baseline for SQLite foreground p99 under mounted secure-storage pressure.",
        "- Non-claim: this is not yet the full Gate B3 two-baseline comparison.",
        "",
        "## Summary",
        "",
        f"- Foreground p99 median ms: `{report['summary']['foreground_p99_ms']['median']}`",
        f"- Deadline misses median: `{report['summary']['deadline_misses']['median']}`",
        f"- Background throughput median MB/s: `{report['summary']['background_throughput_mb_s']['median']}`",
        f"- Kernel-control successful rows: `{report['kernel_control']['successful_rows']}/{report['kernel_control']['writer_start_rows']}`",
        "",
        "## Repetition Rows",
        "",
        "| repetition | acceptable | p99 ms | deadline misses | bg MB/s |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['repetition']} | `{str(row['acceptable']).lower()}` | "
            f"{row['foreground_p99_ms']} | {row['deadline_misses']} | "
            f"{row['background_throughput_mb_s']} |"
        )
    lines.extend(["", "## Raw Logs", ""])
    for row in report["rows"]:
        lines.append(f"- repetition `{row['repetition']}`: `{row['mode_summary']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_hero_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        deadline_ms=args.deadline_ms,
        telemetry_poll_ms=args.telemetry_poll_ms,
        daemon_throttle_sleep_us=args.daemon_throttle_sleep_us,
        enter_util=args.enter_util,
        exit_util=args.exit_util,
        hold_samples=args.hold_samples,
        telemetry_window=args.telemetry_window,
        transactions=args.transactions,
        rows_per_txn=args.rows_per_txn,
        sqlite_payload_bytes=args.sqlite_payload_bytes,
        inter_transaction_sleep_ms=args.inter_transaction_sleep_ms,
        background_writers=args.background_writers,
        background_chunk_bytes=args.background_chunk_bytes,
        background_fsync_every=args.background_fsync_every,
        background_warmup_ms=args.background_warmup_ms,
        telemetry_interval_ms=args.telemetry_interval_ms,
        post_foreground_drain_ms=args.post_foreground_drain_ms,
        controller_warmup_transactions=args.controller_warmup_transactions,
        background_pressure_util=args.background_pressure_util,
        _background_pressure_pattern_values=hero.parse_pressure_pattern(args.background_pressure_pattern),
        harness_throttle_sleep_us=args.harness_throttle_sleep_us,
        require_daemon_throttle=False,
        background_ionice_class=args.ionice_class,
        background_ionice_level=args.ionice_level,
    )


def blocked_report(args: argparse.Namespace, reasons: list[str]) -> dict[str, Any]:
    return {
        "artifact": "sqlite_kernel_qos_baseline",
        "overall_pass": False,
        "verdict": "environment-blocked",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "blocking_reasons": reasons,
        "scope": [
            "No SQLite kernel QoS baseline was measured in this run.",
            "Do not compare AEGIS-Q against kernel QoS from this artifact.",
        ],
        "kernel_control": {
            "control": args.control,
            "available": shutil.which("ionice") is not None,
            "class": args.ionice_class,
            "level": args.ionice_level,
            "systemd_run_available": shutil.which("systemd-run") is not None,
        },
        "platform": hero.platform_manifest(),
        "rows": [],
        "summary": {
            "foreground_p99_ms": metric_summary([]),
            "deadline_misses": metric_summary([]),
            "background_throughput_mb_s": metric_summary([]),
        },
    }


def run_systemd_io_weight_repetition(
    out_dir: Path,
    args: argparse.Namespace,
    hero_args: SimpleNamespace,
) -> dict[str, Any]:
    mode_name = "kernel_systemd_io_weight"
    mode_dir = out_dir / mode_name
    if mode_dir.exists():
        shutil.rmtree(mode_dir)
    mode_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="aegis_kernel_qos_cgroup_"))
    storage_dir = tmp / "store"
    mount_dir = tmp / "mnt"
    storage_dir.mkdir()
    mount_dir.mkdir()
    telemetry_file = mode_dir / "runtime_telemetry.txt"
    admission_trace = mode_dir / "runtime_fuse_admission_trace.jsonl"
    throttle_trace = mode_dir / "runtime_fuse_throttle_trace.jsonl"
    sqlite_log = mode_dir / "foreground_sqlite_latency.jsonl"
    writer_log = mode_dir / "background_writer.jsonl"
    stop_file = mode_dir / "background_writer.stop"
    systemd_stdout = mode_dir / "systemd_run.stdout.txt"
    systemd_stderr = mode_dir / "systemd_run.stderr.txt"
    hero.write_runtime_telemetry(
        telemetry_file,
        0.10,
        0.10,
        int(args.deadline_ms * 1_000_000),
        0,
    )
    config = hero.ModeConfig(
        name=mode_name,
        background_writer=True,
        daemon_throttle=False,
        harness_throttle=False,
        controller_label="systemd_io_weight",
    )
    fuse: hero.FuseHandle | None = None
    proc: subprocess.Popen[str] | None = None
    error: str | None = None
    unit_probe: dict[str, Any] = {"available": False, "reason": "not_started"}
    foreground: dict[str, Any]
    try:
        fuse = hero.start_fuse(
            storage_dir,
            mount_dir,
            mode_dir,
            telemetry_file,
            admission_trace,
            throttle_trace,
            config,
            hero_args,
        )
        argv = [
            "systemd-run",
            "--user",
            "--scope",
            "-p",
            f"IOWeight={args.cgroup_io_weight}",
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-background-writer",
            "--mount-dir",
            str(mount_dir),
            "--writer-id",
            "0",
            "--stop-file",
            str(stop_file),
            "--log-path",
            str(writer_log),
            "--chunk-bytes",
            str(args.background_chunk_bytes),
            "--fsync-every",
            str(args.background_fsync_every),
            "--io-weight",
            str(args.cgroup_io_weight),
        ]
        with systemd_stdout.open("w", encoding="utf-8") as stdout, \
                systemd_stderr.open("w", encoding="utf-8") as stderr:
            proc = subprocess.Popen(
                argv,
                cwd=ROOT,
                text=True,
                stdout=stdout,
                stderr=stderr,
            )
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                rows = hero.load_jsonl(writer_log)
                if any(row.get("event") == "writer_start" for row in rows):
                    break
                if proc.poll() is not None:
                    break
                time.sleep(0.05)
            systemd_text = ""
            if systemd_stderr.exists():
                systemd_text += systemd_stderr.read_text(encoding="utf-8", errors="replace")
            if systemd_stdout.exists():
                systemd_text += "\n" + systemd_stdout.read_text(encoding="utf-8", errors="replace")
            unit_probe = systemctl_user_show(parse_systemd_scope_unit(systemd_text))
            if args.background_warmup_ms > 0:
                time.sleep(args.background_warmup_ms / 1000.0)
            latencies: deque[float] = deque(maxlen=max(hero_args.transactions * 2, 64))
            foreground = hero.run_sqlite_foreground(
                mount_dir,
                sqlite_log,
                latencies,
                threading.Lock(),
                hero_args,
            )
            time.sleep(args.post_foreground_drain_ms / 1000.0)
            stop_file.write_text("stop\n", encoding="ascii")
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5.0)
    except Exception as exc:
        error = str(exc)
        foreground = {
            "samples": 0,
            "rows_written": 0,
            "row_count": 0,
            "integrity_check": "error",
            "deadline_misses": 0,
            "latency_log": rel(sqlite_log),
            "error": error,
        }
    finally:
        if not stop_file.exists():
            stop_file.write_text("stop\n", encoding="ascii")
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)
        hero.stop_fuse(fuse, mount_dir)
        shutil.rmtree(tmp, ignore_errors=True)

    writer_rows = hero.load_jsonl(writer_log)
    background = hero.summarize_background(writer_rows)
    start_rows = [row for row in writer_rows if row.get("event") == "writer_start"]
    systemd_stdout_text = systemd_stdout.read_text(encoding="utf-8", errors="replace") if systemd_stdout.exists() else ""
    systemd_stderr_text = systemd_stderr.read_text(encoding="utf-8", errors="replace") if systemd_stderr.exists() else ""
    acceptable = (
        error is None
        and foreground.get("samples") == args.transactions
        and foreground.get("integrity_check") == "ok"
        and background["bytes_written"] > 0
        and bool(start_rows)
        and proc is not None
        and proc.returncode == 0
    )
    mode_summary = {
        "mode": mode_name,
        "acceptable": acceptable,
        "error": error,
        "config": {
            "background_writer": True,
            "daemon_throttle": False,
            "harness_throttle": False,
            "controller_label": "systemd_io_weight",
        },
        "foreground": foreground,
        "background": background,
        "kernel_control": {
            "control": "systemd_io_weight",
            "io_weight": args.cgroup_io_weight,
            "writer_start_rows": len(start_rows),
            "systemd_run_returncode": proc.returncode if proc is not None else None,
            "systemd_run_stdout": systemd_stdout_text.strip(),
            "systemd_run_stderr": systemd_stderr_text.strip(),
            "writer_kernel_qos_rows": [
                row.get("kernel_qos") for row in start_rows
            ],
            "systemd_unit_probe": unit_probe,
        },
        "logs": {
            "foreground_jsonl": rel(sqlite_log),
            "background_jsonl": rel(writer_log),
            "runtime_telemetry": rel(telemetry_file),
            "runtime_fuse_admission_trace": rel(admission_trace),
            "runtime_fuse_throttle_trace": rel(throttle_trace),
            "systemd_run_stdout": rel(systemd_stdout),
            "systemd_run_stderr": rel(systemd_stderr),
            "fuse_stdout": rel(mode_dir / "mount_logs" / "pqc_fuse.stdout.txt"),
            "fuse_stderr": rel(mode_dir / "mount_logs" / "pqc_fuse.stderr.txt"),
        },
    }
    (mode_dir / "mode_summary.json").write_text(
        json.dumps(mode_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return mode_summary


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker-background-writer":
        raise SystemExit(systemd_worker_main(sys.argv[2:]))

    parser = argparse.ArgumentParser()
    parser.add_argument("--control", choices=["ionice", "systemd_io_weight"], default="ionice")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--thermal-interval-ms", type=int, default=100)
    parser.add_argument("--transactions", type=int, default=24)
    parser.add_argument("--rows-per-txn", type=int, default=8)
    parser.add_argument("--sqlite-payload-bytes", type=int, default=256)
    parser.add_argument("--deadline-ms", type=float, default=10.0)
    parser.add_argument("--inter-transaction-sleep-ms", type=float, default=1.0)
    parser.add_argument("--post-foreground-drain-ms", type=int, default=150)
    parser.add_argument("--background-writers", type=int, default=1)
    parser.add_argument("--background-chunk-bytes", type=int, default=65536)
    parser.add_argument("--background-fsync-every", type=int, default=1)
    parser.add_argument("--background-warmup-ms", type=int, default=100)
    parser.add_argument("--telemetry-interval-ms", type=int, default=20)
    parser.add_argument("--telemetry-poll-ms", type=int, default=10)
    parser.add_argument("--telemetry-window", type=int, default=12)
    parser.add_argument("--controller-warmup-transactions", type=int, default=2)
    parser.add_argument("--background-pressure-util", type=float, default=0.85)
    parser.add_argument("--background-pressure-pattern", default="")
    parser.add_argument("--enter-util", type=float, default=0.70)
    parser.add_argument("--exit-util", type=float, default=0.60)
    parser.add_argument("--hold-samples", type=int, default=1)
    parser.add_argument("--harness-throttle-sleep-us", type=int, default=5000)
    parser.add_argument("--daemon-throttle-sleep-us", type=int, default=30000)
    parser.add_argument("--ionice-class", default="3")
    parser.add_argument("--ionice-level", type=int, default=None)
    parser.add_argument("--cgroup-io-weight", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be at least 1")
    if args.out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.out_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.out_dir / "sqlite_kernel_qos_baseline.json"
    md_path = args.out_dir / "sqlite_kernel_qos_baseline.md"
    blocking_reasons: list[str] = []
    if args.control == "ionice" and shutil.which("ionice") is None:
        blocking_reasons.append("ionice_missing")
    if args.control == "systemd_io_weight" and shutil.which("systemd-run") is None:
        blocking_reasons.append("systemd_run_missing")
    if not hero.FUSE_BIN.exists():
        blocking_reasons.append("pqc_fuse_missing")
    if blocking_reasons:
        report = blocked_report(args, blocking_reasons)
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_markdown(report, md_path)
        print(json.dumps({
            "overall_pass": False,
            "verdict": "environment-blocked",
            "blocking_reasons": blocking_reasons,
            "json": rel(json_path),
        }, sort_keys=True))
        return 0

    hero_args = build_hero_args(args)
    thermal_proc, thermal_fp, thermal_status = hero.start_thermal_log(
        args.out_dir,
        args.thermal_interval_ms,
    )
    modes: list[dict[str, Any]] = []
    try:
        for index in range(args.repetitions):
            if args.control == "ionice":
                config = hero.ModeConfig(
                    name="kernel_ionice_idle",
                    background_writer=True,
                    daemon_throttle=False,
                    harness_throttle=False,
                    controller_label="kernel_ionice_idle",
                )
                modes.append(hero.run_mode(config, args.out_dir / f"rep_{index:02d}", hero_args))
            else:
                modes.append(
                    run_systemd_io_weight_repetition(
                        args.out_dir / f"rep_{index:02d}",
                        args,
                        hero_args,
                    )
                )
    finally:
        thermal_status = hero.stop_thermal_log(thermal_proc, thermal_fp, thermal_status)

    rows: list[dict[str, Any]] = []
    for index, mode in enumerate(modes):
        foreground = mode["foreground"]
        background = mode["background"]
        rows.append({
            "repetition": index,
            "acceptable": mode["acceptable"],
            "foreground_p99_ms": foreground.get("p99_ms"),
            "foreground_p95_ms": foreground.get("p95_ms"),
            "deadline_misses": foreground.get("deadline_misses"),
            "background_throughput_mb_s": background.get("throughput_mb_s"),
            "background_bytes_written": background.get("bytes_written"),
            "mode_summary": rel(args.out_dir / f"rep_{index:02d}" / mode["mode"] / "mode_summary.json"),
        })
    if args.control == "ionice":
        kernel_control = summarize_kernel_control(modes)
        kernel_control.update({
            "class": args.ionice_class,
            "level": args.ionice_level,
        })
    else:
        start_rows = sum(int((mode.get("kernel_control") or {}).get("writer_start_rows", 0)) for mode in modes)
        success_rows = sum(
            1
            for mode in modes
            if (mode.get("kernel_control") or {}).get("systemd_run_returncode") == 0
            and int((mode.get("kernel_control") or {}).get("writer_start_rows", 0)) > 0
        )
        kernel_control = {
            "control": "systemd_io_weight",
            "io_weight": args.cgroup_io_weight,
            "writer_start_rows": start_rows,
            "applied_rows": start_rows,
            "successful_rows": success_rows,
            "failed_rows": max(0, args.repetitions - success_rows),
            "missing_rows": max(0, args.repetitions - start_rows),
            "details": [
                mode.get("kernel_control") for mode in modes
            ],
        }
    p99_samples = [float(row["foreground_p99_ms"]) for row in rows if row["foreground_p99_ms"] is not None]
    miss_samples = [float(row["deadline_misses"]) for row in rows if row["deadline_misses"] is not None]
    throughput_samples = [
        float(row["background_throughput_mb_s"])
        for row in rows
        if row["background_throughput_mb_s"] is not None
    ]
    overall_pass = (
        bool(rows)
        and all(row["acceptable"] for row in rows)
        and kernel_control["successful_rows"] == kernel_control["writer_start_rows"]
        and kernel_control["writer_start_rows"] > 0
        and bool(p99_samples)
        and bool(throughput_samples)
    )
    report = {
        "artifact": "sqlite_kernel_qos_baseline",
        "overall_pass": overall_pass,
        "verdict": "measured" if overall_pass else "invalid-run",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repetitions": args.repetitions,
        "kernel_control": {
            **kernel_control,
        },
        "scope": [
            "First kernel-control baseline for SQLite foreground p99 under mounted secure-storage pressure.",
            "Applies the selected Linux kernel control only to the background secure-storage writer.",
            "Does not close Gate B3 by itself because the parent gate requires at least two kernel-level baselines plus paper text.",
        ],
        "non_claims": [
            "not evidence that AEGIS-Q beats kernel QoS",
            "not a complete kernel QoS comparison by itself",
            "not foreground AI/TensorRT QoS recovery evidence",
        ],
        "args": {
            key: value for key, value in vars(args).items()
            if key != "out_dir"
        },
        "platform": hero.platform_manifest(),
        "thermal_logging": thermal_status,
        "rows": rows,
        "modes": modes,
        "summary": {
            "foreground_p99_ms": metric_summary(p99_samples),
            "deadline_misses": metric_summary(miss_samples),
            "background_throughput_mb_s": metric_summary(throughput_samples),
        },
        "artifacts": {
            "json": rel(json_path),
            "markdown": rel(md_path),
            "thermal_log": thermal_status.get("path"),
            "raw_dir": rel(args.out_dir),
        },
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({
        "overall_pass": overall_pass,
        "verdict": report["verdict"],
        "json": rel(json_path),
        "foreground_p99_ms_median": report["summary"]["foreground_p99_ms"]["median"],
        "deadline_misses_median": report["summary"]["deadline_misses"]["median"],
        "background_throughput_mb_s_median": report["summary"]["background_throughput_mb_s"]["median"],
        "kernel_control": kernel_control["control"],
        "kernel_successful_rows": kernel_control["successful_rows"],
        "kernel_writer_start_rows": kernel_control["writer_start_rows"],
    }, sort_keys=True))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
