#!/usr/bin/env python3
"""Collect a production-path lock profile for Gate 0.15.

This runner is intentionally a partial contract.  It exercises the mounted
AEGIS-Q path with same-file and disjoint-file writers while
``PQC_LOCK_PROFILE_PATH`` is enabled, then summarizes observed lock wait and
hold times.  It does not close Gate 0.15; it creates evidence for the next
fine-grained locking refactor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import queue
import re
import shutil
import signal
import statistics
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FUSE_BIN = ROOT / "build" / "pqc_fuse"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "concurrency_contract"


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


def fusermount_command() -> str:
    for name in ("fusermount3", "fusermount"):
        if shutil.which(name):
            return name
    return "fusermount3"


def start_fuse(
    storage_dir: Path,
    mount_dir: Path,
    out_dir: Path,
    password: str,
    extra_env: dict[str, str] | None = None,
    command_prefix: list[str] | None = None,
) -> FuseHandle:
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
            "PQC_ADMISSION_TRACE_PATH": str(out_dir / "admission_trace.jsonl"),
            "PQC_LOCK_PROFILE_PATH": str(out_dir / "lock_profile_trace.jsonl"),
            "PQC_ENABLE_QOS_THROTTLE_ON_WRITE": "0",
            "PQC_KEY_ROTATION_INTERVAL_S": "0",
        }
    )
    if extra_env:
        env.update(extra_env)
    command = [str(FUSE_BIN), str(storage_dir), str(mount_dir), "-f"]
    if command_prefix:
        command = command_prefix + command
    proc = subprocess.Popen(
        command,
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
    result: dict[str, Any] = {
        "argv": [fusermount_command(), "-u", str(mount_dir)],
        "returncode": unmount.returncode,
    }
    if handle is None:
        return result
    if handle.proc.poll() is None:
        handle.proc.send_signal(signal.SIGINT)
        try:
            handle.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            handle.proc.kill()
            handle.proc.wait(timeout=5)
    handle.stdout.close()
    handle.stderr.close()
    result["fuse_returncode"] = handle.proc.returncode
    return result


def writer_worker(
    mount_dir: Path,
    filename: str,
    worker_id: int,
    iterations: int,
    block_size: int,
    errors: list[dict[str, Any]],
) -> None:
    path = mount_dir / filename
    fd = -1
    try:
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        base = (worker_id % 1000) * 1024 * 1024
        for index in range(iterations):
            byte = 65 + (worker_id + index) % 26
            payload = bytes([byte]) * block_size
            written = os.pwrite(fd, payload, base + index * block_size)
            if written != block_size:
                raise OSError(f"short pwrite: {written} != {block_size}")
            os.fdatasync(fd)
    except BaseException as exc:  # noqa: BLE001 - retain worker failure evidence.
        errors.append(
            {
                "filename": filename,
                "worker_id": worker_id,
                "error": repr(exc),
            }
        )
    finally:
        if fd >= 0:
            os.close(fd)


def lifecycle_worker(
    mount_dir: Path,
    filename: str,
    worker_id: int,
    iterations: int,
    block_size: int,
    errors: list[dict[str, Any]],
) -> None:
    path = mount_dir / filename
    base = (worker_id % 1000) * 1024 * 1024
    try:
        for index in range(iterations):
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                byte = 97 + (worker_id + index) % 26
                payload = bytes([byte]) * block_size
                offset = base + index * block_size
                written = os.pwrite(fd, payload, offset)
                if written != block_size:
                    raise OSError(f"short lifecycle pwrite: {written} != {block_size}")
                os.fdatasync(fd)
                observed = os.pread(fd, block_size, offset)
                if observed != payload:
                    raise OSError("lifecycle read-after-fdatasync mismatch")
            finally:
                os.close(fd)
    except BaseException as exc:  # noqa: BLE001 - retain worker failure evidence.
        errors.append(
            {
                "filename": filename,
                "worker_id": worker_id,
                "error": repr(exc),
            }
        )


def lifecycle_process_worker(
    mount_dir: str,
    filename: str,
    worker_id: int,
    iterations: int,
    block_size: int,
    result_queue: mp.Queue,
) -> None:
    errors: list[dict[str, Any]] = []
    lifecycle_worker(Path(mount_dir), filename, worker_id, iterations, block_size, errors)
    for error in errors:
        result_queue.put(error)


def parse_thread_counts(text: str) -> list[int]:
    counts: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 1:
            raise ValueError("thread counts must be positive")
        if value not in counts:
            counts.append(value)
    if not counts:
        raise ValueError("at least one thread count is required")
    return counts


def run_process_lifecycle_phase(
    mount_dir: Path,
    name: str,
    client_count: int,
    iterations: int,
    block_size: int,
    phase_timeout_s: float,
    phase_index: int,
    same_file: bool,
) -> dict[str, Any]:
    started_ns = time.time_ns()
    result_queue: mp.Queue = mp.Queue()
    phase_id_base = phase_index * 1000
    processes: list[tuple[int, str, mp.Process]] = []

    for worker in range(client_count):
        filename = (
            f"process-same-lifecycle-{client_count}.dat"
            if same_file
            else f"process-disjoint-lifecycle-{client_count}-{worker}.dat"
        )
        worker_id = phase_id_base + worker
        proc = mp.Process(
            target=lifecycle_process_worker,
            args=(
                str(mount_dir),
                filename,
                worker_id,
                iterations,
                block_size,
                result_queue,
            ),
        )
        proc.start()
        processes.append((worker_id, filename, proc))

    timed_out = False
    deadline = time.monotonic() + phase_timeout_s
    for _, _, proc in processes:
        remaining = deadline - time.monotonic()
        proc.join(timeout=max(0.0, remaining))
        if proc.is_alive():
            timed_out = True

    if timed_out:
        for _, _, proc in processes:
            if proc.is_alive():
                proc.terminate()
        for _, _, proc in processes:
            if proc.is_alive():
                proc.join(timeout=5.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5.0)

    queue_errors: list[dict[str, Any]] = []
    while True:
        try:
            queue_errors.append(result_queue.get_nowait())
        except queue.Empty:
            break

    process_failures: list[dict[str, Any]] = []
    for worker_id, filename, proc in processes:
        if proc.exitcode not in (0, None):
            process_failures.append(
                {
                    "filename": filename,
                    "worker_id": worker_id,
                    "error": f"process exitcode {proc.exitcode}",
                    "exitcode": proc.exitcode,
                }
            )
        elif proc.exitcode is None:
            process_failures.append(
                {
                    "filename": filename,
                    "worker_id": worker_id,
                    "error": "process did not exit before timeout handling completed",
                    "exitcode": None,
                }
            )

    ended_ns = time.time_ns()
    return {
        "name": name,
        "worker": "lifecycle_process_worker",
        "thread_count": 0,
        "process_count": len(processes),
        "client_count": len(processes),
        "iterations_per_thread": iterations,
        "block_size": block_size,
        "timed_out": timed_out,
        "worker_errors": queue_errors + process_failures,
        "process_exitcodes": [
            {"worker_id": worker_id, "exitcode": proc.exitcode}
            for worker_id, _, proc in processes
        ],
        "wall_seconds": (ended_ns - started_ns) / 1_000_000_000,
    }


def run_workload(
    mount_dir: Path,
    iterations: int,
    block_size: int,
    thread_counts: list[int],
    phase_timeout_s: float,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    phase_results: list[dict[str, Any]] = []
    phase_index = 0
    for thread_count in thread_counts:
        phases = [
            {
                "name": "same_file",
                "worker": writer_worker,
                "filename": lambda worker: f"same-file-{thread_count}.dat",
            },
            {
                "name": "disjoint_files",
                "worker": writer_worker,
                "filename": lambda worker: f"disjoint-{thread_count}-{worker}.dat",
            },
            {
                "name": "same_file_lifecycle",
                "worker": lifecycle_worker,
                "filename": lambda worker: f"same-lifecycle-{thread_count}.dat",
            },
            {
                "name": "disjoint_lifecycle",
                "worker": lifecycle_worker,
                "filename": lambda worker: f"disjoint-lifecycle-{thread_count}-{worker}.dat",
            },
        ]
        for phase in phases:
            started_ns = time.time_ns()
            worker_fn = phase.get("worker", writer_worker)
            phase_id_base = phase_index * 1000
            phase_threads = [
                (phase["filename"](worker), phase_id_base + worker)
                for worker in range(thread_count)
            ]
            threads = [
                threading.Thread(
                    target=worker_fn,
                    args=(mount_dir, filename, worker_id, iterations, block_size, errors),
                )
                for filename, worker_id in phase_threads
            ]
            for thread in threads:
                thread.start()
            timed_out = False
            deadline = time.monotonic() + phase_timeout_s
            for thread in threads:
                remaining = deadline - time.monotonic()
                thread.join(timeout=max(0.0, remaining))
                if thread.is_alive():
                    timed_out = True
            ended_ns = time.time_ns()
            phase_errors = [
                error for error in errors
                if phase_id_base <= int(error["worker_id"]) < phase_id_base + 1000
            ]
            phase_results.append(
                {
                    "name": phase["name"],
                    "worker": getattr(worker_fn, "__name__", "unknown"),
                    "thread_count": len(threads),
                    "iterations_per_thread": iterations,
                    "block_size": block_size,
                    "timed_out": timed_out,
                    "worker_errors": phase_errors,
                    "wall_seconds": (ended_ns - started_ns) / 1_000_000_000,
                }
            )
            phase_index += 1
    for client_count in thread_counts:
        for phase_name, same_file in (
            ("same_file_process_lifecycle", True),
            ("disjoint_process_lifecycle", False),
        ):
            phase = run_process_lifecycle_phase(
                mount_dir,
                phase_name,
                client_count,
                iterations,
                block_size,
                phase_timeout_s,
                phase_index,
                same_file,
            )
            phase_results.append(phase)
            errors.extend(phase["worker_errors"])
            phase_index += 1
    return {
        "phases": phase_results,
        "errors": errors,
        "coverage": {
            "max_thread_count": max(thread_counts) if thread_counts else 0,
            "same_file_writer_phases": len(
                [phase for phase in phase_results if phase["name"] == "same_file"]
            ),
            "disjoint_writer_phases": len(
                [phase for phase in phase_results if phase["name"] == "disjoint_files"]
            ),
            "same_file_lifecycle_phases": len(
                [phase for phase in phase_results if phase["name"] == "same_file_lifecycle"]
            ),
            "disjoint_lifecycle_phases": len(
                [phase for phase in phase_results if phase["name"] == "disjoint_lifecycle"]
            ),
            "same_file_process_lifecycle_phases": len(
                [
                    phase for phase in phase_results
                    if phase["name"] == "same_file_process_lifecycle"
                ]
            ),
            "disjoint_process_lifecycle_phases": len(
                [
                    phase for phase in phase_results
                    if phase["name"] == "disjoint_process_lifecycle"
                ]
            ),
            "max_process_client_count": max(
                [
                    int(phase.get("client_count", 0))
                    for phase in phase_results
                    if phase.get("worker") == "lifecycle_process_worker"
                ]
                or [0]
            ),
        },
    }


def wait_for_file(path: Path, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists() and path.stat().st_size > 0:
            return True
        time.sleep(0.01)
    return False


def read_exact_prefix(path: Path, size: int) -> bytes:
    with path.open("rb", buffering=0) as handle:
        return handle.read(size)


def run_reader_visibility_probe(out_dir: Path,
                                password: str,
                                block_size: int) -> dict[str, Any]:
    probe_dir = out_dir / "reader_visibility_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    marker_path = probe_dir / "pause_marker.jsonl"
    initial = b"A" * block_size
    update = b"B" * block_size
    writer_errors: list[str] = []
    result: dict[str, Any] = {
        "scope": (
            "Mounted-path probe: pause an update after journal append but before "
            "journal fdatasync, then verify a concurrent reader sees only the "
            "previous committed generation."
        ),
        "pause_cutpoint": "journal_append_after",
        "pause_marker": relpath(marker_path),
        "block_size": block_size,
        "marker_observed": False,
        "during_pause_observed_old": False,
        "after_publish_observed_new": False,
        "writer_errors": writer_errors,
        "overall_pass": False,
    }

    with tempfile.TemporaryDirectory(prefix="aegisq_vis_store_") as storage_tmp, \
            tempfile.TemporaryDirectory(prefix="aegisq_vis_mnt_") as mount_tmp:
        storage_dir = Path(storage_tmp)
        mount_dir = Path(mount_tmp)
        mounted_path = mount_dir / "visibility.dat"
        initial_handle: FuseHandle | None = None
        paused_handle: FuseHandle | None = None
        try:
            initial_handle = start_fuse(
                storage_dir, mount_dir, probe_dir / "initial_mount", password)
            fd = os.open(mounted_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                os.pwrite(fd, initial, 0)
                os.fdatasync(fd)
            finally:
                os.close(fd)
            stop_fuse(initial_handle, mount_dir, probe_dir / "initial_mount")
            initial_handle = None

            paused_handle = start_fuse(
                storage_dir,
                mount_dir,
                probe_dir / "paused_mount",
                password,
                {
                    "PQC_PAUSE_CUTPOINT": "journal_append_after",
                    "PQC_PAUSE_MARKER_PATH": str(marker_path),
                    "PQC_PAUSE_US": "750000",
                },
            )

            def writer() -> None:
                fd = -1
                try:
                    fd = os.open(mounted_path, os.O_RDWR)
                    os.pwrite(fd, update, 0)
                    os.fdatasync(fd)
                except BaseException as exc:  # noqa: BLE001 - retain evidence.
                    writer_errors.append(repr(exc))
                finally:
                    if fd >= 0:
                        os.close(fd)

            thread = threading.Thread(target=writer)
            thread.start()
            result["marker_observed"] = wait_for_file(marker_path, 5.0)
            if result["marker_observed"]:
                during = read_exact_prefix(mounted_path, block_size)
                result["during_pause_sha256"] = hashlib.sha256(during).hexdigest()
                result["during_pause_observed_old"] = during == initial
                result["during_pause_observed_new"] = during == update
            thread.join(timeout=10.0)
            result["writer_thread_alive_after_timeout"] = thread.is_alive()
            after = read_exact_prefix(mounted_path, block_size)
            result["after_publish_sha256"] = hashlib.sha256(after).hexdigest()
            result["after_publish_observed_new"] = after == update
            result["overall_pass"] = (
                result["marker_observed"]
                and result["during_pause_observed_old"]
                and not result.get("during_pause_observed_new", False)
                and result["after_publish_observed_new"]
                and not result["writer_thread_alive_after_timeout"]
                and not writer_errors
            )
        except BaseException as exc:  # noqa: BLE001 - retain setup failure evidence.
            result["error"] = repr(exc)
        finally:
            if paused_handle is not None:
                result["paused_unmount"] = stop_fuse(
                    paused_handle, mount_dir, probe_dir / "paused_mount")
            if initial_handle is not None:
                result["initial_unmount"] = stop_fuse(
                    initial_handle, mount_dir, probe_dir / "initial_mount")

    (probe_dir / "reader_visibility_probe.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


STRACE_SYSCALL_RE = re.compile(
    r"^\s*(?:\d+\s+)?(?:\d+\.\d+\s+)?"
    r"(?P<syscall>[A-Za-z0-9_]+)\(.*\)\s+=\s+"
    r"(?P<result>.*?)\s+<(?P<seconds>[0-9.]+)>$"
)


def summarize_blocking_syscalls(trace_path: Path) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    malformed = 0
    total_lines = 0
    if trace_path.exists():
        for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
            total_lines += 1
            match = STRACE_SYSCALL_RE.match(line)
            if not match:
                if line.strip() and not line.startswith("strace:"):
                    malformed += 1
                continue
            syscall = match.group("syscall")
            seconds = float(match.group("seconds"))
            grouped.setdefault(syscall, []).append(seconds * 1_000_000_000)

    by_syscall: dict[str, Any] = {}
    for syscall, values in sorted(grouped.items()):
        ns = [int(value) for value in values]
        by_syscall[syscall] = {
            "count": len(ns),
            "total_ns": sum(ns),
            "p50_ns": percentile(ns, 0.50),
            "p95_ns": percentile(ns, 0.95),
            "p99_ns": percentile(ns, 0.99),
            "max_ns": max(ns) if ns else 0,
        }
    return {
        "raw_trace": relpath(trace_path),
        "line_count": total_lines,
        "malformed_line_count": malformed,
        "syscall_count": sum(len(values) for values in grouped.values()),
        "observed_syscalls": sorted(grouped.keys()),
        "by_syscall": by_syscall,
    }


def run_blocking_syscall_profile(
    out_dir: Path,
    password: str,
    block_size: int,
    phase_timeout_s: float,
) -> dict[str, Any]:
    profile_dir = out_dir / "blocking_syscall_profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    raw_trace = profile_dir / "strace_raw.txt"
    stdout_path = profile_dir / "client.stdout.txt"
    stderr_path = profile_dir / "client.stderr.txt"
    strace = shutil.which("strace")
    result: dict[str, Any] = {
        "scope": (
            "Mounted-path client blocking-syscall profile collected with "
            "strace -f -T during a bounded strict-mode workload.  The traced "
            "pwrite/fdatasync/pread calls include user-visible FUSE wait time."
        ),
        "tool": "strace",
        "tool_available": bool(strace),
        "overall_pass": False,
    }
    if not strace:
        result["unavailable_reason"] = "strace not found in PATH"
        (profile_dir / "blocking_syscall_profile.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result

    storage_dir = Path(tempfile.mkdtemp(prefix="aegisq_blocking_storage_"))
    mount_dir = Path(tempfile.mkdtemp(prefix="aegisq_blocking_mnt_"))
    fuse: FuseHandle | None = None
    client_run: dict[str, Any] = {}
    unmount: dict[str, Any] = {}
    abort_reason: str | None = None
    try:
        fuse = start_fuse(storage_dir, mount_dir, profile_dir, password)
        client_script = r"""
import hashlib
import json
import os
import sys

mount_dir = sys.argv[1]
block_size = int(sys.argv[2])
iterations = int(sys.argv[3])
rows = []
for name, base in (("blocking_same.dat", 0), ("blocking_disjoint.dat", 1048576)):
    path = os.path.join(mount_dir, name)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        for index in range(iterations):
            payload = bytes([65 + index]) * block_size
            offset = base + index * block_size
            written = os.pwrite(fd, payload, offset)
            os.fdatasync(fd)
            observed = os.pread(fd, block_size, offset)
            if written != block_size or observed != payload:
                raise RuntimeError("blocking profile payload mismatch")
            rows.append({
                "file": name,
                "offset": offset,
                "sha256": hashlib.sha256(observed).hexdigest(),
            })
    finally:
        os.close(fd)
print(json.dumps({"rows": rows, "row_count": len(rows)}, sort_keys=True))
"""
        command = [
            str(strace),
            "-f",
            "-ttt",
            "-T",
            "-o",
            str(raw_trace),
            "-e",
            "trace=fdatasync,pwrite64,pread64,openat,close",
            "--",
            "python3",
            "-c",
            client_script,
            str(mount_dir),
            str(block_size),
            "4",
        ]
        proc = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=phase_timeout_s,
        )
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        client_run = {
            "argv": [
                str(strace),
                "-f",
                "-ttt",
                "-T",
                "-o",
                relpath(raw_trace),
                "-e",
                "trace=fdatasync,pwrite64,pread64,openat,close",
                "--",
                "python3",
                "-c",
                "<inline-mounted-path-client>",
                "<mount_dir>",
                str(block_size),
                "4",
            ],
            "returncode": proc.returncode,
            "stdout": relpath(stdout_path),
            "stderr": relpath(stderr_path),
        }
        if proc.returncode != 0:
            abort_reason = f"blocking profile client returned {proc.returncode}"
    except BaseException as exc:  # noqa: BLE001 - retained as evidence.
        abort_reason = repr(exc)
    finally:
        unmount = stop_fuse(fuse, mount_dir, profile_dir)
        shutil.rmtree(storage_dir, ignore_errors=True)
        shutil.rmtree(mount_dir, ignore_errors=True)

    summary = summarize_blocking_syscalls(raw_trace)
    expected_syscalls = {"fdatasync", "pwrite64", "pread64"}
    result.update(
        {
            "strace_version": command_capture([str(strace), "-V"], timeout_s=5.0),
            "command": [
                str(strace),
                "-f",
                "-ttt",
                "-T",
                "-o",
                relpath(raw_trace),
                "-e",
                "trace=fdatasync,pwrite64,pread64,openat,close",
                "--",
                "python3",
                "-c",
                "<inline-mounted-path-client>",
                "<mount_dir>",
                str(block_size),
                "4",
            ],
            "abort_reason": abort_reason,
            "client_run": client_run,
            "unmount": unmount,
            "summary": summary,
            "expected_syscalls_observed": sorted(
                expected_syscalls.intersection(summary["observed_syscalls"])
            ),
            "overall_pass": (
                abort_reason is None
                and unmount.get("returncode") == 0
                and client_run.get("returncode") == 0
                and summary["syscall_count"] > 0
                and expected_syscalls.issubset(set(summary["observed_syscalls"]))
            ),
            "negative_claim_guard": (
                "This strace profile is user-visible blocking-syscall evidence "
                "for the strict mounted path; it is not daemon scheduler off-CPU "
                "profiling and does not justify scalability or parallel-commit claims."
            ),
        }
    )
    (profile_dir / "blocking_syscall_profile.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Blocking Syscall Profile",
        "",
        f"- Overall pass: `{str(result['overall_pass']).lower()}`",
        f"- Raw trace: `{relpath(raw_trace)}`",
        f"- Syscalls observed: `{', '.join(summary['observed_syscalls'])}`",
        "",
        "| Syscall | Count | p50 ns | p95 ns | p99 ns | max ns | total ns |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for syscall, stats in summary["by_syscall"].items():
        lines.append(
            f"| `{syscall}` | {stats['count']} | {stats['p50_ns']:.0f} | "
            f"{stats['p95_ns']:.0f} | {stats['p99_ns']:.0f} | "
            f"{stats['max_ns']} | {stats['total_ns']} |"
        )
    (profile_dir / "blocking_syscall_profile.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return result


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * q)
    return float(ordered[index])


def summarize_lock_trace(trace_path: Path) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    malformed = 0
    if trace_path.exists():
        for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if payload.get("event") == "lock_hold":
                events.append(payload)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        key = f"{event.get('lock', 'unknown')}@{event.get('site', 'unknown')}"
        grouped.setdefault(key, []).append(event)

    summaries: dict[str, Any] = {}
    for key, group in sorted(grouped.items()):
        wait = [int(item.get("wait_ns", 0) or 0) for item in group]
        hold = [int(item.get("hold_ns", 0) or 0) for item in group]
        summaries[key] = {
            "count": len(group),
            "wait_ns": {
                "p50": percentile(wait, 0.50),
                "p95": percentile(wait, 0.95),
                "p99": percentile(wait, 0.99),
                "max": max(wait) if wait else 0,
            },
            "hold_ns": {
                "p50": percentile(hold, 0.50),
                "p95": percentile(hold, 0.95),
                "p99": percentile(hold, 0.99),
                "max": max(hold) if hold else 0,
            },
        }

    observed_locks = sorted({str(event.get("lock", "unknown")) for event in events})
    return {
        "trace": relpath(trace_path),
        "line_count": len(trace_path.read_text(encoding="utf-8", errors="replace").splitlines()) if trace_path.exists() else 0,
        "lock_hold_event_count": len(events),
        "malformed_line_count": malformed,
        "observed_locks": observed_locks,
        "by_lock_and_site": summaries,
    }


def lock_order_table() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scope": "Current strict-path lock order and known violations before Gate 0.15 refactor.",
        "implemented_mitigations": [
            {
                "name": "dirty-sidecar fdatasync outside fd_lock",
                "code_path": "pqc_fd_context_prepare_dirty_sidecar_sync_locked + pqc_fd_context_run_dirty_sidecar_sync",
                "invariant": "A duplicated sidecar fd is synced outside fd_lock; dirty flags are cleared only if the captured dirty epoch still matches after reacquiring fd_lock.",
                "remaining_limit": "Dirty-flag fault coverage and broad same-FD concurrency evidence are still missing.",
            },
            {
                "name": "strict generation reservation and publish ticket",
                "code_path": "pqc_checkpoint_reserve_generation + file_state publish_ticket + pqc_writeback_flush_locked",
                "invariant": "A generation high-water mark is checkpointed before ciphertext generation; same-file publish turns remain ordered while prepare/crypto run outside commit_lock.",
                "remaining_limit": "Durable-publish lock splitting and broad contention evidence are still missing.",
            },
            {
                "name": "write-buffer snapshot outside fd_lock",
                "code_path": "pqc_writeback_flush_locked + writeback_snapshot_take_locked",
                "invariant": "The live fd write buffer is copied under fd_lock, cleared, and protected by a pending-job reference before prepare/crypto/publish run without fd_lock.",
                "remaining_limit": "External process client-count smoke and strace blocking-syscall evidence exist, but long-duration client-count and scheduler off-CPU evidence are still missing.",
            },
            {
                "name": "authenticated-read snapshot outside fd_lock",
                "code_path": "pqc_read + pqc_recovery_load_authenticated_block_committed",
                "invariant": "The read path snapshots sidecar fds, key material, file id, visible size, and committed generation under fd_lock, then authenticates blocks outside fd_lock while a pending-job reference prevents release from closing the fds.",
                "remaining_limit": "External process client-count smoke and strace blocking-syscall evidence exist, but scheduler off-CPU profiling and long-duration client-count paths still need lock-contract evidence.",
            },
            {
                "name": "metadata resize publication outside hot locks",
                "code_path": "pqc_truncate + pqc_fallocate + metadata_publish_turn_begin",
                "invariant": "Truncate and fallocate snapshot fd state, hold a publish turn for same-file ordering, perform xattr/ftruncate/checkpoint work outside fd_lock and commit_lock, then update visible state with short lock holds.",
                "remaining_limit": "Strace blocking-syscall evidence exists, but scheduler off-CPU profiling and long-duration deadlock/livelock stress still need lock-contract evidence.",
            },
            {
                "name": "release lifecycle detach outside fd_lock",
                "code_path": "pqc_fd_context_clear + retired_resources_release",
                "invariant": "Release waits for pending jobs, detaches sidecar fds, write buffer, and file-state reference under fd_lock, marks the context invalid, then closes/frees/releases detached resources after unlocking.",
                "remaining_limit": "Close/free latency is no longer in fd_lock and the smoke includes repeated thread/process lifecycle phases plus blocking-syscall evidence, but scheduler off-CPU and long-duration evidence are still missing.",
            }
        ],
        "order": [
            {
                "rank": 10,
                "lock": "fd_lock",
                "owner_module": "pqc_fd_context / pqc_file_io",
                "protected_state": "per-FD validity, write buffer, logical size cache, sidecar dirty flags, QoS class, pending-job counter",
                "allowed_call_sites": ["pqc_read", "pqc_write", "pqc_fsync", "pqc_flush", "pqc_truncate", "pqc_fallocate", "pqc_release"],
                "forbidden_blocking_calls_for_closure": ["crypto", "pwrite", "fdatasync", "checkpoint update", "anchor update", "condition wait"],
                "current_status": "writeback_authenticated_read_truncate_fallocate_and_release_lifecycle_snapshot_or_detach_outside_fd_lock_but_broad_stress_still_needs_audit",
            },
            {
                "rank": 20,
                "lock": "commit_lock",
                "owner_module": "pqc_state / pqc_writeback",
                "protected_state": "per-inode generation high-water, reader-visible committed generation, publish tickets, and checkpoint-visible logical size",
                "allowed_call_sites": ["pqc_read", "pqc_writeback_flush_locked", "pqc_fallocate"],
                "forbidden_blocking_calls_for_closure": ["GPU/CPU crypto", "pwrite", "fdatasync", "checkpoint write", "TPM/anchor access"],
                "current_status": "strict_publish_split_but_truncate_fallocate_and_broad_lock_contract_still_need_audit",
            },
        ],
        "negative_claim_guard": "This table is a measurement scaffold, not evidence that Gate 0.15 is closed.",
    }


def source_contains(path: Path, pattern: str) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return re.search(pattern, text, re.MULTILINE | re.DOTALL) is not None


def call_runs_under_profiled_lock(path: Path, lock_name: str, call: str) -> bool:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return False
    in_target_lock = False
    for line in lines:
        if "pqc_profiled_mutex_lock(&" in line and lock_name in line:
            in_target_lock = True
        if call in line and in_target_lock:
            return True
        if "pqc_profiled_mutex_unlock(&" in line and lock_name in line:
            in_target_lock = False
    return False


def call_runs_under_commit_lock(path: Path, call: str) -> bool:
    return call_runs_under_profiled_lock(path, "commit_lock", call)


def call_runs_under_fd_lock(path: Path, call: str) -> bool:
    return call_runs_under_profiled_lock(path, "fd_lock", call)


def commit_lock_refactor_feasibility(out_dir: Path) -> dict[str, Any]:
    writeback = ROOT / "code" / "storage" / "pqc_writeback.c"
    flush_batch = ROOT / "code" / "storage" / "pqc_flush_batch.c"
    strict_publish = ROOT / "code" / "storage" / "pqc_strict_publish.c"
    state = ROOT / "code" / "storage" / "pqc_state.h"
    checkpoint = ROOT / "code" / "storage" / "pqc_checkpoint.c"
    fd_context = ROOT / "code" / "fs" / "pqc_fd_context.c"
    journal = ROOT / "code" / "storage" / "pqc_journal.c"
    file_io = ROOT / "code" / "fs" / "pqc_file_io.c"
    recovery = ROOT / "code" / "fs" / "pqc_recovery.c"

    evidence = {
        "commit_lock_wraps_flush_batch_prepare": call_runs_under_commit_lock(
            writeback, "pqc_flush_batch_prepare"
        ),
        "commit_lock_wraps_crypto_encrypt": call_runs_under_commit_lock(
            writeback, "pqc_flush_crypto_encrypt"
        ),
        "commit_lock_wraps_strict_publish": call_runs_under_commit_lock(
            writeback, "pqc_strict_publish_commit"
        ),
        "fd_lock_still_wraps_writeback_flush": source_contains(
            file_io,
            r"pqc_profiled_mutex_lock\(&ctx->fd_lock.*?pqc_writeback_flush_locked",
        ),
        "fd_lock_wraps_authenticated_recovery": call_runs_under_fd_lock(
            file_io, "pqc_recovery_load_authenticated_block_committed"
        ),
        "read_uses_pending_job": source_contains(
            file_io,
            r"pqc_fd_context_pending_job_begin\(ctx\).*?pqc_recovery_load_authenticated_block_committed.*?pqc_fd_context_pending_job_end\(ctx\)",
        ),
        "read_copies_key_material_before_unlock": source_contains(
            file_io,
            r"memcpy\(ss,\s*ctx->ss,\s*ss_len\).*?pqc_profiled_mutex_unlock\(&ctx->fd_lock,\s*\"fd_lock\"",
        ),
        "metadata_publish_turn_helper_exists": source_contains(
            file_io,
            r"metadata_publish_turn_begin",
        ),
        "truncate_uses_pending_job_and_publish_turn": source_contains(
            file_io,
            r"int pqc_truncate.*?pqc_fd_context_pending_job_begin\(ctx\).*?metadata_publish_turn_begin",
        ),
        "truncate_zero_updates_visible_size_before_journal_truncate": source_contains(
            file_io,
            r"if \(res == 0 && size == 0\).*?state->logical_size\s*=\s*0;.*?ftruncate\(journal_fd,\s*0\)",
        ),
        "fallocate_uses_pending_job_and_publish_turn": source_contains(
            file_io,
            r"int pqc_fallocate.*?pqc_fd_context_pending_job_begin\(ctx\).*?metadata_publish_turn_begin",
        ),
        "fallocate_checkpoint_after_fd_unlock": source_contains(
            file_io,
            r"pqc_fd_context_pending_job_begin\(ctx\).*?pqc_profiled_mutex_unlock\(&ctx->fd_lock,\s*\"fd_lock\".*?pqc_checkpoint_store_and_stage_anchor",
        ),
        "release_waits_pending_before_retire": source_contains(
            fd_context,
            r"void pqc_fd_context_clear.*?pqc_fd_context_wait_pending_locked\(ctx\).*?ctx->valid\s*=\s*0",
        ),
        "release_detaches_resources_before_unlock": source_contains(
            fd_context,
            r"void pqc_fd_context_clear.*?retired\.wbuf\s*=\s*ctx->wbuf.*?retired\.data_fd\s*=\s*ctx->data_fd.*?retired\.journal_fd\s*=\s*ctx->journal_fd.*?ctx->data_fd\s*=\s*-1.*?ctx->journal_fd\s*=\s*-1.*?pthread_mutex_unlock\(&ctx->fd_lock\)",
        ),
        "release_closes_resources_after_unlock": source_contains(
            fd_context,
            r"void pqc_fd_context_clear.*?pthread_mutex_unlock\(&ctx->fd_lock\).*?retired_resources_release\(&retired\)",
        ),
        "file_state_has_committed_generation": source_contains(
            state,
            r"uint64_t\s+committed_generation",
        ),
        "read_snapshots_committed_generation": source_contains(
            file_io,
            r"visible_generation\s*=\s*ctx->state->committed_generation",
        ),
        "read_uses_committed_recovery_bound": source_contains(
            file_io,
            r"pqc_recovery_load_authenticated_block_committed",
        ),
        "open_promotes_journal_tail_only_on_state_initialization": source_contains(
            fd_context,
            r"initializing_visible_state\s*=\s*!state->logical_size_valid.*?initializing_visible_state.*?state->committed_generation\s*<\s*journal_max_generation",
        ),
        "recovery_uses_committed_journal_lookup": source_contains(
            recovery,
            r"pqc_journal_lookup_mapping_committed",
        ),
        "journal_lookup_filters_max_generation": source_contains(
            journal,
            r"record\.mapping\.generation\s*<=\s*max_generation",
        ),
        "journal_lookup_uses_pread": source_contains(
            journal,
            r"pread\(journal_fd,\s*&record,\s*sizeof\(record\),\s*pos\)",
        ),
        "flush_batch_reads_next_generation_plus_one": source_contains(
            flush_batch,
            r"desc->generation\s*=\s*next_generation\s*\+\s*1\s*\+\s*bi",
        ),
        "strict_publish_advances_generation_after_journal_fsync": source_contains(
            strict_publish,
            r"journal_fsync_after.*?\*req->next_generation\s*\+=",
        ),
        "file_state_has_publish_order_condition": source_contains(
            state,
            r"pthread_cond_t\s+.*publish|publish_.*ticket|ticket_.*publish",
        ),
        "writeback_accepts_fd_scope": source_contains(
            writeback,
            r"pqc_writeback_flush_locked\(int storage_fd,\s*pqc_fd_ctx_t \*ctx,\s*pqc_lock_profile_scope_t \*fd_scope",
        ),
        "writeback_snapshots_wbuf": source_contains(
            writeback,
            r"writeback_snapshot_take_locked.*?memcpy\(snapshot->buf,\s*ctx->wbuf",
        ),
        "writeback_clears_wbuf_before_unlock": source_contains(
            writeback,
            r"ctx->wbuf_used\s*=\s*0;.*?pqc_fd_context_pending_job_begin",
        ),
        "writeback_unlocks_fd_lock": source_contains(
            writeback,
            r"pqc_profiled_mutex_unlock\(&ctx->fd_lock,\s*\"fd_lock\"",
        ),
        "writeback_relocks_fd_lock_before_return": source_contains(
            writeback,
            r"pqc_profiled_mutex_lock\(&ctx->fd_lock,\s*\"fd_lock\"",
        ),
        "generation_reservation_checkpoint_exists": source_contains(
            checkpoint,
            r"int\s+pqc_checkpoint_reserve_generation",
        ),
        "remount_uses_checkpoint_max_generation": source_contains(
            fd_context,
            r"state->next_generation\s*<\s*ckpt\.max_generation",
        ),
        "checkpoint_persists_max_generation_only": source_contains(
            checkpoint,
            r"pqc_publish_checkpoint_store_xattr\(path, file_id, sequence,\s*logical_size,\s*max_generation\)",
        ),
        "journal_scans_max_generation_for_remount": source_contains(
            journal,
            r"uint64_t\s+pqc_journal_max_generation",
        ),
    }
    blockers = []
    if evidence["flush_batch_reads_next_generation_plus_one"]:
        blockers.append(
            "Generation numbers are derived from the current committed high-water mark during flush preparation."
        )
    if evidence["strict_publish_advances_generation_after_journal_fsync"]:
        blockers.append(
            "The committed high-water mark advances only after data and journal fdatasync complete."
        )
    if not evidence["file_state_has_publish_order_condition"]:
        blockers.append(
            "There is no per-file publish ticket/condition variable to preserve reservation order if crypto runs outside commit_lock."
        )
    if not evidence["generation_reservation_checkpoint_exists"]:
        blockers.append(
            "There is no persistent generation reservation checkpoint before ciphertext generation."
        )
    if not evidence["remount_uses_checkpoint_max_generation"]:
        blockers.append(
            "Remount does not advance the in-memory high-water mark from checkpoint max_generation."
        )
    fd_snapshot_ready = (
        evidence["writeback_accepts_fd_scope"]
        and evidence["writeback_snapshots_wbuf"]
        and evidence["writeback_clears_wbuf_before_unlock"]
        and evidence["writeback_unlocks_fd_lock"]
        and evidence["writeback_relocks_fd_lock_before_return"]
    )
    if evidence["fd_lock_still_wraps_writeback_flush"] and not fd_snapshot_ready:
        blockers.append(
            "The caller still holds fd_lock across strict writeback prepare, crypto, and durable publication."
        )
    if evidence["fd_lock_wraps_authenticated_recovery"]:
        blockers.append(
            "Authenticated read recovery still runs while fd_lock is held."
        )
    if not evidence["read_uses_pending_job"]:
        blockers.append(
            "Read path does not use the pending-job lifetime guard while recovery runs outside fd_lock."
        )
    if not (
        evidence["metadata_publish_turn_helper_exists"]
        and evidence["truncate_uses_pending_job_and_publish_turn"]
        and evidence["truncate_zero_updates_visible_size_before_journal_truncate"]
        and evidence["fallocate_uses_pending_job_and_publish_turn"]
        and evidence["fallocate_checkpoint_after_fd_unlock"]
    ):
        blockers.append(
            "Truncate/fallocate metadata publication is not yet proven to run outside hot locks with publish-turn ordering."
        )
    if not (
        evidence["release_waits_pending_before_retire"]
        and evidence["release_detaches_resources_before_unlock"]
        and evidence["release_closes_resources_after_unlock"]
    ):
        blockers.append(
            "Release/lifecycle teardown is not yet proven to detach fds, buffers, and file-state references before close/free/release work."
        )
    if evidence["commit_lock_wraps_strict_publish"]:
        blockers.append(
            "Strict publish still holds commit_lock across data/journal fdatasync and checkpoint update."
        )
    if not (
        evidence["file_state_has_committed_generation"]
        and evidence["read_snapshots_committed_generation"]
        and evidence["read_uses_committed_recovery_bound"]
        and evidence["open_promotes_journal_tail_only_on_state_initialization"]
        and evidence["recovery_uses_committed_journal_lookup"]
        and evidence["journal_lookup_filters_max_generation"]
        and evidence["journal_lookup_uses_pread"]
    ):
        blockers.append(
            "Reader visibility is not yet bounded by a committed-generation journal lookup."
        )
    decision = (
        "strict_publish_outside_commit_lock_but_gate_open"
        if not evidence["commit_lock_wraps_strict_publish"]
        and not evidence["commit_lock_wraps_flush_batch_prepare"]
        and not evidence["commit_lock_wraps_crypto_encrypt"]
        and evidence["generation_reservation_checkpoint_exists"]
        and evidence["file_state_has_publish_order_condition"]
        and evidence["file_state_has_committed_generation"]
        and evidence["read_uses_committed_recovery_bound"]
        and not evidence["fd_lock_wraps_authenticated_recovery"]
        and evidence["read_uses_pending_job"]
        and evidence["read_copies_key_material_before_unlock"]
        and evidence["truncate_uses_pending_job_and_publish_turn"]
        and evidence["fallocate_uses_pending_job_and_publish_turn"]
        and evidence["release_waits_pending_before_retire"]
        and evidence["release_detaches_resources_before_unlock"]
        and evidence["release_closes_resources_after_unlock"]
        and fd_snapshot_ready
        else "strict_prepare_crypto_outside_fd_and_commit_locks_but_gate_open"
        if not evidence["commit_lock_wraps_flush_batch_prepare"]
        and not evidence["commit_lock_wraps_crypto_encrypt"]
        and evidence["generation_reservation_checkpoint_exists"]
        and evidence["file_state_has_publish_order_condition"]
        and fd_snapshot_ready
        else "do_not_move_crypto_out_of_commit_lock_yet"
    )

    payload = {
        "schema_version": 1,
        "generated_by": "experiments/run_concurrency_contract_smoke.py",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scope": "Gate 0.15/0.16 feasibility check for moving crypto out of commit_lock in strict mode.",
        "source_evidence": evidence,
        "blockers": blockers,
        "required_before_safe_commit_lock_shrink": [
            "expand strict durable-publication fault coverage after commit_lock splitting",
            "fault matrix covering reserved-but-unpublished generations, crypto failure, data write failure, journal append failure, and remount",
            "production lock-hold evidence before and after the refactor",
        ],
        "decision": decision,
        "negative_claim_guard": (
            "Do not claim fine-grained locking or parallel publication until fd_lock "
            "snapshotting, durable-publish lock splitting, fault evidence, and broad "
            "contention evidence exist."
        ),
    }
    (out_dir / "commit_lock_refactor_feasibility.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Commit Lock Refactor Feasibility",
        "",
        f"- Decision: `{payload['decision']}`",
        "",
        "## Blockers",
        "",
    ]
    for blocker in blockers:
        lines.append(f"- {blocker}")
    lines.extend(["", "## Required Before Safe Refactor", ""])
    for item in payload["required_before_safe_commit_lock_shrink"]:
        lines.append(f"- {item}")
    (out_dir / "commit_lock_refactor_feasibility.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return payload


def render_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Gate 0.15 Concurrency Contract Smoke",
        "",
        f"- Overall pass: `{str(summary['overall_pass']).lower()}`",
        f"- Scope: {summary['scope']}",
        f"- Lock hold events: `{summary['lock_profile']['lock_hold_event_count']}`",
        f"- Observed locks: `{', '.join(summary['lock_profile']['observed_locks'])}`",
        f"- Workload phases: `{len(summary['workload'].get('phases', []))}`",
        "",
        "## Workload Sweep",
        "",
        "| Phase | Worker | Threads | Processes | Iterations/client | Timed out | Wall seconds | Worker errors |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for phase in summary["workload"].get("phases", []):
        lines.append(
            f"| `{phase['name']}` | `{phase.get('worker', 'unknown')}` | "
            f"{phase.get('thread_count', 0)} | {phase.get('process_count', 0)} | "
            f"{phase['iterations_per_thread']} | `{str(phase['timed_out']).lower()}` | "
            f"{phase['wall_seconds']:.6f} | {len(phase.get('worker_errors', []))} |"
        )
    lines.extend(
        [
        "",
        "## Stress Coverage",
        "",
        f"- Max thread count: `{summary['workload'].get('coverage', {}).get('max_thread_count', 0)}`",
        f"- Same-file lifecycle phases: `{summary['workload'].get('coverage', {}).get('same_file_lifecycle_phases', 0)}`",
        f"- Disjoint lifecycle phases: `{summary['workload'].get('coverage', {}).get('disjoint_lifecycle_phases', 0)}`",
        f"- Same-file process lifecycle phases: `{summary['workload'].get('coverage', {}).get('same_file_process_lifecycle_phases', 0)}`",
        f"- Disjoint process lifecycle phases: `{summary['workload'].get('coverage', {}).get('disjoint_process_lifecycle_phases', 0)}`",
        f"- Max process client count: `{summary['workload'].get('coverage', {}).get('max_process_client_count', 0)}`",
        f"- Timed-out phases: `{summary['deadlock_livelock_negative']['timed_out_phase_count']}`",
        f"- Worker errors: `{summary['deadlock_livelock_negative']['worker_error_count']}`",
        f"- Blocking syscall profile pass: `{str(summary['blocking_syscall_profile']['overall_pass']).lower()}`",
        "",
        "## Feasibility Decision",
        "",
            f"- Commit-lock refactor decision: `{summary['commit_lock_refactor_feasibility']['decision']}`",
            f"- Reader visibility probe pass: `{str(summary['reader_visibility_probe']['overall_pass']).lower()}`",
            f"- Reader observed old data during paused publish: `{str(summary['reader_visibility_probe']['during_pause_observed_old']).lower()}`",
            f"- Reader observed new data after publish: `{str(summary['reader_visibility_probe']['after_publish_observed_new']).lower()}`",
            "",
            "## Lock Hold Summary",
            "",
            "| Lock@site | Count | hold p50 ns | hold p95 ns | hold p99 ns | wait p99 ns |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, stats in summary["lock_profile"]["by_lock_and_site"].items():
        lines.append(
            f"| `{key}` | {stats['count']} | {stats['hold_ns']['p50']:.0f} | "
            f"{stats['hold_ns']['p95']:.0f} | {stats['hold_ns']['p99']:.0f} | "
            f"{stats['wait_ns']['p99']:.0f} |"
        )
    lines.extend(
        [
            "",
            "## Blocking Syscall Profile",
            "",
            "| Syscall | Count | p50 ns | p95 ns | p99 ns | max ns |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    blocking_summary = summary["blocking_syscall_profile"].get("summary", {})
    for syscall, stats in blocking_summary.get("by_syscall", {}).items():
        lines.append(
            f"| `{syscall}` | {stats['count']} | {stats['p50_ns']:.0f} | "
            f"{stats['p95_ns']:.0f} | {stats['p99_ns']:.0f} | "
            f"{stats['max_ns']} |"
        )
    lines.extend(
        [
            "",
            "## Non-Closure",
            "",
        ]
    )
    for item in summary["not_closed"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--password", default="aegisq-concurrency-contract-password")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--thread-counts", default="1,2,4")
    parser.add_argument("--phase-timeout-s", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()
    thread_counts = parse_thread_counts(args.thread_counts)

    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{out_dir} exists; pass --overwrite to replace this harness output")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not FUSE_BIN.exists():
        raise SystemExit(f"missing FUSE binary: {FUSE_BIN}")

    platform = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": command_capture(["git", "rev-parse", "HEAD"], timeout_s=5.0),
        "git_dirty_short": command_capture(["git", "status", "--short"], timeout_s=5.0),
        "uname": command_capture(["uname", "-a"], timeout_s=5.0),
        "mountpoint_tool": command_capture(["mountpoint", "--version"], timeout_s=5.0),
        "fusermount": command_capture([fusermount_command(), "--version"], timeout_s=5.0),
    }
    (out_dir / "platform_manifest.json").write_text(
        json.dumps(platform, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "lock_order_table.json").write_text(
        json.dumps(lock_order_table(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    feasibility = commit_lock_refactor_feasibility(out_dir)

    self_test = command_capture([str(FUSE_BIN), "--self-test"], timeout_s=60.0)
    (out_dir / "pqc_fuse_self_test.json").write_text(
        json.dumps(self_test, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if self_test.get("returncode") != 0:
        raise SystemExit("pqc_fuse --self-test failed; refusing to run concurrency smoke")

    storage_dir = Path(tempfile.mkdtemp(prefix="aegisq_concurrency_storage_"))
    mount_dir = Path(tempfile.mkdtemp(prefix="aegisq_concurrency_mnt_"))
    cleanup: dict[str, Any] = {
        "storage_dir": str(storage_dir),
        "mount_dir": str(mount_dir),
        "removed": False,
    }
    fuse: FuseHandle | None = None
    workload: dict[str, Any] = {}
    unmount: dict[str, Any] = {}
    abort_reason: str | None = None
    try:
        fuse = start_fuse(storage_dir, mount_dir, out_dir, args.password)
        workload = run_workload(
            mount_dir,
            args.iterations,
            args.block_size,
            thread_counts,
            args.phase_timeout_s,
        )
    except BaseException as exc:  # noqa: BLE001 - retained as artifact evidence.
        abort_reason = repr(exc)
    finally:
        unmount = stop_fuse(fuse, mount_dir, out_dir)
        if not args.keep_temp:
            shutil.rmtree(storage_dir, ignore_errors=True)
            shutil.rmtree(mount_dir, ignore_errors=True)
            cleanup["removed"] = True

    trace_path = out_dir / "lock_profile_trace.jsonl"
    lock_profile = summarize_lock_trace(trace_path)
    reader_visibility = run_reader_visibility_probe(
        out_dir, args.password, args.block_size)
    blocking_profile = run_blocking_syscall_profile(
        out_dir, args.password, args.block_size, args.phase_timeout_s)
    observed = set(lock_profile["observed_locks"])
    worker_errors = workload.get("errors", []) if workload else []
    timed_out_phases = [
        phase for phase in workload.get("phases", [])
        if phase.get("timed_out")
    ] if workload else []
    coverage = workload.get("coverage", {}) if workload else {}
    expected_phase_count = len(thread_counts)
    lifecycle_coverage_pass = (
        coverage.get("same_file_lifecycle_phases") == expected_phase_count
        and coverage.get("disjoint_lifecycle_phases") == expected_phase_count
    )
    process_client_coverage_pass = (
        coverage.get("same_file_process_lifecycle_phases") == expected_phase_count
        and coverage.get("disjoint_process_lifecycle_phases") == expected_phase_count
        and coverage.get("max_process_client_count") == max(thread_counts)
    )
    overall_pass = (
        abort_reason is None
        and not worker_errors
        and not timed_out_phases
        and unmount.get("returncode") == 0
        and lock_profile["lock_hold_event_count"] > 0
        and {"fd_lock", "commit_lock"}.issubset(observed)
        and lifecycle_coverage_pass
        and process_client_coverage_pass
        and reader_visibility.get("overall_pass") is True
        and blocking_profile.get("overall_pass") is True
    )
    summary = {
        "schema_version": 1,
        "generated_by": "experiments/run_concurrency_contract_smoke.py",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scope": "production mounted-path lock timing smoke for current strict mode",
        "overall_pass": overall_pass,
        "abort_reason": abort_reason,
        "workload": workload,
        "thread_counts": thread_counts,
        "phase_timeout_s": args.phase_timeout_s,
        "deadlock_livelock_negative": {
            "scope": (
                "In-process mounted-path same-file, disjoint-file, same-file "
                "lifecycle, disjoint lifecycle, and external process lifecycle "
                "phases use a bounded timeout; "
                "absence of timed-out phases is a negative deadlock/livelock smoke, "
                "not a complete off-CPU or long-duration stress campaign."
            ),
            "phase_timeout_s": args.phase_timeout_s,
            "timed_out_phase_count": len(timed_out_phases),
            "worker_error_count": len(worker_errors),
            "lifecycle_coverage_pass": lifecycle_coverage_pass,
            "process_client_coverage_pass": process_client_coverage_pass,
            "blocking_syscall_profile_pass": blocking_profile.get("overall_pass") is True,
        },
        "unmount": unmount,
        "cleanup": cleanup,
        "artifacts": {
            "json": relpath(out_dir / "lock_profile_summary.json"),
            "markdown": relpath(out_dir / "lock_profile_summary.md"),
            "trace": relpath(trace_path),
            "lock_order_table": relpath(out_dir / "lock_order_table.json"),
            "commit_lock_refactor_feasibility": relpath(out_dir / "commit_lock_refactor_feasibility.json"),
            "reader_visibility_probe": relpath(out_dir / "reader_visibility_probe" / "reader_visibility_probe.json"),
            "blocking_syscall_profile": relpath(out_dir / "blocking_syscall_profile" / "blocking_syscall_profile.json"),
            "platform_manifest": relpath(out_dir / "platform_manifest.json"),
            "mount_logs": relpath(out_dir / "mount_logs"),
        },
        "lock_profile": lock_profile,
        "reader_visibility_probe": reader_visibility,
        "blocking_syscall_profile": blocking_profile,
        "commit_lock_refactor_feasibility": feasibility,
        "not_closed": [
            "Gate 0.15 is not closed: this thread/process-count sweep is still narrow and lacks off-CPU profiling and long-duration client-count coverage.",
            "Strict full-tier writeback prepare, crypto, durable publication, authenticated-read recovery, truncate metadata publication, and fallocate metadata publication no longer run under fd_lock or commit_lock on their hot paths.",
            "Release now detaches fds, buffers, and file-state references under fd_lock and closes/frees/releases detached resources after unlocking.",
            "Same-file and disjoint-file lifecycle stress now exercises repeated open/write/fdatasync/read/close on the mounted path under bounded timeouts.",
            "External process lifecycle stress now exercises same-file and disjoint-file client phases up to the maximum configured client count.",
            "A strace blocking-syscall profile now records mounted-path client fdatasync/pwrite/pread syscall durations, but scheduler off-CPU sampling is still absent.",
            "Reader visibility is bounded by committed_generation and covered by one paused-publish probe, but lifecycle stress is still short-duration.",
            "Commit-lock shrinking remains incomplete until reserved-generation fault cases, concurrent reader visibility, and client-count sweeps are expanded.",
            "No paper scalability or parallel-commit claim is justified by this artifact.",
        ],
        "negative_claim_guard": (
            "Do not claim fine-grained locking, scalability, multicore scaling, "
            "or parallel commit from this smoke artifact."
        ),
    }
    (out_dir / "lock_profile_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "lock_profile_summary.md").write_text(render_md(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "overall_pass": overall_pass,
                "json": relpath(out_dir / "lock_profile_summary.json"),
                "trace": relpath(trace_path),
                "lock_hold_event_count": lock_profile["lock_hold_event_count"],
                "observed_locks": lock_profile["observed_locks"],
                "reader_visibility_pass": reader_visibility.get("overall_pass"),
                "blocking_syscall_profile_pass": blocking_profile.get("overall_pass"),
                "thread_counts": thread_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
