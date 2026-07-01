#!/usr/bin/env python3
"""Stdout-only second macrobenchmark smoke for secure inference logging.

The runner exercises the mounted production FUSE path with an inference-log
style append workload, remounts the same storage root, and verifies record order
and payload hashes. It deliberately writes no retained repository outputs.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FUSE_BIN = Path(os.environ.get("PQC_FUSE_BIN", ROOT / "build" / "pqc_fuse"))


@dataclass
class FuseHandle:
    proc: subprocess.Popen[bytes]
    stdout: Any
    stderr: Any


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * q)
    return ordered[index]


def summarize_us(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "avg_us": 0.0,
            "p50_us": 0.0,
            "p95_us": 0.0,
            "p99_us": 0.0,
            "max_us": 0.0,
        }
    return {
        "count": len(values),
        "avg_us": sum(values) / len(values),
        "p50_us": percentile(values, 0.50),
        "p95_us": percentile(values, 0.95),
        "p99_us": percentile(values, 0.99),
        "max_us": max(values),
    }


def deterministic_payload(seq: int, payload_bytes: int) -> bytes:
    seed = hashlib.sha256(f"aegisq-secure-inference-log:{seq}".encode()).digest()
    out = bytearray()
    counter = 0
    while len(out) < payload_bytes:
        out.extend(hashlib.sha256(seed + counter.to_bytes(8, "little")).digest())
        counter += 1
    return bytes(out[:payload_bytes])


def mount_is_visible(mount_dir: Path) -> bool:
    return subprocess.run(
        ["mountpoint", "-q", str(mount_dir)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def fusermount_command() -> str:
    for name in ("fusermount3", "fusermount"):
        if shutil.which(name):
            return name
    return "fusermount3"


def start_fuse(storage_dir: Path, mount_dir: Path, work_dir: Path,
               password: str, publication_mode: str) -> FuseHandle:
    log_dir = work_dir / "daemon_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout = (log_dir / "pqc_fuse.stdout.txt").open("wb")
    stderr = (log_dir / "pqc_fuse.stderr.txt").open("wb")
    env = os.environ.copy()
    env.update({
        "PQC_MASTER_PASSWORD": password,
        "PQC_FRESHNESS_ANCHOR_BACKEND": "file",
        "PQC_FRESHNESS_ANCHOR_PATH": str(storage_dir / ".anchor"),
        "PQC_ENABLE_QOS_THROTTLE_ON_WRITE": "0",
        "PQC_KEY_ROTATION_INTERVAL_S": "0",
    })
    if publication_mode != "strict":
        env["PQC_PUBLICATION_MODE"] = publication_mode

    proc = subprocess.Popen(
        [str(FUSE_BIN), str(storage_dir), str(mount_dir), "-f"],
        cwd=ROOT,
        env=env,
        stdout=stdout,
        stderr=stderr,
    )
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if mount_is_visible(mount_dir):
            return FuseHandle(proc=proc, stdout=stdout, stderr=stderr)
        if proc.poll() is not None:
            stdout.close()
            stderr.close()
            raise RuntimeError(f"FUSE exited before mount: rc={proc.returncode}")
        time.sleep(0.05)

    stdout.close()
    stderr.close()
    raise TimeoutError("timed out waiting for AEGIS-Q FUSE mount")


def stop_fuse(handle: FuseHandle | None, mount_dir: Path) -> dict[str, Any]:
    unmount_rc: int | None = None
    unmount_stderr = ""
    if mount_is_visible(mount_dir):
        unmount = subprocess.run(
            [fusermount_command(), "-u", str(mount_dir)],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        unmount_rc = unmount.returncode
        unmount_stderr = unmount.stderr.strip()
        if unmount_rc != 0:
            lazy = subprocess.run(
                [fusermount_command(), "-uz", str(mount_dir)],
                cwd=ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            unmount_rc = lazy.returncode
            unmount_stderr = lazy.stderr.strip()

    fuse_rc: int | None = None
    if handle is not None:
        if handle.proc.poll() is None:
            handle.proc.send_signal(signal.SIGINT)
            try:
                handle.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                handle.proc.kill()
                handle.proc.wait(timeout=5)
        fuse_rc = handle.proc.returncode
        handle.stdout.close()
        handle.stderr.close()

    return {
        "unmount_rc": unmount_rc,
        "unmount_stderr": unmount_stderr,
        "fuse_rc": fuse_rc,
    }


def make_record(seq: int, payload_bytes: int) -> tuple[bytes, dict[str, Any]]:
    payload = deterministic_payload(seq, payload_bytes)
    expected = {
        "seq": seq,
        "payload_bytes": payload_bytes,
        "payload_sha256": sha256_bytes(payload),
    }
    record = {
        "schema": "aegisq.secure_inference_log.v1",
        "seq": seq,
        "trace_id": f"inference-{seq:08d}",
        "model": "macro-smoke-classifier",
        "deadline_ms": 33,
        "result": "accepted",
        "payload_bytes": payload_bytes,
        "payload_sha256": expected["payload_sha256"],
        "payload_b64": base64.b64encode(payload).decode("ascii"),
    }
    return (
        json.dumps(record, separators=(",", ":"), sort_keys=True).encode() + b"\n",
        expected,
    )


def append_records(log_path: Path, records: int,
                   payload_bytes: int) -> dict[str, Any]:
    expected: list[dict[str, Any]] = []
    latencies: list[dict[str, Any]] = []
    append_start = time.monotonic_ns()
    fd = os.open(log_path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
    try:
        for seq in range(records):
            record, expected_record = make_record(seq, payload_bytes)
            write_start = time.monotonic_ns()
            total = 0
            while total < len(record):
                written = os.write(fd, record[total:])
                if written <= 0:
                    raise OSError("short write while appending inference log")
                total += written
            write_end = time.monotonic_ns()
            if hasattr(os, "fdatasync"):
                os.fdatasync(fd)
            else:
                os.fsync(fd)
            fsync_end = time.monotonic_ns()
            write_us = (write_end - write_start) / 1000.0
            sync_us = (fsync_end - write_end) / 1000.0
            expected.append(expected_record)
            latencies.append({
                "seq": seq,
                "record_bytes": len(record),
                "write_us": write_us,
                "sync_us": sync_us,
                "durable_append_us": write_us + sync_us,
            })
    finally:
        os.close(fd)
    append_end = time.monotonic_ns()
    append_wall_us = (append_end - append_start) / 1000.0
    total_record_bytes = sum(int(entry["record_bytes"]) for entry in latencies)
    total_payload_bytes = records * payload_bytes
    durable_append_us = [
        float(entry["durable_append_us"]) for entry in latencies
    ]
    write_us = [float(entry["write_us"]) for entry in latencies]
    sync_us = [float(entry["sync_us"]) for entry in latencies]
    append_seconds = append_wall_us / 1_000_000.0
    summary = {
        "records": records,
        "payload_bytes_per_record": payload_bytes,
        "total_payload_bytes": total_payload_bytes,
        "total_record_bytes": total_record_bytes,
        "append_wall_us": append_wall_us,
        "durable_append_latency": summarize_us(durable_append_us),
        "write_latency": summarize_us(write_us),
        "sync_latency": summarize_us(sync_us),
        "payload_mib_s": (
            (total_payload_bytes / (1024.0 * 1024.0)) / append_seconds
            if append_seconds > 0.0 else 0.0
        ),
        "record_mib_s": (
            (total_record_bytes / (1024.0 * 1024.0)) / append_seconds
            if append_seconds > 0.0 else 0.0
        ),
        "records_s": records / append_seconds if append_seconds > 0.0 else 0.0,
    }
    return {"expected": expected, "latencies": latencies, "summary": summary}


def parse_log(data: bytes) -> dict[str, Any]:
    decoded: list[dict[str, Any]] = []
    malformed = 0
    hash_mismatches = 0
    for line in data.splitlines():
        try:
            record = json.loads(line.decode("utf-8"))
            payload = base64.b64decode(record["payload_b64"].encode("ascii"))
            if sha256_bytes(payload) != record.get("payload_sha256"):
                hash_mismatches += 1
            decoded.append({
                "seq": int(record["seq"]),
                "payload_bytes": int(record["payload_bytes"]),
                "payload_sha256": str(record["payload_sha256"]),
                "record_sha256": sha256_bytes(line),
            })
        except Exception:  # noqa: BLE001 - malformed records are test output.
            malformed += 1
    return {
        "record_count": len(decoded),
        "malformed": malformed,
        "hash_mismatches": hash_mismatches,
        "records": decoded,
        "ordered": [record["seq"] for record in decoded] == list(range(len(decoded))),
    }


def visible_sidecars(mount_dir: Path) -> list[str]:
    hidden: list[str] = []
    try:
        for name in os.listdir(mount_dir):
            if name.startswith(".pqc") or name.endswith(".pqcdata") or name.endswith(".pqcmeta"):
                hidden.append(name)
    except OSError:
        pass
    return sorted(hidden)


def verify_expected(parsed: dict[str, Any],
                    expected: list[dict[str, Any]]) -> bool:
    if parsed["record_count"] != len(expected):
        return False
    if parsed["malformed"] != 0 or parsed["hash_mismatches"] != 0:
        return False
    if not parsed["ordered"]:
        return False
    for observed, wanted in zip(parsed["records"], expected):
        if observed["seq"] != wanted["seq"]:
            return False
        if observed["payload_bytes"] != wanted["payload_bytes"]:
            return False
        if observed["payload_sha256"] != wanted["payload_sha256"]:
            return False
    return True


def run_smoke(records: int, payload_bytes: int,
              publication_mode: str) -> dict[str, Any]:
    if not FUSE_BIN.exists():
        raise SystemExit(f"missing build binary: {FUSE_BIN}")

    work_dir = Path(tempfile.mkdtemp(prefix="aegisq_inference_log_macro_"))
    storage_dir = work_dir / "storage"
    mount_dir = work_dir / "mount"
    storage_dir.mkdir()
    mount_dir.mkdir()
    password = "secure-inference-log-macro-smoke"
    handle: FuseHandle | None = None
    remount_handle: FuseHandle | None = None
    first_unmount: dict[str, Any] = {}
    second_unmount: dict[str, Any] = {}
    result: dict[str, Any] = {
        "schema_version": 1,
        "generated_by": "code/experiments/run_secure_inference_log_macro_smoke.py",
        "generated_utc": now_utc(),
        "workload": "secure_inference_logging",
        "publication_mode": publication_mode,
        "records_requested": records,
        "payload_bytes": payload_bytes,
        "stdout_only": True,
        "retained_repository_outputs": False,
    }

    try:
        handle = start_fuse(storage_dir, mount_dir, work_dir, password,
                            publication_mode)
        log_path = mount_dir / "inference.log"
        append = append_records(log_path, records, payload_bytes)
        mounted_data = log_path.read_bytes()
        mounted_parse = parse_log(mounted_data)
        mounted_sidecars = visible_sidecars(mount_dir)
        first_unmount = stop_fuse(handle, mount_dir)
        handle = None

        remount_handle = start_fuse(storage_dir, mount_dir, work_dir, password,
                                    publication_mode)
        remount_data = (mount_dir / "inference.log").read_bytes()
        remount_parse = parse_log(remount_data)
        remount_sidecars = visible_sidecars(mount_dir)
        second_unmount = stop_fuse(remount_handle, mount_dir)
        remount_handle = None

        mounted_ok = verify_expected(mounted_parse, append["expected"])
        remount_ok = verify_expected(remount_parse, append["expected"])
        unmounts_ok = (
            first_unmount.get("unmount_rc") == 0 and
            second_unmount.get("unmount_rc") == 0
        )
        result.update({
            "append": append,
            "mounted_readback": {
                "bytes": len(mounted_data),
                "sha256": sha256_bytes(mounted_data),
                "parse": mounted_parse,
                "visible_sidecars": mounted_sidecars,
                "pass": mounted_ok,
            },
            "remount_readback": {
                "bytes": len(remount_data),
                "sha256": sha256_bytes(remount_data),
                "parse": remount_parse,
                "visible_sidecars": remount_sidecars,
                "pass": remount_ok,
            },
            "unmount": {
                "first": first_unmount,
                "second": second_unmount,
                "pass": unmounts_ok,
            },
            "overall_pass": mounted_ok and remount_ok and unmounts_ok,
        })
    except Exception as exc:  # noqa: BLE001 - stdout result is the proof.
        result.update({
            "error": repr(exc),
            "first_unmount": first_unmount,
            "second_unmount": second_unmount,
            "overall_pass": False,
        })
    finally:
        if handle is not None:
            result["cleanup_first_unmount"] = stop_fuse(handle, mount_dir)
        if remount_handle is not None:
            result["cleanup_second_unmount"] = stop_fuse(remount_handle, mount_dir)
        shutil.rmtree(work_dir, ignore_errors=True)

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=int, default=1)
    parser.add_argument("--payload-bytes", type=int, default=16384)
    parser.add_argument(
        "--publication-mode",
        choices=("strict", "epoch-redo-log"),
        default="strict",
    )
    args = parser.parse_args()
    if args.records <= 0:
        raise SystemExit("--records must be positive")
    if args.payload_bytes <= 0:
        raise SystemExit("--payload-bytes must be positive")

    result = run_smoke(args.records, args.payload_bytes, args.publication_mode)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("overall_pass") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
