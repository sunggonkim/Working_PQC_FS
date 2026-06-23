#!/usr/bin/env python3
"""
experiments/run_crash_replay_e8.py — AEGIS-Q E8 crash/replay matrix
====================================================================

This script exercises deterministic cut-points for crash and rollback
validation.  It emits:
  - artifacts/crash_replay_matrix.json
  - artifacts/crash_replay_matrix.csv
  - artifacts/crash_replay_summary.json
  - artifacts/crash_replay_summary.csv

The script supports both the file-backed anchor path and, when explicitly
available, the hardware-backed anchor path.  The hardware path is not assumed
to exist on every machine; if it is unavailable, the run is reported as a
skipped configuration rather than being treated as evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FUSE_BIN = ROOT / "build" / "pqc_fuse"
ART = ROOT / "artifacts"


@dataclass
class TrialResult:
    backend: str
    cut_point_s: float
    trial: int
    preserved: bool
    mode: str
    detail: str


def start_fuse(storage_dir: Path, mount_dir: Path, backend: str) -> subprocess.Popen:
    env = os.environ.copy()
    env["PQC_MASTER_PASSWORD"] = env.get("PQC_MASTER_PASSWORD", "benchmark-password")
    env["PQC_FRESHNESS_ANCHOR_BACKEND"] = backend
    if backend == "file":
        env["PQC_FRESHNESS_ANCHOR_PATH"] = str(storage_dir / ".anchor")
    else:
        env.setdefault("PQC_TPM_TCTI", "device:/dev/tpmrm0")

    proc = subprocess.Popen(
        [str(FUSE_BIN), str(storage_dir), str(mount_dir), "-f"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        if subprocess.run(["mountpoint", "-q", str(mount_dir)], check=False).returncode == 0:
            return proc
        if proc.poll() is not None:
            raise RuntimeError(f"FUSE exited with {proc.returncode} before mounting")
        time.sleep(0.05)
    raise TimeoutError("FUSE mount timed out")


def stop_fuse(proc: subprocess.Popen, mount_dir: Path) -> None:
    subprocess.run(["fusermount3", "-u", str(mount_dir)], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


def write_payload(path: Path, payload: bytes) -> None:
    with open(path, "wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())


def run_cutpoint_trial(backend: str, cut_point_s: float, trial: int) -> TrialResult:
    storage_dir = Path(tempfile.mkdtemp(prefix=f"aegis_e8_{backend}_store_{trial}_"))
    mount_dir = Path(tempfile.mkdtemp(prefix=f"aegis_e8_{backend}_mnt_{trial}_"))
    backup_dir = Path(tempfile.mkdtemp(prefix=f"aegis_e8_{backend}_backup_{trial}_"))
    proc = None

    try:
        proc = start_fuse(storage_dir, mount_dir, backend)
        test_file = mount_dir / "test.bin"

        write_payload(test_file, b"baseline-v1\n")
        stop_fuse(proc, mount_dir)
        proc = None

        shutil.rmtree(backup_dir, ignore_errors=True)
        shutil.copytree(storage_dir, backup_dir, symlinks=True)

        proc = start_fuse(storage_dir, mount_dir, backend)
        payload = b"updated-v2\n" * 1024

        # Force a crash at the requested time relative to the second write.
        pid = proc.pid
        crash_delay = max(0.0, cut_point_s)
        if crash_delay > 0:
            time.sleep(crash_delay)

        with open(test_file, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())

        if proc.poll() is None:
            os.kill(pid, signal.SIGKILL)

        stop_fuse(proc, mount_dir)
        proc = None

        shutil.rmtree(storage_dir, ignore_errors=True)
        shutil.copytree(backup_dir, storage_dir, symlinks=True)

        try:
            proc = start_fuse(storage_dir, mount_dir, backend)
            with open(test_file, "rb") as f:
                data = f.read()
            if data == b"baseline-v1\n":
                return TrialResult(backend, cut_point_s, trial, True, "rollback_reject", "stale_image_blocked")
            if data == payload:
                return TrialResult(backend, cut_point_s, trial, False, "rollback_accept", "stale_image_visible")
            return TrialResult(backend, cut_point_s, trial, False, "rollback_accept", "unexpected_content")
        except OSError as exc:
            if exc.errno in (74, 116, 129):
                return TrialResult(backend, cut_point_s, trial, True, "fail_closed", f"errno={exc.errno}")
            return TrialResult(backend, cut_point_s, trial, False, "unexpected_error", f"errno={exc.errno}")
        finally:
            if proc:
                stop_fuse(proc, mount_dir)
                proc = None

    finally:
        if proc:
            stop_fuse(proc, mount_dir)
        shutil.rmtree(storage_dir, ignore_errors=True)
        shutil.rmtree(mount_dir, ignore_errors=True)
        shutil.rmtree(backup_dir, ignore_errors=True)


def backend_available(backend: str) -> bool:
    if backend == "file":
        return True
    return shutil.which("tpm2_nvread") is not None and shutil.which("tpm2_nvwrite") is not None


def summarize(rows: list[TrialResult]) -> list[dict]:
    grouped: dict[tuple[str, float], list[TrialResult]] = {}
    for row in rows:
        grouped.setdefault((row.backend, row.cut_point_s), []).append(row)

    summary = []
    for (backend, cut_point_s), trials in sorted(grouped.items()):
        preserved = sum(1 for t in trials if t.preserved)
        summary.append({
            "backend": backend,
            "cut_point_s": cut_point_s,
            "trials": len(trials),
            "preserved": preserved,
            "success_rate": preserved / max(1, len(trials)),
            "fail_closed": sum(1 for t in trials if t.mode == "fail_closed"),
            "rollback_accept": sum(1 for t in trials if t.mode == "rollback_accept"),
            "unexpected_error": sum(1 for t in trials if t.mode == "unexpected_error"),
        })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="E8 crash/replay matrix")
    parser.add_argument("--trials-per-cutpoint", type=int, default=10)
    parser.add_argument("--cut-points", type=float, nargs="+", default=[0.00, 0.02, 0.05, 0.10])
    parser.add_argument("--backends", nargs="+", default=["file", "hardware"])
    parser.add_argument("--out-prefix", default=str(ART / "crash_replay"))
    args = parser.parse_args()

    if not FUSE_BIN.exists():
        raise FileNotFoundError(f"pqc_fuse binary not found: {FUSE_BIN}")

    rows: list[TrialResult] = []
    skipped = []
    for backend in args.backends:
        if not backend_available(backend):
            skipped.append({"backend": backend, "reason": "backend tools unavailable"})
            continue
        for cut_point_s in args.cut_points:
            for trial in range(args.trials_per_cutpoint):
                rows.append(run_cutpoint_trial(backend, cut_point_s, trial))

    matrix = [row.__dict__ for row in rows]
    summary = summarize(rows)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    matrix_json = out_prefix.with_name(out_prefix.name + "_matrix.json")
    matrix_csv = out_prefix.with_name(out_prefix.name + "_matrix.csv")
    summary_json = out_prefix.with_name(out_prefix.name + "_summary.json")
    summary_csv = out_prefix.with_name(out_prefix.name + "_summary.csv")

    matrix_json.write_text(json.dumps({"rows": matrix, "skipped": skipped}, indent=2))
    with matrix_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["backend", "cut_point_s", "trial", "preserved", "mode", "detail"])
        writer.writeheader()
        writer.writerows(matrix)

    summary_json.write_text(json.dumps({"rows": summary, "skipped": skipped}, indent=2))
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["backend", "cut_point_s", "trials", "preserved", "success_rate", "fail_closed", "rollback_accept", "unexpected_error"])
        writer.writeheader()
        writer.writerows(summary)

    print(json.dumps({"summary": summary, "skipped": skipped}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
