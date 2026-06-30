#!/usr/bin/env python3
"""Mounted D2-S0 smoke for crypto-plane route separation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FUSE_BIN = ROOT / "build" / "pqc_fuse"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "crypto_plane_separation"


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"exists": True, "json_error": str(exc)}
    if isinstance(data, dict):
        data["exists"] = True
        return data
    return {"exists": True, "json_error": "top-level value is not an object"}


def mount_is_visible(mount_dir: Path) -> bool:
    mount_path = mount_dir.resolve()
    try:
        with open("/proc/mounts", "r", encoding="utf-8",
                  errors="replace") as proc_mounts:
            for line in proc_mounts:
                fields = line.split()
                if len(fields) >= 3 and fields[2].startswith("fuse"):
                    try:
                        if Path(fields[1]).resolve() == mount_path:
                            return True
                    except OSError:
                        pass
    except FileNotFoundError:
        pass
    return subprocess.run(
        ["mountpoint", "-q", str(mount_dir)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def wait_for_mount(proc: subprocess.Popen[bytes], mount_dir: Path,
                   timeout_s: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if mount_is_visible(mount_dir):
            return True
        if proc.poll() is not None:
            return False
        time.sleep(0.05)
    return False


def wait_for_rekey_log(stderr_path: Path, timeout_s: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if stderr_path.exists():
            text = stderr_path.read_text(encoding="utf-8", errors="replace")
            if "REKEY WORKER: batched" in text:
                return True
        time.sleep(0.05)
    return False


def stop_fuse(proc: subprocess.Popen[bytes] | None, mount_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"unmounted": False, "returncode": None}
    if mount_is_visible(mount_dir):
        for tool in ("fusermount3", "fusermount"):
            exe = shutil.which(tool)
            if not exe:
                continue
            unmount = subprocess.run(
                [exe, "-uz", str(mount_dir)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if unmount.returncode == 0:
                result["unmounted"] = True
                break
    if result["unmounted"] and proc is not None and proc.poll() is None:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    if proc is not None and proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    if proc is not None:
        result["returncode"] = proc.returncode
    return result


def source_checks() -> dict[str, bool]:
    crypto = (ROOT / "code" / "crypto" / "pqc_crypto.c").read_text(
        encoding="utf-8", errors="replace"
    )
    recovery = (ROOT / "code" / "fs" / "pqc_recovery.c").read_text(
        encoding="utf-8", errors="replace"
    )
    writeback = (ROOT / "code" / "storage" / "pqc_writeback.c").read_text(
        encoding="utf-8", errors="replace"
    )
    rekey = (ROOT / "code" / "runtime" / "pqc_rekey.c").read_text(
        encoding="utf-8", errors="replace"
    )
    runtime = (ROOT / "code" / "runtime" / "pqc_runtime.c").read_text(
        encoding="utf-8", errors="replace"
    )
    plane_trace = (ROOT / "code" / "runtime" / "pqc_plane_trace.c").read_text(
        encoding="utf-8", errors="replace"
    )
    return {
        "data_crypto_uses_aes_gcm":
            "EVP_aes_256_gcm" in crypto and
            "PQC_ALGO_AES_256_GCM" in recovery,
        "bulk_data_path_has_no_oqs_kem":
            "OQS_KEM" not in crypto and
            "OQS_KEM" not in recovery and
            "OQS_KEM" not in writeback,
        "keyplane_worker_uses_mlkem_or_kyber":
            "OQS_KEM_encaps" in rekey and
            ("OQS_KEM_alg_ml_kem_768" in runtime or
             "OQS_KEM_alg_kyber_768" in runtime),
        "plane_trace_dump_is_runtime_cleanup":
            "pqc_plane_trace_dump_if_requested" in runtime,
        "plane_trace_records_aes_not_bulk_pqc":
            "data_plane_algorithm" in plane_trace and
            "AES-256-GCM" in plane_trace and
            "key_plane_algorithm" in plane_trace,
    }


def write_read_fsync(path: Path, payload: bytes) -> dict[str, Any]:
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        total = 0
        while total < len(payload):
            written = os.write(fd, payload[total:])
            if written <= 0:
                raise OSError("short write")
            total += written
        os.fsync(fd)
    finally:
        os.close(fd)
    observed = path.read_bytes()
    return {
        "payload_bytes": len(payload),
        "readback_bytes": len(observed),
        "readback_matches": observed == payload,
    }


def start_fuse(storage_dir: Path, mount_dir: Path, out_dir: Path,
               label: str, trace_path: Path, force_rekey: bool
               ) -> tuple[subprocess.Popen[bytes], Path, Path]:
    stdout_path = out_dir / f"{label}.stdout.txt"
    stderr_path = out_dir / f"{label}.stderr.txt"
    env = os.environ.copy()
    env.update({
        "PQC_MASTER_PASSWORD": f"crypto-plane-{label}",
        "PQC_PLANE_TRACE_PATH": str(trace_path),
        "PQC_ADMISSION_TRACE_PATH": str(out_dir / f"{label}.admission.jsonl"),
        "PQC_FRESHNESS_ANCHOR_BACKEND": "file",
        "PQC_FRESHNESS_ANCHOR_PATH": str(storage_dir / ".anchor"),
        "PQC_FRESHNESS_WINDOW_N": "1",
        "PQC_REKEY_BATCH_MAX": "1",
        "PQC_REKEY_BATCH_COLLECT_MS": "1",
    })
    if force_rekey:
        env["PQC_FORCE_REKEY_ON_WRITE"] = "1"
    stdout = stdout_path.open("wb")
    stderr = stderr_path.open("wb")
    try:
        proc = subprocess.Popen(
            [str(FUSE_BIN), str(storage_dir), str(mount_dir), "-f"],
            cwd=ROOT,
            env=env,
            stdout=stdout,
            stderr=stderr,
        )
    finally:
        stdout.close()
        stderr.close()
    return proc, stdout_path, stderr_path


def run_case(out_dir: Path, label: str, force_rekey: bool) -> dict[str, Any]:
    work_dir = Path(tempfile.mkdtemp(prefix=f"aegisq_{label}_"))
    storage_dir = work_dir / "store"
    mount_dir = work_dir / "mnt"
    storage_dir.mkdir()
    mount_dir.mkdir()
    trace_path = out_dir / f"{label}.plane_trace.json"
    if trace_path.exists():
        trace_path.unlink()
    proc: subprocess.Popen[bytes] | None = None
    mounted = False
    workload: dict[str, Any] = {}
    rekey_log_seen = False
    stop_result: dict[str, Any] = {}
    try:
        proc, stdout_path, stderr_path = start_fuse(
            storage_dir, mount_dir, out_dir, label, trace_path, force_rekey
        )
        mounted = wait_for_mount(proc, mount_dir)
        if mounted:
            payload = (f"{label}-aes-gcm-mounted-data-plane\n".encode() *
                       384)
            workload = write_read_fsync(mount_dir / "payload.bin", payload)
            if force_rekey:
                rekey_log_seen = wait_for_rekey_log(stderr_path)
        else:
            stdout_path = out_dir / f"{label}.stdout.txt"
            stderr_path = out_dir / f"{label}.stderr.txt"
        stop_result = stop_fuse(proc, mount_dir)
    finally:
        if proc is not None and proc.poll() is None:
            stop_result = stop_fuse(proc, mount_dir)
        if mount_is_visible(mount_dir):
            stop_result = stop_fuse(proc, mount_dir)
    trace = read_json(trace_path)
    return {
        "label": label,
        "force_rekey": force_rekey,
        "mounted": mounted,
        "work_dir": str(work_dir),
        "storage_dir": str(storage_dir),
        "mount_dir": str(mount_dir),
        "trace_path": relpath(trace_path),
        "stdout": relpath(out_dir / f"{label}.stdout.txt"),
        "stderr": relpath(out_dir / f"{label}.stderr.txt"),
        "workload": workload,
        "rekey_log_seen": rekey_log_seen,
        "stop": stop_result,
        "trace": trace,
    }


def classify(payload: dict[str, Any]) -> dict[str, bool]:
    ordinary = payload["runs"]["ordinary"]
    keyplane = payload["runs"]["forced_keyplane"]
    source = payload["source_checks"]
    ordinary_trace = ordinary["trace"]
    keyplane_trace = keyplane["trace"]
    ordinary_ok = (
        ordinary["mounted"] and
        ordinary["workload"].get("readback_matches") is True and
        ordinary_trace.get("exists") is True and
        int(ordinary_trace.get("data_aes_gcm_encrypt_blocks", 0)) > 0 and
        int(ordinary_trace.get("data_aes_gcm_decrypt_blocks", 0)) > 0 and
        int(ordinary_trace.get("keyplane_batches", 0)) == 0 and
        int(ordinary_trace.get("keyplane_refreshed_files", 0)) == 0 and
        ordinary_trace.get("data_plane_algorithm") == "AES-256-GCM"
    )
    keyplane_ok = (
        keyplane["mounted"] and
        keyplane["workload"].get("readback_matches") is True and
        keyplane["rekey_log_seen"] and
        keyplane_trace.get("exists") is True and
        int(keyplane_trace.get("data_aes_gcm_encrypt_blocks", 0)) > 0 and
        int(keyplane_trace.get("data_aes_gcm_decrypt_blocks", 0)) > 0 and
        int(keyplane_trace.get("keyplane_batches", 0)) > 0 and
        int(keyplane_trace.get("keyplane_refreshed_files", 0)) > 0 and
        keyplane_trace.get("data_plane_algorithm") == "AES-256-GCM" and
        "ML-KEM" in str(keyplane_trace.get("key_plane_algorithm", ""))
    )
    source_ok = all(source.values())
    return {
        "ordinary_read_write_aes_gcm_only": ordinary_ok,
        "forced_rekey_is_keyplane_only": keyplane_ok,
        "source_plane_boundary_present": source_ok,
        "overall_pass": ordinary_ok and keyplane_ok and source_ok,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    verdict = payload["verdict"]
    ordinary_trace = payload["runs"]["ordinary"]["trace"]
    keyplane_trace = payload["runs"]["forced_keyplane"]["trace"]
    lines = [
        "# Crypto Plane Separation Smoke",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Overall pass: `{str(verdict['overall_pass']).lower()}`",
        f"- Ordinary read/write AES-GCM only: "
        f"`{str(verdict['ordinary_read_write_aes_gcm_only']).lower()}`",
        f"- Forced rekey key-plane only: "
        f"`{str(verdict['forced_rekey_is_keyplane_only']).lower()}`",
        "",
        "## Ordinary Mounted I/O",
        "",
        f"- Trace: `{payload['runs']['ordinary']['trace_path']}`",
        f"- AES-GCM encrypt blocks: "
        f"`{ordinary_trace.get('data_aes_gcm_encrypt_blocks')}`",
        f"- AES-GCM decrypt blocks: "
        f"`{ordinary_trace.get('data_aes_gcm_decrypt_blocks')}`",
        f"- Key-plane batches: `{ordinary_trace.get('keyplane_batches')}`",
        f"- Key-plane refreshed files: "
        f"`{ordinary_trace.get('keyplane_refreshed_files')}`",
        "",
        "## Forced Key-Plane Workflow",
        "",
        f"- Trace: `{payload['runs']['forced_keyplane']['trace_path']}`",
        f"- Rekey log seen: "
        f"`{str(payload['runs']['forced_keyplane']['rekey_log_seen']).lower()}`",
        f"- AES-GCM encrypt blocks: "
        f"`{keyplane_trace.get('data_aes_gcm_encrypt_blocks')}`",
        f"- AES-GCM decrypt blocks: "
        f"`{keyplane_trace.get('data_aes_gcm_decrypt_blocks')}`",
        f"- Key-plane batches: `{keyplane_trace.get('keyplane_batches')}`",
        f"- Key-plane refreshed files: "
        f"`{keyplane_trace.get('keyplane_refreshed_files')}`",
        f"- Key-plane work bytes: `{keyplane_trace.get('keyplane_work_bytes')}`",
        "",
        "## Scope",
        "",
        "This smoke proves the mounted implementation separates AES-GCM block "
        "I/O counters from the ML-KEM envelope-refresh workflow. It does not "
        "claim that bulk file data is encrypted directly by PQC primitives.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(out_dir: Path) -> dict[str, Any]:
    if not FUSE_BIN.exists():
        raise SystemExit(f"missing build artifact: {relpath(FUSE_BIN)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "artifact": "crypto_plane_separation",
        "generated_at": now_utc(),
        "generated_by": "code/experiments/run_crypto_plane_trace_smoke.py",
        "fuse_binary": relpath(FUSE_BIN),
        "source_checks": source_checks(),
        "runs": {},
    }
    payload["runs"]["ordinary"] = run_case(out_dir, "ordinary", False)
    payload["runs"]["forced_keyplane"] = run_case(
        out_dir, "forced_keyplane", True
    )
    payload["verdict"] = classify(payload)
    json_path = out_dir / "crypto_plane_trace_smoke.json"
    md_path = out_dir / "crypto_plane_trace_smoke.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True),
                         encoding="utf-8")
    write_markdown(md_path, payload)
    payload["json_path"] = relpath(json_path)
    payload["md_path"] = relpath(md_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    payload = run(args.out_dir)
    print(json.dumps({
        "overall_pass": payload["verdict"]["overall_pass"],
        "json_path": payload["json_path"],
        "md_path": payload["md_path"],
    }, indent=2, sort_keys=True))
    return 0 if payload["verdict"]["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
