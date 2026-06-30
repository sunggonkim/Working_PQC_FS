#!/usr/bin/env python3
"""Retained nonce/generation fault matrix for the implemented FUSE format.

The matrix is deliberately narrow: it exercises the final ``build/pqc_fuse``
binary and the persisted ``.pqcmeta`` journal, then classifies each trial with
the same oracle vocabulary used by the paper checklist.  It does not claim full
power-loss or daemon-crash certification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import struct
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FUSE_BIN = ROOT / "build" / "pqc_fuse"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "generation_fault_matrix"

PQC_LOGICAL_BLOCK_SIZE = 4096
PQC_JOURNAL_MAGIC = 0x5051434A4E4C3031
PQC_JOURNAL_VERSION = 1
PQC_JOURNAL_COMMITTED = 0x434F4D4D
PQC_ALGO_AES_256_GCM = 0
PQC_XATTR_CHECKPOINT = "user.pqc_checkpoint"
RECORD_SIZE = 96
CHECKPOINT_SIZE = 80

ALLOWED_VERDICTS = {
    "previous_committed",
    "latest_committed",
    "fail_closed",
    "silent_corruption",
    "unexpected_liveness_failure",
}


@dataclass
class FuseProc:
    proc: subprocess.Popen[bytes]
    stdout: Any
    stderr: Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_command(command: list[str], out_dir: Path, name: str) -> dict[str, Any]:
    stdout_path = out_dir / f"{name}.stdout.txt"
    stderr_path = out_dir / f"{name}.stderr.txt"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        proc = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
    stdout_data = stdout_path.read_bytes()
    stderr_data = stderr_path.read_bytes()
    return {
        "command": command,
        "returncode": proc.returncode,
        "stdout": str(stdout_path.relative_to(ROOT)),
        "stderr": str(stderr_path.relative_to(ROOT)),
        "stdout_sha256": sha256_bytes(stdout_data),
        "stderr_sha256": sha256_bytes(stderr_data),
    }


def start_fuse(
    storage_dir: Path,
    mount_dir: Path,
    password: str,
    out_dir: Path,
    label: str,
    extra_env: dict[str, str] | None = None,
) -> FuseProc:
    env = os.environ.copy()
    env["PQC_MASTER_PASSWORD"] = password
    env["PQC_FRESHNESS_ANCHOR_BACKEND"] = "file"
    env["PQC_FRESHNESS_ANCHOR_PATH"] = str(storage_dir / ".anchor")
    if extra_env:
        env.update(extra_env)

    log_dir = out_dir / "mount_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout = (log_dir / f"{label}.stdout.txt").open("wb")
    stderr = (log_dir / f"{label}.stderr.txt").open("wb")
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
            raise RuntimeError(f"FUSE exited before mount for {label}: rc={proc.returncode}")
        time.sleep(0.05)

    stdout.close()
    stderr.close()
    raise TimeoutError(f"timed out waiting for FUSE mount for {label}")


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


def write_fsync(path: Path, payload: bytes, offset: int | None = None) -> None:
    mode = "r+b" if offset is not None else "wb"
    with path.open(mode) as f:
        if offset is not None:
            f.seek(offset)
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())


def read_oracle(path: Path, previous: bytes | None, latest: bytes) -> tuple[str, str | None]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        return "fail_closed", f"errno={exc.errno}"
    if data == latest:
        return "latest_committed", None
    if previous is not None and data == previous:
        return "previous_committed", None
    return "silent_corruption", f"len={len(data)} sha256={sha256_bytes(data)}"


def retain_file(src: Path, out_dir: Path, name: str) -> dict[str, Any]:
    retained_dir = out_dir / "journals"
    retained_dir.mkdir(parents=True, exist_ok=True)
    dest = retained_dir / name
    shutil.copy2(src, dest)
    data = dest.read_bytes()
    return {
        "path": str(dest.relative_to(ROOT)),
        "bytes": len(data),
        "sha256": sha256_bytes(data),
    }


def parse_record(raw: bytes, index: int) -> dict[str, Any]:
    prefix = raw[:64]
    digest = raw[64:96]
    magic = int.from_bytes(raw[0:8], "little")
    version = int.from_bytes(raw[8:12], "little")
    committed = int.from_bytes(raw[12:16], "little")
    logical_block = int.from_bytes(raw[16:24], "little")
    generation = int.from_bytes(raw[24:32], "little")
    ciphertext_offset = int.from_bytes(raw[32:40], "little")
    plaintext_length = int.from_bytes(raw[40:44], "little")
    algorithm_id = int.from_bytes(raw[44:48], "little")
    computed = hashlib.sha256(prefix).digest()
    valid = (
        magic == PQC_JOURNAL_MAGIC
        and version == PQC_JOURNAL_VERSION
        and committed == PQC_JOURNAL_COMMITTED
        and plaintext_length <= PQC_LOGICAL_BLOCK_SIZE
        and algorithm_id == PQC_ALGO_AES_256_GCM
        and computed == digest
    )
    return {
        "index": index,
        "valid": valid,
        "logical_block": logical_block,
        "generation": generation,
        "ciphertext_offset": ciphertext_offset,
        "plaintext_length": plaintext_length,
        "algorithm_id": algorithm_id,
        "raw_sha256": sha256_bytes(raw),
        "_raw": raw,
    }


def parse_journal(journal_path: Path) -> dict[str, Any]:
    data = journal_path.read_bytes() if journal_path.exists() else b""
    full = len(data) // RECORD_SIZE
    records = [parse_record(data[i * RECORD_SIZE : (i + 1) * RECORD_SIZE], i) for i in range(full)]
    valid = [r for r in records if r["valid"]]
    pairs: dict[tuple[int, int], int] = {}
    duplicates = []
    for record in valid:
        pair = (record["logical_block"], record["generation"])
        pairs[pair] = pairs.get(pair, 0) + 1
    for (logical_block, generation), count in sorted(pairs.items()):
        if count > 1:
            duplicates.append({
                "logical_block": logical_block,
                "generation": generation,
                "count": count,
            })
    visible = [{k: v for k, v in r.items() if k != "_raw"} for r in records]
    return {
        "path": str(journal_path.relative_to(ROOT)) if journal_path.is_relative_to(ROOT) else str(journal_path),
        "bytes": len(data),
        "record_size": RECORD_SIZE,
        "full_records": full,
        "torn_tail_bytes": len(data) % RECORD_SIZE,
        "valid_records": len(valid),
        "records": visible,
        "duplicate_valid_pairs": duplicates,
        "_records_with_raw": records,
    }


def journal_max_generation(summary: dict[str, Any]) -> int:
    generations = [
        int(record["generation"])
        for record in summary.get("records", [])
        if record.get("valid")
    ]
    return max(generations) if generations else 0


def parse_checkpoint_xattr(marker_path: Path) -> dict[str, Any]:
    try:
        data = os.getxattr(marker_path, PQC_XATTR_CHECKPOINT)
    except OSError as exc:
        return {
            "path": str(marker_path),
            "present": False,
            "errno": exc.errno,
            "error": repr(exc),
        }
    result: dict[str, Any] = {
        "path": str(marker_path),
        "present": True,
        "bytes": len(data),
        "sha256": sha256_bytes(data),
    }
    if len(data) == CHECKPOINT_SIZE:
        magic, version, reserved, file_id, sequence, logical_size, max_generation = struct.unpack(
            "<QIIQQQQ", data[:48]
        )
        result.update(
            {
                "magic": magic,
                "version": version,
                "reserved": reserved,
                "file_id": file_id,
                "sequence": sequence,
                "logical_size": logical_size,
                "max_generation": max_generation,
            }
        )
    return result


def case_partial_update_remount(out_dir: Path, password: str) -> dict[str, Any]:
    storage_dir = Path(tempfile.mkdtemp(prefix="gen_matrix_partial_store_"))
    mount_dir = Path(tempfile.mkdtemp(prefix="gen_matrix_partial_mnt_"))
    handle: FuseProc | None = None
    logical = "partial.bin"
    try:
        base = bytes((i % 251 for i in range(4 * PQC_LOGICAL_BLOCK_SIZE)))
        patch = b"GENERATION-PARTIAL-UPDATE" * 97
        latest = bytearray(base)
        offset = PQC_LOGICAL_BLOCK_SIZE - 137
        latest[offset : offset + len(patch)] = patch
        latest_bytes = bytes(latest)

        handle = start_fuse(storage_dir, mount_dir, password, out_dir, "partial_mount1")
        logical_path = mount_dir / logical
        write_fsync(logical_path, base)
        write_fsync(logical_path, patch, offset=offset)
        live_verdict, live_detail = read_oracle(logical_path, base, latest_bytes)
        stop_fuse(handle, mount_dir)
        handle = None

        journal_path = storage_dir / f"{logical}.pqcmeta"
        journal_before = parse_journal(journal_path)
        journal_before["retained_file"] = retain_file(journal_path, out_dir, "partial_update_after_write.pqcmeta")

        handle = start_fuse(storage_dir, mount_dir, password, out_dir, "partial_remount")
        remount_verdict, remount_detail = read_oracle(mount_dir / logical, base, latest_bytes)
        stop_fuse(handle, mount_dir)
        handle = None

        acceptable = live_verdict == "latest_committed" and remount_verdict == "latest_committed"
        return {
            "case": "partial_update_and_remount",
            "required_cases": ["partial update", "remount"],
            "oracle_verdict": remount_verdict,
            "live_verdict": live_verdict,
            "detail": remount_detail or live_detail,
            "acceptable": acceptable,
            "previous_sha256": sha256_bytes(base),
            "latest_sha256": sha256_bytes(latest_bytes),
            "journal": {k: v for k, v in journal_before.items() if k != "_records_with_raw"},
            "generated_duplicate_nonce_pairs": journal_before["duplicate_valid_pairs"],
        }
    except Exception as exc:  # retained as an explicit oracle verdict
        return {
            "case": "partial_update_and_remount",
            "required_cases": ["partial update", "remount"],
            "oracle_verdict": "unexpected_liveness_failure",
            "detail": repr(exc),
            "acceptable": False,
        }
    finally:
        stop_fuse(handle, mount_dir)
        shutil.rmtree(storage_dir, ignore_errors=True)
        shutil.rmtree(mount_dir, ignore_errors=True)


def case_torn_journal_tail(out_dir: Path, password: str) -> dict[str, Any]:
    storage_dir = Path(tempfile.mkdtemp(prefix="gen_matrix_torn_store_"))
    mount_dir = Path(tempfile.mkdtemp(prefix="gen_matrix_torn_mnt_"))
    handle: FuseProc | None = None
    logical = "torn.bin"
    try:
        payload = b"T" * (2 * PQC_LOGICAL_BLOCK_SIZE)
        handle = start_fuse(storage_dir, mount_dir, password, out_dir, "torn_mount1")
        write_fsync(mount_dir / logical, payload)
        stop_fuse(handle, mount_dir)
        handle = None

        journal_path = storage_dir / f"{logical}.pqcmeta"
        with journal_path.open("ab") as f:
            f.write(b"TORN")
            f.flush()
            os.fsync(f.fileno())
        journal_after_torn = parse_journal(journal_path)
        journal_after_torn["retained_file"] = retain_file(journal_path, out_dir, "torn_journal_after_tail.pqcmeta")

        handle = start_fuse(storage_dir, mount_dir, password, out_dir, "torn_remount")
        verdict, detail = read_oracle(mount_dir / logical, None, payload)
        stop_fuse(handle, mount_dir)
        handle = None

        acceptable = verdict == "latest_committed" and journal_after_torn["torn_tail_bytes"] == 4
        return {
            "case": "torn_journal_write",
            "required_cases": ["torn journal write"],
            "oracle_verdict": verdict,
            "detail": detail,
            "acceptable": acceptable,
            "latest_sha256": sha256_bytes(payload),
            "journal": {k: v for k, v in journal_after_torn.items() if k != "_records_with_raw"},
            "generated_duplicate_nonce_pairs": journal_after_torn["duplicate_valid_pairs"],
        }
    except Exception as exc:
        return {
            "case": "torn_journal_write",
            "required_cases": ["torn journal write"],
            "oracle_verdict": "unexpected_liveness_failure",
            "detail": repr(exc),
            "acceptable": False,
        }
    finally:
        stop_fuse(handle, mount_dir)
        shutil.rmtree(storage_dir, ignore_errors=True)
        shutil.rmtree(mount_dir, ignore_errors=True)


def case_reserved_generation_skip_after_data_fsync(out_dir: Path, password: str) -> dict[str, Any]:
    storage_dir = Path(tempfile.mkdtemp(prefix="gen_matrix_reserved_store_"))
    mount_dir = Path(tempfile.mkdtemp(prefix="gen_matrix_reserved_mnt_"))
    handle: FuseProc | None = None
    logical = "reserved.bin"
    marker_path = storage_dir / logical
    fault_marker = out_dir / "reserved_generation_fault_marker.jsonl"
    try:
        initial = b"R" * PQC_LOGICAL_BLOCK_SIZE
        fault_payload = b"S" * PQC_LOGICAL_BLOCK_SIZE
        final = b"T" * PQC_LOGICAL_BLOCK_SIZE

        handle = start_fuse(storage_dir, mount_dir, password, out_dir, "reserved_mount1")
        write_fsync(mount_dir / logical, initial)
        stop_fuse(handle, mount_dir)
        handle = None

        fault_env = {
            "PQC_FAULT_CUTPOINT": "data_fsync_after",
            "PQC_FAULT_MARKER_PATH": str(fault_marker),
        }
        handle = start_fuse(
            storage_dir,
            mount_dir,
            password,
            out_dir,
            "reserved_fault_mount",
            extra_env=fault_env,
        )
        fault_write_error = None
        try:
            write_fsync(mount_dir / logical, fault_payload)
        except OSError as exc:
            fault_write_error = {
                "errno": exc.errno,
                "detail": repr(exc),
            }
        deadline = time.monotonic() + 5
        while handle.proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        fault_returncode = handle.proc.poll()
        stop_fuse(handle, mount_dir)
        handle = None

        journal_path = storage_dir / f"{logical}.pqcmeta"
        after_fault = parse_journal(journal_path)
        after_fault["retained_file"] = retain_file(
            journal_path, out_dir, "reserved_generation_after_fault.pqcmeta"
        )
        checkpoint_after_fault = parse_checkpoint_xattr(marker_path)

        handle = start_fuse(storage_dir, mount_dir, password, out_dir, "reserved_remount")
        write_fsync(mount_dir / logical, final)
        verdict, detail = read_oracle(mount_dir / logical, initial, final)
        stop_fuse(handle, mount_dir)
        handle = None

        final_journal = parse_journal(journal_path)
        final_journal["retained_file"] = retain_file(
            journal_path, out_dir, "reserved_generation_after_remount_write.pqcmeta"
        )
        checkpoint_max = int(checkpoint_after_fault.get("max_generation") or 0)
        journal_max_after_fault = journal_max_generation(after_fault)
        final_generations = [
            int(record["generation"])
            for record in final_journal.get("records", [])
            if record.get("valid") and int(record.get("logical_block", -1)) == 0
        ]
        skipped_reserved_generation = (
            checkpoint_after_fault.get("present") is True
            and checkpoint_max > journal_max_after_fault
            and final_generations
            and max(final_generations) > checkpoint_max
            and checkpoint_max not in final_generations
        )
        fault_triggered = fault_marker.exists() and fault_returncode is not None
        acceptable = (
            verdict == "latest_committed"
            and fault_triggered
            and skipped_reserved_generation
            and not final_journal["duplicate_valid_pairs"]
        )
        return {
            "case": "reserved_generation_skip_after_data_fsync_fault",
            "required_cases": [
                "reserved-but-unpublished generation",
                "data fsync fault",
                "remount",
                "duplicate generation prevention",
            ],
            "oracle_verdict": verdict,
            "detail": detail,
            "acceptable": acceptable,
            "fault_cutpoint": "data_fsync_after",
            "fault_returncode": fault_returncode,
            "fault_write_error": fault_write_error,
            "fault_marker": str(fault_marker.relative_to(ROOT)) if fault_marker.exists() else None,
            "checkpoint_after_fault": checkpoint_after_fault,
            "journal_max_after_fault": journal_max_after_fault,
            "final_generations_for_block0": final_generations,
            "skipped_reserved_generation": skipped_reserved_generation,
            "initial_sha256": sha256_bytes(initial),
            "fault_payload_sha256": sha256_bytes(fault_payload),
            "final_sha256": sha256_bytes(final),
            "journal_after_fault": {k: v for k, v in after_fault.items() if k != "_records_with_raw"},
            "journal_after_remount_write": {k: v for k, v in final_journal.items() if k != "_records_with_raw"},
            "generated_duplicate_nonce_pairs": final_journal["duplicate_valid_pairs"],
            "scope": "Daemon SIGKILL after reserved generation data fdatasync; proves remount skips the reserved high-water under this fault model, not physical power loss.",
        }
    except Exception as exc:
        return {
            "case": "reserved_generation_skip_after_data_fsync_fault",
            "required_cases": [
                "reserved-but-unpublished generation",
                "data fsync fault",
                "remount",
                "duplicate generation prevention",
            ],
            "oracle_verdict": "unexpected_liveness_failure",
            "detail": repr(exc),
            "acceptable": False,
        }
    finally:
        stop_fuse(handle, mount_dir)
        shutil.rmtree(storage_dir, ignore_errors=True)
        shutil.rmtree(mount_dir, ignore_errors=True)


def case_older_generation_append(out_dir: Path, password: str) -> dict[str, Any]:
    storage_dir = Path(tempfile.mkdtemp(prefix="gen_matrix_replay_store_"))
    mount_dir = Path(tempfile.mkdtemp(prefix="gen_matrix_replay_mnt_"))
    handle: FuseProc | None = None
    logical = "replay.bin"
    try:
        previous = b"A" * PQC_LOGICAL_BLOCK_SIZE
        latest = b"B" * PQC_LOGICAL_BLOCK_SIZE
        handle = start_fuse(storage_dir, mount_dir, password, out_dir, "replay_mount1")
        logical_path = mount_dir / logical
        write_fsync(logical_path, previous)
        write_fsync(logical_path, latest)
        stop_fuse(handle, mount_dir)
        handle = None

        journal_path = storage_dir / f"{logical}.pqcmeta"
        before = parse_journal(journal_path)
        before["retained_file"] = retain_file(journal_path, out_dir, "older_generation_before_attack.pqcmeta")
        raw_records = before["_records_with_raw"]
        valid_records = [r for r in raw_records if r["valid"] and r["logical_block"] == 0]
        if len(valid_records) < 2:
            raise RuntimeError("expected at least two valid block-0 records")
        older = min(valid_records, key=lambda r: r["generation"])
        newer = max(valid_records, key=lambda r: r["generation"])
        if older["generation"] >= newer["generation"]:
            raise RuntimeError("could not identify older/newer generation")

        with journal_path.open("ab") as f:
            f.write(older["_raw"])
            f.flush()
            os.fsync(f.fileno())
        after = parse_journal(journal_path)
        after["retained_file"] = retain_file(journal_path, out_dir, "older_generation_after_attack.pqcmeta")

        handle = start_fuse(storage_dir, mount_dir, password, out_dir, "replay_remount")
        verdict, detail = read_oracle(mount_dir / logical, previous, latest)
        stop_fuse(handle, mount_dir)
        handle = None

        acceptable = verdict == "latest_committed"
        return {
            "case": "older_generation_append_after_newer_mapping",
            "required_cases": ["older-generation append after newer mapping"],
            "oracle_verdict": verdict,
            "detail": detail,
            "acceptable": acceptable,
            "previous_sha256": sha256_bytes(previous),
            "latest_sha256": sha256_bytes(latest),
            "older_generation": older["generation"],
            "newer_generation": newer["generation"],
            "journal_before_attack": {k: v for k, v in before.items() if k != "_records_with_raw"},
            "journal_after_attack": {k: v for k, v in after.items() if k != "_records_with_raw"},
            "generated_duplicate_nonce_pairs": before["duplicate_valid_pairs"],
            "adversarial_replay_duplicate_pairs": after["duplicate_valid_pairs"],
        }
    except Exception as exc:
        return {
            "case": "older_generation_append_after_newer_mapping",
            "required_cases": ["older-generation append after newer mapping"],
            "oracle_verdict": "unexpected_liveness_failure",
            "detail": repr(exc),
            "acceptable": False,
        }
    finally:
        stop_fuse(handle, mount_dir)
        shutil.rmtree(storage_dir, ignore_errors=True)
        shutil.rmtree(mount_dir, ignore_errors=True)


def case_file_backed_stale_snapshot(out_dir: Path, password: str) -> dict[str, Any]:
    storage_dir = Path(tempfile.mkdtemp(prefix="gen_matrix_snapshot_store_"))
    mount_dir = Path(tempfile.mkdtemp(prefix="gen_matrix_snapshot_mnt_"))
    snapshot_dir = Path(tempfile.mkdtemp(prefix="gen_matrix_snapshot_copy_"))
    handle: FuseProc | None = None
    logical = "snapshot.bin"
    try:
        previous = b"snapshot-v1\n" * 128
        latest = b"snapshot-v2\n" * 128

        handle = start_fuse(storage_dir, mount_dir, password, out_dir, "snapshot_mount1")
        write_fsync(mount_dir / logical, previous)
        stop_fuse(handle, mount_dir)
        handle = None

        shutil.rmtree(snapshot_dir, ignore_errors=True)
        shutil.copytree(storage_dir, snapshot_dir, symlinks=True)

        handle = start_fuse(storage_dir, mount_dir, password, out_dir, "snapshot_mount2")
        write_fsync(mount_dir / logical, latest)
        stop_fuse(handle, mount_dir)
        handle = None

        shutil.rmtree(storage_dir, ignore_errors=True)
        shutil.copytree(snapshot_dir, storage_dir, symlinks=True)

        handle = start_fuse(storage_dir, mount_dir, password, out_dir, "snapshot_replay_mount")
        verdict, detail = read_oracle(mount_dir / logical, previous, latest)
        stop_fuse(handle, mount_dir)
        handle = None

        return {
            "case": "stale_snapshot_replay_file_anchor_negative_control",
            "required_cases": ["stale snapshot replay"],
            "oracle_verdict": verdict,
            "detail": detail,
            "acceptable": verdict in {"previous_committed", "fail_closed"},
            "negative_control": True,
            "previous_sha256": sha256_bytes(previous),
            "latest_sha256": sha256_bytes(latest),
            "scope": "file-backed anchor is replayable with the backing directory; previous_committed is expected negative-control behavior, not rollback protection.",
        }
    except Exception as exc:
        return {
            "case": "stale_snapshot_replay_file_anchor_negative_control",
            "required_cases": ["stale snapshot replay"],
            "oracle_verdict": "unexpected_liveness_failure",
            "detail": repr(exc),
            "acceptable": False,
            "negative_control": True,
        }
    finally:
        stop_fuse(handle, mount_dir)
        shutil.rmtree(storage_dir, ignore_errors=True)
        shutil.rmtree(mount_dir, ignore_errors=True)
        shutil.rmtree(snapshot_dir, ignore_errors=True)


def existing_tpm_snapshot_row() -> dict[str, Any]:
    tpm_json = ROOT / "artifacts" / "validation" / "tpm_monotonic_replay" / "tpm_monotonic_replay.json"
    if not tpm_json.exists():
        return {
            "case": "stale_snapshot_replay_tpm_anchor_existing_artifact",
            "required_cases": ["stale snapshot replay"],
            "oracle_verdict": "unexpected_liveness_failure",
            "detail": "missing retained TPM monotonic replay artifact",
            "acceptable": False,
            "artifact": str(tpm_json.relative_to(ROOT)),
        }
    data = json.loads(tpm_json.read_text(encoding="utf-8"))
    replay = data.get("result", {}).get("replay_result", {})
    mode = replay.get("mode")
    if mode == "fail_closed":
        verdict = "fail_closed"
    elif mode == "rollback_visible":
        verdict = "previous_committed"
    else:
        verdict = "unexpected_liveness_failure"
    return {
        "case": "stale_snapshot_replay_tpm_anchor_existing_artifact",
        "required_cases": ["stale snapshot replay"],
        "oracle_verdict": verdict,
        "detail": replay.get("detail"),
        "acceptable": verdict == "fail_closed",
        "negative_control": False,
        "artifact": str(tpm_json.relative_to(ROOT)),
        "scope": "existing hardware-backed TPM replay-after-advance artifact; not rerun by this generation matrix script.",
    }


def write_markdown(result: dict[str, Any], md_path: Path) -> None:
    lines = [
        "# Generation fault matrix",
        "",
        "Scope: final-binary FUSE generation/nonce regression matrix.  This is not full power-loss or daemon-crash certification.",
        "",
        f"- Command: `{' '.join(result['command'])}`",
        f"- Overall pass: `{result['overall_pass']}`",
        f"- No generated nonce-pair reuse: `{result['no_generated_nonce_reuse']}`",
        f"- No silent corruption verdicts: `{result['no_silent_corruption']}`",
        "",
        "| Case | Verdict | Acceptable | Scope |",
        "|---|---:|---:|---|",
    ]
    for row in result["rows"]:
        scope = row.get("scope", "")
        detail = row.get("detail")
        if detail and not scope:
            scope = str(detail).replace("\n", " ")[:160]
        lines.append(
            f"| `{row['case']}` | `{row['oracle_verdict']}` | `{row['acceptable']}` | {scope} |"
        )
    lines.append("")
    lines.append("Raw JSON retains mount logs, journal summaries, and SHA-256 digests.")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--password", default=os.environ.get("PQC_MASTER_PASSWORD", "generation-matrix-password"))
    args = parser.parse_args()

    if not FUSE_BIN.exists():
        raise FileNotFoundError(f"missing final binary: {FUSE_BIN}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    self_test = run_command([str(FUSE_BIN), "--self-test"], out_dir, "pqc_fuse_self_test")
    rows = [
        {
            "case": "self_test_older_generation_regression",
            "required_cases": ["older-generation append after newer mapping"],
            "oracle_verdict": "latest_committed" if self_test["returncode"] == 0 else "unexpected_liveness_failure",
            "detail": None if self_test["returncode"] == 0 else "pqc_fuse --self-test failed",
            "acceptable": self_test["returncode"] == 0,
            "artifact": {
                "stdout": self_test["stdout"],
                "stderr": self_test["stderr"],
                "stdout_sha256": self_test["stdout_sha256"],
                "stderr_sha256": self_test["stderr_sha256"],
            },
        },
        case_partial_update_remount(out_dir, args.password),
        case_torn_journal_tail(out_dir, args.password),
        case_reserved_generation_skip_after_data_fsync(out_dir, args.password),
        case_older_generation_append(out_dir, args.password),
        case_file_backed_stale_snapshot(out_dir, args.password),
        existing_tpm_snapshot_row(),
    ]

    generated_duplicate_pairs = []
    for row in rows:
        generated_duplicate_pairs.extend(row.get("generated_duplicate_nonce_pairs") or [])

    result = {
        "command": ["experiments/run_generation_fault_matrix.py", "--out-dir", str(out_dir.relative_to(ROOT))],
        "allowed_oracle_verdicts": sorted(ALLOWED_VERDICTS),
        "rows": rows,
        "no_generated_nonce_reuse": len(generated_duplicate_pairs) == 0,
        "generated_duplicate_nonce_pairs": generated_duplicate_pairs,
        "no_silent_corruption": all(row.get("oracle_verdict") != "silent_corruption" for row in rows),
        "unexpected_liveness_failures": sum(row.get("oracle_verdict") == "unexpected_liveness_failure" for row in rows),
    }
    result["overall_pass"] = (
        all(row.get("oracle_verdict") in ALLOWED_VERDICTS for row in rows)
        and all(bool(row.get("acceptable")) for row in rows)
        and result["no_generated_nonce_reuse"]
        and result["no_silent_corruption"]
        and result["unexpected_liveness_failures"] == 0
    )

    json_path = out_dir / "generation_fault_matrix.json"
    md_path = out_dir / "generation_fault_matrix.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_markdown(result, md_path)
    print(json.dumps(result, indent=2))
    return 0 if result["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
