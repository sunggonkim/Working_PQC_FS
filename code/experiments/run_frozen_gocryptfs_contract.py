#!/usr/bin/env python3
"""Run gocryptfs under the frozen filesystem workload contract.

The harness executes only the gocryptfs mode from the retained v2 contract.
It records a contract-valid warm-cache fio row, keeps raw mount/fio/platform
logs, and marks the cold-cache row invalid when privileged cache dropping is
unavailable.  It is a same-platform baseline row, not a complete filesystem
comparison matrix by itself.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
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

import run_frozen_aegisq_contract as common


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = (
    ROOT
    / "artifacts"
    / "validation"
    / "frozen_workload_contract"
    / "frozen_workload_contract.json"
)
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "frozen_gocryptfs_contract"


@dataclass
class GocryptfsHandle:
    proc: subprocess.Popen[bytes]
    stdout: Any
    stderr: Any


def gocryptfs_bin() -> str:
    found = shutil.which("gocryptfs")
    return found or "gocryptfs"


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


def sha256_file(path: Path) -> str | None:
    try:
        return common.sha256_bytes(path.read_bytes())
    except OSError:
        return None


def platform_manifest(contract_path: Path) -> dict[str, Any]:
    binary = Path(gocryptfs_bin())
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
        "cpu_governors": common.read_cpu_governors(),
        "uname": command_capture(["uname", "-a"], timeout_s=5.0),
        "findmnt_root": command_capture(
            ["findmnt", "-T", str(ROOT), "-no", "SOURCE,FSTYPE,TARGET,OPTIONS"],
            timeout_s=5.0,
        ),
        "df_root": command_capture(["df", "-PT", str(ROOT)], timeout_s=5.0),
        "fio_version": command_capture(["fio", "--version"], timeout_s=5.0),
        "gocryptfs_version": command_capture([gocryptfs_bin(), "-version"], timeout_s=5.0),
        "fusermount": command_capture([common.fusermount_command(), "--version"], timeout_s=5.0),
        "nvpmodel_q": command_capture(["nvpmodel", "-q"], timeout_s=10.0),
        "jetson_clocks_show": command_capture(["jetson_clocks", "--show"], timeout_s=10.0),
        "git_head": command_capture(["git", "rev-parse", "HEAD"], timeout_s=5.0),
        "git_dirty_short": command_capture(["git", "status", "--short"], timeout_s=5.0),
        "contract": {
            "path": common.relpath(contract_path),
            "sha256": common.sha256_bytes(contract_path.read_bytes()) if contract_path.exists() else None,
        },
        "gocryptfs_binary": {
            "path": str(binary),
            "exists": binary.exists(),
            "sha256": sha256_file(binary) if binary.exists() else None,
        },
        "process_snapshot": common.process_snapshot(),
    }


def write_passfile(path: Path, password: str) -> dict[str, Any]:
    path.write_text(password + "\n", encoding="utf-8")
    path.chmod(0o600)
    stat = path.stat()
    return {
        "path_recorded": False,
        "mode_octal": oct(stat.st_mode & 0o777),
        "bytes": stat.st_size,
        "secret_material_retained": False,
    }


def init_cipher_dir(cipher_dir: Path, passfile: Path, out_dir: Path) -> dict[str, Any]:
    log_dir = out_dir / "mount_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "gocryptfs_init.stdout.txt"
    stderr_path = log_dir / "gocryptfs_init.stderr.txt"
    command_path = log_dir / "gocryptfs_init.command.json"
    argv = [gocryptfs_bin(), "-quiet", "-init", "-passfile", str(passfile), str(cipher_dir)]
    start_ns = time.time_ns()
    proc = subprocess.run(
        argv,
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    end_ns = time.time_ns()
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    command = {
        "argv": [gocryptfs_bin(), "-quiet", "-init", "-passfile", "${PASSFILE}", str(cipher_dir)],
        "cwd": str(ROOT),
        "returncode": proc.returncode,
        "start_realtime_ns": start_ns,
        "end_realtime_ns": end_ns,
        "wall_seconds": (end_ns - start_ns) / 1_000_000_000,
        "stdout": common.relpath(stdout_path),
        "stderr": common.relpath(stderr_path),
    }
    command_path.write_text(json.dumps(command, indent=2, sort_keys=True), encoding="utf-8")
    command["artifact"] = common.relpath(command_path)
    return command


def start_gocryptfs(cipher_dir: Path, mount_dir: Path, passfile: Path, out_dir: Path) -> GocryptfsHandle:
    log_dir = out_dir / "mount_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout = (log_dir / "gocryptfs.stdout.txt").open("wb")
    stderr = (log_dir / "gocryptfs.stderr.txt").open("wb")
    proc = subprocess.Popen(
        [gocryptfs_bin(), "-quiet", "-fg", "-passfile", str(passfile), str(cipher_dir), str(mount_dir)],
        cwd=ROOT,
        stdout=stdout,
        stderr=stderr,
    )
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if subprocess.run(["mountpoint", "-q", str(mount_dir)], check=False).returncode == 0:
            return GocryptfsHandle(proc=proc, stdout=stdout, stderr=stderr)
        if proc.poll() is not None:
            stdout.close()
            stderr.close()
            raise RuntimeError(f"gocryptfs exited before mount: rc={proc.returncode}")
        time.sleep(0.05)
    stdout.close()
    stderr.close()
    raise TimeoutError("timed out waiting for gocryptfs mount")


def stop_gocryptfs(handle: GocryptfsHandle | None, mount_dir: Path, out_dir: Path) -> dict[str, Any]:
    log_dir = out_dir / "mount_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    unmount = subprocess.run(
        [common.fusermount_command(), "-u", str(mount_dir)],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    (log_dir / "unmount.stdout.txt").write_text(unmount.stdout, encoding="utf-8")
    (log_dir / "unmount.stderr.txt").write_text(unmount.stderr, encoding="utf-8")
    if handle is None:
        return {
            "argv": [common.fusermount_command(), "-u", str(mount_dir)],
            "returncode": unmount.returncode,
        }
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
        "argv": [common.fusermount_command(), "-u", str(mount_dir)],
        "returncode": unmount.returncode,
        "gocryptfs_returncode": handle.proc.returncode,
    }


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
        ci_low, ci_high = common.bootstrap_ci(
            samples,
            f"gocryptfs|warm|{metric}",
            trials=int(ci_contract["resamples"]),
            alpha=1.0 - float(ci_contract["confidence_level"]),
        )
        summary["metrics"][metric] = {
            "samples": samples,
            "median": statistics.median(samples),
            "p05": common.quantile(samples, 0.05),
            "p95": common.quantile(samples, 0.95),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
        }
    return summary


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "label",
        "filesystem_mode",
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
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            enriched = dict(row)
            enriched["filesystem_mode"] = "gocryptfs"
            writer.writerow(enriched)


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload["warm_cache_summary"]
    metrics = summary["metrics"]
    version = payload["platform"]["gocryptfs_version"].get("stdout", "")
    lines = [
        "# gocryptfs Frozen Workload Contract Run",
        "",
        f"- Overall pass: `{str(payload['overall_pass']).lower()}`",
        f"- Contract ID: `{payload['contract_id']}`",
        f"- gocryptfs version: `{version}`",
        f"- Scope: {payload['scope']}",
        f"- File preparation valid: `{str((payload.get('file_preparation') or {}).get('valid')).lower()}`",
        f"- Warm-cache valid repetitions: `{summary['valid_repetitions']}`",
        f"- Cold-cache status: `{payload['cold_cache']['status']}`",
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
            "- This is not a complete plaintext/fscrypt/dm-crypt comparison matrix.",
            "- The cold-cache row is not reported as a result unless privileged cache dropping is available.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--password", default="aegisq-gocryptfs-frozen-password")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--thermal-interval-ms", type=int, default=100)
    parser.add_argument("--fio-timeout-s", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    args.contract = args.contract if args.contract.is_absolute() else ROOT / args.contract
    args.out = args.out if args.out.is_absolute() else ROOT / args.out
    if not args.contract.exists():
        raise SystemExit(f"missing frozen workload contract: {args.contract}")
    if shutil.which("gocryptfs") is None:
        raise SystemExit("gocryptfs is required for the frozen workload contract")
    if shutil.which("fio") is None:
        raise SystemExit("fio is required for the frozen workload contract")
    if args.warmup_runs < 1:
        raise SystemExit("--warmup-runs must be at least 1 for the frozen contract")

    contract_payload = common.load_contract(args.contract)
    profile = common.workload_profile(contract_payload)
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

    rows: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    abort_reason: str | None = None
    cipher_dir = Path(tempfile.mkdtemp(prefix="gocryptfs_frozen_cipher_"))
    mount_dir = Path(tempfile.mkdtemp(prefix="gocryptfs_frozen_mnt_"))
    passfile = Path(tempfile.mkdtemp(prefix="gocryptfs_frozen_secret_")) / "passfile"
    handle: GocryptfsHandle | None = None
    cleanup: dict[str, Any] = {
        "cipher_dir": str(cipher_dir),
        "mount_dir": str(mount_dir),
        "secret_dir": str(passfile.parent),
        "removed": False,
    }
    passfile_metadata = write_passfile(passfile, args.password)
    init_result: dict[str, Any] | None = None
    file_preparation: dict[str, Any] | None = None
    unmount_result: dict[str, Any] | None = None
    backing_snapshot: dict[str, Any] | None = None
    thermal_proc, thermal_fp, thermal_status = common.start_thermal_log(args.out, args.thermal_interval_ms)
    try:
        init_result = init_cipher_dir(cipher_dir, passfile, args.out)
        if init_result.get("returncode") != 0:
            abort_reason = "gocryptfs -init failed"
        if abort_reason is None:
            handle = start_gocryptfs(cipher_dir, mount_dir, passfile, args.out)
            bench_dir = mount_dir / "contract"
            bench_dir.mkdir(parents=True, exist_ok=True)
            file_preparation = common.precreate_fio_file(profile, bench_dir, args.out)
            if not file_preparation.get("valid"):
                abort_reason = "file preparation did not create the contract file"
        argv = common.fio_command(profile, mount_dir / "contract")
        command_template = profile["mount_options"]["command_template"]
        (args.out / "fio_command_template.txt").write_text(command_template + "\n", encoding="utf-8")

        if abort_reason is None:
            for index in range(args.warmup_runs):
                command = common.run_fio_with_timeout(
                    f"warmup_{index:02d}",
                    argv,
                    args.out,
                    args.fio_timeout_s,
                )
                commands.append(command)
                row = common.parse_fio_result(command, "warmup", None)
                rows.append(row)
                if not row.get("valid"):
                    abort_reason = f"warmup {index} did not produce a valid fio result"
                    break
        if abort_reason is None:
            for rep in range(repetitions):
                command = common.run_fio_with_timeout(
                    f"warm_rep_{rep:02d}",
                    argv,
                    args.out,
                    args.fio_timeout_s,
                )
                commands.append(command)
                row = common.parse_fio_result(command, "warm", rep)
                rows.append(row)
                if not row.get("valid"):
                    abort_reason = f"measured repetition {rep} did not produce a valid fio result"
                    break
    finally:
        unmount_result = stop_gocryptfs(handle, mount_dir, args.out)
        backing_snapshot = common.storage_snapshot(cipher_dir, args.out)
        thermal_status = common.stop_thermal_log(thermal_proc, thermal_fp, thermal_status)
        if not args.keep_temp:
            shutil.rmtree(cipher_dir, ignore_errors=True)
            shutil.rmtree(mount_dir, ignore_errors=True)
            shutil.rmtree(passfile.parent, ignore_errors=True)
            cleanup["removed"] = True

    cold_cache = common.probe_cold_cache_capability(args.out)
    csv_path = args.out / "frozen_gocryptfs_repetitions.csv"
    write_csv(rows, csv_path)
    summary = summarize_rows(rows, profile)
    valid_warm = [
        row for row in rows
        if row.get("cache_state") == "warm" and row.get("valid")
    ]
    contract_repetition_match = repetitions == contract_reps
    warm_cache_pass = (
        len(valid_warm) == contract_reps
        and contract_repetition_match
        and (file_preparation or {}).get("valid") is True
        and (init_result or {}).get("returncode") == 0
    )
    methodology = {
        "methodology_id": "gocryptfs-frozen-workload-contract-v1",
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
            "gocryptfs_init_failure": "invalid run with retained stdout/stderr",
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
            "gocryptfs-only warm-cache execution of the frozen filesystem "
            "workload contract.  It can serve as the user-space encrypted "
            "filesystem baseline row once compared with other retained rows."
        ),
        "non_claims": [
            "not a complete plaintext/fscrypt/dm-crypt comparison matrix",
            "not a cold-cache result when drop_caches is unavailable",
            "not a QoS or SQLite hero result",
        ],
        "contract_id": contract_payload["contract"]["contract_id"],
        "contract_sha256_recorded": contract_payload.get("contract_sha256"),
        "contract_path_sha256": platform_data["contract"]["sha256"],
        "filesystem_mode": "gocryptfs",
        "workload_profile": profile["profile_id"],
        "fio_command": common.fio_command(profile, Path("${GOCRYPTFS_MOUNT}") / "contract"),
        "gocryptfs_init": init_result,
        "gocryptfs_mount": {
            "command": "gocryptfs -quiet -fg -passfile ${PASSFILE} ${GOCRYPTFS_CIPHER} ${GOCRYPTFS_MOUNT}",
            "cipher_directory": "${GOCRYPTFS_CIPHER}",
            "plaintext_names": False,
            "reverse_mode": False,
        },
        "passfile": passfile_metadata,
        "methodology": methodology,
        "file_preparation": file_preparation,
        "platform": {
            "gocryptfs_version": platform_data["gocryptfs_version"],
            "fio_version": platform_data["fio_version"],
            "fusermount": platform_data["fusermount"],
        },
        "platform_manifest": common.relpath(platform_path),
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
            "json": common.relpath(args.out / "frozen_gocryptfs_contract.json"),
            "csv": common.relpath(csv_path),
            "markdown": common.relpath(args.out / "frozen_gocryptfs_contract.md"),
            "fio_raw_dir": common.relpath(args.out / "fio_raw"),
            "file_preparation": common.relpath(args.out / "file_preparation.json"),
            "mount_logs": common.relpath(args.out / "mount_logs"),
            "platform_manifest": common.relpath(platform_path),
            "storage_snapshot": common.relpath(args.out / "storage_snapshot.json"),
            "thermal_log": thermal_status.get("path"),
        },
    }
    json_path = args.out / "frozen_gocryptfs_contract.json"
    md_path = args.out / "frozen_gocryptfs_contract.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(build_markdown(payload), encoding="utf-8")

    print(
        json.dumps(
            {
                "overall_pass": payload["overall_pass"],
                "json": common.relpath(json_path),
                "markdown": common.relpath(md_path),
                "csv": common.relpath(csv_path),
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
