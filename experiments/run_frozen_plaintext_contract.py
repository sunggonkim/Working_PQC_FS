#!/usr/bin/env python3
"""Run plaintext lowerfs under the frozen filesystem workload contract."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run_frozen_aegisq_contract as common


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    ROOT
    / "artifacts"
    / "validation"
    / "frozen_workload_contract"
    / "frozen_workload_contract.json"
)
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "frozen_plaintext_contract"


def platform_manifest(contract_path: Path, bench_root: Path) -> dict[str, Any]:
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
        "uname": common.command_capture(["uname", "-a"], timeout_s=5.0),
        "findmnt_root": common.command_capture(
            ["findmnt", "-T", str(ROOT), "-no", "SOURCE,FSTYPE,TARGET,OPTIONS"],
            timeout_s=5.0,
        ),
        "findmnt_bench_root": common.command_capture(
            ["findmnt", "-T", str(bench_root), "-no", "SOURCE,FSTYPE,TARGET,OPTIONS"],
            timeout_s=5.0,
        ),
        "df_bench_root": common.command_capture(["df", "-PT", str(bench_root)], timeout_s=5.0),
        "fio_version": common.command_capture(["fio", "--version"], timeout_s=5.0),
        "nvpmodel_q": common.command_capture(["nvpmodel", "-q"], timeout_s=10.0),
        "jetson_clocks_show": common.command_capture(["jetson_clocks", "--show"], timeout_s=10.0),
        "git_head": common.command_capture(["git", "rev-parse", "HEAD"], timeout_s=5.0),
        "git_dirty_short": common.command_capture(["git", "status", "--short"], timeout_s=5.0),
        "contract": {
            "path": common.relpath(contract_path),
            "sha256": common.sha256_bytes(contract_path.read_bytes()) if contract_path.exists() else None,
        },
        "process_snapshot": common.process_snapshot(),
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
            f"plaintext|warm|{metric}",
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
            enriched["filesystem_mode"] = "plaintext_lowerfs"
            writer.writerow(enriched)


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload["warm_cache_summary"]
    metrics = summary["metrics"]
    lines = [
        "# Plaintext Frozen Workload Contract Run",
        "",
        f"- Overall pass: `{str(payload['overall_pass']).lower()}`",
        f"- Contract ID: `{payload['contract_id']}`",
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
            f"- Platform manifest: `{payload['artifacts']['platform_manifest']}`",
            f"- Thermal log: `{payload['artifacts']['thermal_log']}`",
            "",
            "## Non-Claims",
            "",
            "- This is not a complete fscrypt/dm-crypt comparison matrix.",
            "- The cold-cache row is not reported as a result unless privileged cache dropping is available.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
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

    rows: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    abort_reason: str | None = None
    bench_root = Path(tempfile.mkdtemp(prefix="plaintext_frozen_root_", dir="/tmp"))
    bench_dir = bench_root / "contract"
    bench_dir.mkdir(parents=True, exist_ok=True)
    cleanup: dict[str, Any] = {
        "bench_root": str(bench_root),
        "removed": False,
    }
    platform_path = args.out / "platform_manifest.json"
    platform_data = platform_manifest(args.contract, bench_root)
    platform_path.write_text(json.dumps(platform_data, indent=2, sort_keys=True), encoding="utf-8")
    thermal_proc, thermal_fp, thermal_status = common.start_thermal_log(args.out, args.thermal_interval_ms)
    file_preparation: dict[str, Any] | None = None
    storage_snapshot: dict[str, Any] | None = None
    try:
        file_preparation = common.precreate_fio_file(profile, bench_dir, args.out)
        if not file_preparation.get("valid"):
            abort_reason = "file preparation did not create the contract file"
        argv = common.fio_command(profile, bench_dir)
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
        storage_snapshot = common.storage_snapshot(bench_root, args.out)
        thermal_status = common.stop_thermal_log(thermal_proc, thermal_fp, thermal_status)
        if not args.keep_temp:
            shutil.rmtree(bench_root, ignore_errors=True)
            cleanup["removed"] = True

    cold_cache = common.probe_cold_cache_capability(args.out)
    csv_path = args.out / "frozen_plaintext_repetitions.csv"
    write_csv(rows, csv_path)
    summary = summarize_rows(rows, profile)
    valid_warm = [
        row for row in rows
        if row.get("cache_state") == "warm" and row.get("valid")
    ]
    contract_repetition_match = repetitions == contract_reps
    lower_findmnt = platform_data["findmnt_bench_root"]
    lower_mount_ok = (
        lower_findmnt.get("returncode") == 0
        and " ext4 " in f" {lower_findmnt.get('stdout', '')} "
    )
    warm_cache_pass = (
        len(valid_warm) == contract_reps
        and contract_repetition_match
        and (file_preparation or {}).get("valid") is True
        and lower_mount_ok
    )
    methodology = {
        "methodology_id": "plaintext-frozen-workload-contract-v1",
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
            "unsupported_cold_cache": "reported as invalid_not_run, not folded into warm-cache results",
            "fio_timeout": "invalid row with retained command log and lowerfs snapshot",
        },
    }
    payload = {
        "overall_pass": warm_cache_pass,
        "contract_compliant_warm_cache": warm_cache_pass,
        "comparison_ready": False,
        "scope": (
            "Plaintext lower-filesystem warm-cache execution of the frozen "
            "filesystem workload contract.  It separates raw lowerfs behavior "
            "from encrypted FUSE/kernel rows but is not a full matrix by itself."
        ),
        "non_claims": [
            "not a complete fscrypt/dm-crypt comparison matrix",
            "not a cold-cache result when drop_caches is unavailable",
            "not a QoS or SQLite hero result",
        ],
        "contract_id": contract_payload["contract"]["contract_id"],
        "contract_sha256_recorded": contract_payload.get("contract_sha256"),
        "contract_path_sha256": platform_data["contract"]["sha256"],
        "filesystem_mode": "plaintext_lowerfs",
        "workload_profile": profile["profile_id"],
        "fio_command": common.fio_command(profile, Path("${LOWER_ROOT}") / "plaintext_contract"),
        "mount_options": {
            "filesystem": "ext4",
            "source": platform_data["findmnt_bench_root"].get("stdout", "").split()[0]
            if platform_data["findmnt_bench_root"].get("stdout")
            else None,
            "encryption": "none",
            "findmnt_bench_root": platform_data["findmnt_bench_root"],
        },
        "methodology": methodology,
        "file_preparation": file_preparation,
        "platform_manifest": common.relpath(platform_path),
        "thermal_logging": thermal_status,
        "cold_cache": cold_cache,
        "warm_cache_summary": summary,
        "abort_reason": abort_reason,
        "repetitions": rows,
        "commands": commands,
        "cleanup": cleanup,
        "storage_snapshot": storage_snapshot,
        "artifacts": {
            "json": common.relpath(args.out / "frozen_plaintext_contract.json"),
            "csv": common.relpath(csv_path),
            "markdown": common.relpath(args.out / "frozen_plaintext_contract.md"),
            "fio_raw_dir": common.relpath(args.out / "fio_raw"),
            "file_preparation": common.relpath(args.out / "file_preparation.json"),
            "platform_manifest": common.relpath(platform_path),
            "storage_snapshot": common.relpath(args.out / "storage_snapshot.json"),
            "thermal_log": thermal_status.get("path"),
        },
    }
    json_path = args.out / "frozen_plaintext_contract.json"
    md_path = args.out / "frozen_plaintext_contract.md"
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
