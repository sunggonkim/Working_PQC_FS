#!/usr/bin/env python3
"""Run dm-crypt/ext4 under the frozen filesystem workload contract.

The runner creates a disposable loop-backed LUKS2 volume, formats ext4 inside
the mapper device, runs the same fio contract used by the other filesystem
rows, and tears everything down.  If privileged setup is unavailable, it writes
an environment-blocked verdict instead of silently skipping the baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
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
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "frozen_dmcrypt_contract"


@dataclass
class DmcryptHandle:
    work_dir: Path
    image: Path
    key_file: Path
    mount_dir: Path
    loopdev: str
    mapper_name: str
    mapper_path: Path
    setup_steps: list[dict[str, Any]]


def sudo_capture(
    command: list[str],
    password: str | None,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    argv = ["sudo", "-n", *command] if password is None else ["sudo", "-S", "-p", "", *command]
    if shutil.which("sudo") is None:
        return {
            "argv": argv,
            "available": False,
            "returncode": None,
            "stdout": "",
            "stderr": "",
        }
    try:
        proc = subprocess.run(
            argv,
            cwd=ROOT,
            check=False,
            input=(password + "\n") if password is not None else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "argv": argv,
            "available": True,
            "timeout": True,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "password_supplied": password is not None,
        }
    return {
        "argv": argv,
        "available": True,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "password_supplied": password is not None,
    }


def write_command(path: Path, name: str, result: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"name": name, "result": result}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"name": name, "artifact": common.relpath(path), "result": result}


def platform_manifest(contract_path: Path, bench_root: Path | None) -> dict[str, Any]:
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
        )
        if bench_root is not None
        else None,
        "df_bench_root": common.command_capture(["df", "-PT", str(bench_root)], timeout_s=5.0)
        if bench_root is not None
        else None,
        "fio_version": common.command_capture(["fio", "--version"], timeout_s=5.0),
        "cryptsetup_version": common.command_capture(["cryptsetup", "--version"], timeout_s=5.0),
        "losetup_version": common.command_capture(["losetup", "--version"], timeout_s=5.0),
        "mkfs_ext4_version": common.command_capture(["mkfs.ext4", "-V"], timeout_s=5.0),
        "sudo_probe": common.command_capture(["sudo", "-n", "true"], timeout_s=5.0),
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
            f"dmcrypt|warm|{metric}",
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
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            enriched = dict(row)
            enriched["filesystem_mode"] = "dm_crypt_ext4"
            writer.writerow(enriched)


def blocking_verdict(
    args: argparse.Namespace,
    contract_payload: dict[str, Any],
    reasons: list[str],
    probes: dict[str, Any],
    platform_path: Path,
    platform_data: dict[str, Any],
) -> dict[str, Any]:
    csv_path = args.out / "frozen_dmcrypt_repetitions.csv"
    write_csv([], csv_path)
    payload = {
        "overall_pass": False,
        "contract_compliant_warm_cache": False,
        "verdict": "environment-blocked",
        "blocking_reasons": reasons,
        "scope": (
            "dm-crypt/ext4 frozen-contract baseline was not executed because "
            "the disposable LUKS2 loop device could not be created in this run."
        ),
        "non_claims": [
            "not a measured dm-crypt throughput row",
            "not evidence for an AEGIS-Q speedup over dm-crypt",
            "not a cold-cache result",
        ],
        "contract_id": contract_payload["contract"]["contract_id"],
        "contract_path_sha256": platform_data["contract"]["sha256"],
        "filesystem_mode": "dm_crypt_ext4",
        "workload_profile": common.workload_profile(contract_payload)["profile_id"],
        "required_privilege": "root or AEGISQ_SUDO_PASSWORD for sudo -S",
        "probes": probes,
        "platform_manifest": common.relpath(platform_path),
        "cold_cache": {"status": "invalid_not_run", "reason": "dm-crypt mount was not created"},
        "warm_cache_summary": {
            "cache_state": "warm",
            "valid_repetitions": 0,
            "expected_repetitions": common.workload_profile(contract_payload)["repetition_count"],
            "metrics": {},
        },
        "repetitions": [],
        "commands": [],
        "artifacts": {
            "json": common.relpath(args.out / "frozen_dmcrypt_contract.json"),
            "csv": common.relpath(csv_path),
            "platform_manifest": common.relpath(platform_path),
            "setup_logs": common.relpath(args.out / "setup_logs"),
        },
    }
    return payload


def setup_dmcrypt(
    out_dir: Path,
    image_size_mib: int,
    password: str | None,
) -> DmcryptHandle:
    work_dir = Path(tempfile.mkdtemp(prefix="dmcrypt_frozen_root_", dir="/tmp"))
    image = work_dir / "dmcrypt.img"
    key_file = work_dir / "luks.key"
    mount_dir = work_dir / "mnt"
    mount_dir.mkdir(parents=True, exist_ok=True)
    key_file.write_bytes(os.urandom(64))
    key_file.chmod(0o600)
    with image.open("wb") as fp:
        fp.truncate(image_size_mib * 1024 * 1024)

    setup_logs = out_dir / "setup_logs"
    mapper_name = f"aegisq_dmcrypt_{os.getpid()}"
    mapper_path = Path("/dev/mapper") / mapper_name
    steps: list[dict[str, Any]] = []
    loopdev = ""

    def run_step(name: str, command: list[str], timeout_s: float = 30.0) -> dict[str, Any]:
        result = sudo_capture(command, password, timeout_s=timeout_s)
        step = write_command(setup_logs / f"{len(steps):02d}_{name}.json", name, result)
        steps.append(step)
        if result.get("returncode") != 0:
            raise RuntimeError(f"{name} failed: {result.get('stderr') or result.get('stdout')}")
        return result

    try:
        losetup = run_step("losetup", ["losetup", "--find", "--show", str(image)], timeout_s=10.0)
        loopdev = str(losetup.get("stdout", "")).strip()
        if not loopdev.startswith("/dev/"):
            raise RuntimeError(f"losetup returned invalid loop device: {loopdev!r}")
        run_step(
            "luks_format",
            [
                "cryptsetup",
                "--batch-mode",
                "--type",
                "luks2",
                "--cipher",
                "aes-xts-plain64",
                "--key-size",
                "512",
                "--pbkdf",
                "pbkdf2",
                "--pbkdf-force-iterations",
                "1000",
                "--key-file",
                str(key_file),
                "luksFormat",
                loopdev,
            ],
            timeout_s=60.0,
        )
        run_step(
            "luks_open",
            ["cryptsetup", "open", "--key-file", str(key_file), loopdev, mapper_name],
            timeout_s=30.0,
        )
        run_step("mkfs_ext4", ["mkfs.ext4", "-q", str(mapper_path)], timeout_s=60.0)
        run_step("mount", ["mount", str(mapper_path), str(mount_dir)], timeout_s=30.0)
        run_step(
            "chown_mount",
            ["chown", f"{os.getuid()}:{os.getgid()}", str(mount_dir)],
            timeout_s=10.0,
        )
    except Exception:
        teardown_dmcrypt(
            DmcryptHandle(
                work_dir=work_dir,
                image=image,
                key_file=key_file,
                mount_dir=mount_dir,
                loopdev=loopdev,
                mapper_name=mapper_name,
                mapper_path=mapper_path,
                setup_steps=steps,
            ),
            out_dir,
            password,
            remove_work_dir=True,
        )
        raise

    return DmcryptHandle(
        work_dir=work_dir,
        image=image,
        key_file=key_file,
        mount_dir=mount_dir,
        loopdev=loopdev,
        mapper_name=mapper_name,
        mapper_path=mapper_path,
        setup_steps=steps,
    )


def teardown_dmcrypt(
    handle: DmcryptHandle | None,
    out_dir: Path,
    password: str | None,
    remove_work_dir: bool,
) -> dict[str, Any]:
    logs = out_dir / "teardown_logs"
    logs.mkdir(parents=True, exist_ok=True)
    steps: list[dict[str, Any]] = []
    if handle is None:
        return {"steps": steps, "removed": False}

    for name, command, timeout in [
        ("sync", ["sync"], 30.0),
        ("umount", ["umount", str(handle.mount_dir)], 30.0),
        ("cryptsetup_close", ["cryptsetup", "close", handle.mapper_name], 30.0),
        ("losetup_detach", ["losetup", "-d", handle.loopdev], 10.0),
    ]:
        result = sudo_capture(command, password, timeout_s=timeout)
        steps.append(write_command(logs / f"{len(steps):02d}_{name}.json", name, result))

    removed = False
    if remove_work_dir:
        shutil.rmtree(handle.work_dir, ignore_errors=True)
        removed = True
    return {
        "steps": steps,
        "removed": removed,
        "work_dir": str(handle.work_dir),
        "loopdev": handle.loopdev,
        "mapper_name": handle.mapper_name,
    }


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload["warm_cache_summary"]
    metrics = summary.get("metrics", {})
    lines = [
        "# dm-crypt/ext4 Frozen Workload Contract Run",
        "",
        f"- Overall pass: `{str(payload['overall_pass']).lower()}`",
        f"- Verdict: `{payload.get('verdict', 'measured')}`",
        f"- Contract ID: `{payload['contract_id']}`",
        f"- Scope: {payload['scope']}",
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
    if payload.get("blocking_reasons"):
        lines.extend(["", "## Blocking Reasons", ""])
        for reason in payload["blocking_reasons"]:
            lines.append(f"- `{reason}`")
    lines.extend(
        [
            "",
            "## Retained Artifacts",
            "",
            f"- JSON summary: `{payload['artifacts']['json']}`",
            f"- CSV repetitions: `{payload['artifacts']['csv']}`",
            f"- Platform manifest: `{payload['artifacts']['platform_manifest']}`",
            f"- Setup logs: `{payload['artifacts']['setup_logs']}`",
            "",
            "## Non-Claims",
            "",
        ]
    )
    lines.extend(f"- {claim}" for claim in payload["non_claims"])
    return "\n".join(lines) + "\n"


def write_payload(payload: dict[str, Any], out_dir: Path) -> Path:
    json_path = out_dir / "frozen_dmcrypt_contract.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return json_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--thermal-interval-ms", type=int, default=500)
    parser.add_argument("--fio-timeout-s", type=float, default=None)
    parser.add_argument("--image-size-mib", type=int, default=1536)
    parser.add_argument("--use-sudo-password-env", action="store_true")
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
    if args.image_size_mib < 1280:
        raise SystemExit("--image-size-mib must leave room for the 1 GiB fio file plus ext4/LUKS metadata")
    if args.out.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.out} exists; pass --overwrite to replace this harness output")
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    contract_payload = common.load_contract(args.contract)
    profile = common.workload_profile(contract_payload)
    contract_reps = int(profile["repetition_count"])
    repetitions = contract_reps if args.repetitions is None else args.repetitions
    if repetitions < 1:
        raise SystemExit("--repetitions must be positive")

    password = os.environ.get("AEGISQ_SUDO_PASSWORD") if args.use_sudo_password_env else None
    platform_path = args.out / "platform_manifest.json"
    platform_data = platform_manifest(args.contract, None)
    platform_path.write_text(json.dumps(platform_data, indent=2, sort_keys=True), encoding="utf-8")
    sudo_password_probe = sudo_capture(["true"], password, timeout_s=5.0) if password else None
    probes = {
        "sudo_noninteractive": platform_data["sudo_probe"],
        "sudo_password_env_supplied": password is not None,
        "sudo_password_probe": sudo_password_probe,
        "cryptsetup_version": platform_data["cryptsetup_version"],
        "losetup_version": platform_data["losetup_version"],
        "mkfs_ext4_version": platform_data["mkfs_ext4_version"],
    }
    sudo_ready = platform_data["sudo_probe"].get("returncode") == 0 or (
        sudo_password_probe is not None and sudo_password_probe.get("returncode") == 0
    )
    missing_reasons: list[str] = []
    for binary in ("sudo", "cryptsetup", "losetup", "mkfs.ext4", "mount", "umount"):
        if shutil.which(binary) is None:
            missing_reasons.append(f"{binary}_missing")
    if not sudo_ready:
        missing_reasons.append("noninteractive_root_unavailable")
        if password is None:
            missing_reasons.append("AEGISQ_SUDO_PASSWORD_not_set")
    if missing_reasons:
        payload = blocking_verdict(args, contract_payload, missing_reasons, probes, platform_path, platform_data)
        json_path = write_payload(payload, args.out)
        print(
            json.dumps(
                {
                    "overall_pass": payload["overall_pass"],
                    "verdict": payload["verdict"],
                    "blocking_reasons": payload["blocking_reasons"],
                    "json": common.relpath(json_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    handle: DmcryptHandle | None = None
    rows: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    abort_reason: str | None = None
    setup_error: str | None = None
    teardown: dict[str, Any] | None = None
    storage_snapshot: dict[str, Any] | None = None
    file_preparation: dict[str, Any] | None = None
    thermal_proc = None
    thermal_fp = None
    thermal_status: dict[str, Any] = {}
    try:
        handle = setup_dmcrypt(args.out, args.image_size_mib, password)
        platform_data = platform_manifest(args.contract, handle.mount_dir)
        platform_path.write_text(json.dumps(platform_data, indent=2, sort_keys=True), encoding="utf-8")
        thermal_proc, thermal_fp, thermal_status = common.start_thermal_log(args.out, args.thermal_interval_ms)
        bench_dir = handle.mount_dir / "contract"
        bench_dir.mkdir(parents=True, exist_ok=True)
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
    except Exception as exc:
        setup_error = str(exc)
    finally:
        if handle is not None:
            storage_snapshot = common.storage_snapshot(handle.work_dir, args.out)
        if thermal_status:
            thermal_status = common.stop_thermal_log(thermal_proc, thermal_fp, thermal_status)
        teardown = teardown_dmcrypt(
            handle,
            args.out,
            password,
            remove_work_dir=not args.keep_temp,
        )

    if setup_error is not None:
        payload = blocking_verdict(
            args,
            contract_payload,
            ["dmcrypt_setup_failed"],
            {**probes, "setup_error": setup_error},
            platform_path,
            platform_data,
        )
        payload["teardown"] = teardown
        json_path = write_payload(payload, args.out)
        print(
            json.dumps(
                {
                    "overall_pass": payload["overall_pass"],
                    "verdict": payload["verdict"],
                    "blocking_reasons": payload["blocking_reasons"],
                    "setup_error": setup_error,
                    "json": common.relpath(json_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    cold_cache = common.probe_cold_cache_capability(args.out)
    csv_path = args.out / "frozen_dmcrypt_repetitions.csv"
    write_csv(rows, csv_path)
    summary = summarize_rows(rows, profile)
    valid_warm = [
        row for row in rows
        if row.get("cache_state") == "warm" and row.get("valid")
    ]
    contract_repetition_match = repetitions == contract_reps
    warm_cache_pass = len(valid_warm) == contract_reps and contract_repetition_match and abort_reason is None
    payload = {
        "overall_pass": warm_cache_pass,
        "contract_compliant_warm_cache": warm_cache_pass,
        "verdict": "measured" if warm_cache_pass else "invalid-run",
        "scope": (
            "dm-crypt/ext4 warm-cache execution of the frozen filesystem "
            "workload contract on a disposable loop-backed LUKS2 volume."
        ),
        "non_claims": [
            "not a fscrypt row",
            "not a cold-cache result when drop_caches is unavailable",
            "not an AEGIS-Q speedup claim by itself",
        ],
        "contract_id": contract_payload["contract"]["contract_id"],
        "contract_sha256_recorded": contract_payload.get("contract_sha256"),
        "contract_path_sha256": platform_data["contract"]["sha256"],
        "filesystem_mode": "dm_crypt_ext4",
        "workload_profile": profile["profile_id"],
        "fio_command": common.fio_command(profile, Path("${DMCRYPT_MOUNT}") / "contract"),
        "mount_options": {
            "cryptsetup": "LUKS2, aes-xts-plain64, 512-bit key",
            "discard": "disabled",
            "filesystem": "ext4",
            "backing": "disposable loop image",
            "mapper_name_recorded": False,
        },
        "methodology": {
            "methodology_id": "dmcrypt-frozen-workload-contract-v1",
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
        },
        "file_preparation": file_preparation,
        "platform_manifest": common.relpath(platform_path),
        "thermal_logging": thermal_status,
        "cold_cache": cold_cache,
        "warm_cache_summary": summary,
        "abort_reason": abort_reason,
        "repetitions": rows,
        "commands": commands,
        "setup": {
            "steps": handle.setup_steps if handle is not None else [],
            "image_size_mib": args.image_size_mib,
            "key_file_retained": False,
            "luks_key_material_retained": False,
        },
        "teardown": teardown,
        "storage_snapshot": storage_snapshot,
        "artifacts": {
            "json": common.relpath(args.out / "frozen_dmcrypt_contract.json"),
            "csv": common.relpath(csv_path),
            "fio_raw_dir": common.relpath(args.out / "fio_raw"),
            "file_preparation": common.relpath(args.out / "file_preparation.json"),
            "platform_manifest": common.relpath(platform_path),
            "storage_snapshot": common.relpath(args.out / "storage_snapshot.json"),
            "thermal_log": thermal_status.get("path"),
            "setup_logs": common.relpath(args.out / "setup_logs"),
            "teardown_logs": common.relpath(args.out / "teardown_logs"),
        },
    }
    json_path = write_payload(payload, args.out)
    print(
        json.dumps(
            {
                "overall_pass": payload["overall_pass"],
                "verdict": payload["verdict"],
                "json": common.relpath(json_path),
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
