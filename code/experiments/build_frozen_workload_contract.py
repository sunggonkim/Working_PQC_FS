#!/usr/bin/env python3
"""Emit the frozen workload contract for filesystem comparisons.

This script does not run benchmarks.  It creates a retained, machine-readable
contract that future AEGIS-Q, plaintext, gocryptfs, fscrypt, and dm-crypt runs
must follow before their results can be compared in the paper.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "artifacts" / "validation" / "frozen_workload_contract"

REQUIRED_WORKLOAD_FIELDS = (
    "request_size_bytes",
    "read_write_mix",
    "sync_mode",
    "cache_states",
    "queue_depth",
    "client_count",
    "file_size_bytes",
    "mount_options",
    "repetition_count",
    "confidence_interval_method",
)

REQUIRED_ENVIRONMENT_FIELDS = (
    "cpu_governor",
    "thermal_mode",
    "storage_device",
    "lower_filesystem",
)

REQUIRED_FILESYSTEM_MODES = (
    "plaintext_lowerfs",
    "aegis_q",
    "gocryptfs",
    "fscrypt",
    "dm_crypt_ext4",
)


def run_capture(argv: list[str], timeout_s: float = 5.0) -> dict[str, Any]:
    if shutil.which(argv[0]) is None:
        return {"argv": argv, "available": False, "stdout": "", "stderr": ""}
    try:
        proc = subprocess.run(
            argv,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "argv": argv,
            "available": True,
            "timeout": True,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }
    return {
        "argv": argv,
        "available": True,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def findmnt_for_root() -> dict[str, Any]:
    captured = run_capture(["findmnt", "-T", str(ROOT), "-no", "SOURCE,FSTYPE,TARGET,OPTIONS"])
    fields: dict[str, str] = {}
    if captured.get("returncode") == 0 and captured.get("stdout"):
        parts = captured["stdout"].split(None, 3)
        if len(parts) == 4:
            fields = {
                "source": parts[0],
                "fstype": parts[1],
                "target": parts[2],
                "options": parts[3],
            }
    return {"parsed": fields, "raw": captured}


def lsblk_snapshot() -> dict[str, Any]:
    captured = run_capture(["lsblk", "-J", "-o", "NAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS,MODEL"])
    parsed: Any = None
    if captured.get("returncode") == 0 and captured.get("stdout"):
        try:
            parsed = json.loads(captured["stdout"])
        except json.JSONDecodeError:
            parsed = None
    return {"parsed": parsed, "raw": captured}


def read_cpu_governors() -> dict[str, Any]:
    governors: dict[str, int] = {}
    paths = sorted(glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"))
    for path in paths:
        try:
            value = Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            value = "unreadable"
        governors[value] = governors.get(value, 0) + 1
    return {"paths_observed": len(paths), "governor_counts": governors}


def stable_sha256(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_contract() -> dict[str, Any]:
    root_mount = findmnt_for_root()
    mount_fields = root_mount.get("parsed", {})
    lower_source = mount_fields.get("source", "/dev/nvme0n1p1")
    lower_fstype = mount_fields.get("fstype", "ext4")
    lower_options = mount_fields.get("options", "rw,relatime")

    fio_mount_options = {
        "fio_common": {
            "ioengine": "psync",
            "rw": "randrw",
            "rwmixread": 70,
            "bs": "4k",
            "direct": 0,
            "fdatasync": 1,
            "iodepth": 1,
            "numjobs": 1,
            "size": "1G",
            "fallocate": "none",
            "allow_file_create": 0,
            "overwrite": 1,
            "time_based": True,
            "runtime": "60s",
            "ramp_time": "10s",
            "output_format": "json",
        },
        "file_preparation": {
            "filename": "frozen_randrw_4k_fdatasync.0.0",
            "procedure": (
                "Before each mode's warm-cache sequence, create ${BENCH_DIR}, "
                "create ${BENCH_DIR}/frozen_randrw_4k_fdatasync.0.0, and set "
                "its logical length to 1 GiB with posix_fallocate or an "
                "equivalent sparse truncate path recorded by the harness.  "
                "fio then runs with allow_file_create=0, overwrite=1, and "
                "fallocate=none so file-layout writes are not folded into the "
                "random read/write workload."
            ),
        },
        "command_template": (
            "fio --name=frozen_randrw_4k_fdatasync --ioengine=psync "
            "--rw=randrw --rwmixread=70 --bs=4k --direct=0 --fdatasync=1 "
            "--iodepth=1 --numjobs=1 --size=1G --fallocate=none "
            "--allow_file_create=0 --overwrite=1 --time_based --runtime=60 "
            "--ramp_time=10 --directory=${BENCH_DIR} --output-format=json"
        ),
    }

    workload = {
        "profile_id": "fio_randrw_4k_70r30w_fdatasync_qd1",
        "profile_scope": (
            "Primary mode-aligned filesystem comparison profile; it is not a "
            "SQLite/QoS workload and is not a result by itself."
        ),
        "request_size_bytes": 4096,
        "read_write_mix": {
            "read_percent": 70,
            "write_percent": 30,
            "fio_rw": "randrw",
            "fio_rwmixread": 70,
        },
        "sync_mode": {
            "data_path": "buffered POSIX I/O",
            "fio_ioengine": "psync",
            "direct_io": False,
            "fdatasync_per_write": True,
            "completion_boundary": "record write latency only after fdatasync returns",
        },
        "cache_states": [
            {
                "state": "warm",
                "procedure": (
                    "sparse-create the 1 GiB fio file by the file-preparation "
                    "procedure, run one untimed fio pass with the same profile, "
                    "then run the measured repetitions without dropping caches"
                ),
            },
            {
                "state": "cold",
                "procedure": (
                    "sync, unmount the tested filesystem mode, drop page cache "
                    "with /proc/sys/vm/drop_caches when privileged, remount the "
                    "mode, and run the measured repetition; if this cannot be "
                    "performed, mark the cold-cache row invalid instead of "
                    "folding it into warm-cache results"
                ),
            },
        ],
        "queue_depth": 1,
        "client_count": 1,
        "file_size_bytes": 1_073_741_824,
        "mount_options": fio_mount_options,
        "repetition_count": 5,
        "confidence_interval_method": {
            "name": "nonparametric bootstrap",
            "confidence_level": 0.95,
            "resamples": 10000,
            "random_seed": 12648430,
            "unit": "per-repetition summary rows",
            "metrics": [
                "throughput_mib_s",
                "latency_p50_us",
                "latency_p95_us",
                "latency_p99_us",
                "latency_p99_9_us",
            ],
        },
    }

    filesystem_modes = {
        "plaintext_lowerfs": {
            "role": "raw lower-filesystem control",
            "benchmark_directory": "${LOWER_ROOT}/plaintext_contract",
            "mount_options": {
                "filesystem": lower_fstype,
                "source": lower_source,
                "options": lower_options,
                "encryption": "none",
            },
        },
        "aegis_q": {
            "role": "prototype secure-storage path",
            "benchmark_directory": "${AEGIS_MOUNT}/contract",
            "mount_options": {
                "mount_command": "build/pqc_fuse ${AEGIS_BACKING} ${AEGIS_MOUNT} -f",
                "fuse": "fuse3",
                "required_environment": {
                    "PQC_MASTER_PASSWORD": "set by harness secret file",
                    "PQC_FRESHNESS_ANCHOR_BACKEND": "file",
                    "PQC_QOS_MODE": "disabled for this filesystem-throughput profile",
                },
                "file_tier": "default encrypted data path; no latency-sensitive xattr",
            },
        },
        "gocryptfs": {
            "role": "user-space encrypted-filesystem baseline",
            "benchmark_directory": "${GOCRYPTFS_MOUNT}/contract",
            "mount_options": {
                "mount_command": "gocryptfs -fg -passfile ${PASSFILE} ${GOCRYPTFS_CIPHER} ${GOCRYPTFS_MOUNT}",
                "cipher_directory": "${GOCRYPTFS_CIPHER}",
                "plaintext_names": False,
                "reverse_mode": False,
            },
        },
        "fscrypt": {
            "role": "kernel filesystem-encryption baseline",
            "benchmark_directory": "${FSCRYPT_ROOT}/contract",
            "mount_options": {
                "filesystem": "ext4 with encrypt feature",
                "policy": "fscrypt v2 policy with custom passphrase",
                "mount_command": "mount ${FSCRYPT_BLOCK_DEVICE} ${FSCRYPT_ROOT}",
                "lower_device_policy": "same NVMe device class as plaintext and AEGIS-Q",
            },
        },
        "dm_crypt_ext4": {
            "role": "kernel block-encryption baseline",
            "benchmark_directory": "${DMCRYPT_MOUNT}/contract",
            "mount_options": {
                "cryptsetup": "LUKS2, aes-xts-plain64, 512-bit key",
                "filesystem": "ext4 inside /dev/mapper/${DMCRYPT_NAME}",
                "mount_command": "cryptsetup open ${DMCRYPT_BLOCK_DEVICE} ${DMCRYPT_NAME}; mount /dev/mapper/${DMCRYPT_NAME} ${DMCRYPT_MOUNT}",
                "discard": "disabled",
            },
        },
    }

    environment_contract = {
        "platform": "NVIDIA Jetson AGX Thor Developer Kit",
        "cpu_governor": "performance",
        "thermal_mode": (
            "fixed max-performance/power mode with active cooling; tegrastats "
            "must be retained and any thermal-throttle indication invalidates "
            "the run"
        ),
        "storage_device": "WD PC SN5000S SDEPNSJ-1T00 NVMe, /dev/nvme0n1",
        "lower_filesystem": {
            "filesystem": "ext4",
            "source": "/dev/nvme0n1p1",
            "mount_options": "rw,relatime",
        },
        "background_process_control": (
            "no unrelated foreground GPU/CPU/storage jobs; retain ps/top or "
            "equivalent process snapshot with each benchmark bundle"
        ),
    }

    contract = {
        "contract_id": "aegisq-fs-frozen-v2-2026-06-27",
        "scope": (
            "Mode-aligned filesystem comparison contract.  Results may be "
            "compared only when every filesystem mode follows this contract."
        ),
        "workload_profiles": [workload],
        "filesystem_modes": filesystem_modes,
        "environment_contract": environment_contract,
        "result_validity_rules": [
            "Retain raw fio JSON, stdout/stderr, exact command lines, mount logs, and version strings for every mode.",
            "Do not compare a mode if its warm/cold cache state, queue depth, sync mode, file size, or repetition count deviates from the contract.",
            "Do not include fio file-layout writes in measured rows; the harness must precreate the 1 GiB file and run fio with allow_file_create=0, overwrite=1, and fallocate=none.",
            "Report five-repetition medians and bootstrap 95% confidence intervals; label any smaller run as smoke-only.",
            "Existing sequential fscrypt/dm-crypt fio reference outputs are not contract-compliant until rerun with this profile.",
        ],
    }
    contract_hash = stable_sha256(contract)

    observed_platform = {
        "python_platform": platform.platform(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "findmnt": root_mount,
        "df": run_capture(["df", "-PT", str(ROOT)]),
        "lsblk": lsblk_snapshot(),
        "cpu_governors": read_cpu_governors(),
        "nvpmodel": run_capture(["nvpmodel", "-q"], timeout_s=10.0),
        "jetson_clocks": run_capture(["jetson_clocks", "--show"], timeout_s=10.0),
    }

    report: dict[str, Any] = {
        "schema_version": 1,
        "overall_pass": False,
        "contract_complete": False,
        "current_host_execution_ready": False,
        "contract_sha256": contract_hash,
        "contract": contract,
        "observed_platform": observed_platform,
    }
    report["validation"] = validate_report(report)
    report["contract_complete"] = not report["validation"]["missing_fields"]
    report["current_host_execution_ready"] = not report["validation"]["current_host_warnings"]
    report["overall_pass"] = bool(report["contract_complete"])
    return report


def validate_report(report: dict[str, Any]) -> dict[str, Any]:
    missing: list[str] = []
    contract = report["contract"]
    for idx, profile in enumerate(contract.get("workload_profiles", [])):
        for field in REQUIRED_WORKLOAD_FIELDS:
            if field not in profile or profile[field] in (None, "", [], {}):
                missing.append(f"workload_profiles[{idx}].{field}")

    env = contract.get("environment_contract", {})
    for field in REQUIRED_ENVIRONMENT_FIELDS:
        if field not in env or env[field] in (None, "", [], {}):
            missing.append(f"environment_contract.{field}")

    modes = contract.get("filesystem_modes", {})
    for mode in REQUIRED_FILESYSTEM_MODES:
        if mode not in modes:
            missing.append(f"filesystem_modes.{mode}")
            continue
        if not modes[mode].get("mount_options"):
            missing.append(f"filesystem_modes.{mode}.mount_options")

    warnings: list[str] = []
    governors = report.get("observed_platform", {}).get("cpu_governors", {})
    governor_counts = governors.get("governor_counts", {})
    if governor_counts and set(governor_counts) != {"performance"}:
        warnings.append(
            "current host governor is not uniformly performance; set it before executing the contract"
        )
    nvpmodel = report.get("observed_platform", {}).get("nvpmodel", {})
    if not nvpmodel.get("available") or nvpmodel.get("returncode") not in (0, None):
        warnings.append("nvpmodel state was not captured successfully on this host")
    jetson_clocks = report.get("observed_platform", {}).get("jetson_clocks", {})
    if not jetson_clocks.get("available") or jetson_clocks.get("returncode") not in (0, None):
        warnings.append("jetson_clocks state was not captured successfully on this host")

    return {
        "required_workload_fields": list(REQUIRED_WORKLOAD_FIELDS),
        "required_environment_fields": list(REQUIRED_ENVIRONMENT_FIELDS),
        "required_filesystem_modes": list(REQUIRED_FILESYSTEM_MODES),
        "missing_fields": missing,
        "current_host_warnings": warnings,
    }


def markdown_report(report: dict[str, Any]) -> str:
    contract = report["contract"]
    workload = contract["workload_profiles"][0]
    env = contract["environment_contract"]
    validation = report["validation"]
    lines = [
        "# Frozen Filesystem Workload Contract",
        "",
        f"- Contract ID: `{contract['contract_id']}`",
        f"- Contract SHA-256: `{report['contract_sha256']}`",
        f"- Overall pass: `{str(report['overall_pass']).lower()}`",
        f"- Contract complete: `{str(report['contract_complete']).lower()}`",
        f"- Current host execution ready: `{str(report['current_host_execution_ready']).lower()}`",
        "- Scope: this is a benchmark contract, not a benchmark result.",
        "",
        "## Required workload",
        "",
        f"- Profile: `{workload['profile_id']}`",
        f"- Request size: `{workload['request_size_bytes']}` bytes",
        f"- Read/write mix: `{workload['read_write_mix']['read_percent']}`/`{workload['read_write_mix']['write_percent']}`",
        f"- Sync mode: `{workload['sync_mode']['fio_ioengine']}` with fdatasync per write",
        f"- Queue depth: `{workload['queue_depth']}`",
        f"- Client count: `{workload['client_count']}`",
        f"- File size: `{workload['file_size_bytes']}` bytes",
        f"- Repetitions: `{workload['repetition_count']}`",
        f"- Confidence interval: `{workload['confidence_interval_method']['name']}` "
        f"{int(workload['confidence_interval_method']['confidence_level'] * 100)}% with "
        f"`{workload['confidence_interval_method']['resamples']}` resamples",
        f"- File preparation: `{workload['mount_options']['file_preparation']['procedure']}`",
        f"- Fio command: `{workload['mount_options']['command_template']}`",
        "",
        "## Cache states",
        "",
    ]
    for cache_state in workload["cache_states"]:
        lines.append(f"- `{cache_state['state']}`: {cache_state['procedure']}")

    lines.extend(
        [
            "",
            "## Required environment",
            "",
            f"- CPU governor: `{env['cpu_governor']}`",
            f"- Thermal mode: {env['thermal_mode']}",
            f"- Storage device: `{env['storage_device']}`",
            f"- Lower filesystem: `{env['lower_filesystem']['filesystem']}` on "
            f"`{env['lower_filesystem']['source']}` with "
            f"`{env['lower_filesystem']['mount_options']}`",
            "",
            "## Filesystem modes",
            "",
        ]
    )
    for name, mode in contract["filesystem_modes"].items():
        mount_json = json.dumps(mode["mount_options"], sort_keys=True)
        lines.append(f"- `{name}`: {mode['role']}; mount options `{mount_json}`")

    lines.extend(
        [
            "",
            "## Validity rules",
            "",
        ]
    )
    for rule in contract["result_validity_rules"]:
        lines.append(f"- {rule}")

    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Missing fields: `{len(validation['missing_fields'])}`",
            f"- Current-host warnings: `{len(validation['current_host_warnings'])}`",
        ]
    )
    for warning in validation["current_host_warnings"]:
        lines.append(f"- Warning: {warning}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    report = build_contract()
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "frozen_workload_contract.json"
    md_path = out_dir / "frozen_workload_contract.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(json_path.relative_to(ROOT)),
                "markdown": str(md_path.relative_to(ROOT)),
                "overall_pass": report["overall_pass"],
                "current_host_execution_ready": report["current_host_execution_ready"],
                "contract_sha256": report["contract_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
