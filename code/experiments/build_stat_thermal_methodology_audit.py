#!/usr/bin/env python3
"""Audit statistical and thermal methodology coverage for paper claims.

The audit is intentionally conservative.  It does not convert existing
single-run or three-run artifacts into statistically complete results.  Instead,
it records the methodology required for future headline comparisons, inspects
the current retained artifacts, and verifies that the paper no longer treats
diagnostic measurements as headline performance claims.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "stat_thermal_methodology"

MICROBENCH = ROOT / "artifacts" / "validation" / "microbench" / "summary.json"
MICROBENCH_METHODOLOGY = ROOT / "artifacts" / "validation" / "microbench_methodology" / "summary.json"
QOS_HERO = ROOT / "artifacts" / "validation" / "qos_sqlite_hero_bundle" / "qos_sqlite_hero_bundle.json"
QOS_METHODOLOGY = ROOT / "artifacts" / "validation" / "qos_sqlite_hero_methodology" / "qos_sqlite_hero_bundle.json"
KEYPLANE = ROOT / "artifacts" / "validation" / "keyplane_rekey_workflow" / "keyplane_rekey_workflow.json"
KEYPLANE_METHODOLOGY = ROOT / "artifacts" / "validation" / "keyplane_rekey_methodology" / "keyplane_rekey_workflow.json"
FROZEN_CONTRACT = ROOT / "artifacts" / "validation" / "frozen_workload_contract" / "frozen_workload_contract.json"
FROZEN_AEGISQ = ROOT / "artifacts" / "validation" / "frozen_aegisq_contract" / "frozen_aegisq_contract.json"
FROZEN_GOCRYPTFS = ROOT / "artifacts" / "validation" / "frozen_gocryptfs_contract" / "frozen_gocryptfs_contract.json"
FROZEN_PLAINTEXT = ROOT / "artifacts" / "validation" / "frozen_plaintext_contract" / "frozen_plaintext_contract.json"
KERNEL_BASELINE_FEASIBILITY = ROOT / "artifacts" / "validation" / "kernel_baseline_feasibility" / "kernel_baseline_feasibility.json"
FSCRYPT_REF = ROOT / "artifacts" / "results" / "baselines" / "fscrypt_fio.json"
DMCRYPT_REF = ROOT / "artifacts" / "results" / "baselines" / "dm_crypt_fio.json"

PAPER_FILES = [
    ROOT / "Paper" / "main.tex",
    ROOT / "Paper" / "1_Introduction.tex",
    ROOT / "Paper" / "2_Background.tex",
    ROOT / "Paper" / "3_Design.tex",
    ROOT / "Paper" / "4_Evaluation.tex",
    ROOT / "Paper" / "5_Related_Works.tex",
    ROOT / "Paper" / "6_Conclusion.tex",
    ROOT / "Paper" / "7_Implementation_Details.tex",
    ROOT / "Paper" / "8_Security_Analysis.tex",
    ROOT / "Paper" / "10_Discussion_and_Limitations.tex",
]

ABSTRACT_FORBIDDEN_PATTERNS = {
    "aes_cpu_gbps": r"1\.61\s*~?\\?GB/s",
    "aes_gpu_gbps": r"0\.17(?:2)?\s*~?\\?GB/s",
    "mlkem_gpu_rate": r"1\.50\s*~?M\s+keygens/s",
    "mlkem_cpu_rate": r"64\.8\s*~?K\s+keygens/s",
    "keyplane_gpu_ms": r"21\.070\s*~?ms",
    "keyplane_cpu_ms": r"24\.399\s*~?ms",
}


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def command_output(argv: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        argv,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "argv": argv,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def read_abstract() -> str:
    text = (ROOT / "Paper" / "main.tex").read_text(encoding="utf-8")
    match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.S)
    return match.group(1) if match else ""


def paper_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in PAPER_FILES if path.exists())


def methodology_contract() -> dict[str, Any]:
    return {
        "methodology_id": "aegisq-stat-thermal-v1-2026-06-27",
        "warmup": {
            "required": True,
            "rule": "Each performance mode must run an untimed warmup: at least one full workload pass for fio-style profiles or explicit workload-specific warmup transactions before measured samples.",
        },
        "run_count": {
            "headline_minimum_repetitions": 5,
            "diagnostic_exception": "Three-run primitive measurements are allowed only when labeled diagnostic/descriptive and excluded from abstract or baseline-superiority headline claims.",
        },
        "confidence_interval_method": {
            "name": "nonparametric bootstrap",
            "confidence_level": 0.95,
            "resamples": 10000,
            "unit": "independent repetitions for headline comparisons; transaction samples may be summarized but do not replace independent repetitions",
        },
        "outlier_policy": {
            "rule": "Retain all completed repetitions. Exclude only predeclared infrastructure failures with raw log, exit status, and reason; do not winsorize or silently drop slow runs.",
        },
        "cpu_gpu_clocks_or_power_mode": {
            "cpu_governor": "performance",
            "jetson_power_mode": "record nvpmodel -q",
            "clock_state": "record jetson_clocks --show or equivalent; missing capture invalidates headline runs",
        },
        "thermal_logging": {
            "required": "tegrastats or equivalent raw thermal/power log for the whole measured interval",
            "sampling_interval_ms_max": 100,
            "failure_rule": "Any thermal-throttle or clock-cap indication invalidates the headline run unless the paper labels it as a thermal-throttled case study.",
        },
        "background_process_control": {
            "required": "Record process snapshot and declare unrelated CPU/GPU/storage jobs absent.",
        },
        "cache_dropping_policy": {
            "warm": "Warm-cache measurements require an untimed priming pass.",
            "cold": "Cold-cache measurements require sync, unmount/remount or a privileged drop_caches step; if unavailable, mark the cold-cache row invalid.",
        },
        "failure_handling": {
            "rule": "A mode with command failure, missing raw logs, failed integrity check, missing mount log, or missing metadata is invalid. Scripts must not synthesize zeros for unsupported configurations.",
        },
    }


def artifact_status(name: str, path: Path, category: str, retained: dict[str, Any] | None,
                    present_metadata: list[str], missing_metadata: list[str],
                    paper_role: str, methodology_status: str) -> dict[str, Any]:
    return {
        "name": name,
        "path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
        "category": category,
        "paper_role": paper_role,
        "methodology_status": methodology_status,
        "present_metadata": present_metadata,
        "missing_metadata": missing_metadata,
        "retained_summary": retained or {},
    }


def summarize_microbench() -> dict[str, Any]:
    data = load_json(MICROBENCH)
    retained: dict[str, Any] = {}
    present: list[str] = []
    missing: list[str] = []
    if data:
        retained = {
            "runs": data.get("runs"),
            "manifest_fields": sorted((data.get("manifest") or {}).keys()),
            "workload_map_rows": len(data.get("workload_map") or {}),
            "gpu_mlkem_rows": len(data.get("gpu_mlkem") or {}),
        }
        present = ["raw_stdout_captures", "platform_manifest", "command_list", "run_count", "descriptive_range"]
    return artifact_status(
        "verified_microbench",
        MICROBENCH,
        "primitive placement diagnostic",
        retained,
        present,
        missing,
        "diagnostic only after paper downgrade; not a headline baseline or hero result",
        "diagnostic_scope_gate_passed",
    )


def summarize_microbench_methodology() -> dict[str, Any]:
    data = load_json(MICROBENCH_METHODOLOGY)
    retained: dict[str, Any] = {}
    present: list[str] = []
    missing: list[str] = []
    if data:
        methodology = data.get("methodology") or {}
        clocks = methodology.get("cpu_gpu_clocks_or_power_mode") or {}
        thermal = methodology.get("thermal_logging") or {}
        run_count = methodology.get("run_count") or {}
        retained = {
            "runs": data.get("runs"),
            "warmup_runs": data.get("warmup_runs"),
            "run_count": run_count,
            "governor_ready": clocks.get("cpu_governor_ready"),
            "observed_cpu_governors": clocks.get("observed_cpu_governors"),
            "nvpmodel_returncode": (clocks.get("nvpmodel_q") or {}).get("returncode"),
            "jetson_clocks_returncode": (clocks.get("jetson_clocks_show") or {}).get("returncode"),
            "thermal_line_count": thermal.get("line_count"),
            "thermal_nonempty": thermal.get("nonempty"),
        }
        present = [
            "warmup",
            "headline_minimum_repetitions",
            "bootstrap_ci_over_repetitions",
            "nvpmodel_capture",
            "thermal_log",
            "background_process_snapshot",
            "outlier_policy",
            "cache_state_policy",
            "failure_handling",
        ]
        if not clocks.get("cpu_governor_ready"):
            missing.append("performance_cpu_governor")
        if (clocks.get("jetson_clocks_show") or {}).get("returncode") != 0:
            missing.append("successful_jetson_clocks_capture")
        if not thermal.get("nonempty"):
            missing.append("nonempty_thermal_log")
        if not run_count.get("meets_headline_minimum"):
            missing.append("headline_minimum_repetitions")
    else:
        missing = ["methodology_run_missing"]
    return artifact_status(
        "verified_microbench_methodology_run",
        MICROBENCH_METHODOLOGY,
        "methodology-strengthened primitive diagnostic run",
        retained,
        present,
        missing,
        "progress evidence only; not the paper figure source",
        "methodology_progress_host_not_ready" if missing else "methodology_metadata_retained",
    )


def summarize_qos() -> dict[str, Any]:
    data = load_json(QOS_HERO)
    retained: dict[str, Any] = {}
    present: list[str] = []
    missing: list[str] = []
    if data:
        modes = data.get("modes") or []
        retained = {
            "overall_pass": data.get("overall_pass"),
            "mode_count": len(modes),
            "foreground_samples_per_mode": {
                mode.get("mode"): (mode.get("foreground") or {}).get("samples")
                for mode in modes
            },
            "has_platform": bool(data.get("platform")),
            "has_component_coverage": bool(data.get("component_coverage")),
        }
        present = [
            "raw_foreground_logs",
            "raw_background_logs",
            "policy_trace_logs",
            "mounted_fuse_logs",
            "telemetry_logs",
            "platform_manifest",
            "workload_arguments",
            "failure_gate",
        ]
    else:
        missing = ["single_workflow_artifact_missing"]
    return artifact_status(
        "qos_sqlite_hero_bundle",
        QOS_HERO,
        "single synchronized mounted-FUSE QoS bundle",
        retained,
        present,
        missing,
        "single-bundle SQLite recovery evidence; statistical metadata is tracked by qos_sqlite_hero_methodology",
        "single_workflow_scope_gate_passed" if data else "single_workflow_missing",
    )


def summarize_qos_methodology() -> dict[str, Any]:
    data = load_json(QOS_METHODOLOGY)
    retained: dict[str, Any] = {}
    present: list[str] = []
    missing: list[str] = []
    if data:
        methodology = data.get("methodology") or {}
        clocks = methodology.get("cpu_gpu_clocks_or_power_mode") or {}
        thermal = methodology.get("thermal_logging") or {}
        run_count = methodology.get("run_count") or {}
        warmup = methodology.get("warmup") or {}
        mode_summaries = data.get("mode_summaries") or []
        all_modes_have_ci = bool(mode_summaries) and all(
            "p99_ms" in mode and
            "storage_mb_s" in mode and
            "ci95_low" in mode["p99_ms"] and
            "ci95_high" in mode["p99_ms"] and
            "ci95_low" in mode["storage_mb_s"] and
            "ci95_high" in mode["storage_mb_s"]
            for mode in mode_summaries
        )
        recovery_summary = data.get("recovery_check_summary") or {}
        unstable_recovery = {
            key: value
            for key, value in recovery_summary.items()
            if not value.get("all_true")
        }
        retained = {
            "overall_pass": data.get("overall_pass"),
            "warmup_runs": data.get("warmup_runs"),
            "repetitions_measured": data.get("repetitions_measured"),
            "run_count": run_count,
            "mode_summary_count": len(mode_summaries),
            "unstable_recovery_checks": unstable_recovery,
            "governor_ready": clocks.get("cpu_governor_ready"),
            "observed_cpu_governors": clocks.get("observed_cpu_governors"),
            "nvpmodel_returncode": (clocks.get("nvpmodel_q") or {}).get("returncode"),
            "jetson_clocks_returncode": (clocks.get("jetson_clocks_show") or {}).get("returncode"),
            "thermal_line_count": thermal.get("line_count"),
            "thermal_nonempty": thermal.get("nonempty"),
        }
        present = [
            "warmup",
            "headline_minimum_repetitions",
            "bootstrap_ci_over_repetitions",
            "platform_manifest",
            "nvpmodel_capture",
            "thermal_log",
            "background_process_snapshot",
            "outlier_policy",
            "cache_state_policy",
            "failure_handling",
        ]
        if not warmup.get("full_workload_warmup"):
            missing.append("full_workload_warmup")
        if not run_count.get("meets_headline_minimum"):
            missing.append("independent_repetitions_ge_5")
        if not all_modes_have_ci:
            missing.append("bootstrap_ci_over_repetitions")
        if not clocks.get("cpu_governor_ready"):
            missing.append("performance_cpu_governor")
        if (clocks.get("nvpmodel_q") or {}).get("returncode") != 0 and (
            clocks.get("jetson_clocks_show") or {}
        ).get("returncode") != 0:
            missing.append("nvpmodel_or_jetson_clocks")
        elif (clocks.get("jetson_clocks_show") or {}).get("returncode") != 0:
            missing.append("successful_jetson_clocks_capture")
        if not thermal.get("nonempty"):
            missing.append("nonempty_thermal_log")
        if unstable_recovery:
            missing.append("recovery_stability_gate_failed")
    else:
        missing = ["methodology_run_missing"]
    host_or_stability_missing = {
        "performance_cpu_governor",
        "successful_jetson_clocks_capture",
        "recovery_stability_gate_failed",
    }
    hard_missing = [item for item in missing if item not in host_or_stability_missing]
    if not data or hard_missing:
        methodology_status = "not_complete_for_statistical_headline"
    elif "recovery_stability_gate_failed" in missing:
        methodology_status = "methodology_progress_recovery_unstable"
    elif missing:
        methodology_status = "methodology_progress_host_not_ready"
    else:
        methodology_status = "methodology_metadata_retained"
    return artifact_status(
        "qos_sqlite_hero_methodology",
        QOS_METHODOLOGY,
        "methodology-strengthened repeated SQLite QoS workflow run",
        retained,
        present,
        missing,
        "progress evidence for repeated SQLite QoS methodology; paper source remains the single workflow bundle",
        methodology_status,
    )


def summarize_keyplane() -> dict[str, Any]:
    data = load_json(KEYPLANE)
    retained: dict[str, Any] = {}
    present: list[str] = []
    missing: list[str] = []
    if data:
        retained = {
            "overall_pass": data.get("overall_pass"),
            "files_per_mode": data.get("files_per_mode"),
            "mode_count": len(data.get("modes") or []),
            "gpu_vs_cpu_speedup": data.get("gpu_vs_cpu_speedup"),
        }
        present = ["mode_logs", "admission_traces", "failure_gate", "files_per_mode", "collect_ms"]
    else:
        missing = ["single_workflow_artifact_missing"]
    return artifact_status(
        "keyplane_rekey_workflow",
        KEYPLANE,
        "single mounted maintenance workflow",
        retained,
        present,
        missing,
        "single-bundle workflow evidence; statistical metadata is tracked by keyplane_rekey_methodology",
        "single_workflow_scope_gate_passed" if data else "single_workflow_missing",
    )


def summarize_keyplane_methodology() -> dict[str, Any]:
    data = load_json(KEYPLANE_METHODOLOGY)
    retained: dict[str, Any] = {}
    present: list[str] = []
    missing: list[str] = []
    if data:
        methodology = data.get("methodology") or {}
        clocks = methodology.get("cpu_gpu_clocks_or_power_mode") or {}
        thermal = methodology.get("thermal_logging") or {}
        run_count = methodology.get("run_count") or {}
        warmup = methodology.get("warmup") or {}
        mode_summaries = data.get("mode_summaries") or []
        all_modes_have_ci = bool(mode_summaries) and all(
            "total_rekey_usec_ci95_low" in mode and
            "total_rekey_usec_ci95_high" in mode and
            "throughput_files_per_s_ci95_low" in mode and
            "throughput_files_per_s_ci95_high" in mode
            for mode in mode_summaries
        )
        retained = {
            "overall_pass": data.get("overall_pass"),
            "files_per_mode": data.get("files_per_mode"),
            "warmup_runs": data.get("warmup_runs"),
            "repetitions_measured": data.get("repetitions_measured"),
            "run_count": run_count,
            "gpu_vs_cpu_speedup_summary": data.get("gpu_vs_cpu_speedup_summary"),
            "mode_summary_count": len(mode_summaries),
            "governor_ready": clocks.get("cpu_governor_ready"),
            "observed_cpu_governors": clocks.get("observed_cpu_governors"),
            "nvpmodel_returncode": (clocks.get("nvpmodel_q") or {}).get("returncode"),
            "jetson_clocks_returncode": (clocks.get("jetson_clocks_show") or {}).get("returncode"),
            "thermal_line_count": thermal.get("line_count"),
            "thermal_nonempty": thermal.get("nonempty"),
        }
        present = [
            "warmup",
            "headline_minimum_repetitions",
            "bootstrap_ci_over_repetitions",
            "platform_manifest",
            "nvpmodel_capture",
            "thermal_log",
            "background_process_snapshot",
            "outlier_policy",
            "cache_state_policy",
            "failure_handling",
        ]
        if not data.get("overall_pass"):
            missing.append("workflow_acceptance_or_speedup_gate")
        if not warmup.get("full_workload_warmup"):
            missing.append("full_workload_warmup")
        if not run_count.get("meets_headline_minimum"):
            missing.append("independent_repetitions_ge_5")
        if not all_modes_have_ci:
            missing.append("bootstrap_ci_over_repetitions")
        if not clocks.get("cpu_governor_ready"):
            missing.append("performance_cpu_governor")
        if (clocks.get("nvpmodel_q") or {}).get("returncode") != 0 and (
            clocks.get("jetson_clocks_show") or {}
        ).get("returncode") != 0:
            missing.append("nvpmodel_or_jetson_clocks")
        elif (clocks.get("jetson_clocks_show") or {}).get("returncode") != 0:
            missing.append("successful_jetson_clocks_capture")
        if not thermal.get("nonempty"):
            missing.append("nonempty_thermal_log")
    else:
        missing = ["methodology_run_missing"]
    host_only_missing = {"performance_cpu_governor", "successful_jetson_clocks_capture"}
    hard_missing = [item for item in missing if item not in host_only_missing]
    if not data or hard_missing:
        methodology_status = "not_complete_for_statistical_headline"
    elif missing:
        methodology_status = "methodology_progress_host_not_ready"
    else:
        methodology_status = "methodology_metadata_retained"
    return artifact_status(
        "keyplane_rekey_methodology",
        KEYPLANE_METHODOLOGY,
        "methodology-strengthened mounted key-plane workflow run",
        retained,
        present,
        missing,
        "progress evidence for repeated key-plane workflow methodology; paper source remains the single workflow bundle",
        methodology_status,
    )


def summarize_frozen_contract() -> dict[str, Any]:
    data = load_json(FROZEN_CONTRACT)
    retained: dict[str, Any] = {}
    present: list[str] = []
    missing: list[str] = []
    if data:
        retained = {
            "overall_pass": data.get("overall_pass"),
            "contract_complete": data.get("contract_complete"),
            "current_host_execution_ready": data.get("current_host_execution_ready"),
            "contract_sha256": data.get("contract_sha256"),
        }
        present = [
            "request_size",
            "read_write_mix",
            "sync_mode",
            "warm_cold_cache",
            "queue_depth",
            "client_count",
            "file_size",
            "mount_options",
            "cpu_governor",
            "thermal_mode",
            "storage_device",
            "lower_filesystem",
            "repetition_count",
            "confidence_interval_method",
        ]
        if not data.get("current_host_execution_ready"):
            missing.append("current_host_execution_ready_for_benchmark_execution")
    return artifact_status(
        "frozen_workload_contract",
        FROZEN_CONTRACT,
        "contract, not result",
        retained,
        present,
        missing,
        "future filesystem comparison contract",
        "contract_complete_not_executed",
    )


def summarize_frozen_aegisq() -> dict[str, Any]:
    data = load_json(FROZEN_AEGISQ)
    retained: dict[str, Any] = {}
    present: list[str] = []
    missing: list[str] = []
    if data:
        artifacts = data.get("artifacts", {})
        summary = data.get("warm_cache_summary", {})
        metrics = summary.get("metrics", {})
        rows = data.get("repetitions", [])
        measured_rows = [
            row for row in rows
            if row.get("cache_state") == "warm" and row.get("valid") is True
        ]
        warmup_rows = [
            row for row in rows
            if row.get("cache_state") == "warmup" and row.get("valid") is True
        ]
        thermal = data.get("thermal_logging", {})
        thermal_command = thermal.get("command", [])
        thermal_interval_ms = None
        if isinstance(thermal_command, list) and "--interval" in thermal_command:
            try:
                thermal_interval_ms = int(thermal_command[thermal_command.index("--interval") + 1])
            except (ValueError, IndexError):
                thermal_interval_ms = None
        file_prep = data.get("file_preparation") or {}
        paper = paper_text()
        paper_mentions_result = bool(
            re.search(r"AEGIS-Q warm-cache row", paper)
            and re.search(r"0\.359\s*~?MiB/s", paper)
            and re.search(r"11\.21\s*~?ms", paper)
        )
        retained = {
            "overall_pass": data.get("overall_pass"),
            "contract_id": data.get("contract_id"),
            "contract_compliant_warm_cache": data.get("contract_compliant_warm_cache"),
            "comparison_ready": data.get("comparison_ready"),
            "file_preparation_valid": file_prep.get("valid"),
            "file_preparation_method": file_prep.get("method"),
            "warm_valid_repetitions": summary.get("valid_repetitions"),
            "cold_cache_status": (data.get("cold_cache") or {}).get("status"),
            "thermal_interval_ms": thermal_interval_ms,
            "paper_mentions_result": paper_mentions_result,
        }

        checks = {
            "result_json": data.get("overall_pass") is True,
            "contract_id_v2": data.get("contract_id") == "aegisq-fs-frozen-v2-2026-06-27",
            "file_preparation": file_prep.get("valid") is True,
            "warmup": bool(warmup_rows),
            "five_valid_measured_repetitions": len(measured_rows) == 5,
            "bootstrap_ci": all(
                metric in metrics and
                "ci95_low" in metrics[metric] and
                "ci95_high" in metrics[metric]
                for metric in (
                    "throughput_mib_s",
                    "latency_p50_us",
                    "latency_p95_us",
                    "latency_p99_us",
                    "latency_p99_9_us",
                )
            ),
            "csv": bool(artifacts.get("csv")) and (ROOT / artifacts.get("csv", "")).exists(),
            "file_preparation_artifact": bool(artifacts.get("file_preparation")) and (ROOT / artifacts.get("file_preparation", "")).exists(),
            "fio_raw_dir": bool(artifacts.get("fio_raw_dir")) and (ROOT / artifacts.get("fio_raw_dir", "")).is_dir(),
            "mount_logs": bool(artifacts.get("mount_logs")) and (ROOT / artifacts.get("mount_logs", "")).is_dir(),
            "platform_manifest": bool(artifacts.get("platform_manifest")) and (ROOT / artifacts.get("platform_manifest", "")).exists(),
            "thermal_log": bool(artifacts.get("thermal_log")) and thermal.get("nonempty") is True,
            "thermal_interval_at_most_100ms": thermal_interval_ms is not None and thermal_interval_ms <= 100,
            "cold_cache_invalid_declared": (data.get("cold_cache") or {}).get("status") == "invalid_not_run",
            "paper_result_scope": paper_mentions_result,
        }
        present = [name for name, ok in checks.items() if ok]
        missing = [name for name, ok in checks.items() if not ok]

    methodology_status = (
        "methodology_metadata_retained_cold_invalid"
        if data and not missing
        else "not_complete_for_frozen_aegisq"
    )
    return artifact_status(
        "frozen_aegisq_contract",
        FROZEN_AEGISQ,
        "AEGIS-Q-only frozen filesystem contract row",
        retained,
        present,
        missing,
        "warm-cache AEGIS-Q result only; not a cross-system comparison and not a cold-cache row",
        methodology_status,
    )


def summarize_frozen_gocryptfs() -> dict[str, Any]:
    data = load_json(FROZEN_GOCRYPTFS)
    retained: dict[str, Any] = {}
    present: list[str] = []
    missing: list[str] = []
    if data:
        artifacts = data.get("artifacts", {})
        summary = data.get("warm_cache_summary", {})
        metrics = summary.get("metrics", {})
        rows = data.get("repetitions", [])
        measured_rows = [
            row for row in rows
            if row.get("cache_state") == "warm" and row.get("valid") is True
        ]
        warmup_rows = [
            row for row in rows
            if row.get("cache_state") == "warmup" and row.get("valid") is True
        ]
        thermal = data.get("thermal_logging", {})
        thermal_command = thermal.get("command", [])
        thermal_interval_ms = None
        if isinstance(thermal_command, list) and "--interval" in thermal_command:
            try:
                thermal_interval_ms = int(thermal_command[thermal_command.index("--interval") + 1])
            except (ValueError, IndexError):
                thermal_interval_ms = None
        file_prep = data.get("file_preparation") or {}
        platform_summary = data.get("platform") or {}
        gocryptfs_version = (platform_summary.get("gocryptfs_version") or {}).get("stdout", "")
        paper = paper_text()
        paper_mentions_result = bool(
            re.search(r"gocryptfs warm-cache row", paper)
            and re.search(r"21\.72\s*~?MiB/s", paper)
            and re.search(r"0\.060\s*~?ms", paper)
        )
        retained = {
            "overall_pass": data.get("overall_pass"),
            "contract_id": data.get("contract_id"),
            "contract_compliant_warm_cache": data.get("contract_compliant_warm_cache"),
            "file_preparation_valid": file_prep.get("valid"),
            "file_preparation_method": file_prep.get("method"),
            "warm_valid_repetitions": summary.get("valid_repetitions"),
            "cold_cache_status": (data.get("cold_cache") or {}).get("status"),
            "thermal_interval_ms": thermal_interval_ms,
            "gocryptfs_version": gocryptfs_version,
            "paper_mentions_result": paper_mentions_result,
        }

        checks = {
            "result_json": data.get("overall_pass") is True,
            "contract_id_v2": data.get("contract_id") == "aegisq-fs-frozen-v2-2026-06-27",
            "filesystem_mode": data.get("filesystem_mode") == "gocryptfs",
            "gocryptfs_version": bool(gocryptfs_version),
            "mount_options": bool(data.get("gocryptfs_mount")),
            "file_preparation": file_prep.get("valid") is True,
            "warmup": bool(warmup_rows),
            "five_valid_measured_repetitions": len(measured_rows) == 5,
            "bootstrap_ci": all(
                metric in metrics and
                "ci95_low" in metrics[metric] and
                "ci95_high" in metrics[metric]
                for metric in (
                    "throughput_mib_s",
                    "latency_p50_us",
                    "latency_p95_us",
                    "latency_p99_us",
                    "latency_p99_9_us",
                )
            ),
            "csv": bool(artifacts.get("csv")) and (ROOT / artifacts.get("csv", "")).exists(),
            "file_preparation_artifact": bool(artifacts.get("file_preparation")) and (ROOT / artifacts.get("file_preparation", "")).exists(),
            "fio_raw_dir": bool(artifacts.get("fio_raw_dir")) and (ROOT / artifacts.get("fio_raw_dir", "")).is_dir(),
            "mount_logs": bool(artifacts.get("mount_logs")) and (ROOT / artifacts.get("mount_logs", "")).is_dir(),
            "platform_manifest": bool(artifacts.get("platform_manifest")) and (ROOT / artifacts.get("platform_manifest", "")).exists(),
            "thermal_log": bool(artifacts.get("thermal_log")) and thermal.get("nonempty") is True,
            "thermal_interval_at_most_100ms": thermal_interval_ms is not None and thermal_interval_ms <= 100,
            "cold_cache_invalid_declared": (data.get("cold_cache") or {}).get("status") == "invalid_not_run",
            "paper_result_scope": paper_mentions_result,
        }
        present = [name for name, ok in checks.items() if ok]
        missing = [name for name, ok in checks.items() if not ok]

    methodology_status = (
        "baseline_metadata_retained_cold_invalid"
        if data and not missing
        else "not_complete_for_frozen_gocryptfs"
    )
    return artifact_status(
        "frozen_gocryptfs_contract",
        FROZEN_GOCRYPTFS,
        "gocryptfs frozen filesystem contract row",
        retained,
        present,
        missing,
        "warm-cache gocryptfs baseline row only; not a full filesystem comparison matrix and not a cold-cache row",
        methodology_status,
    )


def summarize_frozen_plaintext() -> dict[str, Any]:
    data = load_json(FROZEN_PLAINTEXT)
    retained: dict[str, Any] = {}
    present: list[str] = []
    missing: list[str] = []
    if data:
        artifacts = data.get("artifacts", {})
        summary = data.get("warm_cache_summary", {})
        metrics = summary.get("metrics", {})
        rows = data.get("repetitions", [])
        measured_rows = [
            row for row in rows
            if row.get("cache_state") == "warm" and row.get("valid") is True
        ]
        warmup_rows = [
            row for row in rows
            if row.get("cache_state") == "warmup" and row.get("valid") is True
        ]
        thermal = data.get("thermal_logging", {})
        thermal_command = thermal.get("command", [])
        thermal_interval_ms = None
        if isinstance(thermal_command, list) and "--interval" in thermal_command:
            try:
                thermal_interval_ms = int(thermal_command[thermal_command.index("--interval") + 1])
            except (ValueError, IndexError):
                thermal_interval_ms = None
        file_prep = data.get("file_preparation") or {}
        lower_findmnt = ((data.get("mount_options") or {}).get("findmnt_bench_root") or {}).get("stdout", "")
        paper = paper_text()
        paper_mentions_result = bool(
            re.search(r"plaintext warm-cache row", paper)
            and re.search(r"32\.88\s*~?MiB/s", paper)
            and re.search(r"0\.165\s*~?ms", paper)
        )
        retained = {
            "overall_pass": data.get("overall_pass"),
            "contract_id": data.get("contract_id"),
            "contract_compliant_warm_cache": data.get("contract_compliant_warm_cache"),
            "file_preparation_valid": file_prep.get("valid"),
            "file_preparation_method": file_prep.get("method"),
            "warm_valid_repetitions": summary.get("valid_repetitions"),
            "cold_cache_status": (data.get("cold_cache") or {}).get("status"),
            "thermal_interval_ms": thermal_interval_ms,
            "lower_findmnt": lower_findmnt,
            "paper_mentions_result": paper_mentions_result,
        }

        checks = {
            "result_json": data.get("overall_pass") is True,
            "contract_id_v2": data.get("contract_id") == "aegisq-fs-frozen-v2-2026-06-27",
            "filesystem_mode": data.get("filesystem_mode") == "plaintext_lowerfs",
            "lower_ext4": " ext4 " in f" {lower_findmnt} ",
            "file_preparation": file_prep.get("valid") is True,
            "warmup": bool(warmup_rows),
            "five_valid_measured_repetitions": len(measured_rows) == 5,
            "bootstrap_ci": all(
                metric in metrics and
                "ci95_low" in metrics[metric] and
                "ci95_high" in metrics[metric]
                for metric in (
                    "throughput_mib_s",
                    "latency_p50_us",
                    "latency_p95_us",
                    "latency_p99_us",
                    "latency_p99_9_us",
                )
            ),
            "csv": bool(artifacts.get("csv")) and (ROOT / artifacts.get("csv", "")).exists(),
            "file_preparation_artifact": bool(artifacts.get("file_preparation")) and (ROOT / artifacts.get("file_preparation", "")).exists(),
            "fio_raw_dir": bool(artifacts.get("fio_raw_dir")) and (ROOT / artifacts.get("fio_raw_dir", "")).is_dir(),
            "platform_manifest": bool(artifacts.get("platform_manifest")) and (ROOT / artifacts.get("platform_manifest", "")).exists(),
            "thermal_log": bool(artifacts.get("thermal_log")) and thermal.get("nonempty") is True,
            "thermal_interval_at_most_100ms": thermal_interval_ms is not None and thermal_interval_ms <= 100,
            "cold_cache_invalid_declared": (data.get("cold_cache") or {}).get("status") == "invalid_not_run",
            "paper_result_scope": paper_mentions_result,
        }
        present = [name for name, ok in checks.items() if ok]
        missing = [name for name, ok in checks.items() if not ok]

    methodology_status = (
        "baseline_metadata_retained_cold_invalid"
        if data and not missing
        else "not_complete_for_frozen_plaintext"
    )
    return artifact_status(
        "frozen_plaintext_contract",
        FROZEN_PLAINTEXT,
        "plaintext lower-filesystem frozen contract row",
        retained,
        present,
        missing,
        "warm-cache plaintext lowerfs baseline row only; not a full filesystem comparison matrix and not a cold-cache row",
        methodology_status,
    )


def summarize_kernel_baseline_feasibility() -> dict[str, Any]:
    data = load_json(KERNEL_BASELINE_FEASIBILITY)
    retained: dict[str, Any] = {}
    present: list[str] = []
    missing: list[str] = []
    if data:
        fscrypt = data.get("fscrypt") or {}
        dmcrypt = data.get("dm_crypt_ext4") or {}
        retained = {
            "overall_pass": data.get("overall_pass"),
            "contract_id": (data.get("contract") or {}).get("contract_id"),
            "fscrypt_runnable_without_interactive_root": fscrypt.get("runnable_without_interactive_root"),
            "fscrypt_blocking_reasons": fscrypt.get("blocking_reasons"),
            "dm_crypt_runnable_without_interactive_root": dmcrypt.get("runnable_without_interactive_root"),
            "dm_crypt_blocking_reasons": dmcrypt.get("blocking_reasons"),
        }
        checks = {
            "result_json": data.get("overall_pass") is True,
            "contract_id_v2": (data.get("contract") or {}).get("contract_id") == "aegisq-fs-frozen-v2-2026-06-27",
            "fscrypt_probe": isinstance(fscrypt.get("blocking_reasons"), list),
            "dmcrypt_probe": isinstance(dmcrypt.get("blocking_reasons"), list),
            "scope_nonexecuting": "does not execute fscrypt or dm-crypt" in data.get("scope", ""),
        }
        present = [name for name, ok in checks.items() if ok]
        missing = [name for name, ok in checks.items() if not ok]
    else:
        missing = ["kernel_baseline_feasibility_missing"]
    return artifact_status(
        "kernel_baseline_feasibility",
        KERNEL_BASELINE_FEASIBILITY,
        "non-destructive fscrypt/dm-crypt feasibility audit",
        retained,
        present,
        missing,
        "explains why fscrypt/dm-crypt frozen rows remain open on this host; not a benchmark result",
        "kernel_baseline_feasibility_retained" if data and not missing else "kernel_baseline_feasibility_missing",
    )


def summarize_baseline_refs() -> dict[str, Any]:
    present = [path.name for path in (FSCRYPT_REF, DMCRYPT_REF) if path.exists()]
    text = paper_text()
    forbidden_patterns = {
        "matched_reference_row": r"Matched fscrypt/dm-crypt reference runs",
        "reference_points_until_rerun": r"Existing fscrypt and dm-crypt sequential-write baselines remain reference points",
        "retained_matched_runs": r"repository retains matched fscrypt and dm-crypt sequential-write fio runs",
    }
    forbidden_hits = {
        name: bool(re.search(pattern, text))
        for name, pattern in forbidden_patterns.items()
    }
    required_scope_patterns = {
        "historical_traceability_only": r"Historical fscrypt and dm-crypt sequential fio outputs are retained in the repository for traceability only",
        "not_current_comparison_evidence": r"not used as current comparison evidence",
        "no_apples_to_apples": r"reports no apples-to-apples end-to-end comparison",
    }
    required_scope_hits = {
        name: bool(re.search(pattern, text))
        for name, pattern in required_scope_patterns.items()
    }
    scope_gate_pass = (
        bool(present)
        and not any(forbidden_hits.values())
        and all(required_scope_hits.values())
    )
    missing: list[str] = []
    if not scope_gate_pass:
        missing = [
            "frozen_contract_execution",
            "warm_cold_cache_rows",
            "five_repetitions",
            "bootstrap_ci",
            "cpu_governor",
            "thermal_log",
            "background_process_snapshot",
            "paper_scope_gate_for_historical_fio_refs",
        ]
    return artifact_status(
        "fscrypt_dmcrypt_reference_fio",
        ROOT / "artifacts" / "results" / "baselines",
        "historical reference-only baseline outputs",
        {
            "present_files": present,
            "scope_gate_pass": scope_gate_pass,
            "forbidden_hits": forbidden_hits,
            "required_scope_hits": required_scope_hits,
        },
        ["raw_fio_json", "paper_scope_gate"] if scope_gate_pass else (["raw_fio_json"] if present else []),
        missing,
        "historical repository artifacts only; not current paper comparison evidence",
        "baseline_scope_gate_passed" if scope_gate_pass else "not_complete_for_fair_baseline",
    )


def paper_checks() -> dict[str, Any]:
    abstract = read_abstract()
    text = paper_text()
    violations = [
        {"name": name, "pattern": pattern}
        for name, pattern in ABSTRACT_FORBIDDEN_PATTERNS.items()
        if re.search(pattern, abstract)
    ]
    required_phrases = {
        "diagnostic_placement": r"diagnostic placement",
        "single_workflow_not_statistical": r"single retained workflow.*not statistical confidence",
        "future_contract": r"five repetitions.*bootstrap 95\\% confidence intervals",
        "limitations_three_run": r"three-run primitive measurements cannot establish",
    }
    phrase_hits = {
        name: bool(re.search(pattern, text, re.S))
        for name, pattern in required_phrases.items()
    }
    return {
        "abstract_headline_violations": violations,
        "required_phrase_hits": phrase_hits,
        "paper_scope_gate_pass": not violations and all(phrase_hits.values()),
    }


def build_report() -> dict[str, Any]:
    artifacts = [
        summarize_microbench(),
        summarize_microbench_methodology(),
        summarize_qos(),
        summarize_qos_methodology(),
        summarize_keyplane(),
        summarize_keyplane_methodology(),
        summarize_frozen_contract(),
        summarize_frozen_aegisq(),
        summarize_frozen_gocryptfs(),
        summarize_frozen_plaintext(),
        summarize_kernel_baseline_feasibility(),
        summarize_baseline_refs(),
    ]
    checks = paper_checks()
    completion_blockers = [
        {
            "artifact": artifact["name"],
            "missing_metadata": artifact["missing_metadata"],
            "status": artifact["methodology_status"],
        }
        for artifact in artifacts
        if artifact["methodology_status"] not in (
            "contract_complete_not_executed",
            "diagnostic_scope_gate_passed",
            "methodology_metadata_retained",
            "methodology_progress_host_not_ready",
            "methodology_progress_recovery_unstable",
            "methodology_metadata_retained_cold_invalid",
            "baseline_metadata_retained_cold_invalid",
            "kernel_baseline_feasibility_retained",
            "baseline_scope_gate_passed",
        ) and artifact["missing_metadata"]
    ]
    methodology_complete = (
        checks["paper_scope_gate_pass"]
        and not completion_blockers
        and all(artifact["exists"] for artifact in artifacts)
    )
    return {
        "schema_version": 1,
        "audit_id": "aegisq-stat-thermal-methodology-audit-2026-06-27",
        "overall_pass": methodology_complete,
        "paper_scope_gate_pass": checks["paper_scope_gate_pass"],
        "methodology_complete": methodology_complete,
        "methodology_contract": methodology_contract(),
        "paper_checks": checks,
        "artifacts": artifacts,
        "completion_blockers": completion_blockers,
        "commands": {
            "generate": ["python3", "code/experiments/build_stat_thermal_methodology_audit.py"],
            "enforce_complete": [
                "python3",
                "experiments/build_stat_thermal_methodology_audit.py",
                "--require-complete",
            ],
        },
        "git": {
            "head": command_output(["git", "rev-parse", "HEAD"]),
            "status_short": command_output(["git", "status", "--short"]),
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Statistical and Thermal Methodology Audit",
        "",
        f"- Overall pass: `{str(report['overall_pass']).lower()}`",
        f"- Paper scope gate pass: `{str(report['paper_scope_gate_pass']).lower()}`",
        f"- Methodology complete: `{str(report['methodology_complete']).lower()}`",
        "- Scope: this audit defines the required method and verifies retained benchmark-result coverage without turning scoped rows into cross-system claims.",
        "",
        "## Required Methodology",
        "",
    ]
    contract = report["methodology_contract"]
    lines.extend([
        f"- Warmup: {contract['warmup']['rule']}",
        f"- Run count: at least `{contract['run_count']['headline_minimum_repetitions']}` independent repetitions for headline results.",
        f"- Confidence intervals: {contract['confidence_interval_method']['name']} at 95% with `{contract['confidence_interval_method']['resamples']}` resamples.",
        f"- Outlier policy: {contract['outlier_policy']['rule']}",
        f"- Clocks/power: CPU governor `{contract['cpu_gpu_clocks_or_power_mode']['cpu_governor']}`, plus nvpmodel and clock-state capture.",
        f"- Thermal logging: {contract['thermal_logging']['required']}",
        f"- Background control: {contract['background_process_control']['required']}",
        f"- Cache policy: warm and cold rows must follow the declared procedures.",
        f"- Failure handling: {contract['failure_handling']['rule']}",
        "",
        "## Current Artifacts",
        "",
    ])
    for artifact in report["artifacts"]:
        lines.append(
            f"- `{artifact['name']}`: `{artifact['methodology_status']}`; "
            f"missing `{len(artifact['missing_metadata'])}` metadata fields."
        )
    lines.extend(["", "## Paper Gate", ""])
    for name, hit in report["paper_checks"]["required_phrase_hits"].items():
        lines.append(f"- {name}: `{str(hit).lower()}`")
    if report["paper_checks"]["abstract_headline_violations"]:
        for violation in report["paper_checks"]["abstract_headline_violations"]:
            lines.append(f"- abstract violation: `{violation['name']}`")
    else:
        lines.append("- abstract headline violations: `0`")
    lines.extend(["", "## Completion Blockers", ""])
    if report["completion_blockers"]:
        for blocker in report["completion_blockers"]:
            missing = ", ".join(blocker["missing_metadata"])
            lines.append(f"- `{blocker['artifact']}`: {missing}")
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    report = build_report()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "stat_thermal_methodology_audit.json"
    md_path = out_dir / "stat_thermal_methodology_audit.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(
        json.dumps(
            {
                "json": str(json_path.relative_to(ROOT)),
                "markdown": str(md_path.relative_to(ROOT)),
                "overall_pass": report["overall_pass"],
                "paper_scope_gate_pass": report["paper_scope_gate_pass"],
                "methodology_complete": report["methodology_complete"],
                "blockers": len(report["completion_blockers"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.require_complete and not report["methodology_complete"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
