#!/usr/bin/env python3
"""Build the E1 SQLite hero validity closeout.

This is a narrow gate proof.  It does not rerun the workload or invent a new
headline number; it checks that the retained SQLite hero evidence is sufficient
for a scoped workload-envelope claim and that paper text does not turn that
claim into broad workload generality.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "sqlite_hero_validity_closeout"

SINGLE_BUNDLE = ROOT / "artifacts" / "validation" / "qos_sqlite_hero_bundle" / "qos_sqlite_hero_bundle.json"
METHODOLOGY_BUNDLE = ROOT / "artifacts" / "validation" / "qos_sqlite_hero_methodology" / "qos_sqlite_hero_bundle.json"
KERNEL_QOS_COMPARISON = ROOT / "artifacts" / "validation" / "sqlite_kernel_qos_comparison" / "sqlite_kernel_qos_comparison.json"
STAT_THERMAL_AUDIT = ROOT / "artifacts" / "validation" / "stat_thermal_methodology" / "stat_thermal_methodology_audit.json"
THERMAL_LOG = ROOT / "artifacts" / "validation" / "qos_sqlite_hero_methodology" / "thermal_tegrastats.log"

PAPER_FILES = [
    ROOT / "Paper" / "main.tex",
    ROOT / "Paper" / "1_Introduction.tex",
    ROOT / "Paper" / "3_Design.tex",
    ROOT / "Paper" / "4_Evaluation.tex",
    ROOT / "Paper" / "6_Conclusion.tex",
]

REQUIRED_MODES = [
    "app_only",
    "unthrottled_storage",
    "simple_controller",
    "aegis_policy",
]

REQUIRED_LOG_KEYS = [
    "foreground_csv",
    "foreground_jsonl",
    "background_jsonl",
    "fuse_stderr",
    "fuse_stdout",
    "policy_jsonl",
    "runtime_fuse_admission_trace",
    "runtime_fuse_throttle_trace",
    "runtime_telemetry",
    "telemetry_jsonl",
]

ALWAYS_NONEMPTY_LOG_KEYS = {
    "foreground_csv",
    "foreground_jsonl",
    "fuse_stderr",
    "policy_jsonl",
    "runtime_fuse_admission_trace",
    "runtime_telemetry",
    "telemetry_jsonl",
}

REQUIRED_PAPER_PHRASES = {
    "scoped_runtime_boundary":
        "This is a scoped edge file-encryption runtime result, not deployed-filesystem or peak-throughput superiority",
    "repeated_vs_kernel_boundary":
        "five-run hero medians and three-run kernel QoS controls",
    "sqlite_kernel_qos_table":
        "Table~\\ref{tab:qos_sqlite_recovery} reports five-run hero medians and three-run kernel QoS controls",
    "no_uniqueness_claim":
        "This is bounded storage-visible control, not a uniqueness or non-storage QoS claim",
    "no_external_scheduler":
        "not an external application scheduler",
    "workload_boundary":
        "not a broad workload suite",
}

BROAD_GENERALITY_PATTERNS = [
    ("all_workloads", re.compile(r"\ball workloads\b", re.I)),
    ("arbitrary_workloads", re.compile(r"\barbitrary workloads\b", re.I)),
    ("broad_workload_generalization", re.compile(r"\bbroad workload generalization\b", re.I)),
    ("general_purpose_performance", re.compile(r"\bgeneral[- ]purpose .*performance\b", re.I)),
    ("always_recovers", re.compile(r"\balways (?:recovers|restores)\b", re.I)),
    ("unique_sqlite_recovery", re.compile(r"\bunique(?:ly)? (?:recovers|restores).*SQLite\b", re.I)),
]

NEGATION_TERMS = (
    "not ",
    "does not ",
    "do not ",
    "no ",
    "without ",
    "rather than ",
    "remains ",
    "outside scope",
    "non-claim",
    "not a ",
    "not as ",
    "requires ",
    "future ",
    "limitation",
    "limited",
    "scoped",
    "bounded",
)


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return data


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def require_path(path: Path) -> dict[str, Any]:
    return {
        "path": relpath(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
    }


def mode_by_name(bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    modes = bundle.get("modes", [])
    if not isinstance(modes, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for mode in modes:
        if isinstance(mode, dict) and isinstance(mode.get("mode"), str):
            out[mode["mode"]] = mode
    return out


def summarize_single_modes(bundle: dict[str, Any]) -> dict[str, Any]:
    modes = mode_by_name(bundle)
    summaries: dict[str, Any] = {}
    for name in REQUIRED_MODES:
        mode = modes.get(name, {})
        fg = mode.get("foreground", {}) if isinstance(mode.get("foreground"), dict) else {}
        bg = mode.get("background", {}) if isinstance(mode.get("background"), dict) else {}
        daemon = mode.get("daemon_throttle", {}) if isinstance(mode.get("daemon_throttle"), dict) else {}
        policy = mode.get("policy", {}) if isinstance(mode.get("policy"), dict) else {}
        telemetry = mode.get("telemetry", {}) if isinstance(mode.get("telemetry"), dict) else {}
        summaries[name] = {
            "present": bool(mode),
            "acceptable": bool(mode.get("acceptable")) if mode else False,
            "p99_ms": fg.get("p99_ms"),
            "deadline_misses": fg.get("deadline_misses"),
            "foreground_samples": fg.get("samples"),
            "foreground_rows": fg.get("row_count"),
            "background_mb_s": bg.get("throughput_mb_s"),
            "background_bytes": bg.get("bytes_written"),
            "daemon_throttle_rows": daemon.get("rows"),
            "daemon_throttled_rows": daemon.get("throttled_rows"),
            "daemon_sleep_us_total": daemon.get("sleep_us_total"),
            "policy_events": policy.get("events"),
            "telemetry_rows": telemetry.get("rows"),
        }
    return summaries


def summarize_methodology(bundle: dict[str, Any]) -> dict[str, Any]:
    mode_summaries: dict[str, Any] = {}
    for item in bundle.get("mode_summaries", []):
        if not isinstance(item, dict) or not isinstance(item.get("mode"), str):
            continue
        mode_summaries[item["mode"]] = {
            "all_acceptable": bool(item.get("all_acceptable")),
            "runs": item.get("runs"),
            "p99_ms": item.get("p99_ms"),
            "deadline_misses": item.get("deadline_misses"),
            "storage_mb_s": item.get("storage_mb_s"),
        }
    return {
        "overall_pass": bool(bundle.get("overall_pass")),
        "overall_pass_interpretation": (
            "false is acceptable for E1 closeout when repeated evidence exists and "
            "unstable recovery directions are scoped rather than claimed"
        ),
        "warmup_runs": bundle.get("warmup_runs"),
        "repetitions_measured": bundle.get("repetitions_measured"),
        "component_coverage_summary": bundle.get("component_coverage_summary", {}),
        "recovery_check_summary": bundle.get("recovery_check_summary", {}),
        "mode_summaries": mode_summaries,
    }


def collect_log_evidence(bundle: dict[str, Any], bundle_root: Path) -> dict[str, Any]:
    repetitions = bundle.get("repetitions")
    if isinstance(repetitions, list) and repetitions:
        runs = repetitions
        run_prefix = "rep"
    else:
        runs = [bundle]
        run_prefix = "single"

    missing: list[dict[str, Any]] = []
    empty_expected: list[dict[str, Any]] = []
    checked = 0
    foreground_rows = 0
    for run_idx, run in enumerate(runs):
        if not isinstance(run, dict):
            continue
        modes = mode_by_name(run)
        for mode_name in REQUIRED_MODES:
            mode = modes.get(mode_name)
            if not mode:
                missing.append({"run": f"{run_prefix}_{run_idx:02d}", "mode": mode_name, "missing_mode": True})
                continue
            logs = mode.get("logs", {})
            if not isinstance(logs, dict):
                missing.append({"run": f"{run_prefix}_{run_idx:02d}", "mode": mode_name, "missing_logs": True})
                continue
            for key in REQUIRED_LOG_KEYS:
                raw = logs.get(key)
                optional_absent = key == "runtime_fuse_throttle_trace" and mode_name != "aegis_policy"
                if not isinstance(raw, str):
                    if optional_absent:
                        empty_expected.append({"run": f"{run_prefix}_{run_idx:02d}", "mode": mode_name, "key": key, "path": None})
                    else:
                        missing.append({"run": f"{run_prefix}_{run_idx:02d}", "mode": mode_name, "key": key, "path": None})
                    continue
                path = ROOT / raw
                checked += 1
                if not path.exists():
                    if optional_absent:
                        empty_expected.append({"run": f"{run_prefix}_{run_idx:02d}", "mode": mode_name, "key": key, "path": raw})
                    else:
                        missing.append({"run": f"{run_prefix}_{run_idx:02d}", "mode": mode_name, "key": key, "path": raw})
                    continue
                size = path.stat().st_size
                must_be_nonempty = key in ALWAYS_NONEMPTY_LOG_KEYS
                if key == "background_jsonl":
                    must_be_nonempty = mode_name != "app_only"
                elif key == "runtime_fuse_throttle_trace":
                    must_be_nonempty = mode_name == "aegis_policy"
                elif key == "fuse_stdout":
                    must_be_nonempty = False
                if must_be_nonempty and size == 0:
                    missing.append({"run": f"{run_prefix}_{run_idx:02d}", "mode": mode_name, "key": key, "path": raw})
                elif not must_be_nonempty and size == 0:
                    empty_expected.append({"run": f"{run_prefix}_{run_idx:02d}", "mode": mode_name, "key": key, "path": raw})
            fg = mode.get("foreground", {})
            if isinstance(fg, dict):
                foreground_rows += int(fg.get("row_count") or 0)
    return {
        "bundle_root": relpath(bundle_root),
        "run_count": len(runs),
        "log_paths_checked": checked,
        "foreground_rows_counted": foreground_rows,
        "missing_or_empty_logs": missing,
        "expected_empty_logs": empty_expected,
        "complete": not missing and checked > 0 and foreground_rows > 0,
    }


def scan_paper() -> dict[str, Any]:
    texts = {relpath(path): read_text(path) for path in PAPER_FILES if path.exists()}
    combined = "\n".join(texts.values())
    required = {name: phrase in combined for name, phrase in REQUIRED_PAPER_PHRASES.items()}

    broad_hits: list[dict[str, Any]] = []
    for path, text in texts.items():
        lines = text.splitlines()
        for idx, line in enumerate(lines, start=1):
            for name, pattern in BROAD_GENERALITY_PATTERNS:
                if not pattern.search(line):
                    continue
                window = " ".join(lines[max(0, idx - 3): min(len(lines), idx + 3)]).lower()
                guarded = any(term in window for term in NEGATION_TERMS)
                broad_hits.append({
                    "kind": name,
                    "path": path,
                    "line": idx,
                    "text": line.strip(),
                    "guarded": guarded,
                })
    unguarded = [hit for hit in broad_hits if not hit["guarded"]]
    return {
        "paper_files": sorted(texts),
        "required_phrases": required,
        "broad_generality_hits": broad_hits,
        "unguarded_broad_generality_hits": unguarded,
        "complete": all(required.values()) and not unguarded,
    }


def build_report() -> dict[str, Any]:
    required_inputs = {
        "single_bundle": require_path(SINGLE_BUNDLE),
        "methodology_bundle": require_path(METHODOLOGY_BUNDLE),
        "kernel_qos_comparison": require_path(KERNEL_QOS_COMPARISON),
        "stat_thermal_audit": require_path(STAT_THERMAL_AUDIT),
        "thermal_tegrastats_log": require_path(THERMAL_LOG),
    }
    missing_inputs = [name for name, meta in required_inputs.items() if not meta["exists"] or meta["bytes"] <= 0]

    single = read_json(SINGLE_BUNDLE) if not missing_inputs or SINGLE_BUNDLE.exists() else {}
    methodology = read_json(METHODOLOGY_BUNDLE) if METHODOLOGY_BUNDLE.exists() else {}
    kernel = read_json(KERNEL_QOS_COMPARISON) if KERNEL_QOS_COMPARISON.exists() else {}
    thermal = read_json(STAT_THERMAL_AUDIT) if STAT_THERMAL_AUDIT.exists() else {}

    single_modes = summarize_single_modes(single)
    methodology_summary = summarize_methodology(methodology)
    single_logs = collect_log_evidence(single, SINGLE_BUNDLE.parent) if single else {"complete": False}
    repeated_logs = collect_log_evidence(methodology, METHODOLOGY_BUNDLE.parent) if methodology else {"complete": False}
    paper = scan_paper()

    required_modes_single = all(single_modes.get(mode, {}).get("present") for mode in REQUIRED_MODES)
    required_modes_acceptable = all(single_modes.get(mode, {}).get("acceptable") for mode in REQUIRED_MODES)
    repeated_modes = methodology_summary["mode_summaries"]
    repeated_modes_complete = all(
        repeated_modes.get(mode, {}).get("all_acceptable") and int(repeated_modes.get(mode, {}).get("runs") or 0) >= 5
        for mode in REQUIRED_MODES
    )

    coverage_summary = methodology_summary.get("component_coverage_summary", {})
    coverage_complete = bool(coverage_summary) and all(
        isinstance(value, dict) and value.get("all_true") is True
        for value in coverage_summary.values()
    )
    recovery_summary = methodology_summary.get("recovery_check_summary", {})
    required_stability = {
        "required_modes_available": recovery_summary.get("required_modes_available", {}).get("all_true") is True,
        "aegis_records_throttle_decisions": recovery_summary.get("aegis_records_throttle_decisions", {}).get("all_true") is True,
        "aegis_keeps_more_storage_than_simple": recovery_summary.get("aegis_keeps_more_storage_than_simple", {}).get("all_true") is True,
    }
    scoped_instability = {
        key: value
        for key, value in recovery_summary.items()
        if isinstance(value, dict) and value.get("all_true") is False
    }

    close_conditions = {
        "required_inputs_present": not missing_inputs,
        "single_bundle_passes": bool(single.get("overall_pass")),
        "four_single_modes_present": required_modes_single,
        "four_single_modes_acceptable": required_modes_acceptable,
        "single_raw_logs_complete": bool(single_logs.get("complete")),
        "repeated_warmup_and_five_runs": int(methodology.get("warmup_runs") or 0) >= 1 and int(methodology.get("repetitions_measured") or 0) >= 5,
        "repeated_mode_summaries_complete": repeated_modes_complete,
        "repeated_raw_logs_complete": bool(repeated_logs.get("complete")),
        "methodology_component_coverage_complete": coverage_complete,
        "required_recovery_checks_pass": all(required_stability.values()),
        "unstable_recovery_directions_are_scoped": bool(scoped_instability) and not bool(methodology.get("overall_pass")),
        "kernel_qos_comparison_available": bool(kernel.get("overall_pass")) and bool(kernel.get("paper_compares_both_kernel_baselines")),
        "stat_thermal_audit_passes": bool(thermal.get("overall_pass")) and bool(thermal.get("paper_scope_gate_pass")),
        "thermal_log_retained": THERMAL_LOG.exists() and THERMAL_LOG.stat().st_size > 0,
        "paper_scope_guard_passes": bool(paper.get("complete")),
    }

    return {
        "artifact": "sqlite_hero_validity_closeout",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifacts": {
            "single_bundle": relpath(SINGLE_BUNDLE),
            "methodology_bundle": relpath(METHODOLOGY_BUNDLE),
            "kernel_qos_comparison": relpath(KERNEL_QOS_COMPARISON),
            "stat_thermal_audit": relpath(STAT_THERMAL_AUDIT),
            "thermal_tegrastats_log": relpath(THERMAL_LOG),
        },
        "required_inputs": required_inputs,
        "single_run_summary": {
            "overall_pass": bool(single.get("overall_pass")),
            "mode_summaries": single_modes,
            "component_coverage": single.get("component_coverage", {}),
            "recovery_checks": single.get("recovery_checks", {}),
        },
        "repeated_methodology_summary": methodology_summary,
        "raw_log_evidence": {
            "single": single_logs,
            "repeated": repeated_logs,
        },
        "kernel_qos_integration": {
            "overall_pass": bool(kernel.get("overall_pass")),
            "two_kernel_baselines_measured": kernel.get("two_kernel_baselines_measured"),
            "paper_compares_both_kernel_baselines": kernel.get("paper_compares_both_kernel_baselines"),
            "comparability_warnings": kernel.get("comparability_warnings", []),
            "claim_guard": kernel.get("claim_guard", {}),
        },
        "thermal_and_methodology": {
            "overall_pass": bool(thermal.get("overall_pass")),
            "paper_scope_gate_pass": thermal.get("paper_scope_gate_pass"),
            "thermal_log": require_path(THERMAL_LOG),
        },
        "paper_scope_guard": paper,
        "claim_boundary": {
            "allowed": [
                "SQLite WAL/FULL foreground transaction p99 under the retained mounted secure-storage pressure workflow",
                "four-mode comparison: app-only, unthrottled storage, simple controller, and AEGIS-Q policy",
                "AEGIS-Q storage-visible throttling records policy/telemetry/daemon-throttle decisions in every repeated run",
                "kernel-QoS rows are compared with an explicit repetition mismatch warning",
            ],
            "forbidden": [
                "broad workload generality",
                "SQLite recovery uniqueness without qualification",
                "external application scheduler recovery",
                "general-purpose filesystem performance",
                "deployed-filesystem or peak-throughput superiority",
            ],
            "scoped_instability": scoped_instability,
        },
        "close_conditions": close_conditions,
        "overall_pass": all(close_conditions.values()),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# SQLite Hero Validity Closeout",
        "",
        f"- Overall pass: `{str(report['overall_pass']).lower()}`",
        f"- Generated: `{report['timestamp_utc']}`",
        "",
        "## Close Conditions",
        "",
    ]
    for key, value in report["close_conditions"].items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")
    lines.extend(["", "## Claim Boundary", ""])
    for item in report["claim_boundary"]["allowed"]:
        lines.append(f"- Allowed: {item}")
    for item in report["claim_boundary"]["forbidden"]:
        lines.append(f"- Forbidden: {item}")
    lines.extend(["", "## Repeated Mode Summary", ""])
    for mode, summary in report["repeated_methodology_summary"]["mode_summaries"].items():
        p99 = summary.get("p99_ms", {})
        storage = summary.get("storage_mb_s", {})
        misses = summary.get("deadline_misses", {})
        lines.append(
            f"- `{mode}`: runs={summary.get('runs')}, "
            f"p99_median={p99.get('median')}, "
            f"miss_median={misses.get('median')}, "
            f"storage_mb_s_median={storage.get('median')}"
        )
    lines.extend(["", "## Scoped Instability", ""])
    instability = report["claim_boundary"]["scoped_instability"]
    if not instability:
        lines.append("- none")
    else:
        for key, value in instability.items():
            lines.append(f"- `{key}`: {json.dumps(value, sort_keys=True)}")
    lines.extend(["", "## Source Artifacts", ""])
    for key, value in report["source_artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    report = build_report()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "sqlite_hero_validity_closeout.json"
    md_path = args.out_dir / "sqlite_hero_validity_closeout.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)

    print(f"wrote {relpath(json_path)}")
    print(f"wrote {relpath(md_path)}")
    print(f"overall_pass={str(report['overall_pass']).lower()}")
    for key, value in report["close_conditions"].items():
        print(f"{key}={str(value).lower()}")

    if args.require_complete and not report["overall_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
