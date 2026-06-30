#!/usr/bin/env python3
"""Build the E5 energy/thermal metadata closeout.

This closeout does not rerun benchmarks. It verifies that current paper-facing
performance numbers are either linked to retained Jetson/platform-state
metadata, explicitly scoped as diagnostic/progress evidence, or marked
environment-unavailable.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "energy_thermal_metadata_closeout"

JETSON_CONTRACT = ROOT / "artifacts" / "validation" / "jetson_power_thermal_contract" / "jetson_power_thermal_contract.json"
STAT_THERMAL_AUDIT = ROOT / "artifacts" / "validation" / "stat_thermal_methodology" / "stat_thermal_methodology_audit.json"
SQLITE_CLOSEOUT = ROOT / "artifacts" / "validation" / "sqlite_hero_validity_closeout" / "sqlite_hero_validity_closeout.json"
KERNEL_QOS_CLOSEOUT = ROOT / "artifacts" / "validation" / "kernel_qos_hero_integration_closeout" / "kernel_qos_hero_integration_closeout.json"
SECOND_MACRO_CLOSEOUT = ROOT / "artifacts" / "validation" / "second_macrobenchmark_closeout" / "second_macrobenchmark_closeout.json"

FROZEN_ROWS = {
    "aegisq": ROOT / "artifacts" / "validation" / "frozen_aegisq_contract" / "frozen_aegisq_contract.json",
    "plaintext": ROOT / "artifacts" / "validation" / "frozen_plaintext_contract" / "frozen_plaintext_contract.json",
    "gocryptfs": ROOT / "artifacts" / "validation" / "frozen_gocryptfs_contract" / "frozen_gocryptfs_contract.json",
    "dmcrypt": ROOT / "artifacts" / "validation" / "frozen_dmcrypt_contract" / "frozen_dmcrypt_contract.json",
}

PAPER_FILES = [
    ROOT / "Paper" / "main.tex",
    ROOT / "Paper" / "1_Introduction.tex",
    ROOT / "Paper" / "3_Design.tex",
    ROOT / "Paper" / "4_Evaluation.tex",
    ROOT / "Paper" / "6_Conclusion.tex",
]

REQUIRED_PAPER_PHRASES = {
    "platform_identified": "The platform was an NVIDIA Jetson AGX Thor Developer Kit",
    "future_headline_metadata": "future headline comparisons require fixed clock/power/thermal capture",
    "frozen_cost_boundary": "This is the cost boundary for authenticated publication, not the headline win",
    "diagnostic_placement_scope": "Figure~\\ref{fig:baseline_comparison} is a diagnostic three-run placement result, not a confidence claim",
    "sqlite_scope": "This is bounded storage-visible control, not a uniqueness or AI-inference QoS claim",
    "second_macro_scope": "This is not a broad workload suite or an AI-inference QoS recovery claim",
}

FORBIDDEN_PATTERNS = [
    ("energy_efficiency", re.compile(r"\benergy[- ]efficien(?:t|cy)\b", re.I)),
    ("power_efficiency", re.compile(r"\bpower[- ]efficien(?:t|cy)\b", re.I)),
    ("thermally_stable", re.compile(r"\bthermally stable\b|\bstable thermal\b", re.I)),
    ("no_throttling_claim", re.compile(r"\bno throttling\b|\bthrottling[- ]free\b", re.I)),
    ("fixed_clock_claim", re.compile(r"\bfixed clocks?\b|\blocked clocks?\b", re.I)),
]

NEGATION_TERMS = (
    "not ",
    "no ",
    "without ",
    "unless ",
    "future ",
    "require",
    "requires",
    "unavailable",
    "missing",
    "invalid",
    "diagnostic",
    "scope",
    "scoped",
    "non-claim",
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


def path_meta(path: Path) -> dict[str, Any]:
    return {
        "path": relpath(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
    }


def jetson_contract() -> dict[str, Any]:
    meta = path_meta(JETSON_CONTRACT)
    if not JETSON_CONTRACT.exists():
        return {"complete": False, "contract": meta, "reason": "missing Jetson power/thermal contract"}
    data = read_json(JETSON_CONTRACT)
    tegra = data.get("tegrastats", {}) if isinstance(data.get("tegrastats"), dict) else {}
    nvp = data.get("nvpmodel", {}) if isinstance(data.get("nvpmodel"), dict) else {}
    required = data.get("required_sections_present", {}) if isinstance(data.get("required_sections_present"), dict) else {}
    headline = data.get("headline_run_eligibility", {}) if isinstance(data.get("headline_run_eligibility"), dict) else {}
    artifact_verdict = data.get("artifact_verdict", {})
    if not isinstance(artifact_verdict, dict):
        artifact_verdict = {"overall_pass": bool(artifact_verdict)}
    eligible = headline.get("headline_run_eligible", headline.get("eligible"))
    return {
        "complete": bool(artifact_verdict.get("overall_pass")) and bool(eligible),
        "contract": meta,
        "artifact_verdict": bool(artifact_verdict.get("overall_pass")),
        "headline_run_eligible": bool(eligible),
        "invalid_reasons": headline.get("invalid_reasons", []),
        "tegrastats_available": bool(tegra.get("available")),
        "nvpmodel_available": bool(nvp.get("available")),
        "required_sections_present": required,
    }


def thermal_log_ok(path: Path | None) -> bool:
    return bool(path and path.exists() and path.stat().st_size > 0)


def summarize_frozen_row(name: str, path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "name": name,
            "path": relpath(path),
            "status": "missing",
            "complete": False,
            "reason": "missing frozen-row artifact",
        }
    data = read_json(path)
    artifacts = data.get("artifacts", {}) if isinstance(data.get("artifacts"), dict) else {}
    thermal_path = ROOT / artifacts["thermal_log"] if isinstance(artifacts.get("thermal_log"), str) else None
    platform_path = ROOT / artifacts["platform_manifest"] if isinstance(artifacts.get("platform_manifest"), str) else None
    verdict = data.get("verdict")
    blocked = verdict == "environment-blocked" or bool(data.get("blocking_reasons"))
    measured = bool(data.get("overall_pass")) and bool(data.get("contract_compliant_warm_cache"))
    thermal = thermal_log_ok(thermal_path)
    platform = bool(platform_path and platform_path.exists() and platform_path.stat().st_size > 0)
    warm = data.get("warm_cache_summary", {}) if isinstance(data.get("warm_cache_summary"), dict) else {}
    valid_reps = warm.get("valid_repetitions", 0)
    complete = (measured and thermal and platform and valid_reps >= 5) or blocked
    return {
        "name": name,
        "path": relpath(path),
        "status": "environment-blocked" if blocked else "measured" if measured else "incomplete",
        "complete": complete,
        "overall_pass": bool(data.get("overall_pass")),
        "contract_compliant_warm_cache": bool(data.get("contract_compliant_warm_cache")),
        "valid_repetitions": valid_reps,
        "thermal_log": path_meta(thermal_path) if thermal_path else {"exists": False},
        "platform_manifest": path_meta(platform_path) if platform_path else {"exists": False},
        "blocking_reasons": data.get("blocking_reasons", []),
        "metrics": (warm.get("metrics") or {}) if isinstance(warm.get("metrics"), dict) else {},
    }


def stat_thermal_artifacts() -> dict[str, Any]:
    meta = path_meta(STAT_THERMAL_AUDIT)
    if not STAT_THERMAL_AUDIT.exists():
        return {"complete": False, "artifact": meta, "reason": "missing stat/thermal methodology audit"}
    data = read_json(STAT_THERMAL_AUDIT)
    artifacts = data.get("artifacts", [])
    by_name = {item.get("name"): item for item in artifacts if isinstance(item, dict)}
    required = {
        "verified_microbench": "diagnostic_scope_gate_passed",
        "keyplane_rekey_workflow": "single_workflow_scope_gate_passed",
        "keyplane_rekey_methodology": "methodology_progress_host_not_ready",
        "qos_sqlite_hero_methodology": "methodology_progress_recovery_unstable",
        "frozen_aegisq_contract": "methodology_metadata_retained_cold_invalid",
        "frozen_gocryptfs_contract": "baseline_metadata_retained_cold_invalid",
        "frozen_plaintext_contract": "baseline_metadata_retained_cold_invalid",
    }
    checks: dict[str, Any] = {}
    for name, expected_status in required.items():
        item = by_name.get(name, {})
        checks[name] = {
            "present": bool(item.get("exists")),
            "status": item.get("methodology_status"),
            "expected_status": expected_status,
            "missing_metadata": item.get("missing_metadata", []),
            "paper_role": item.get("paper_role"),
            "complete": bool(item.get("exists")) and item.get("methodology_status") == expected_status,
        }
    return {
        "complete": bool(data.get("overall_pass")) and bool(data.get("paper_scope_gate_pass")) and all(c["complete"] for c in checks.values()),
        "artifact": meta,
        "overall_pass": bool(data.get("overall_pass")),
        "paper_scope_gate_pass": bool(data.get("paper_scope_gate_pass")),
        "checks": checks,
    }


def sqlite_closeout() -> dict[str, Any]:
    meta = path_meta(SQLITE_CLOSEOUT)
    if not SQLITE_CLOSEOUT.exists():
        return {"complete": False, "artifact": meta, "reason": "missing SQLite closeout"}
    data = read_json(SQLITE_CLOSEOUT)
    thermal = data.get("thermal_and_methodology", {}) if isinstance(data.get("thermal_and_methodology"), dict) else {}
    conditions = data.get("close_conditions", {}) if isinstance(data.get("close_conditions"), dict) else {}
    return {
        "complete": bool(data.get("overall_pass")) and bool(conditions.get("stat_thermal_audit_passes")) and bool(conditions.get("thermal_log_retained")),
        "artifact": meta,
        "overall_pass": bool(data.get("overall_pass")),
        "thermal_and_methodology": thermal,
        "conditions": {
            "stat_thermal_audit_passes": conditions.get("stat_thermal_audit_passes"),
            "thermal_log_retained": conditions.get("thermal_log_retained"),
            "repeated_warmup_and_five_runs": conditions.get("repeated_warmup_and_five_runs"),
            "paper_scope_guard_passes": conditions.get("paper_scope_guard_passes"),
        },
    }


def kernel_qos_closeout() -> dict[str, Any]:
    meta = path_meta(KERNEL_QOS_CLOSEOUT)
    if not KERNEL_QOS_CLOSEOUT.exists():
        return {"complete": False, "artifact": meta, "reason": "missing kernel QoS closeout"}
    data = read_json(KERNEL_QOS_CLOSEOUT)
    conditions = data.get("close_conditions", {}) if isinstance(data.get("close_conditions"), dict) else {}
    return {
        "complete": bool(data.get("overall_pass")) and bool(conditions.get("e1_closeout_passes")),
        "artifact": meta,
        "overall_pass": bool(data.get("overall_pass")),
        "conditions": {
            "e1_closeout_passes": conditions.get("e1_closeout_passes"),
            "kernel_rows_are_measured": conditions.get("kernel_rows_are_measured"),
            "paper_claim_guard_passes": conditions.get("paper_claim_guard_passes"),
        },
    }


def second_macro_closeout() -> dict[str, Any]:
    meta = path_meta(SECOND_MACRO_CLOSEOUT)
    if not SECOND_MACRO_CLOSEOUT.exists():
        return {"complete": False, "artifact": meta, "reason": "missing second macro closeout"}
    data = read_json(SECOND_MACRO_CLOSEOUT)
    conditions = data.get("close_conditions", {}) if isinstance(data.get("close_conditions"), dict) else {}
    thermal = data.get("thermal_metadata", {}) if isinstance(data.get("thermal_metadata"), dict) else {}
    return {
        "complete": bool(data.get("overall_pass")) and bool(conditions.get("thermal_metadata_linked")) and bool(thermal.get("complete")),
        "artifact": meta,
        "overall_pass": bool(data.get("overall_pass")),
        "thermal_metadata": thermal,
        "conditions": {
            "thermal_metadata_linked": conditions.get("thermal_metadata_linked"),
            "raw_latency_throughput_retained": conditions.get("raw_latency_throughput_retained"),
            "process_resource_usage_retained": conditions.get("process_resource_usage_retained"),
            "paper_scope_guard_passes": conditions.get("paper_scope_guard_passes"),
        },
    }


def paper_guard() -> dict[str, Any]:
    texts = {relpath(path): read_text(path) for path in PAPER_FILES if path.exists()}
    combined = "\n".join(texts.values())
    required = {name: phrase in combined for name, phrase in REQUIRED_PAPER_PHRASES.items()}
    hits: list[dict[str, Any]] = []
    unguarded: list[dict[str, Any]] = []
    for rel, text in texts.items():
        lines = text.splitlines()
        for idx, line in enumerate(lines, start=1):
            for name, pattern in FORBIDDEN_PATTERNS:
                if not pattern.search(line):
                    continue
                context = " ".join(lines[max(0, idx - 3): min(len(lines), idx + 3)]).lower()
                hit = {"file": rel, "line": idx, "pattern": name, "text": line.strip()}
                hits.append(hit)
                if not any(term in context for term in NEGATION_TERMS):
                    unguarded.append(hit)
    return {
        "complete": all(required.values()) and not unguarded,
        "paper_files": list(texts),
        "required_phrases": required,
        "forbidden_hits": hits,
        "unguarded_forbidden_hits": unguarded,
    }


def build_closeout() -> dict[str, Any]:
    jetson = jetson_contract()
    frozen = {name: summarize_frozen_row(name, path) for name, path in FROZEN_ROWS.items()}
    stat = stat_thermal_artifacts()
    sqlite = sqlite_closeout()
    kernel = kernel_qos_closeout()
    second = second_macro_closeout()
    paper = paper_guard()

    headline_results = {
        "frozen_filesystem_rows": {
            "complete": all(row["complete"] for row in frozen.values()),
            "rows": frozen,
        },
        "sqlite_qos_headline": sqlite,
        "kernel_qos_comparison_context": kernel,
        "secure_inference_log_second_macro": second,
        "primitive_and_keyplane_diagnostic_context": stat,
        "jetson_platform_contract": jetson,
    }
    close_conditions = {
        "jetson_contract_complete": jetson["complete"],
        "frozen_rows_have_metadata_or_unavailable_status": headline_results["frozen_filesystem_rows"]["complete"],
        "sqlite_hero_has_thermal_methodology": sqlite["complete"],
        "kernel_qos_context_links_sqlite_closeout": kernel["complete"],
        "second_macro_has_linked_platform_metadata": second["complete"],
        "diagnostic_and_keyplane_rows_are_scoped_by_stat_thermal_audit": stat["complete"],
        "paper_guard_passes": paper["complete"],
    }
    return {
        "artifact": "E5 energy/thermal metadata closeout",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": all(close_conditions.values()),
        "close_conditions": close_conditions,
        "headline_results": headline_results,
        "paper_guard": paper,
        "claim_boundary": {
            "allowed": [
                "measured filesystem headline rows carry retained platform and thermal logs",
                "SQLite headline numbers are tied to repeated methodology and retained thermal logs",
                "second macrobenchmark latency/throughput rows link to Jetson platform-state metadata",
                "primitive and key-plane numbers remain diagnostic or progress-scoped when thermal methodology is incomplete",
            ],
            "forbidden": [
                "energy-efficiency or power-efficiency claims",
                "throttling-free or fixed-clock claims without per-run proof",
                "treating diagnostic primitive results as full headline comparisons",
                "treating environment-blocked dm-crypt/fscrypt rows as measured",
            ],
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Energy/Thermal Metadata Closeout",
        "",
        f"- Overall pass: `{str(report['overall_pass']).lower()}`",
        f"- Generated: `{report['timestamp_utc']}`",
        "",
        "## Close Conditions",
        "",
    ]
    for key, value in report["close_conditions"].items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")
    lines.extend(["", "## Frozen Filesystem Rows", ""])
    frozen = report["headline_results"]["frozen_filesystem_rows"]["rows"]
    for name, row in frozen.items():
        lines.append(
            f"- `{name}`: status=`{row['status']}`, complete=`{str(row['complete']).lower()}`, "
            f"thermal=`{row.get('thermal_log', {}).get('path', 'n/a')}`"
        )
    lines.extend(["", "## Headline Context", ""])
    lines.append(f"- SQLite thermal/methodology complete: `{str(report['headline_results']['sqlite_qos_headline']['complete']).lower()}`")
    lines.append(f"- Kernel QoS context complete: `{str(report['headline_results']['kernel_qos_comparison_context']['complete']).lower()}`")
    lines.append(f"- Second macro thermal metadata complete: `{str(report['headline_results']['secure_inference_log_second_macro']['complete']).lower()}`")
    lines.append(f"- Diagnostic/key-plane stat-thermal audit complete: `{str(report['headline_results']['primitive_and_keyplane_diagnostic_context']['complete']).lower()}`")
    lines.extend(["", "## Paper Guard", ""])
    lines.append(f"- Required phrases complete: `{str(all(report['paper_guard']['required_phrases'].values())).lower()}`")
    lines.append(f"- Unguarded forbidden hits: `{len(report['paper_guard']['unguarded_forbidden_hits'])}`")
    lines.extend(["", "## Claim Boundary", ""])
    for item in report["claim_boundary"]["allowed"]:
        lines.append(f"- Allowed: {item}")
    for item in report["claim_boundary"]["forbidden"]:
        lines.append(f"- Forbidden: {item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_closeout()
    json_path = out_dir / "energy_thermal_metadata_closeout.json"
    md_path = out_dir / "energy_thermal_metadata_closeout.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({
        "overall_pass": report["overall_pass"],
        "json": relpath(json_path),
        "markdown": relpath(md_path),
        "close_conditions": report["close_conditions"],
        "unguarded_forbidden_hits": len(report["paper_guard"]["unguarded_forbidden_hits"]),
    }, indent=2, sort_keys=True))
    if args.require_complete and not report["overall_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
