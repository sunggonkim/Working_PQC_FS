#!/usr/bin/env python3
"""Build the FINAL-S2 closeout for remaining final resubmission blockers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "reports" / "final_s2_closeout"
JSON_OUT = OUT / "final_s2_closeout.json"
MD_OUT = OUT / "final_s2_closeout.md"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: str) -> dict[str, Any]:
    full = ROOT / path
    try:
        data = json.loads(full.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"_path": path, "_present": full.exists(), "_error": str(exc)}
    if not isinstance(data, dict):
        return {"_path": path, "_present": True, "_error": "json root is not object"}
    data["_path"] = path
    data["_present"] = True
    return data


def text(path: str) -> str:
    try:
        return (ROOT / path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def artifact_pass(data: dict[str, Any], key: str = "overall_pass") -> bool:
    return data.get("_present") is True and data.get(key) is True


def build_report() -> dict[str, Any]:
    source = load_json("artifacts/validation/refactor_inventory/source_ownership_map.json")
    phase1 = load_json("artifacts/validation/refactor_inventory/phase1_behavior_equivalence.json")
    concurrency = load_json("artifacts/validation/concurrency_contract/concurrency_contract.json")
    publication = load_json("artifacts/validation/publication_protocol_fault_matrix/publication_protocol_closeout.json")
    parallel = load_json("artifacts/validation/parallel_commit_contract/parallel_commit_closure_audit.json")
    a1 = load_json("artifacts/validation/a1_throughput_decision/a1_throughput_decision.json")
    a3 = load_json("artifacts/validation/a3_batching_decision/a3_batching_decision.json")
    a4 = load_json("artifacts/validation/a4_hidden_overhead_accounting/a4_hidden_overhead_closeout.json")
    a5 = load_json("artifacts/validation/vfs_ebpf_fdatasync_storm/a5_fdatasync_storm_closeout.json")
    tpm = load_json("artifacts/validation/async_merkle_tpm_epoch/tpm_epoch_freshness_probe.json")
    dangerous = load_json("artifacts/reports/dangerous_claim_lint/dangerous_claim_lint.json")
    firewall = load_json("artifacts/reports/architecture_claim_firewall/architecture_claim_firewall.json")
    manifest = load_json("artifacts/reports/final_claim_evidence_manifest/final_claim_evidence_manifest.json")

    source_summary = source.get("summary", {})
    phase1_summary = phase1.get("summary", {})
    paper_text = "\n".join(text(path) for path in [
        "Paper/1_Introduction.tex",
        "Paper/3_Design.tex",
        "Paper/4_Evaluation.tex",
        "Paper/10_Discussion_and_Limitations.tex",
        "Paper/6_Conclusion.tex",
    ])
    lower_paper = paper_text.lower()

    checks = {
        "source_ownership_no_blockers":
            source_summary.get("ownership_blocker_count") == 0
            and source_summary.get("all_current_production_functions_mapped") is True
            and source_summary.get("all_current_global_state_mapped") is True
            and source_summary.get("missing_production_source_files_count") == 0
            and source_summary.get("listed_but_missing_source_files_count") == 0,
        "mechanical_decomposition_behavior_evidence":
            phase1_summary.get("gate014_code_artifact_evidence_pass") is True
            and phase1_summary.get("ownership_blocker_count") == 0
            and phase1_summary.get("ambiguous_function_count") == 0
            and phase1_summary.get("ambiguous_global_state_count") == 0,
        "publication_protocol_closed":
            publication.get("closeout_complete") is True
            and publication.get("parent_checklist_closed") is True,
        "parallel_commit_closed":
            parallel.get("overall_pass") is True
            and parallel.get("closure_verdict") == "closed",
        "fine_grained_lock_contract_closed":
            artifact_pass(concurrency)
            and not concurrency.get("blocking_items")
            and int(concurrency.get("observed_hot_lock_count", 0) or 0) >= 1,
        "throughput_cost_boundary_closed":
            artifact_pass(a1)
            and a1.get("decision", {}).get("verdict") == "cost-boundary-closeout"
            and "low frozen-contract throughput row is a known cost boundary" in lower_paper,
        "batching_boundary_closed":
            artifact_pass(a3)
            and a3.get("derived", {}).get("grouped_sync_reduced_current") is True
            and a3.get("derived", {}).get("grouped_has_win_and_loss") is True,
        "hidden_overhead_closed": artifact_pass(a4),
        "fdatasync_storm_closed": artifact_pass(a5),
        "async_merkle_tpm_nonclaim_guarded":
            artifact_pass(tpm)
            and tpm.get("hardware_epoch_committed") is False
            and tpm.get("environment_blocked") is True
            and artifact_pass(dangerous)
            and artifact_pass(firewall)
            and "persistent pcr-bound freshness" in lower_paper
            and "does not claim" in lower_paper,
        "final_manifest_still_passes": artifact_pass(manifest),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    gate0_closed = all(checks[name] for name in [
        "source_ownership_no_blockers",
        "mechanical_decomposition_behavior_evidence",
        "publication_protocol_closed",
        "parallel_commit_closed",
        "fine_grained_lock_contract_closed",
    ])
    final_s2_closed = gate0_closed and not blockers

    return {
        "schema_version": 1,
        "generated_utc": now_utc(),
        "scope": "FINAL-S2 aggregate closeout for the remaining final blockers.",
        "checks": checks,
        "blockers": blockers,
        "gate0_closed": gate0_closed,
        "final_s2_closed": final_s2_closed,
        "evidence": {
            "source_ownership": source.get("_path"),
            "phase1_behavior_equivalence": phase1.get("_path"),
            "concurrency_contract": concurrency.get("_path"),
            "publication_closeout": publication.get("_path"),
            "parallel_commit_closure": parallel.get("_path"),
            "a1_throughput_decision": a1.get("_path"),
            "a3_batching_decision": a3.get("_path"),
            "a4_hidden_overhead": a4.get("_path"),
            "a5_fdatasync_storm": a5.get("_path"),
            "tpm_epoch_probe": tpm.get("_path"),
            "dangerous_claim_lint": dangerous.get("_path"),
            "architecture_claim_firewall": firewall.get("_path"),
            "final_claim_manifest": manifest.get("_path"),
        },
        "closed_checklist_candidates": [
            "Gate 0.4 Source Ownership and Module Decomposition",
            "Gate 0.14 Mechanical Decomposition Gate",
            "Gate 0.15 Fine-Grained Concurrency and Lock Contract",
            "Gate A1 AEGIS-Q Frozen-Contract Throughput",
            "Gate A3 Batching or Redesign Evidence",
            "Gate A4 Hidden Overhead Accounting",
            "Gate A5 VFS/eBPF Passthrough and fdatasync Storm",
            "Final: Gate 0 is closed before any SOSP/OSDI readiness claim",
            "Final: AEGIS-Q throughput bottleneck breakdown exists and is reflected in the paper",
            "Final: Fine-grained lock contract exists with measured lock hold-time and deadlock/livelock stress evidence",
            "Final: Async Merkle + TPM epoch freshness exists before rollback-resistance claims",
        ] if final_s2_closed else [],
        "non_claim_boundary": (
            "C6 production async Merkle + persistent PCR-bound TPM epoch freshness "
            "is not implemented in this environment; the final condition is closed "
            "only as a guarded non-claim because the paper forbids rollback-resistance "
            "and persistent PCR-bound freshness wording."
        ),
    }


def write_outputs(report: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# FINAL-S2 Closeout",
        "",
        f"- Generated: `{report['generated_utc']}`",
        f"- Gate 0 closed: `{str(report['gate0_closed']).lower()}`",
        f"- FINAL-S2 closed: `{str(report['final_s2_closed']).lower()}`",
        "",
        "## Checks",
        "",
    ]
    for name, passed in report["checks"].items():
        lines.append(f"- `{name}`: `{str(passed).lower()}`")
    if report["blockers"]:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- `{item}`" for item in report["blockers"])
    lines.extend(["", "## Evidence", ""])
    for name, path in report["evidence"].items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Non-Claim Boundary", "", report["non_claim_boundary"], ""])
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    report = build_report()
    write_outputs(report)
    print(json.dumps({
        "final_s2_closed": report["final_s2_closed"],
        "gate0_closed": report["gate0_closed"],
        "blockers": report["blockers"],
        "json": rel(JSON_OUT),
        "markdown": rel(MD_OUT),
    }, indent=2, sort_keys=True))
    return 0 if report["final_s2_closed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
