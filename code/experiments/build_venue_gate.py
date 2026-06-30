#!/usr/bin/env python3
"""Gate 0.3-S0 venue-readiness gate builder."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "artifacts" / "validation" / "refactor_inventory"
REVIEW_MAP = OUT_DIR / "review_objection_map.json"
WORKTREE_FREEZE = OUT_DIR / "worktree_freeze.json"
JSON_OUT = OUT_DIR / "venue_gate.json"
MD_OUT = OUT_DIR / "venue_gate.md"


@dataclass(frozen=True)
class VenueSpec:
    venue_class: str
    readiness_verdict: str
    required_bar: tuple[str, ...]
    blocker_ids: tuple[str, ...]
    fallback_classification_rule: str
    paper_nonclaim_boundary: str


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def venue_specs() -> list[VenueSpec]:
    return [
        VenueSpec(
            venue_class="SOSP/OSDI",
            readiness_verdict="blocked",
            required_bar=(
                "one-sentence systems thesis centered on a storage-runtime bottleneck",
                "implemented production-path mechanism with explicit invariant and failure boundary",
                "clean code architecture or retained behavior-equivalence evidence for decomposition",
                "mode-aligned filesystem baselines including kernel-native alternatives or blocking proof",
                "kernel QoS baselines for any QoS uniqueness claim",
                "macrobenchmark evidence beyond a single SQLite/FUSE vignette",
                "correctness, POSIX, crash, freshness, and security boundaries backed by artifacts",
                "claim firewall that prevents unimplemented GPU, TPM, eBPF, POSIX, and deployment wording",
            ),
            blocker_ids=(
                "O01",
                "O02",
                "O03",
                "O04",
                "O05",
                "O06",
                "O07",
                "O08",
                "O09",
                "O10",
                "O11",
                "O12",
                "O13",
                "O14",
            ),
            fallback_classification_rule=(
                "SOSP/OSDI is the active target. It cannot be claimed by scoped paper wording alone; every hard blocker "
                "must be closed by production-path code, retained artifact, paper text, and negative guard, or the "
                "corresponding venue-level claim must be removed."
            ),
            paper_nonclaim_boundary=(
                "Do not say SOSP-ready, OSDI-ready, deployment-ready, general-purpose filesystem, broad POSIX support, "
                "or full rollback/crash/GPU-isolation defense until the listed blockers close."
            ),
        ),
        VenueSpec(
            venue_class="ATC",
            readiness_verdict="conditional-blocked",
            required_bar=(
                "production-quality implementation with a clear engineering contribution",
                "reproducible evaluation harness and retained raw artifacts",
                "practical baselines against deployed alternatives",
                "honest limitations for POSIX, crash, freshness, hardware, and threat model",
                "paper narrative that emphasizes measured implementation behavior instead of broad systems novelty",
            ),
            blocker_ids=(
                "O01",
                "O02",
                "O03",
                "O04",
                "O05",
                "O07",
                "O09",
                "O12",
                "O14",
            ),
            fallback_classification_rule=(
                "ATC may become a fallback only if the user explicitly rescopes from SOSP/OSDI; it still needs clean "
                "implementation evidence, practical baselines, reproducibility, and explicit limitations."
            ),
            paper_nonclaim_boundary=(
                "Do not use ATC fallback to keep SOSP/OSDI claims. If rescoped to ATC, remove or downgrade claims that "
                "depend on unclosed venue-level novelty, broad POSIX, rollback resistance, or second-workload generality."
            ),
        ),
        VenueSpec(
            venue_class="FAST",
            readiness_verdict="conditional-blocked",
            required_bar=(
                "storage-specific mechanism with durability, write amplification, sync, and recovery accounting",
                "mode-aligned filesystem or block-layer baselines",
                "crash and replay semantics tied to implementation artifacts",
                "tail-latency and throughput attribution for the mounted path",
                "related-work positioning against logging, journaling, fast-commit, and kernel encryption systems",
            ),
            blocker_ids=(
                "O02",
                "O03",
                "O04",
                "O06",
                "O07",
                "O08",
                "O13",
                "O12",
            ),
            fallback_classification_rule=(
                "FAST may become a fallback only if the paper is reframed as a storage publication/runtime mechanism; "
                "filesystem viability, baselines, crash semantics, and sync/fdatasync accounting remain required."
            ),
            paper_nonclaim_boundary=(
                "Do not claim a FAST-style storage mechanism until epoch publication, fdatasync amortization, recovery, "
                "and mode-aligned storage baselines are measured in the production mounted path."
            ),
        ),
    ]


def freeze_summary() -> dict[str, Any]:
    freeze = read_json(WORKTREE_FREEZE)
    git = freeze.get("git", {})
    return {
        "source": relpath(WORKTREE_FREEZE),
        "exists": WORKTREE_FREEZE.exists(),
        "generated_utc": freeze.get("generated_utc"),
        "branch": git.get("branch"),
        "diff_name_only_count": len(git.get("diff_name_only", [])) if isinstance(git.get("diff_name_only"), list) else None,
        "untracked_files_count": git.get("untracked_files_count"),
        "code_refactor_strategy_absent": freeze.get("authoritative_checklist", {}).get("code_refactor_strategy_absent"),
        "parent_checklist_closed": freeze.get("parent_checklist_closed"),
        "paper_text_status": freeze.get("paper_text_status"),
    }


def review_map_summary(review: dict[str, Any]) -> dict[str, Any]:
    verdict = review.get("artifact_verdict", {})
    return {
        "source": relpath(REVIEW_MAP),
        "exists": REVIEW_MAP.exists(),
        "schema_version": review.get("schema_version"),
        "generated_utc": review.get("generated_utc"),
        "objection_count": review.get("objection_count"),
        "overall_pass": verdict.get("overall_pass"),
        "status_counts": verdict.get("status_counts", {}),
    }


def short_missing_artifacts(item: dict[str, Any]) -> list[str]:
    missing = item.get("missing_artifacts", [])
    if not isinstance(missing, list):
        return []
    out: list[str] = []
    for entry in missing[:4]:
        if isinstance(entry, dict):
            out.append(str(entry.get("path") or entry.get("label") or "unknown"))
        else:
            out.append(str(entry))
    if len(missing) > 4:
        out.append(f"+{len(missing) - 4} more")
    return out


def build_blocker(ident: str, objections: dict[str, dict[str, Any]]) -> dict[str, Any]:
    item = objections.get(ident)
    if not item:
        return {
            "id": ident,
            "found": False,
            "status": "missing-objection",
            "title": "missing review objection",
            "primary_code_module": None,
            "blocking_reason": "The venue gate references an objection that is not present in the review-objection map.",
            "negative_guard": None,
            "missing_artifacts": [],
        }
    return {
        "id": ident,
        "found": True,
        "status": item.get("status"),
        "title": item.get("title"),
        "primary_code_module": item.get("primary_code_module"),
        "gate_links": item.get("gate_links", []),
        "blocking_reason": item.get("reviewer_objection"),
        "paper_claim_boundary": item.get("paper_claim_boundary"),
        "negative_guard": item.get("negative_guard"),
        "missing_artifacts": short_missing_artifacts(item),
    }


def build_venue(spec: VenueSpec, objections: dict[str, dict[str, Any]]) -> dict[str, Any]:
    blockers = [build_blocker(ident, objections) for ident in spec.blocker_ids]
    unresolved_status_counts: dict[str, int] = {}
    for blocker in blockers:
        status = str(blocker.get("status"))
        unresolved_status_counts[status] = unresolved_status_counts.get(status, 0) + 1
    return {
        "venue_class": spec.venue_class,
        "readiness_verdict": spec.readiness_verdict,
        "required_bar": list(spec.required_bar),
        "required_blockers": blockers,
        "unresolved_status_counts": unresolved_status_counts,
        "fallback_classification_rule": spec.fallback_classification_rule,
        "paper_nonclaim_boundary": spec.paper_nonclaim_boundary,
    }


def artifact_verdict(venues: list[dict[str, Any]], review: dict[str, Any], freeze: dict[str, Any]) -> dict[str, Any]:
    present_classes = {venue.get("venue_class") for venue in venues}
    missing_classes = sorted({"SOSP/OSDI", "ATC", "FAST"} - present_classes)
    missing_blockers = [
        f"{venue.get('venue_class')}:{blocker.get('id')}"
        for venue in venues
        for blocker in venue.get("required_blockers", [])
        if not blocker.get("found")
    ]
    missing_boundaries = [
        str(venue.get("venue_class"))
        for venue in venues
        if not venue.get("fallback_classification_rule") or not venue.get("paper_nonclaim_boundary")
    ]
    all_blocked_or_conditional = all(
        venue.get("readiness_verdict") in {"blocked", "conditional-blocked"}
        for venue in venues
    )
    review_loaded = REVIEW_MAP.exists() and review.get("schema_version") == 2 and review.get("artifact_verdict", {}).get("overall_pass") is True
    freeze_loaded = WORKTREE_FREEZE.exists() and freeze.get("code_refactor_strategy_absent") is True
    return {
        "overall_pass": (
            review_loaded
            and freeze_loaded
            and not missing_classes
            and not missing_blockers
            and not missing_boundaries
            and all_blocked_or_conditional
        ),
        "review_map_loaded": review_loaded,
        "worktree_freeze_loaded": freeze_loaded,
        "venue_classes_present": sorted(present_classes),
        "missing_venue_classes": missing_classes,
        "missing_blocker_refs": missing_blockers,
        "missing_policy_boundaries": missing_boundaries,
        "ready_for_sosp_osdi_claim": False,
        "fallback_targets_are_claim_closures": False,
    }


def write_markdown(payload: dict[str, Any]) -> None:
    lines = [
        "# Venue Gate",
        "",
        "This Gate 0.3-S0 artifact separates the active SOSP/OSDI target from ATC/FAST fallback classifications. It does not close venue readiness; it records why the current evidence remains blocked or conditional.",
        "",
        f"- Generated: `{payload['generated_utc']}`",
        f"- Review map: `{payload['review_map']['source']}`",
        f"- Worktree freeze: `{payload['worktree_freeze']['source']}`",
        f"- Ready for SOSP/OSDI claim: `{payload['artifact_verdict']['ready_for_sosp_osdi_claim']}`",
        f"- Parent checklist closed by this artifact: `{payload['parent_checklist_closed']}`",
        "",
        "## Venue Verdicts",
        "",
        "| Venue | Verdict | Blocker statuses | Classification rule |",
        "| --- | --- | --- | --- |",
    ]
    for venue in payload["venues"]:
        counts = ", ".join(
            f"`{status}`={count}" for status, count in sorted(venue["unresolved_status_counts"].items())
        )
        rule = venue["fallback_classification_rule"].replace("|", "/")
        lines.append(
            f"| {venue['venue_class']} | `{venue['readiness_verdict']}` | {counts} | {rule} |"
        )
    lines.extend(["", "## SOSP/OSDI Hard Blockers", ""])
    sosp = next(venue for venue in payload["venues"] if venue["venue_class"] == "SOSP/OSDI")
    for blocker in sosp["required_blockers"]:
        lines.append(
            f"- `{blocker['id']}` `{blocker['status']}` `{blocker['primary_code_module']}`: {blocker['negative_guard']}"
        )
    lines.extend(
        [
            "",
            "## Non-Claim Boundary",
            "",
        ]
    )
    for venue in payload["venues"]:
        lines.append(f"- {venue['venue_class']}: {venue['paper_nonclaim_boundary']}")
    lines.extend(
        [
            "",
            "## Next Cursor",
            "",
            f"`{payload['next_cursor']['row_id']}`: {payload['next_cursor']['reason']}",
            "",
        ]
    )
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    review = read_json(REVIEW_MAP)
    freeze = freeze_summary()
    objections = {
        str(item.get("id")): item
        for item in review.get("objections", [])
        if isinstance(item, dict) and item.get("id")
    }
    venues = [build_venue(spec, objections) for spec in venue_specs()]
    payload = {
        "schema_version": 2,
        "generated_by": relpath(Path(__file__)),
        "generated_utc": now_utc(),
        "scope": "Gate 0.3-S0 venue readiness against current review-objection map and frozen worktree.",
        "target_policy": {
            "primary_target": "SOSP/OSDI",
            "fallback_targets": ["ATC", "FAST"],
            "rule": (
                "SOSP/OSDI remains the active target. ATC/FAST are fallback classifications only if the user explicitly "
                "rescopes; they do not close SOSP/OSDI blockers."
            ),
        },
        "review_map": review_map_summary(review),
        "worktree_freeze": freeze,
        "venues": venues,
        "paper_text_policy": {
            "allowed_now": (
                "scoped prototype",
                "retained evidence for specific mounted-path workflows",
                "blocked SOSP/OSDI target pending Gate 0 and dependent gates",
            ),
            "forbidden_until_closed": (
                "SOSP-ready",
                "OSDI-ready",
                "general-purpose filesystem",
                "ready for deployment",
                "direct NVMe-to-UVM DMA",
                "GPUDirect/RDMA",
                "dma-buf zero-copy",
                "eBPF/io_uring completion bypass",
                "persistent PCR-bound freshness",
                "TPM rollback resistance",
                "foreground AI QoS recovery",
                "full crash certification",
                "side-channel protection",
                "portability",
            ),
        },
        "artifact_verdict": artifact_verdict(venues, review, freeze),
        "paper_text_status": "not_updated",
        "parent_checklist_closed": False,
        "next_cursor": {
            "row_id": "0.4-S0",
            "reason": "Refresh source ownership and module-decomposition inventory for the current code tree.",
        },
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(payload)
    print(json.dumps(payload["artifact_verdict"], indent=2, sort_keys=True))
    return 0 if payload["artifact_verdict"]["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
