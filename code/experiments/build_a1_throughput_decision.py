#!/usr/bin/env python3
"""Build the Gate A1 throughput-blocker decision record.

The decision is intentionally narrow: it checks the current retained frozen
AEGIS-Q row, the A2 gap-attribution probe, production strict-publication source
guards, and paper wording. It does not rerun broad benchmark suites.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "a1_throughput_decision"
FROZEN_AEGISQ = ROOT / "artifacts" / "validation" / "frozen_aegisq_contract" / "frozen_aegisq_contract.json"
A2_ATTRIBUTION = ROOT / "artifacts" / "validation" / "a2_gap_attribution" / "a2_gap_attribution.json"
STRICT_PUBLISH = ROOT / "code" / "storage" / "pqc_strict_publish.c"
FS_IO = ROOT / "code" / "fs" / "pqc_file_io.c"
EVAL_TEX = ROOT / "Paper" / "4_Evaluation.tex"
DISCUSSION_TEX = ROOT / "Paper" / "10_Discussion_and_Limitations.tex"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nested_get(obj: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def scan_source() -> dict[str, Any]:
    strict = STRICT_PUBLISH.read_text(encoding="utf-8")
    fs_io = FS_IO.read_text(encoding="utf-8")
    return {
        "strict_publish_file": str(STRICT_PUBLISH.relative_to(ROOT)),
        "fs_io_file": str(FS_IO.relative_to(ROOT)),
        "data_before_metadata_comment": "data-before-metadata durability boundary" in strict,
        "strict_data_fdatasync": "PQC_DURABILITY_SITE_DATA_SIDECAR" in strict
        and "pqc_durability_fdatasync" in strict,
        "strict_journal_fdatasync": "PQC_DURABILITY_SITE_JOURNAL_SIDECAR" in strict
        and "pqc_durability_fdatasync" in strict,
        "marker_syncfs_tail": "marker_metadata_syncfs" in fs_io
        and "PQC_DURABILITY_SITE_MARKER_METADATA" in fs_io,
        "anchor_external_sync": "pqc_anchor_worker_flush_now_external_sync" in fs_io,
        "windowed_anchor_opt_in_only": "pqc_anchor_worker_windowed_file_anchor_enabled()" in fs_io,
    }


def scan_paper() -> dict[str, Any]:
    eval_text = EVAL_TEX.read_text(encoding="utf-8")
    discussion = DISCUSSION_TEX.read_text(encoding="utf-8")
    combined = eval_text + "\n" + discussion
    return {
        "evaluation_file": str(EVAL_TEX.relative_to(ROOT)),
        "discussion_file": str(DISCUSSION_TEX.relative_to(ROOT)),
        "cost_boundary_wording": "cost boundary for authenticated publication" in eval_text,
        "a2_attribution_wording": "data-sidecar, journal-sidecar, and marker/checkpoint durability boundaries" in eval_text,
        "no_headline_win_wording": "not the headline win" in eval_text,
        "no_high_throughput_replacement_wording": "high-throughput general-purpose encryption" in discussion,
        "dangerous_high_throughput_claim": "high-throughput general-purpose filesystem" in combined
        or "general-purpose encrypted-filesystem replacement" in combined,
    }


def build_report() -> dict[str, Any]:
    frozen = load_json(FROZEN_AEGISQ)
    a2 = load_json(A2_ATTRIBUTION)
    frozen_tput = nested_get(
        frozen,
        ["warm_cache_summary", "metrics", "throughput_mib_s", "median"],
        0.0,
    )
    frozen_p99 = nested_get(
        frozen,
        ["warm_cache_summary", "metrics", "latency_p99_us", "median"],
        0.0,
    )
    a2_summary = a2.get("summary", {})
    durability = a2_summary.get("durability_calls", {})
    source = scan_source()
    paper = scan_paper()
    proof_checks = {
        "frozen_aegisq_overall_pass": bool(frozen.get("overall_pass")),
        "a2_overall_pass": bool(a2.get("overall_pass")),
        "strict_publication_boundaries_present": all(
            source[key]
            for key in (
                "data_before_metadata_comment",
                "strict_data_fdatasync",
                "strict_journal_fdatasync",
                "marker_syncfs_tail",
            )
        ),
        "paper_cost_boundary_present": all(
            paper[key]
            for key in (
                "cost_boundary_wording",
                "a2_attribution_wording",
                "no_headline_win_wording",
                "no_high_throughput_replacement_wording",
            )
        ),
        "no_broad_high_throughput_claim": not paper["dangerous_high_throughput_claim"],
    }
    decision = {
        "verdict": "cost-boundary-closeout",
        "why_not_local_syscall_swap": (
            "The remaining strict path has data-sidecar durability before "
            "journal mapping durability, followed by marker/checkpoint "
            "publication. Replacing those barriers with one later sync would "
            "change crash-ordering obligations and requires a publication "
            "redesign/fault matrix, not a local optimization."
        ),
        "next_valid_optimization_target": (
            "Use epoch/group commit only for batched or concurrent work where "
            "a shared barrier can amortize publication, or design a new "
            "strict-compatible compact publication format with fresh crash "
            "evidence."
        ),
    }
    return {
        "overall_pass": all(proof_checks.values()),
        "schema": "a1-throughput-decision-v1",
        "inputs": {
            "frozen_aegisq": str(FROZEN_AEGISQ.relative_to(ROOT)),
            "a2_attribution": str(A2_ATTRIBUTION.relative_to(ROOT)),
            "strict_publish_source": str(STRICT_PUBLISH.relative_to(ROOT)),
            "fs_io_source": str(FS_IO.relative_to(ROOT)),
            "evaluation_text": str(EVAL_TEX.relative_to(ROOT)),
            "discussion_text": str(DISCUSSION_TEX.relative_to(ROOT)),
        },
        "metrics": {
            "frozen_aegisq_throughput_mib_s_median": frozen_tput,
            "frozen_aegisq_p99_us_median": frozen_p99,
            "a2_aegisq_throughput_mib_s": a2_summary.get("aegisq_throughput_mib_s"),
            "a2_gocryptfs_throughput_mib_s": a2_summary.get("gocryptfs_throughput_mib_s"),
            "a2_aegisq_to_gocryptfs_ratio": a2_summary.get("aegisq_to_gocryptfs_ratio"),
            "a2_durability_calls": durability,
        },
        "source_checks": source,
        "paper_checks": paper,
        "proof_checks": proof_checks,
        "decision": decision,
        "non_claims": [
            "not a high-throughput filesystem claim",
            "not a general-purpose encrypted-filesystem replacement claim",
            "not proof that local syscall coalescing can preserve crash semantics",
        ],
    }


def build_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    decision = report["decision"]
    checks = report["proof_checks"]
    lines = [
        "# A1 Throughput Decision",
        "",
        f"- Overall pass: `{report['overall_pass']}`",
        f"- Verdict: `{decision['verdict']}`",
        "",
        "## Metrics",
        "",
        f"- Frozen AEGIS-Q median throughput: `{metrics['frozen_aegisq_throughput_mib_s_median']:.3f} MiB/s`",
        f"- Frozen AEGIS-Q median p99: `{metrics['frozen_aegisq_p99_us_median']:.3f} us`",
        f"- A2 short AEGIS-Q throughput: `{metrics['a2_aegisq_throughput_mib_s']:.3f} MiB/s`",
        f"- A2 short gocryptfs throughput: `{metrics['a2_gocryptfs_throughput_mib_s']:.3f} MiB/s`",
        f"- A2 AEGIS-Q/gocryptfs ratio: `{metrics['a2_aegisq_to_gocryptfs_ratio']:.3f}`",
        "",
        "## Decision",
        "",
        decision["why_not_local_syscall_swap"],
        "",
        decision["next_valid_optimization_target"],
        "",
        "## Proof Checks",
        "",
    ]
    for key, value in checks.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)

    report = build_report()
    json_path = args.out_dir / "a1_throughput_decision.json"
    md_path = args.out_dir / "a1_throughput_decision.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(build_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "overall_pass": report["overall_pass"],
                "json": str(json_path),
                "markdown": str(md_path),
                "verdict": report["decision"]["verdict"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
