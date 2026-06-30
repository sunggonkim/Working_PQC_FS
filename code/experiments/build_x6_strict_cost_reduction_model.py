#!/usr/bin/env python3
"""Build the X6 strict-path cost-reduction model.

This is a narrow verifier for the production change that replaces the strict
fsync-return marker/checkpoint `syncfs` with a marker-file `fsync`. It uses the
retained frozen-path attribution row only to quantify the targeted barrier; it
does not claim a new throughput benchmark, kernel upstreaming, or power-loss
certification.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "x6_strict_cost_reduction"
A2_ATTRIBUTION = ROOT / "artifacts" / "validation" / "a2_gap_attribution" / "a2_gap_attribution.json"
FILE_IO = ROOT / "code" / "fs" / "pqc_file_io.c"
FD_CONTEXT = ROOT / "code" / "fs" / "pqc_fd_context.c"
EVAL_TEX = ROOT / "Paper" / "4_Evaluation.tex"
DISCUSSION_TEX = ROOT / "Paper" / "10_Discussion_and_Limitations.tex"


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_checks() -> dict[str, Any]:
    file_io = read_text(FILE_IO)
    fd_context = read_text(FD_CONTEXT)
    return {
        "marker_metadata_uses_fsync": "static int marker_metadata_fsync" in file_io
        and "pqc_durability_fsync(fd, PQC_DURABILITY_SITE_MARKER_METADATA)" in file_io,
        "marker_metadata_no_syncfs_helper": "marker_metadata_syncfs" not in file_io,
        "anchor_external_sync_uses_marker_fsync": "marker_metadata_fsync, &fd" in file_io,
        "epoch_syncfs_still_present": "pqc_durability_syncfs(" in fd_context,
        "epoch_syncfs_scope": "PQC_DURABILITY_SITE_EPOCH_LOG" in fd_context,
    }


def attribution_model() -> dict[str, Any]:
    data = load_json(A2_ATTRIBUTION)
    durability = ((data.get("aegisq") or {}).get("durability") or {})
    fdatasync_calls = int(durability.get("fdatasync") or 0)
    syncfs_calls = int(durability.get("syncfs") or 0)
    marker_calls = int(durability.get("marker_metadata") or 0)
    data_calls = int(durability.get("data_sidecar") or 0)
    journal_calls = int(durability.get("journal_sidecar") or 0)
    targeted_syncfs = min(syncfs_calls, marker_calls)
    modeled_remaining_blocking_syncs = fdatasync_calls + marker_calls
    modeled_before_blocking_syncs = fdatasync_calls + syncfs_calls
    return {
        "source_artifact": relpath(A2_ATTRIBUTION),
        "retained_counts": {
            "fdatasync_calls": fdatasync_calls,
            "syncfs_calls": syncfs_calls,
            "marker_metadata_calls": marker_calls,
            "data_sidecar_calls": data_calls,
            "journal_sidecar_calls": journal_calls,
        },
        "targeted_barrier": "strict fsync-return marker/checkpoint syncfs",
        "targeted_syncfs_calls": targeted_syncfs,
        "modeled_before_blocking_syncs": modeled_before_blocking_syncs,
        "modeled_after_blocking_syncs": modeled_remaining_blocking_syncs,
        "modeled_filesystem_wide_syncfs_before": targeted_syncfs,
        "modeled_filesystem_wide_syncfs_after": 0,
        "modeled_syncfs_removed": targeted_syncfs,
        "modeled_syncfs_removed_fraction": (
            targeted_syncfs / syncfs_calls if syncfs_calls else 0.0
        ),
        "scope": (
            "Counts are from retained A2 attribution. The model changes the "
            "type of the marker/checkpoint durability primitive from syncfs to "
            "fsync; it does not remove data or journal fdatasync and does not "
            "predict end-to-end throughput."
        ),
    }


def paper_checks() -> dict[str, Any]:
    combined = read_text(EVAL_TEX) + "\n" + read_text(DISCUSSION_TEX)
    required = {
        "mentions_marker_fsync_narrowing": "marker/checkpoint step now uses marker-file \\texttt{fsync}" in combined,
        "does_not_claim_kernel_upstreaming": "not kernel upstreaming" in combined,
        "does_not_claim_power_loss": "not physical power-loss" in combined
        or "power-loss certification" in combined,
        "keeps_strict_cost_boundary": "strict path still pays data and journal barriers" in combined,
    }
    return {
        "required_phrases": required,
        "complete": all(required.values()),
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    model = result["attribution_model"]
    checks = result["source_checks"]
    lines = [
        "# X6 Strict Cost-Reduction Model",
        "",
        f"- Overall pass: `{str(result['overall_pass']).lower()}`",
        f"- Marker metadata uses fsync: `{str(checks['marker_metadata_uses_fsync']).lower()}`",
        f"- Marker syncfs helper removed: `{str(checks['marker_metadata_no_syncfs_helper']).lower()}`",
        f"- Retained A2 syncfs calls targeted: `{model['targeted_syncfs_calls']}`",
        f"- Modeled filesystem-wide syncfs calls before/after: `{model['modeled_filesystem_wide_syncfs_before']}` / `{model['modeled_filesystem_wide_syncfs_after']}`",
        f"- Modeled blocking sync primitives before/after: `{model['modeled_before_blocking_syncs']}` / `{model['modeled_after_blocking_syncs']}`",
        "",
        "Scope: production strict-path marker/checkpoint sync narrowing plus retained-count model; no throughput, kernel-upstreaming, or power-loss claim.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_result() -> dict[str, Any]:
    source = source_checks()
    model = attribution_model()
    paper = paper_checks()
    checks = {
        **source,
        "retained_attribution_present": A2_ATTRIBUTION.exists(),
        "targeted_syncfs_positive": model["targeted_syncfs_calls"] > 0,
        "paper_scope_complete": paper["complete"],
    }
    return {
        "schema": "aegisq.x6_strict_cost_reduction_model.v1",
        "generated_utc": now_utc(),
        "scope": [
            "Production strict-path marker/checkpoint syncfs narrowing.",
            "Retained attribution-count model only; not a new throughput benchmark.",
            "No kernel upstreaming, broad POSIX, physical power-loss, or drive-cache certification claim.",
        ],
        "source_files": [relpath(FILE_IO), relpath(FD_CONTEXT)],
        "source_checks": source,
        "attribution_model": model,
        "paper_checks": paper,
        "checks": checks,
        "overall_pass": all(checks.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = build_result()
    json_path = args.out_dir / "x6_strict_cost_reduction_model.json"
    md_path = args.out_dir / "x6_strict_cost_reduction_model.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(result, md_path)
    print(json.dumps({
        "overall_pass": result["overall_pass"],
        "json": relpath(json_path),
        "markdown": relpath(md_path),
        "targeted_syncfs_calls": result["attribution_model"]["targeted_syncfs_calls"],
        "modeled_blocking_syncs_before": result["attribution_model"]["modeled_before_blocking_syncs"],
        "modeled_blocking_syncs_after": result["attribution_model"]["modeled_after_blocking_syncs"],
        "modeled_syncfs_before": result["attribution_model"]["modeled_filesystem_wide_syncfs_before"],
        "modeled_syncfs_after": result["attribution_model"]["modeled_filesystem_wide_syncfs_after"],
    }, indent=2, sort_keys=True))
    return 0 if result["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
