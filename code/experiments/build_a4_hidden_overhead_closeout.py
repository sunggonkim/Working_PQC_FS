#!/usr/bin/env python3
"""Build the Gate A4 hidden-overhead closeout.

This is intentionally narrow. It reads the mounted A4 smoke trace plus the
already-retained Jetson/CUDA gate artifacts and classifies each overhead named
by A4 as measured, diagnostic-only, unavailable, or not claimed.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "a4_hidden_overhead_accounting"

A4_SMOKE = DEFAULT_OUT / "a4_overhead_trace_smoke.json"
JETSON_MEMORY = ROOT / "artifacts" / "validation" / "jetson_memory_contract" / "jetson_memory_contract.json"
CUDA_QOS = ROOT / "artifacts" / "validation" / "cuda_qos_contract" / "cuda_qos_contract.json"
JETSON_LADDER = ROOT / "artifacts" / "validation" / "jetson_optimization_ladder" / "jetson_optimization_ladder.json"
EVAL_TEX = ROOT / "Paper" / "4_Evaluation.tex"
DESIGN_TEX = ROOT / "Paper" / "3_Design.tex"
DISCUSSION_TEX = ROOT / "Paper" / "10_Discussion_and_Limitations.tex"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def number(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def average_ns(op: dict[str, Any]) -> int:
    calls = number(op.get("calls"))
    if calls <= 0:
        return 0
    return number(op.get("total_ns")) // calls


def fuse_op(smoke: dict[str, Any], name: str) -> dict[str, Any]:
    for op in smoke.get("fuse_trace", {}).get("operations", []):
        if isinstance(op, dict) and op.get("op") == name:
            return op
    return {}


def paper_scan() -> dict[str, Any]:
    texts = {
        "design": DESIGN_TEX.read_text(encoding="utf-8", errors="replace"),
        "evaluation": EVAL_TEX.read_text(encoding="utf-8", errors="replace"),
        "discussion": DISCUSSION_TEX.read_text(encoding="utf-8", errors="replace"),
    }
    combined = "\n".join(texts.values())
    dangerous_positive = [
        "implements a completed eBPF completion path",
        "completed eBPF completion path is part of the mounted implementation",
        "implements a closed-loop hardware-QoS controller",
        "direct NVMe-to-UVM DMA is implemented",
        "uses GPUDirect Storage in the mounted path",
        "proves foreground AI-QoS recovery",
        "proves foreground AI p99 recovery",
        "uses storage-DMA registration",
    ]
    scoped_phrases = [
        "not a claim that a FUSE write becomes 23.2",
        "not a CUDA optimization claim by itself",
        "not a kernel scheduler context-switch trace",
        "not storage-DMA registration",
        "no foreground AI p99 recovery claim",
        "not a completed closed-loop QoS controller",
        "not yet a complete measured kernel-encryption",
        "not to rank deployed encryption systems",
        "does not expose an \\texttt{O\\_DIRECT} storage path",
    ]
    return {
        "files": [rel(DESIGN_TEX), rel(EVAL_TEX), rel(DISCUSSION_TEX)],
        "dangerous_positive_hits": [
            phrase for phrase in dangerous_positive if phrase in combined
        ],
        "scoped_phrase_hits": [
            phrase for phrase in scoped_phrases if phrase in combined
        ],
        "reports_fuse_cost_boundary": "FUSE prototype impose" in texts["evaluation"]
        and "cost boundary for authenticated publication" in texts["evaluation"],
        "reports_cuda_staging_scope": "managed-buffer staging" in texts["evaluation"]
        and "data plane is CPU-first" in texts["evaluation"],
        "reports_no_storage_dma": "not storage-DMA registration" in texts["discussion"]
        or "not a storage-DMA path" in texts["design"],
    }


def classify(smoke: dict[str, Any], memory: dict[str, Any],
             cuda_qos: dict[str, Any], ladder: dict[str, Any]) -> list[dict[str, Any]]:
    durability = smoke.get("durability_mounted_operation_stats", {})
    publication = smoke.get("publication_trace", {})
    plane = smoke.get("plane_trace", {})
    ladder_summary = ladder.get("summary", {})
    cuda_claims = cuda_qos.get("claim_verdicts", [])
    kernel_launch = next(
        (claim for claim in cuda_claims if claim.get("claim") == "Kernel launch shape accounting"),
        {},
    )
    profiler_claim = next(
        (claim for claim in cuda_claims if claim.get("claim") == "Nsight/CUPTI-backed GPU QoS"),
        {},
    )
    memory_claim_names = {
        item.get("name"): item for item in memory.get("claim_contract", [])
        if isinstance(item, dict)
    }

    return [
        {
            "overhead": "FUSE daemon operation latency",
            "classification": "measured",
            "evidence": [
                f"create_avg_ns={average_ns(fuse_op(smoke, 'create'))}",
                f"write_avg_ns={average_ns(fuse_op(smoke, 'write'))}",
                f"fsync_avg_ns={average_ns(fuse_op(smoke, 'fsync'))}",
                f"read_avg_ns={average_ns(fuse_op(smoke, 'read'))}",
                "daemon-side proxy only",
            ],
            "paper_guard": "Do not call this a kernel context-switch count or eBPF/io_uring bypass proof.",
        },
        {
            "overhead": "durability syscalls and publication barriers",
            "classification": "measured",
            "evidence": [
                f"fdatasync={durability.get('fdatasync', 0)}",
                f"syncfs={durability.get('syncfs', 0)}",
                f"data_sidecar={durability.get('data_sidecar', 0)}",
                f"journal_sidecar={durability.get('journal_sidecar', 0)}",
                f"marker_metadata={durability.get('marker_metadata', 0)}",
                f"publication_sync_count_total={publication.get('publication_sync_count_total', 0)}",
                f"publication_elapsed_ns_total={publication.get('publication_elapsed_ns_total', 0)}",
            ],
            "paper_guard": "Report strict publication cost as a boundary, not a general filesystem ranking.",
        },
        {
            "overhead": "journal/checkpoint update",
            "classification": "measured",
            "evidence": [
                f"journal_sidecar={durability.get('journal_sidecar', 0)}",
                f"marker_metadata={durability.get('marker_metadata', 0)}",
                f"publication_count={publication.get('publication_count', 0)}",
            ],
            "paper_guard": "Keep journal/checkpoint wording tied to D/J/C publication evidence.",
        },
        {
            "overhead": "freshness anchor refresh",
            "classification": "measured",
            "evidence": [
                f"freshness_anchor_events={plane.get('freshness_anchor_events', 0)}",
                f"freshness_anchor_successes={plane.get('freshness_anchor_successes', 0)}",
                f"freshness_anchor_file_backend={plane.get('freshness_anchor_file_backend', 0)}",
                f"freshness_anchor_hardware_backend={plane.get('freshness_anchor_hardware_backend', 0)}",
            ],
            "paper_guard": "Do not upgrade file-backed or replay-after-advance evidence to PCR-bound rollback resistance.",
        },
        {
            "overhead": "AES-GCM data-plane routing",
            "classification": "measured",
            "evidence": [
                f"encrypt_blocks={plane.get('data_aes_gcm_encrypt_blocks', 0)}",
                f"decrypt_blocks={plane.get('data_aes_gcm_decrypt_blocks', 0)}",
                f"cpu_blocks={plane.get('data_route_cpu_blocks', 0)}",
                f"gpu_blocks={plane.get('data_route_gpu_blocks', 0)}",
            ],
            "paper_guard": "Do not imply bulk file data is encrypted directly by PQC primitives.",
        },
        {
            "overhead": "CUDA launch and stream synchronization",
            "classification": "diagnostic-only",
            "evidence": [
                f"production_kernel_launches={kernel_launch.get('production_kernel_launches', 0)}",
                f"cuda_qos_paper_eligible={kernel_launch.get('paper_mechanism_eligible', False)}",
                f"profiler_trace_retained={profiler_claim.get('retained_profiler_trace', False)}",
            ],
            "paper_guard": "CUDA launch evidence is not a paper mechanism until same-run mounted-path traces show benefit.",
        },
        {
            "overhead": "GPU initialization",
            "classification": "diagnostic-only",
            "evidence": [
                f"device_count={memory.get('probe_payload', {}).get('device_count')}",
                f"selected_device={memory.get('probe_payload', {}).get('selected_device')}",
                "A4 smoke does not time initialization on the mounted path",
            ],
            "paper_guard": "Do not report GPU initialization as hidden mounted-path overhead until a production trace times it.",
        },
        {
            "overhead": "staging copy / UVM / pinned memory",
            "classification": "diagnostic-only",
            "evidence": [
                f"managed_memory_class={memory_claim_names.get('managed memory / UVM allocation', {}).get('evidence_level')}",
                f"ladder_paper_mechanism_eligible_count={ladder_summary.get('paper_mechanism_eligible_count')}",
                "no production mounted-path benefit retained",
            ],
            "paper_guard": "Keep UVM, pinned, and registered-memory wording diagnostic unless production mounted-path benefit exists.",
        },
        {
            "overhead": "dma-buf, GPUDirect/RDMA, direct NVMe-to-UVM DMA",
            "classification": "not-claimed",
            "evidence": [
                f"non_claim_terms={ladder_summary.get('non_claim_terms', [])}",
                "no mounted-path peer-DMA proof",
            ],
            "paper_guard": "These terms remain non-claims.",
        },
    ]


def build_report() -> dict[str, Any]:
    smoke = load_json(A4_SMOKE)
    memory = load_json(JETSON_MEMORY)
    cuda_qos = load_json(CUDA_QOS)
    ladder = load_json(JETSON_LADDER)
    overheads = classify(smoke, memory, cuda_qos, ladder)
    paper = paper_scan()
    proof_checks = {
        "a4_smoke_pass": bool(smoke.get("overall_pass")),
        "jetson_memory_pass": bool(memory.get("overall_pass")),
        "cuda_qos_pass": bool(cuda_qos.get("artifact_verdict", {}).get("overall_pass")),
        "optimization_ladder_pass": bool(ladder.get("artifact_verdict", {}).get("overall_pass")),
        "measured_fuse_latency": any(
            row["overhead"] == "FUSE daemon operation latency"
            and row["classification"] == "measured"
            for row in overheads
        ),
        "measured_publication_barriers": number(
            smoke.get("publication_trace", {}).get("publication_sync_count_total")
        ) > 0,
        "paper_has_scope_guards": paper["reports_fuse_cost_boundary"]
        and paper["reports_cuda_staging_scope"]
        and paper["reports_no_storage_dma"],
        "no_unscoped_dangerous_positive_hits": not paper["dangerous_positive_hits"],
    }
    return {
        "overall_pass": all(proof_checks.values()),
        "schema": "a4-hidden-overhead-closeout-v1",
        "generated_utc": now_utc(),
        "inputs": {
            "a4_smoke": rel(A4_SMOKE),
            "jetson_memory_contract": rel(JETSON_MEMORY),
            "cuda_qos_contract": rel(CUDA_QOS),
            "jetson_optimization_ladder": rel(JETSON_LADDER),
            "paper_files": paper["files"],
        },
        "overhead_classification": overheads,
        "paper_scan": paper,
        "proof_checks": proof_checks,
        "paper_text_status": "already_scoped_no_update",
        "parent_checklist_closed": False,
        "non_claims": [
            "not a kernel context-switch count",
            "not proof of eBPF/io_uring bypass",
            "not a CUDA optimization paper mechanism",
            "not a direct NVMe-to-UVM, GPUDirect/RDMA, or dma-buf zero-copy claim",
            "not a TPM rollback-resistance claim",
        ],
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# A4 Hidden Overhead Closeout",
        "",
        f"- Overall pass: `{report['overall_pass']}`",
        f"- Paper text status: `{report['paper_text_status']}`",
        f"- Parent checklist closed: `{report['parent_checklist_closed']}`",
        "",
        "## Overhead Classification",
        "",
        "| Overhead | Classification | Evidence | Guard |",
        "| --- | --- | --- | --- |",
    ]
    for row in report["overhead_classification"]:
        evidence = "<br>".join(f"`{item}`" for item in row["evidence"])
        lines.append(
            f"| {row['overhead']} | `{row['classification']}` | "
            f"{evidence} | {row['paper_guard']} |"
        )
    lines.extend(["", "## Proof Checks", ""])
    for key, value in report["proof_checks"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Non-Claims", ""])
    for item in report["non_claims"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report()
    json_path = out_dir / "a4_hidden_overhead_closeout.json"
    md_path = out_dir / "a4_hidden_overhead_closeout.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({
        "overall_pass": report["overall_pass"],
        "json": str(json_path),
        "markdown": str(md_path),
        "measured_count": sum(
            1 for row in report["overhead_classification"]
            if row["classification"] == "measured"
        ),
        "diagnostic_only_count": sum(
            1 for row in report["overhead_classification"]
            if row["classification"] == "diagnostic-only"
        ),
        "not_claimed_count": sum(
            1 for row in report["overhead_classification"]
            if row["classification"] == "not-claimed"
        ),
    }, indent=2, sort_keys=True))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
