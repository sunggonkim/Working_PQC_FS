#!/usr/bin/env python3
"""Audit that eBPF/io_uring remains scoped out of the submitted claim.

The repository retains standalone tracepoint and io_uring diagnostics.  This
audit checks that the main paper does not turn those diagnostics into a mounted
FUSE notification path or completion-bypass claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "Paper"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "ebpf_iouring_scope_audit"
FUSE_SOURCE = ROOT / "pqc_fuse.c"
TRACE_SCRIPT = ROOT / "experiments" / "trace_nvme_lat.bt"
TRACE_OUT = ROOT / "artifacts" / "probes" / "evidence" / "trace_nvme_lat.out"
IOURING_PROBE = ROOT / "experiments" / "bench_io_uring_ebpf.cu"


@dataclass(frozen=True)
class Pattern:
    name: str
    regex: re.Pattern[str]
    explanation: str


HIGH_RISK_PATTERNS = [
    Pattern(
        "ebpf",
        re.compile(r"eBPF", re.IGNORECASE),
        "eBPF wording must be negated, scoped to a standalone tracepoint, or future work.",
    ),
    Pattern(
        "io_uring",
        re.compile(r"io\\?_uring", re.IGNORECASE),
        "io_uring wording must not imply a mounted completion-bypass path.",
    ),
    Pattern(
        "completion_bypass",
        re.compile(r"(?:completion|notification|fast-notification)\s+bypass", re.IGNORECASE),
        "Completion-bypass language must be explicitly rejected or future work.",
    ),
    Pattern(
        "triggered_cuda_work",
        re.compile(r"eBPF-triggered\s+CUDA\s+work", re.IGNORECASE),
        "eBPF-triggered CUDA work is not implemented in the FUSE path.",
    ),
    Pattern(
        "tracepoint_notification",
        re.compile(r"tracepoint\s+notification", re.IGNORECASE),
        "Tracepoint-notification language must say it is not claimed.",
    ),
]

SCOPE_TERMS = (
    "does not",
    "do not",
    "not ",
    " no ",
    "unless",
    "future",
    "scoped",
    "unsupported",
    "incorrectly",
    "initial draft",
    "standalone",
    "raw",
    "histogram",
    "artifact is retained",
    "not claimed",
    "would need",
    "must not",
    "ordinary",
    "out of the contribution",
    "out of the paper",
)

REQUIRED_PAPER_PHRASES = [
    "does not claim direct NVMe-to-UVM DMA, \\texttt{io\\_uring}/eBPF completion bypass",
    "does not claim verified \\texttt{O\\_DIRECT} NVMe-to-UVM DMA, an \\texttt{io\\_uring}/eBPF completion bypass",
    "No final eBPF notification path",
    "eBPF/io\\_uring completion bypass",
    "does not claim direct NVMe-to-UVM DMA, eBPF/io\\_uring completion bypass",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def context(text: str, start: int, end: int, radius: int = 170) -> str:
    begin = max(0, start - radius)
    finish = min(len(text), end + radius)
    return " ".join(text[begin:finish].split())


def is_scoped(match_context: str) -> bool:
    lowered = f" {match_context.lower()} "
    return any(term in lowered for term in SCOPE_TERMS)


def paper_files() -> list[Path]:
    return sorted(path for path in PAPER_DIR.glob("*.tex") if path.is_file())


def scan_paper() -> dict[str, Any]:
    files = paper_files()
    combined = "\n".join(path.read_text(encoding="utf-8") for path in files)
    mentions: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []

    for path in files:
        text = path.read_text(encoding="utf-8")
        for pattern in HIGH_RISK_PATTERNS:
            for match in pattern.regex.finditer(text):
                ctx = context(text, match.start(), match.end())
                scoped = is_scoped(ctx)
                record = {
                    "pattern": pattern.name,
                    "file": str(path.relative_to(ROOT)),
                    "line": text.count("\n", 0, match.start()) + 1,
                    "match": match.group(0),
                    "scoped": scoped,
                    "context": ctx,
                    "explanation": pattern.explanation,
                }
                mentions.append(record)
                if not scoped:
                    violations.append(record)

    required = [{"phrase": phrase, "found": phrase in combined} for phrase in REQUIRED_PAPER_PHRASES]
    return {
        "paper_files": [str(path.relative_to(ROOT)) for path in files],
        "high_risk_mentions": mentions,
        "unscoped_violations": violations,
        "required_scope_phrases": required,
        "paper_scope_pass": not violations and all(item["found"] for item in required),
    }


def audit_fuse_source() -> dict[str, Any]:
    text = FUSE_SOURCE.read_text(encoding="utf-8", errors="replace") if FUSE_SOURCE.exists() else ""
    mentions = []
    for regex in (re.compile(r"io_uring", re.IGNORECASE), re.compile(r"eBPF", re.IGNORECASE)):
        for match in regex.finditer(text):
            ctx = context(text, match.start(), match.end())
            mentions.append(
                {
                    "line": text.count("\n", 0, match.start()) + 1,
                    "match": match.group(0),
                    "scoped": is_scoped(ctx),
                    "context": ctx,
                }
            )
    return {
        "path": str(FUSE_SOURCE.relative_to(ROOT)),
        "sha256": sha256_bytes(FUSE_SOURCE.read_bytes()) if FUSE_SOURCE.exists() else None,
        "mentions": mentions,
        "fuse_source_pass": FUSE_SOURCE.exists() and mentions and all(item["scoped"] for item in mentions),
        "scope": "Mounted FUSE source contains only an explicit no-io_uring/eBPF-path boundary.",
    }


def audit_trace_artifact() -> dict[str, Any]:
    trace_text = TRACE_OUT.read_text(encoding="utf-8", errors="replace") if TRACE_OUT.exists() else ""
    script_text = TRACE_SCRIPT.read_text(encoding="utf-8", errors="replace") if TRACE_SCRIPT.exists() else ""
    checks = [
        {
            "name": "trace_output_exists",
            "passed": TRACE_OUT.exists(),
            "detail": str(TRACE_OUT.relative_to(ROOT)),
        },
        {
            "name": "trace_output_is_histogram",
            "passed": "@usecs" in trace_text and "Attaching" in trace_text,
            "detail": "bpftrace histogram output only",
        },
        {
            "name": "tracepoint_script_targets_nvme_completion",
            "passed": "tracepoint:nvme:nvme_complete_rq" in script_text,
            "detail": str(TRACE_SCRIPT.relative_to(ROOT)),
        },
        {
            "name": "trace_output_not_mounted_fuse_path",
            "passed": "pqc_fuse" not in trace_text and "FUSE" not in trace_text,
            "detail": "no mounted-storage notification evidence in the retained trace output",
        },
    ]
    return {
        "path": str(TRACE_OUT.relative_to(ROOT)),
        "sha256": sha256_bytes(TRACE_OUT.read_bytes()) if TRACE_OUT.exists() else None,
        "checks": checks,
        "trace_artifact_pass": all(item["passed"] for item in checks),
        "scope": "Standalone nvme_complete_rq latency histogram; not a mounted FUSE notification path.",
    }


def audit_iouring_probe() -> dict[str, Any]:
    text = IOURING_PROBE.read_text(encoding="utf-8", errors="replace") if IOURING_PROBE.exists() else ""
    checks = [
        {
            "name": "probe_source_exists",
            "passed": IOURING_PROBE.exists(),
            "detail": str(IOURING_PROBE.relative_to(ROOT)),
        },
        {
            "name": "probe_warns_not_validated_bypass",
            "passed": "does not emit validated bypass evidence" in text,
            "detail": "prototype warning retained",
        },
        {
            "name": "placeholder_values_labeled",
            "passed": "Illustrative placeholder, not a measured bypass result" in text,
            "detail": "generated bypass rows are not retained as evidence",
        },
    ]
    return {
        "path": str(IOURING_PROBE.relative_to(ROOT)),
        "sha256": sha256_bytes(IOURING_PROBE.read_bytes()) if IOURING_PROBE.exists() else None,
        "checks": checks,
        "iouring_probe_pass": all(item["passed"] for item in checks),
        "scope": "Standalone prototype/simulation source; not a paper result.",
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# eBPF/io_uring Scope Audit",
        "",
        f"- Overall pass: `{str(result['overall_pass']).lower()}`",
        f"- Paper scope pass: `{str(result['paper_scan']['paper_scope_pass']).lower()}`",
        f"- FUSE source pass: `{str(result['fuse_source']['fuse_source_pass']).lower()}`",
        f"- Trace artifact pass: `{str(result['trace_artifact']['trace_artifact_pass']).lower()}`",
        f"- io_uring probe pass: `{str(result['iouring_probe']['iouring_probe_pass']).lower()}`",
        "",
        "## Paper Mentions",
    ]
    for mention in result["paper_scan"]["high_risk_mentions"]:
        lines.append(
            f"- `{mention['file']}:{mention['line']}` `{mention['pattern']}` "
            f"scoped=`{str(mention['scoped']).lower()}`: {mention['context']}"
        )
    if not result["paper_scan"]["high_risk_mentions"]:
        lines.append("- None.")
    lines.extend(["", "## Required Scope Phrases"])
    for item in result["paper_scan"]["required_scope_phrases"]:
        lines.append(f"- `{item['phrase']}` found=`{str(item['found']).lower()}`")
    lines.extend(["", "## Artifact Scope"])
    lines.append(f"- Trace output: `{result['trace_artifact']['path']}`")
    lines.append(f"- Trace scope: {result['trace_artifact']['scope']}")
    lines.append(f"- Probe scope: {result['iouring_probe']['scope']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    paper_scan = scan_paper()
    fuse_source = audit_fuse_source()
    trace_artifact = audit_trace_artifact()
    iouring_probe = audit_iouring_probe()
    result = {
        "artifact": "ebpf_iouring_scope_audit",
        "paper_scan": paper_scan,
        "fuse_source": fuse_source,
        "trace_artifact": trace_artifact,
        "iouring_probe": iouring_probe,
        "decision": "scoped_out_no_mounted_ebpf_iouring_completion_bypass",
    }
    result["overall_pass"] = (
        paper_scan["paper_scope_pass"]
        and fuse_source["fuse_source_pass"]
        and trace_artifact["trace_artifact_pass"]
        and iouring_probe["iouring_probe_pass"]
    )

    json_path = args.out / "ebpf_iouring_scope_audit.json"
    md_path = args.out / "ebpf_iouring_scope_audit.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(result, md_path)
    print(json.dumps({"overall_pass": result["overall_pass"], "out": str(json_path)}, sort_keys=True))
    return 0 if result["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
