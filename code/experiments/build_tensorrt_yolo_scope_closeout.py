#!/usr/bin/env python3
"""Close the TensorRT/YOLO scope boundary without upgrading the paper claim.

The retained TensorRT/YOLO traces are useful interference and policy-direction
evidence.  They are not a same-run closed-loop foreground-inference recovery
experiment.  This closeout makes that boundary explicit and fails if paper or
README wording treats the retained traces as foreground AI QoS recovery.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "tensorrt_yolo_scope_closeout"

CI_REPORT = ROOT / "artifacts" / "reports" / "tensorrt_ci_report" / "tensorrt_ci_report.json"
TRACE_PATHS = [
    ROOT / "artifacts" / "results" / "motivation" / "tensorrt_interference.json",
    ROOT / "artifacts" / "results" / "qos" / "m6_tradeoff_yolov8_adaptive_tight" / "tensorrt_interference.json",
    ROOT / "artifacts" / "results" / "qos" / "m6_tradeoff_yolov8_elasticcontend" / "tensorrt_interference.json",
]
CUPTI_BRIDGE = ROOT / "artifacts" / "validation" / "qos_cupti_pm_fuse_bridge" / "qos_cupti_pm_fuse_bridge.json"

PAPER_FILES = sorted((ROOT / "Paper").glob("*.tex"))
README = ROOT / "README.md"

REQUIRED_MODES = {
    "yolov8:inference_only",
    "yolov8:cpu_only",
    "yolov8:gpu_only",
    "yolov8:adaptive",
}

REQUIRED_PAPER_PHRASES = [
    "TensorRT/AI p99 recovery",
    "not a uniqueness or AI-inference QoS claim",
    "foreground AI p99 recovery",
    "does not justify foreground AI-QoS recovery",
    "does not claim",
]

DANGEROUS_PATTERNS = [
    ("foreground_ai_qos_recovery", re.compile(r"foreground\s+AI[- ]QoS\s+recovery", re.I)),
    ("foreground_ai_p99_recovery", re.compile(r"foreground\s+AI\s+p99\s+recovery", re.I)),
    ("foreground_ai_tensorrt_recovery", re.compile(r"foreground\s+AI/TensorRT\s+recovery", re.I)),
    ("tensorrt_p99_recovery", re.compile(r"TensorRT\s+p99\s+recovery", re.I)),
    ("ai_qos_restoration", re.compile(r"AI[- ]QoS\s+restoration", re.I)),
    ("closed_loop_controller", re.compile(r"closed-loop\s+(?:QoS\s+)?controller", re.I)),
    ("fully_restores_qos", re.compile(r"fully\s+restores\s+QoS", re.I)),
    ("restores_inference_p99", re.compile(r"restores\s+inference\s+p99", re.I)),
]

NEGATION_TOKENS = [
    "not ",
    "not a ",
    "not full",
    "not foreground",
    "not tensor",
    "no ",
    "does not",
    "do not",
    "cannot",
    "without ",
    "out of",
    "does not establish",
    "unless ",
    "still requires",
    "requires separate",
    "future",
    "prototype",
    "trace evidence",
    "not yet",
    "non-claim",
    "아직",
    "아니다",
    "아니라",
    "않는다",
    "않았",
    "승격하지",
]


def path_meta(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_trace(path: Path) -> dict[str, Any]:
    meta = path_meta(path)
    if not path.exists():
        return {**meta, "complete": False, "modes": [], "mode_summaries": {}}

    rows = load_json(path)
    if not isinstance(rows, list):
        return {**meta, "complete": False, "modes": [], "mode_summaries": {}, "error": "not a list"}

    mode_summaries: dict[str, dict[str, Any]] = {}
    for mode in sorted({str(row.get("mode", "")) for row in rows}):
        mode_rows = [row for row in rows if str(row.get("mode", "")) == mode]
        p99s = [float(row["p99_ms"]) for row in mode_rows if "p99_ms" in row]
        samples = sum(len(row.get("samples_ms", [])) for row in mode_rows)
        mode_summaries[mode] = {
            "trials": len(mode_rows),
            "sample_count_total": samples,
            "p99_median_ms": statistics.median(p99s) if p99s else None,
        }

    modes = set(mode_summaries)
    return {
        **meta,
        "row_count": len(rows),
        "modes": sorted(modes),
        "mode_summaries": mode_summaries,
        "complete": REQUIRED_MODES.issubset(modes)
        and all(mode_summaries[m]["sample_count_total"] > 0 for m in REQUIRED_MODES),
    }


def summarize_ci_report(path: Path) -> dict[str, Any]:
    meta = path_meta(path)
    if not path.exists():
        return {**meta, "complete": False, "rows": []}
    data = load_json(path)
    rows = data.get("rows", []) if isinstance(data, dict) else []
    modes = sorted(str(row.get("mode", "")) for row in rows)
    return {
        **meta,
        "source": data.get("source") if isinstance(data, dict) else None,
        "row_count": len(rows),
        "modes": modes,
        "complete": REQUIRED_MODES.issubset(set(modes)),
        "rows": [
            {
                "mode": row.get("mode"),
                "trials": row.get("trials"),
                "sample_count_total": row.get("sample_count_total"),
                "median_of_trial_medians_ms": row.get("median_of_trial_medians_ms"),
            }
            for row in rows
        ],
    }


def summarize_cupti_bridge(path: Path) -> dict[str, Any]:
    meta = path_meta(path)
    if not path.exists():
        return {**meta, "complete": False}
    data = load_json(path)
    criteria = data.get("success_criteria", {}) if isinstance(data, dict) else {}
    return {
        **meta,
        "note": data.get("note"),
        "verified": data.get("verified") is True,
        "cupti_returncode": data.get("cupti_returncode"),
        "runtime_throttle_trace_rows": data.get("runtime_throttle_trace_rows"),
        "runtime_admission_trace_rows": data.get("runtime_admission_trace_rows"),
        "runtime_throttle_counts": data.get("runtime_throttle_counts"),
        "runtime_throttle_sleep_us_total": data.get("runtime_throttle_sleep_us_total"),
        "success_criteria": criteria,
        "complete": (
            data.get("verified") is True
            and data.get("cupti_returncode") == 0
            and all(criteria.get(key) is True for key in (
                "cupti_samples_present",
                "mounted_fuse_throttle_present",
                "writer_harness_throttle_disabled",
                "cupti_returncode_zero",
            ))
        ),
        "scope": "same-run CUPTI PM to mounted-FUSE throttle wiring; not foreground inference p99 recovery",
    }


def guarded_context(text: str, start: int, end: int) -> bool:
    window = text[max(0, start - 520): min(len(text), end + 220)].lower()
    return any(token in window for token in NEGATION_TOKENS)


def scan_claims(paths: list[Path]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    unguarded: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        line_offsets: list[int] = []
        pos = 0
        for line in text.splitlines(keepends=True):
            line_offsets.append(pos)
            pos += len(line)
        for name, pattern in DANGEROUS_PATTERNS:
            for match in pattern.finditer(text):
                line_no = 1
                for idx, offset in enumerate(line_offsets):
                    if offset > match.start():
                        break
                    line_no = idx + 1
                guarded = guarded_context(text, match.start(), match.end())
                item = {
                    "file": str(path.relative_to(ROOT)),
                    "line": line_no,
                    "pattern": name,
                    "match": match.group(0),
                    "guarded": guarded,
                    "line_text": text.splitlines()[line_no - 1].strip() if text.splitlines() else "",
                }
                candidates.append(item)
                if not guarded:
                    unguarded.append(item)
    return {
        "candidate_count": len(candidates),
        "unguarded_count": len(unguarded),
        "candidates": candidates,
        "unguarded": unguarded,
        "passes": not unguarded,
    }


def paper_phrase_checks() -> dict[str, bool]:
    paper_text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in PAPER_FILES)
    return {phrase: phrase in paper_text for phrase in REQUIRED_PAPER_PHRASES}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    traces = [summarize_trace(path) for path in TRACE_PATHS]
    ci_report = summarize_ci_report(CI_REPORT)
    cupti_bridge = summarize_cupti_bridge(CUPTI_BRIDGE)
    claim_scan = scan_claims(PAPER_FILES + [README] + sorted((ROOT / "docs").glob("**/*.md")))
    phrase_checks = paper_phrase_checks()

    conditions = {
        "ci_report_complete": ci_report["complete"],
        "primary_trace_complete": traces[0]["complete"] if traces else False,
        "all_retained_traces_complete": all(trace["complete"] for trace in traces),
        "cupti_bridge_complete": cupti_bridge["complete"],
        "paper_scope_phrases_present": all(phrase_checks.values()),
        "claim_scan_passes": claim_scan["passes"],
    }
    overall_pass = all(conditions.values())

    report = {
        "schema": "aegisq.tensorrt_yolo_scope_closeout.v1",
        "overall_pass": overall_pass,
        "verdict": "trace-scoped-non-claim" if overall_pass else "blocked",
        "conditions": conditions,
        "allowed_claims": [
            "TensorRT/YOLO interference traces show foreground inference can suffer under GPU/storage pressure.",
            "Adaptive/prototype traces may be described as policy-direction evidence.",
            "CUPTI PM samples can drive the mounted FUSE throttle path in the retained bridge run.",
        ],
        "forbidden_claims": [
            "foreground AI QoS recovery",
            "foreground AI or TensorRT p99 recovery",
            "closed-loop inference QoS controller",
            "full AI-QoS restoration",
            "uniqueness or deployment claim based on TensorRT traces",
        ],
        "ci_report": ci_report,
        "traces": traces,
        "cupti_bridge": cupti_bridge,
        "paper_phrase_checks": phrase_checks,
        "claim_scan": claim_scan,
    }

    json_path = args.out_dir / "tensorrt_yolo_scope_closeout.json"
    md_path = args.out_dir / "tensorrt_yolo_scope_closeout.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# TensorRT/YOLO Scope Closeout",
        "",
        f"- overall_pass: `{overall_pass}`",
        f"- verdict: `{report['verdict']}`",
        "",
        "## Conditions",
    ]
    for key, value in conditions.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Retained Trace Boundary"])
    for trace in traces:
        lines.append(f"- `{trace['path']}`: complete=`{trace['complete']}`, modes={', '.join(trace.get('modes', []))}")
        for mode, summary in trace.get("mode_summaries", {}).items():
            lines.append(
                f"  - {mode}: trials={summary['trials']}, samples={summary['sample_count_total']}, "
                f"median p99={summary['p99_median_ms']} ms"
            )
    lines.extend([
        "",
        "## Interpretation",
        "- The TensorRT/YOLO rows remain interference/prototype evidence.",
        "- The CUPTI bridge is same-run telemetry-to-mounted-FUSE-throttle wiring evidence.",
        "- The closeout does not permit a foreground AI/TensorRT p99 recovery claim.",
        "",
        "## Claim Scan",
        f"- candidates: `{claim_scan['candidate_count']}`",
        f"- unguarded: `{claim_scan['unguarded_count']}`",
    ])
    for item in claim_scan["unguarded"][:20]:
        lines.append(f"  - `{item['file']}:{item['line']}` {item['pattern']}: {item['line_text']}")
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    print(json.dumps({
        "overall_pass": overall_pass,
        "out_dir": str(args.out_dir),
        "conditions": conditions,
        "unguarded_claims": claim_scan["unguarded_count"],
    }, indent=2))
    if args.require_complete and not overall_pass:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
