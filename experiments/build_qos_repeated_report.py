#!/usr/bin/env python3
"""Build a conservative report from repeated QoS prototype runs.

This script only packages retained outputs from `run_qos_repeated.py`.
It does not infer PMU-backed QoS stability and does not invent missing traces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN = ROOT / "artifacts" / "validation" / "qos_repeated_run"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "qos_repeated_report"


def load_run(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists() or not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def summarize_trace(path: Path) -> dict:
    rows = load_jsonl(path)
    events: dict[str, int] = {}
    states = set()
    throttle = []
    gpu_power = []
    for row in rows:
        if "hysteresis_event" in row:
            event = str(row["hysteresis_event"])
            events[event] = events.get(event, 0) + 1
        if "hysteresis_state" in row:
            states.add(str(row["hysteresis_state"]))
        if "throttle" in row:
            throttle.append(int(row["throttle"]))
        if "gpu_power_mw" in row:
            gpu_power.append(float(row["gpu_power_mw"]))
    return {
        "sample_count": len(rows),
        "hysteresis_event_counts": events,
        "hysteresis_states": sorted(states),
        "throttle_high_fraction": (sum(throttle) / len(throttle)) if throttle else 0.0,
        "gpu_power_mw_median": statistics.median(gpu_power) if gpu_power else None,
    }


def load_summary(path: Path) -> dict:
    if not path.exists() or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_files = sorted(
        p for p in args.in_dir.glob("run_*.json")
        if p.stem.split("_")[-1].isdigit()
    )
    runs = [load_run(p) for p in run_files]
    run_summaries = []
    for r in runs:
        trace_file = r.get("trace_file")
        summary_file = r.get("summary_file")
        trace_path = Path(trace_file) if trace_file else Path("__missing_trace__")
        summary_path = Path(summary_file) if summary_file else Path("__missing_summary__")
        trace_summary = summarize_trace(trace_path)
        summary = load_summary(summary_path)
        run_summaries.append({
            "run": r.get("run"),
            "returncode": r.get("returncode"),
            "trace_file": str(trace_path),
            "summary_file": str(summary_path),
            "trace": trace_summary,
            "ai_latency_ms": summary.get("ai_latency_ms"),
            "enabled": summary.get("enabled"),
        })
    latencies = [
        float(r["ai_latency_ms"])
        for r in run_summaries
        if r.get("ai_latency_ms") is not None
    ]
    report = {
        "note": "Prototype repetition summary only; not a PMU/CUPTI-backed QoS result.",
        "input_dir": str(args.in_dir),
        "run_count": len(runs),
        "latency_ms_median": statistics.median(latencies) if latencies else None,
        "latency_ms_min": min(latencies) if latencies else None,
        "latency_ms_max": max(latencies) if latencies else None,
        "runs": run_summaries,
    }
    (args.out_dir / "qos_repeated_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# QoS repeated-run report",
        "",
        "This bundle packages repeated telemetry-prototype runs only.",
        "",
        f"- Input directory: `{args.in_dir}`",
        f"- Runs found: `{len(runs)}`",
        "",
        "## Runs",
        "",
    ]
    if latencies:
        md.append(f"- latency_ms_median: `{report['latency_ms_median']:.2f}`")
        md.append(f"- latency_ms_range: `{report['latency_ms_min']:.2f}`--`{report['latency_ms_max']:.2f}`")
        md.append("")
    for r in run_summaries:
        md.append(
            f"- run {r.get('run')}: returncode={r.get('returncode')}, "
            f"latency_ms={r.get('ai_latency_ms')}, "
            f"events={r['trace']['hysteresis_event_counts']}, "
            f"states={r['trace']['hysteresis_states']}, "
            f"trace={r.get('trace_file')}"
        )
    md.append("")
    md.append("This report does not claim PMU/CUPTI-backed stability.")
    (args.out_dir / "qos_repeated_report.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps({"out_dir": str(args.out_dir), "runs": len(runs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
