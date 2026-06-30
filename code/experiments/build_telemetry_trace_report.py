#!/usr/bin/env python3
"""Summarize the retained tegrastats telemetry traces.

This script derives a compact report from the existing JSONL trace artifacts.
It does not rerun telemetry or change the claim boundary; it only makes the
current hardware traces easier to audit.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "telemetry_trace_report"


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def transitions(values: list[int]) -> int:
    if not values:
        return 0
    count = 0
    prev = values[0]
    for v in values[1:]:
        if v != prev:
            count += 1
            prev = v
    return count


def event_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        event = str(row.get("hysteresis_event", "missing"))
        counts[event] = counts.get(event, 0) + 1
    return counts


def summarize(path: Path) -> dict[str, object]:
    rows = load_jsonl(path)
    throttle = [int(r.get("throttle", 0)) for r in rows]
    loads = [float(r["avg_cpu_load"]) for r in rows if "avg_cpu_load" in r]
    gpu_power = [float(r["gpu_power_mw"]) for r in rows if "gpu_power_mw" in r]
    hysteresis_rows = [r for r in rows if "hysteresis_state" in r]
    states = sorted({str(r.get("hysteresis_state")) for r in hysteresis_rows})
    out = {
        "source": str(path),
        "sample_count": len(rows),
        "throttle_high_fraction": sum(throttle) / len(throttle) if throttle else 0.0,
        "throttle_transitions": transitions(throttle),
        "hysteresis_sample_count": len(hysteresis_rows),
        "hysteresis_states": states,
        "hysteresis_event_counts": event_counts(hysteresis_rows),
        "avg_cpu_load_mean": statistics.mean(loads) if loads else None,
        "avg_cpu_load_median": statistics.median(loads) if loads else None,
        "gpu_power_mw_mean": statistics.mean(gpu_power) if gpu_power else None,
        "gpu_power_mw_median": statistics.median(gpu_power) if gpu_power else None,
        "first_lines": [r.get("raw") for r in rows[:3]],
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "tegra_qos_daemon": summarize(ROOT / "artifacts" / "validation" / "tegra_qos_daemon_trace.jsonl"),
        "run_qos_gpu": summarize(ROOT / "artifacts" / "validation" / "run_qos_gpu_trace.jsonl"),
    }
    (args.out_dir / "telemetry_trace_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# Telemetry trace report",
        "",
        "This report summarizes the retained tegrastats JSONL traces.",
        "",
    ]
    for name, row in report.items():
        md_lines.append(f"## {name}")
        md_lines.append(f"- sample_count: {row['sample_count']}")
        md_lines.append(f"- throttle_high_fraction: {row['throttle_high_fraction']:.3f}")
        md_lines.append(f"- throttle_transitions: {row['throttle_transitions']}")
        md_lines.append(f"- hysteresis_sample_count: {row['hysteresis_sample_count']}")
        md_lines.append(f"- hysteresis_states: {row['hysteresis_states']}")
        md_lines.append(f"- hysteresis_event_counts: {row['hysteresis_event_counts']}")
        if row["avg_cpu_load_mean"] is not None:
            md_lines.append(f"- avg_cpu_load_mean: {row['avg_cpu_load_mean']:.3f}")
            md_lines.append(f"- avg_cpu_load_median: {row['avg_cpu_load_median']:.3f}")
        if row["gpu_power_mw_mean"] is not None:
            md_lines.append(f"- gpu_power_mw_mean: {row['gpu_power_mw_mean']:.3f}")
            md_lines.append(f"- gpu_power_mw_median: {row['gpu_power_mw_median']:.3f}")
        md_lines.append("")
    (args.out_dir / "telemetry_trace_report.md").write_text("\n".join(md_lines).rstrip() + "\n", encoding="utf-8")

    print(json.dumps({"out_dir": str(args.out_dir), "entries": len(report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
