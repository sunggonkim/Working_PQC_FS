#!/usr/bin/env python3
"""Derive a CI report from the retained TensorRT/YOLO interference traces.

This script does not rerun the workload. It reads the checked-in per-trial raw
samples, computes bootstrap confidence intervals for the median latency of each
configuration, and writes a small audit report. The intent is to make the
retained evidence easier to inspect without inventing new results.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts" / "reports" / "tensorrt_ci_report"


def bootstrap_ci(samples: list[float], trials: int = 10000, alpha: float = 0.05, seed: int = 19) -> tuple[float, float]:
    if not samples:
        raise ValueError("empty samples")
    if len(samples) == 1:
        return samples[0], samples[0]
    rng = random.Random(seed)
    meds = []
    n = len(samples)
    for _ in range(trials):
        resampled = [samples[rng.randrange(n)] for _ in range(n)]
        meds.append(statistics.median(resampled))
    meds.sort()
    lo = max(0, min(trials - 1, int((alpha / 2) * trials)))
    hi = max(0, min(trials - 1, int((1 - alpha / 2) * trials) - 1))
    return meds[lo], meds[hi]


def mode_label(mode: str) -> str:
    return mode.split(":", 1)[1] if ":" in mode else mode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads((ROOT / "artifacts" / "motivation" / "tensorrt_interference.json").read_text())
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["mode"])].append(row)

    report_rows = []
    for mode, trials in sorted(grouped.items()):
        trial_summaries = []
        trial_medians = []
        pooled_samples: list[float] = []
        for trial in sorted(trials, key=lambda r: int(r["trial"])):
            samples = [float(x) for x in trial.get("samples_ms", [])]
            if not samples:
                continue
            pooled_samples.extend(samples)
            med = statistics.median(samples)
            lo, hi = bootstrap_ci(samples)
            trial_medians.append(med)
            trial_summaries.append({
                "trial": int(trial["trial"]),
                "duration_s": float(trial["duration_s"]),
                "engine": str(trial["engine"]),
                "count": int(trial["count"]),
                "sample_count": len(samples),
                "median_ms": med,
                "median_ci95_low_ms": lo,
                "median_ci95_high_ms": hi,
                "p95_ms": float(trial["p95_ms"]),
                "p99_ms": float(trial["p99_ms"]),
            })

        if not trial_medians:
            continue

        median_of_trial_medians = statistics.median(trial_medians)
        ci_lo, ci_hi = bootstrap_ci(trial_medians)
        pooled_median = statistics.median(pooled_samples) if pooled_samples else None

        report_rows.append({
            "mode": mode,
            "mode_label": mode_label(mode),
            "trials": len(trial_summaries),
            "engine": trial_summaries[0]["engine"],
            "duration_s": trial_summaries[0]["duration_s"],
            "sample_count_total": len(pooled_samples),
            "median_of_trial_medians_ms": median_of_trial_medians,
            "median_ci95_low_ms": ci_lo,
            "median_ci95_high_ms": ci_hi,
            "pooled_median_ms": pooled_median,
            "trial_summaries": trial_summaries,
        })

    report = {
        "source": "artifacts/results/motivation/tensorrt_interference.json",
        "bootstrap": {
            "trials": 10000,
            "alpha": 0.05,
            "statistic": "median",
            "seed": 19,
        },
        "rows": report_rows,
    }

    (args.out_dir / "tensorrt_ci_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# TensorRT/YOLO CI report",
        "",
        "This report is derived from the retained per-trial TensorRT/YOLO interference traces.",
        "",
        f"- bootstrap trials: {report['bootstrap']['trials']}",
        f"- statistic: {report['bootstrap']['statistic']}",
        f"- confidence level: {int((1 - report['bootstrap']['alpha']) * 100)}%",
        "",
    ]
    for row in report_rows:
        lines.append(f"## {row['mode']}")
        lines.append(f"- engine: {row['engine']}")
        lines.append(f"- duration_s: {row['duration_s']}")
        lines.append(f"- trials: {row['trials']}")
        lines.append(f"- pooled sample count: {row['sample_count_total']}")
        lines.append(
            f"- median of trial medians: {row['median_of_trial_medians_ms']:.6f} ms "
            f"(95% CI {row['median_ci95_low_ms']:.6f}--{row['median_ci95_high_ms']:.6f})"
        )
        if row["pooled_median_ms"] is not None:
            lines.append(f"- pooled median: {row['pooled_median_ms']:.6f} ms")
        lines.append("- trial summaries:")
        for trial in row["trial_summaries"]:
            lines.append(
                f"  - trial {trial['trial']}: median {trial['median_ms']:.6f} ms "
                f"(95% CI {trial['median_ci95_low_ms']:.6f}--{trial['median_ci95_high_ms']:.6f}), "
                f"p95 {trial['p95_ms']:.6f} ms, p99 {trial['p99_ms']:.6f} ms, n={trial['sample_count']}"
            )
        lines.append("")
    (args.out_dir / "tensorrt_ci_report.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    print(json.dumps({"out_dir": str(args.out_dir), "rows": len(report_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
