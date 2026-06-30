#!/usr/bin/env python3
"""Derive a CI report from the retained M3 QoS raw rows.

The script does not rerun the TensorRT / secure-I/O workload. It only
summarizes the checked-in per-trial rows into a small report with bootstrap
confidence intervals for the median of each metric, grouped by configuration.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts" / "reports" / "m3_qos_ci_report"


def bootstrap_ci(samples: list[float], trials: int = 10000, alpha: float = 0.05, seed: int = 11) -> tuple[float, float]:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads((ROOT / "artifacts" / "m3_qos" / "m3_qos_results.json").read_text())
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[row["Configuration"]].append(row)

    metrics = ["LLM_TTFT_ms", "LLM_TPS", "YOLO_p99_ms", "SQLite_latency_ms"]
    active_metrics = []
    for metric in metrics:
        values = [float(row[metric]) for row in rows]
        if any(abs(v) > 1e-12 for v in values):
            active_metrics.append(metric)
    report_rows = []
    for config, samples in sorted(grouped.items()):
        entry = {"Configuration": config, "trials": len(samples)}
        for metric in active_metrics:
            values = [float(sample[metric]) for sample in samples]
            entry[metric] = {
                "median": statistics.median(values),
                "ci95_low": bootstrap_ci(values)[0],
                "ci95_high": bootstrap_ci(values)[1],
                "samples": values,
            }
        report_rows.append(entry)

    report = {
        "source": "artifacts/results/qos/m3_qos/m3_qos_results.json",
        "bootstrap": {
            "trials": 10000,
            "alpha": 0.05,
            "statistic": "median",
            "seed": 11,
        },
        "metrics": active_metrics,
        "rows": report_rows,
    }

    (args.out_dir / "m3_qos_ci_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# M3 QoS CI report",
        "",
        "This report is derived from the retained per-trial M3 QoS rows.",
        "",
    ]
    for row in report_rows:
        lines.append(f"## {row['Configuration']}")
        lines.append(f"- trials: {row['trials']}")
        for metric in active_metrics:
            m = row[metric]
            lines.append(
                f"- {metric}: median {m['median']:.6f} "
                f"(95% CI {m['ci95_low']:.6f}--{m['ci95_high']:.6f})"
            )
        lines.append("")
    (args.out_dir / "m3_qos_ci_report.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    print(json.dumps({"out_dir": str(args.out_dir), "rows": len(report_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
