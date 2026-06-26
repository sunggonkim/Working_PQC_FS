#!/usr/bin/env python3
"""Generate a CI report from the existing SQLite workload samples.

This script does not rerun the workload. It reads the checked-in raw sample
artifacts, computes bootstrap confidence intervals for the median latency, and
writes a small reproducibility report. The purpose is to make the workload
evidence easier to audit, not to manufacture new benchmark claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts" / "sqlite_ci_report"


def bootstrap_ci(samples: list[float], trials: int = 10000, alpha: float = 0.05, seed: int = 7) -> tuple[float, float]:
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
    lo_idx = int((alpha / 2) * trials)
    hi_idx = int((1 - alpha / 2) * trials) - 1
    lo_idx = max(0, min(trials - 1, lo_idx))
    hi_idx = max(0, min(trials - 1, hi_idx))
    return meds[lo_idx], meds[hi_idx]


def load_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = dict(row)
            if row.get("samples_ms"):
                parsed["samples_ms"] = json.loads(row["samples_ms"])
            rows.append(parsed)
    return rows


def summarize(path: Path) -> list[dict[str, object]]:
    out = []
    for row in load_rows(path):
        samples = row.get("samples_ms", [])
        if not isinstance(samples, list) or not samples:
            continue
        med = statistics.median(samples)
        lo, hi = bootstrap_ci(samples)
        out.append({
            "tier": row["tier"],
            "requested_mode": row["requested_mode"],
            "actual_mode": row["actual_mode"],
            "sync_mode": row["sync_mode"],
            "integrity_check": row["integrity_check"],
            "fallback_error": row["fallback_error"],
            "sample_count": len(samples),
            "median_ms": med,
            "median_ci95_low_ms": lo,
            "median_ci95_high_ms": hi,
            "p95_ms": row.get("p95_ms"),
            "samples_ms": samples,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    workload_report = {
        "source": "artifacts/motivation/sqlite_latency.csv",
        "description": "End-to-end SQLite workload on the current FUSE mount, preserved as raw samples.",
        "rows": summarize(ROOT / "artifacts" / "motivation" / "sqlite_latency.csv"),
    }
    contention_report = {
        "source": "artifacts/motivation/sqlite_contention_latency.csv",
        "description": "Concurrent reader/writer SQLite contention artifact, preserved as raw samples.",
        "rows": summarize(ROOT / "artifacts" / "motivation" / "sqlite_contention_latency.csv"),
    }

    report = {
        "generated_from": [
            str(ROOT / "artifacts" / "motivation" / "sqlite_latency.csv"),
            str(ROOT / "artifacts" / "motivation" / "sqlite_contention_latency.csv"),
        ],
        "bootstrap": {
            "trials": 10000,
            "alpha": 0.05,
            "seed": 7,
            "statistic": "median",
        },
        "workload": workload_report,
        "contention": contention_report,
    }

    (args.out_dir / "sqlite_ci_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# SQLite workload CI report",
        "",
        "This report is derived from the checked-in SQLite raw samples.",
        "",
        f"- bootstrap trials: {report['bootstrap']['trials']}",
        f"- statistic: {report['bootstrap']['statistic']}",
        f"- confidence level: {int((1 - report['bootstrap']['alpha']) * 100)}%",
        "",
        "## Workload",
    ]
    for row in workload_report["rows"]:
        md.append(
            f"- {row['tier']}: {row['requested_mode']}/{row['actual_mode']} {row['sync_mode']} "
            f"median {row['median_ms']:.6f} ms "
            f"(95% CI {row['median_ci95_low_ms']:.6f}--{row['median_ci95_high_ms']:.6f}), "
            f"n={row['sample_count']}, integrity={row['integrity_check']}"
        )
    md.append("")
    md.append("## Contention")
    for row in contention_report["rows"]:
        md.append(
            f"- {row['tier']}: {row['requested_mode']}/{row['actual_mode']} {row['sync_mode']} "
            f"median {row['median_ms']:.6f} ms "
            f"(95% CI {row['median_ci95_low_ms']:.6f}--{row['median_ci95_high_ms']:.6f}), "
            f"n={row['sample_count']}, integrity={row['integrity_check']}"
        )
    (args.out_dir / "sqlite_ci_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps({
        "out_dir": str(args.out_dir),
        "rows": len(workload_report["rows"]) + len(contention_report["rows"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
