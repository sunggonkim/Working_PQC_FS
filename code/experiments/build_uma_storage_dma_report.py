#!/usr/bin/env python3
"""Package repeated UMA / storage-DMA probe outputs into a conservative report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = ROOT / "artifacts" / "validation" / "uma_storage_dma_repeated"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "uma_storage_dma_report"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_files = sorted(args.in_dir.glob("run_*.json"))
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in run_files]
    report = {
        "note": "Probe-report packaging only; not a proof of NVMe-to-UVM DMA semantics.",
        "input_dir": str(args.in_dir),
        "run_count": len(runs),
        "runs": [
            {
                "run": r.get("run"),
                "returncode": r.get("returncode"),
                "probe_dir": r.get("probe_dir"),
            }
            for r in runs
        ],
    }
    (args.out_dir / "uma_storage_dma_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# UMA / storage-DMA report",
        "",
        "This bundle packages repeated probe runs only.",
        "",
        f"- Input directory: `{args.in_dir}`",
        f"- Runs found: `{len(runs)}`",
        "",
        "## Runs",
        "",
    ]
    for r in runs:
        md.append(f"- run {r.get('run')}: returncode={r.get('returncode')}, probe_dir={r.get('probe_dir')}")
    md.append("")
    md.append("This report does not claim verified NVMe-to-UVM DMA semantics.")
    (args.out_dir / "uma_storage_dma_report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "runs": len(runs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
