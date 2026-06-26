#!/usr/bin/env python3
"""Summarize the retained UVM proxy and semantic-gap artifacts.

This report is intentionally conservative. It does not promote the proxy into
hardware-counter evidence. It merely packages the current measured proxy and
the emulated semantic-gap projection into a compact audit artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts" / "motivation" / "uvm_proxy_report"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    um_counters = json.loads((ROOT / "artifacts" / "motivation" / "um_counters.json").read_text())
    semantic_gap = json.loads((ROOT / "artifacts" / "motivation" / "semantic_gap.json").read_text())

    report = {
        "um_counters": um_counters,
        "semantic_gap": semantic_gap,
        "interpretation": {
            "um_counters": "managed-memory smoke evidence only; not a storage-DMA proof",
            "semantic_gap": "emulated projection because eBPF/trace collection was unavailable",
        },
    }

    (args.out_dir / "uvm_proxy_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# UVM proxy report",
        "",
        "This report packages the existing managed-memory smoke evidence and the emulated semantic-gap projection.",
        "",
        "## Managed-memory smoke",
    ]
    for k, v in um_counters[0].items():
        md.append(f"- {k}: {v}")
    md.append("")
    md.append("## Semantic gap projection")
    for row in semantic_gap:
        md.append(
            f"- {row['workload']}: page_faults={row['page_faults']}, page_migrations={row['page_migrations']}, "
            f"read_ahead_waste_pct={row['read_ahead_waste_pct']}, provenance={row['data_provenance']}"
        )
    md.append("")
    md.append("## Interpretation")
    md.append("- managed-memory smoke: proxy evidence only, not DMA or migration suppression proof")
    md.append("- semantic-gap projection: emulated because the trace backend was unavailable")
    (args.out_dir / "uvm_proxy_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps({"out_dir": str(args.out_dir), "sections": 2}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
