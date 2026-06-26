#!/usr/bin/env python3
"""Repeat the UMA / storage-DMA probe and keep a conservative run bundle."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "uma_storage_dma_repeated"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--nvme-device", default="/dev/nvme0n1")
    ap.add_argument("--profile-cmd", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for idx in range(args.runs):
        cmd = [
            "python3",
            "experiments/run_uma_storage_dma_probe.py",
            "--nvme-device",
            args.nvme_device,
            "--out-dir",
            str(out_dir / f"probe_run_{idx + 1}"),
        ]
        if args.profile_cmd:
            cmd.extend(["--profile-cmd", args.profile_cmd])
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        record = {
            "run": idx + 1,
            "command": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "probe_dir": str(out_dir / f"probe_run_{idx + 1}"),
        }
        (out_dir / f"run_{idx + 1}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        records.append(record)

    report = {
        "runs": args.runs,
        "note": "Repeat/probe bundle only; not evidence of verified NVMe-to-UVM DMA semantics.",
        "records": [
            {
                "run": r["run"],
                "returncode": r["returncode"],
                "probe_dir": r["probe_dir"],
            }
            for r in records
        ],
    }
    (out_dir / "uma_storage_dma_repeated.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "runs": args.runs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
