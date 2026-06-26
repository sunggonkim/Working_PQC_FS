#!/usr/bin/env python3
"""Run the retained TPM freshness-related harnesses behind one command.

This is an orchestration wrapper, not a proof of hardware-backed freshness.
It reuses the existing TPM freshness bundle and records the retained outputs
under a single bundle directory so the checklist can point at one command.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "tpm_only_bundle"


def run_cmd(cmd: list[str], out_dir: Path, name: str) -> dict:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (out_dir / f"{name}.stderr.txt").write_text(proc.stderr, encoding="utf-8")
    return {
        "name": name,
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": str(out_dir / f"{name}.stdout.txt"),
        "stderr": str(out_dir / f"{name}.stderr.txt"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    checks = []
    checks.append(run_cmd(["python3", "experiments/run_tpm_freshness_bundle.py"], out_dir, "tpm_freshness_bundle"))
    checks.append(run_cmd(["python3", "experiments/build_tpm_freshness_report.py"], out_dir, "tpm_freshness_report"))

    report = {
        "note": "TPM-only bundle wrapper; monotonic freshness update remains open evidence, while the recovery verdict is retained separately.",
        "checks": checks,
    }
    (out_dir / "tpm_only_bundle.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "checks": len(checks)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
