#!/usr/bin/env python3
"""Probe harness for the UMA / storage-DMA evidence chain.

This script is intentionally conservative:
  - It can verify the implemented pinning and raw-O_DIRECT smoke paths.
  - It can wrap an optional external profiler command (Nsight Systems / CUPTI /
    other user-provided tooling) without inventing counter values.
  - It does not claim storage-DMA semantics by itself; it only packages the
    evidence needed to decide whether the stronger claim is justified.

Typical usage:

    python3 code/experiments/run_uma_storage_dma_probe.py \
      --nvme-device /dev/nvme0n1 \
      --profile-cmd "nsys profile --trace=uvm --output=artifacts/uma_probe/nsys <cmd...>"

The script emits a JSON report with command lines, return codes, and stdout/stderr
paths for later review.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build"
DEFAULT_OUT = ROOT / "artifacts" / "probes" / "uma_storage_dma_probe"


@dataclass
class CmdResult:
    name: str
    command: list[str]
    returncode: int
    stdout_path: str
    stderr_path: str


def run_cmd(name: str, command: list[str], out_dir: Path) -> CmdResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / f"{name}.stdout.txt"
    stderr_path = out_dir / f"{name}.stderr.txt"
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    return CmdResult(
        name=name,
        command=command,
        returncode=proc.returncode,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nvme-device", default="/dev/nvme0n1")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--profile-cmd",
        default=None,
        help=(
            "Optional external profiler wrapper command to run as a shell-like "
            "string. Use this to attach Nsight Systems or CUPTI-backed tooling "
            "around the same smoke command."
        ),
    )
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "scope": "Evidence harness only; does not itself prove NVMe-to-UVM DMA semantics.",
        "platform_hint": {
            "kernel": os.uname().release,
            "machine": os.uname().machine,
        },
        "checks": [],
    }

    repro = BUILD / "repro_malloc_register"
    io_uring_uvm = BUILD / "io_uring_uvm"

    if repro.exists():
        report["checks"].append(
            asdict(run_cmd("repro_malloc_register", [str(repro)], out_dir))
        )
    else:
        report["checks"].append(
            {
                "name": "repro_malloc_register",
                "command": [str(repro)],
                "returncode": 127,
                "stdout_path": "",
                "stderr_path": "",
                "missing": "build/repro_malloc_register not found; build the project first.",
            }
        )

    if io_uring_uvm.exists():
        report["checks"].append(
            asdict(run_cmd("io_uring_uvm_nvme", [str(io_uring_uvm), args.nvme_device], out_dir))
        )
    else:
        report["checks"].append(
            {
                "name": "io_uring_uvm_nvme",
                "command": [str(io_uring_uvm), args.nvme_device],
                "returncode": 127,
                "stdout_path": "",
                "stderr_path": "",
                "missing": "build/io_uring_uvm not found; build the project first.",
            }
        )

    if args.profile_cmd:
        # The user explicitly provides the profiler command. We keep it generic
        # and conservative, so the harness can be used with Nsight Systems or
        # any equivalent tool that the environment supports.
        prof_cmd = shlex.split(args.profile_cmd)
        report["checks"].append(asdict(run_cmd("profile_wrapper", prof_cmd, out_dir)))

    report_path = out_dir / "uma_storage_dma_probe.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "checks": len(report["checks"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
