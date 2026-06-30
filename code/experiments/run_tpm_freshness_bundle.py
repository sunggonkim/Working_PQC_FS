#!/usr/bin/env python3
"""Run the existing TPM freshness-related harnesses as one conservative bundle.

This does not invent hardware-backed freshness proof.  It reruns the current
PCR-policy probe, anchor latency, monotonic replay, analytical freshness-window,
and crash-replay helpers and records where their outputs land.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "tpm_freshness_bundle"


def run_cmd(cmd: list[str], out_dir: Path, name: str, *, use_sudo: bool = False) -> dict:
    if use_sudo:
        password = os.environ.get("PQC_SUDO_PASSWORD")
        if not password:
            raise RuntimeError("PQC_SUDO_PASSWORD is required for sudo bundle steps")
        proc = subprocess.run(
            ["sudo", "-S", "-p", "", "env", f"PQC_SUDO_PASSWORD={password}", *cmd],
            cwd=ROOT,
            text=True,
            capture_output=True,
            input=password + "\n",
        )
    else:
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
    checks.append(run_cmd(["python3", "code/experiments/run_tpm_pcr_policy_probe.py"], out_dir, "tpm_pcr_policy_probe"))
    checks.append(run_cmd(["python3", "code/experiments/benchmark_anchor_latency.py"], out_dir, "anchor_latency"))
    checks.append(run_cmd(["python3", "code/experiments/run_tpm_monotonic_replay.py"], out_dir, "tpm_monotonic_replay", use_sudo=True))
    checks.append(run_cmd(["python3", "code/experiments/run_power_fail_test.py"], out_dir, "freshness_window_model"))
    checks.append(run_cmd(["python3", "code/experiments/run_crash_replay_e8.py"], out_dir, "crash_replay_e8"))
    checks.append(run_cmd(["python3", "code/experiments/build_tpm_recovery_verdict.py"], out_dir, "tpm_recovery_verdict"))

    report = {
        "note": (
            "TPM freshness bundle only; PCR-policy probe is transient and this "
            "is not a full hardware-backed freshness proof. The freshness-window "
            "row is analytical and is not power-loss, kernel-crash, or drive-cache evidence."
        ),
        "checks": checks,
    }
    (out_dir / "tpm_freshness_bundle.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "checks": len(checks)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
