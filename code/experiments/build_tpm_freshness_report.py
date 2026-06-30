#!/usr/bin/env python3
"""Package retained TPM freshness-related outputs into a conservative report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = ROOT / "artifacts" / "validation" / "tpm_freshness_bundle"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "tpm_freshness_report"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = args.in_dir / "tpm_freshness_bundle.json"
    if bundle_path.exists():
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    else:
        bundle = {"note": "bundle missing; run experiments/run_tpm_freshness_bundle.py first", "checks": []}
    pcr_probe_path = ROOT / "artifacts" / "validation" / "tpm_pcr_policy_probe" / "tpm_pcr_policy_probe.json"
    if pcr_probe_path.exists():
        pcr_probe = json.loads(pcr_probe_path.read_text(encoding="utf-8"))
    else:
        pcr_probe = None
    recovery_verdict_path = ROOT / "artifacts" / "validation" / "tpm_recovery_verdict" / "tpm_recovery_verdict.json"
    monotonic_replay_path = ROOT / "artifacts" / "validation" / "tpm_monotonic_replay" / "tpm_monotonic_replay.json"
    if monotonic_replay_path.exists():
        monotonic_replay = json.loads(monotonic_replay_path.read_text(encoding="utf-8"))
    else:
        monotonic_replay = None
    if recovery_verdict_path.exists():
        recovery_verdict = json.loads(recovery_verdict_path.read_text(encoding="utf-8"))
    else:
        recovery_verdict = None
    report = {
        "note": "Report packaging only; PCR-policy probe is transient and the recovery verdict remains conservative.",
        "bundle": bundle,
        "pcr_policy_probe": {
            "path": str(pcr_probe_path),
            "present": pcr_probe_path.exists(),
            "drift_rejected": (pcr_probe or {}).get("results", {}).get("drift_rejected"),
        },
        "monotonic_replay": {
            "path": str(monotonic_replay_path),
            "present": monotonic_replay_path.exists(),
            "mode": (((monotonic_replay or {}).get("result") or {}).get("replay_result") or {}).get("mode"),
        },
        "recovery_verdict": {
            "path": str(recovery_verdict_path),
            "present": recovery_verdict_path.exists(),
            "hardware_backend_present": (recovery_verdict or {}).get("verdict", {}).get("hardware_backend_present"),
            "hardware_backend_fail_closed_count": (recovery_verdict or {}).get("verdict", {}).get("hardware_backend_fail_closed_count"),
        },
    }
    (args.out_dir / "tpm_freshness_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# TPM freshness report",
        "",
        "This bundle packages the retained PCR-policy, anchor, crash-replay, and analytical freshness-window helpers.",
        "",
        f"- Input directory: `{args.in_dir}`",
        f"- Checks: `{len(bundle.get('checks', []))}`",
        f"- Bundle present: `{bundle_path.exists()}`",
        "",
        f"- PCR policy probe present: `{pcr_probe_path.exists()}`",
        f"- PCR drift rejected: `{(pcr_probe or {}).get('results', {}).get('drift_rejected')}`",
        f"- Monotonic replay present: `{monotonic_replay_path.exists()}`",
        f"- Monotonic replay mode: `{(((monotonic_replay or {}).get('result') or {}).get('replay_result') or {}).get('mode')}`",
        f"- Recovery verdict present: `{recovery_verdict_path.exists()}`",
        f"- Hardware backend rows present: `{(recovery_verdict or {}).get('verdict', {}).get('hardware_backend_present')}`",
        f"- Hardware backend fail-closed count: `{(recovery_verdict or {}).get('verdict', {}).get('hardware_backend_fail_closed_count')}`",
        "",
        "This report does not claim persistent filesystem PCR sealing, hardware-backed freshness, physical power-loss safety, kernel-crash safety, or drive-cache safety.",
    ]
    (args.out_dir / "tpm_freshness_report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
