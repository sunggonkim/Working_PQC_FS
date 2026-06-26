#!/usr/bin/env python3
"""Package the retained hardware-backed crash/replay rows as a recovery verdict.

This report is conservative. It packages the retained hardware-backend rows
from the E8 crash/replay matrix alongside the existing hardware anchor round-
trip evidence so the mount/recovery behavior is explicit in one retained
artifact. The separate replay-after-advance freshness result lives in
artifacts/validation/tpm_monotonic_replay/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "tpm_recovery_verdict"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    crash_summary_path = ROOT / "artifacts" / "crash_replay_e8_test_summary.json"
    crash_matrix_path = ROOT / "artifacts" / "crash_replay_e8_test_matrix.json"
    anchor_path = ROOT / "artifacts" / "anchor_refresh" / "hardware_anchor_latency.json"
    if crash_summary_path.exists():
        crash_summary = json.loads(crash_summary_path.read_text(encoding="utf-8"))
    else:
        crash_summary = {"rows": [], "skipped": ["missing crash replay summary"]}
    if crash_matrix_path.exists():
        crash_matrix = json.loads(crash_matrix_path.read_text(encoding="utf-8"))
    else:
        crash_matrix = {"rows": [], "skipped": ["missing crash replay matrix"]}
    if anchor_path.exists():
        anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    else:
        anchor = {"rows": [], "skipped": ["missing hardware anchor latency"]}

    hardware_rows = [row for row in crash_matrix.get("rows", []) if row.get("backend") == "hardware"]
    hardware_summary_rows = [row for row in crash_summary.get("rows", []) if row.get("backend") == "hardware"]
    verdict = {
        "note": "Recovery verdict artifact only; combine it with the separate monotonic replay artifact for freshness claims.",
        "inputs": {
            "crash_replay_e8_summary": str(crash_summary_path),
            "crash_replay_e8_matrix": str(crash_matrix_path),
            "hardware_anchor_latency": str(anchor_path),
        },
        "hardware_rows": hardware_rows,
        "hardware_summary_rows": hardware_summary_rows,
        "hardware_anchor_latency": anchor,
        "verdict": {
            "hardware_backend_present": bool(hardware_rows),
            "hardware_backend_fail_closed_count": sum(1 for row in hardware_rows if row.get("mode") == "fail_closed"),
            "hardware_backend_rollback_accept_count": sum(1 for row in hardware_rows if row.get("mode") == "rollback_visible"),
            "hardware_backend_success_rate_min": min((row.get("success_rate", 0.0) for row in hardware_summary_rows), default=None),
            "hardware_backend_success_rate_max": max((row.get("success_rate", 0.0) for row in hardware_summary_rows), default=None),
        },
    }

    (args.out_dir / "tpm_recovery_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    md = [
        "# TPM recovery verdict",
        "",
        "This artifact packages the retained hardware-backend crash/replay rows together with the hardware anchor round-trip measurement.",
        "",
        f"- crash replay summary: `{crash_summary_path}`",
        f"- crash replay matrix: `{crash_matrix_path}`",
        f"- hardware anchor latency: `{anchor_path}`",
        f"- hardware backend rows present: `{verdict['verdict']['hardware_backend_present']}`",
        f"- hardware backend fail-closed count: `{verdict['verdict']['hardware_backend_fail_closed_count']}`",
        f"- hardware backend rollback-accept count: `{verdict['verdict']['hardware_backend_rollback_accept_count']}`",
        f"- hardware backend success-rate range: `{verdict['verdict']['hardware_backend_success_rate_min']}` .. `{verdict['verdict']['hardware_backend_success_rate_max']}`",
        "",
        "This artifact should be read together with the separate monotonic replay result; it still does not close the combined durability claim.",
    ]
    (args.out_dir / "tpm_recovery_verdict.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "hardware_rows": len(hardware_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
