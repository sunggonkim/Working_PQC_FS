#!/usr/bin/env python3
"""Summarize the current retained platform evidence.

This report is intentionally conservative. It records the current platform
manifest and makes explicit that no second hardware / driver matrix is retained
in the repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts" / "reports" / "platform_inventory_report"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    micro = json.loads((ROOT / "artifacts" / "validation" / "microbench" / "summary.json").read_text())
    manifest = micro.get("manifest", {})
    gpu_headers = [
        line.strip()
        for line in (ROOT / "artifacts" / "validation" / "gpu_mlkem.csv").read_text().splitlines()
        if line.startswith("#")
    ]
    report = {
        "current_platform": {
            "device_model": manifest.get("device_model"),
            "kernel": manifest.get("kernel"),
            "cuda": manifest.get("nvcc"),
            "notes": "These retained artifacts all come from the same Thor platform; no second hardware / driver matrix is retained in the repository.",
        },
        "raw_headers": gpu_headers,
        "gap": "second hardware / driver matrix remains open",
        "second_platform_contract": {
            "minimum_commands": [
                "python3 experiments/run_verified_microbench.py --runs 3 --out artifacts/validation/microbench",
                "python3 experiments/run_m5_admission_sweep.py",
                "python3 experiments/run_app_recovery_bundle.py",
                "PQC_SUDO_PASSWORD=<password> python3 experiments/run_combined_durability_bundle.py --out-dir artifacts/validation/combined_durability_bundle",
                "PQC_SUDO_PASSWORD=<password> python3 experiments/run_sqlite_syscall_crash_tpm.py --out-dir artifacts/validation/sqlite_syscall_crash_tpm --when 1 2 3",
            ],
            "required_output_schema": [
                "platform manifest with device model, kernel, CUDA compiler, and driver identifiers",
                "microbenchmark summary with per-run medians and observed range",
                "queue-depth / slack trace rows for admission sweeps",
                "SQLite selected-boundary oracle verdicts and retained crash-audit bundle",
                "combined SQLite/dbm.dumb stale-snapshot replay verdicts",
                "SQLite syscall-exact app-crash verdicts",
            ],
            "required_platform_fields": [
                "device model",
                "kernel",
                "CUDA compiler / runtime",
                "driver / firmware identifiers",
                "available accelerator notes",
            ],
            "status": "placeholder only; no second-platform raw outputs are retained",
        },
    }

    (args.out_dir / "platform_inventory_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    placeholder_manifest = {
        "status": "no second-platform raw outputs retained",
        "platform": None,
        "commands": report["second_platform_contract"]["minimum_commands"],
        "schema": report["second_platform_contract"]["required_output_schema"],
        "required_platform_fields": report["second_platform_contract"]["required_platform_fields"],
    }
    (args.out_dir / "second_platform_placeholder_manifest.json").write_text(
        json.dumps(placeholder_manifest, indent=2),
        encoding="utf-8",
    )
    md = [
        "# Platform inventory report",
        "",
        "This report records the current retained platform evidence.",
        "",
        f"- device model: {report['current_platform']['device_model']}",
        f"- kernel: {report['current_platform']['kernel']}",
        f"- cuda: {report['current_platform']['cuda'].splitlines()[-1] if report['current_platform']['cuda'] else None}",
        "- no second hardware / driver matrix is retained in the repository",
        "",
        "## Second-platform contract",
        "- minimum commands:",
        "  - `python3 experiments/run_verified_microbench.py --runs 3 --out artifacts/validation/microbench`",
        "  - `python3 experiments/run_m5_admission_sweep.py`",
        "  - `python3 experiments/run_app_recovery_bundle.py`",
        "  - `PQC_SUDO_PASSWORD=<password> python3 experiments/run_combined_durability_bundle.py --out-dir artifacts/validation/combined_durability_bundle`",
        "  - `PQC_SUDO_PASSWORD=<password> python3 experiments/run_sqlite_syscall_crash_tpm.py --out-dir artifacts/validation/sqlite_syscall_crash_tpm --when 1 2 3`",
        "- required output schema: platform manifest, per-run medians and observed range, queue-depth / slack traces, SQLite selected-boundary verdicts, crash-audit bundle, combined SQLite/dbm.dumb replay verdicts, and SQLite syscall-exact app-crash verdicts",
        "- required platform fields: device model, kernel, CUDA compiler / runtime, driver / firmware identifiers, accelerator notes",
        "- status: placeholder only; no second-platform raw outputs are retained",
        "",
        "## Gap",
        "- second hardware / driver matrix remains open",
    ]
    (args.out_dir / "platform_inventory_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (args.out_dir / "second_platform_placeholder_manifest.md").write_text(
        "\n".join(
            [
                "# Second-platform placeholder manifest",
                "",
                "- status: no second-platform raw outputs retained",
                "- platform: none",
                "- commands:",
                *[f"  - `{cmd}`" for cmd in report["second_platform_contract"]["minimum_commands"]],
                "- required output schema: platform manifest, per-run medians and observed range, queue-depth / slack traces, SQLite selected-boundary verdicts, crash-audit bundle, combined SQLite/dbm.dumb replay verdicts, SQLite syscall-exact app-crash verdicts",
                "- required platform fields: device model, kernel, CUDA compiler / runtime, driver / firmware identifiers, accelerator notes",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"out_dir": str(args.out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
