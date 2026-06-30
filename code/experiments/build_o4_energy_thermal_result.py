#!/usr/bin/env python3
"""Build the O4 same-run power/thermal summary from retained tegrastats logs.

This is a retained-log analysis, not a new benchmark.  It parses tegrastats
samples that were captured next to the QoS, key-plane, and frozen-contract
runs and emits a conservative power/thermal observation.  The report may be
used to answer "no energy/thermal evidence" only as an observation over the
retained runs, not as an energy-efficiency or thermally-stable claim.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts" / "reports" / "o4_energy_thermal_result"

LOG_SPECS = {
    "sqlite_qos_methodology": {
        "path": ROOT / "artifacts" / "validation" / "qos_sqlite_hero_methodology" / "thermal_tegrastats.log",
        "interval_ms": 100,
        "role": "mounted SQLite QoS methodology run",
    },
    "keyplane_rekey_methodology": {
        "path": ROOT / "artifacts" / "validation" / "keyplane_rekey_methodology" / "thermal_tegrastats.log",
        "interval_ms": 100,
        "role": "mounted ML-KEM key-plane methodology run",
    },
    "frozen_aegisq_contract": {
        "path": ROOT / "artifacts" / "validation" / "frozen_aegisq_contract" / "thermal_tegrastats.log",
        "interval_ms": 500,
        "role": "strict AEGIS-Q frozen contract",
    },
    "frozen_dmcrypt_contract": {
        "path": ROOT / "artifacts" / "validation" / "frozen_dmcrypt_contract" / "thermal_tegrastats.log",
        "interval_ms": 500,
        "role": "dm-crypt frozen contract",
    },
    "frozen_gocryptfs_contract": {
        "path": ROOT / "artifacts" / "validation" / "frozen_gocryptfs_contract" / "thermal_tegrastats.log",
        "interval_ms": 100,
        "role": "gocryptfs frozen contract",
    },
    "frozen_plaintext_contract": {
        "path": ROOT / "artifacts" / "validation" / "frozen_plaintext_contract" / "thermal_tegrastats.log",
        "interval_ms": 100,
        "role": "plaintext frozen contract",
    },
}

PAPER_FILES = [
    ROOT / "Paper" / "4_Evaluation.tex",
    ROOT / "Paper" / "10_Discussion_and_Limitations.tex",
]
CHECKLIST = ROOT / "SUBMISSION_CHECKLIST.md"

POWER_RE = re.compile(r"\b(VIN|VDD_GPU|VDD_CPU_SOC_MSS|VIN_SYS_5V0)\s+(\d+)mW/(\d+)mW")
TEMP_RE = re.compile(r"\b(cpu|tj|gpu|soc012|soc345)@([0-9.]+)C")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def summarize_values(values: list[float]) -> dict[str, float | int]:
    return {
        "samples": len(values),
        "mean": mean(values) if values else 0.0,
        "p95": percentile(values, 0.95),
        "max": max(values) if values else 0.0,
    }


def parse_log(path: Path, interval_ms: int) -> dict[str, Any]:
    powers: dict[str, list[float]] = {}
    temps: dict[str, list[float]] = {}
    line_count = 0
    if not path.exists():
        return {"path": rel(path), "present": False, "line_count": 0}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        line_count += 1
        for rail, current, _avg in POWER_RE.findall(line):
            powers.setdefault(rail, []).append(float(current))
        for sensor, temp in TEMP_RE.findall(line):
            temps.setdefault(sensor, []).append(float(temp))
    seconds = line_count * interval_ms / 1000.0
    vin = powers.get("VIN", [])
    summary = {
        "path": rel(path),
        "present": True,
        "line_count": line_count,
        "interval_ms": interval_ms,
        "estimated_duration_s": seconds,
        "power_mw": {rail: summarize_values(values) for rail, values in sorted(powers.items())},
        "temperature_c": {sensor: summarize_values(values) for sensor, values in sorted(temps.items())},
        "estimated_vin_energy_j": (mean(vin) / 1000.0 * seconds) if vin else 0.0,
    }
    return summary


def build_report() -> dict[str, Any]:
    rows = {}
    for name, spec in LOG_SPECS.items():
        parsed = parse_log(spec["path"], int(spec["interval_ms"]))
        parsed["role"] = spec["role"]
        rows[name] = parsed
    required = {
        "sqlite_log_present": rows["sqlite_qos_methodology"].get("line_count", 0) > 0,
        "keyplane_log_present": rows["keyplane_rekey_methodology"].get("line_count", 0) > 0,
        "frozen_rows_present": all(rows[name].get("line_count", 0) > 0 for name in (
            "frozen_aegisq_contract",
            "frozen_dmcrypt_contract",
            "frozen_gocryptfs_contract",
            "frozen_plaintext_contract",
        )),
        "paper_mentions_o4": any("same-run power/thermal" in path.read_text(encoding="utf-8", errors="replace") for path in PAPER_FILES),
        "paper_no_energy_efficiency_win_claim": not any(
            bad in path.read_text(encoding="utf-8", errors="replace").lower()
            for path in PAPER_FILES
            for bad in ("energy-efficient", "improves energy", "energy win", "energy savings")
        ),
        "checklist_o4_done": "| O4 | DONE |" in CHECKLIST.read_text(encoding="utf-8", errors="replace"),
    }
    representative = {
        name: {
            "mean_vin_w": rows[name]["power_mw"]["VIN"]["mean"] / 1000.0,
            "max_tj_c": rows[name]["temperature_c"].get("tj", {}).get("max", 0.0),
            "estimated_vin_energy_j": rows[name]["estimated_vin_energy_j"],
            "samples": rows[name]["line_count"],
        }
        for name in rows
        if rows[name].get("present") and "VIN" in rows[name].get("power_mw", {})
    }
    return {
        "artifact": "o4_energy_thermal_result",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Same-run tegrastats power/thermal summary for retained QoS, key-plane, and frozen-contract runs.",
        "non_claim": "This is not an energy-efficiency claim, not a fixed-clock proof, and not a replacement for per-mode energy experiments.",
        "logs": rows,
        "representative_summary": representative,
        "checks": required,
        "overall_pass": all(required.values()),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# O4 energy/thermal result",
        "",
        f"Overall pass: `{report['overall_pass']}`",
        "",
        report["non_claim"],
        "",
        "## Representative rows",
        "",
        "| Row | Samples | Mean VIN W | Max tj C | Estimated VIN J |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in report["representative_summary"].items():
        lines.append(
            f"| `{name}` | {row['samples']} | {row['mean_vin_w']:.2f} | "
            f"{row['max_tj_c']:.1f} | {row['estimated_vin_energy_j']:.1f} |"
        )
    lines.extend(["", "## Checks", ""])
    for name, ok in report["checks"].items():
        lines.append(f"- `{name}`: `{ok}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report()
    json_path = args.out_dir / "o4_energy_thermal_result.json"
    md_path = args.out_dir / "o4_energy_thermal_result.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({
        "overall_pass": report["overall_pass"],
        "json": rel(json_path),
        "markdown": rel(md_path),
        "failed_checks": [name for name, ok in report["checks"].items() if not ok],
    }, indent=2, sort_keys=True))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
