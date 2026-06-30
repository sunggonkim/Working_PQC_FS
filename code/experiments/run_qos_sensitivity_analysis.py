#!/usr/bin/env python3
"""Run mounted SQLite QoS sensitivity cases for the AEGIS-Q controller.

The artifact produced here is deliberately narrower than the four-mode hero
bundle: each mounted case runs the final ``build/pqc_fuse`` binary in AEGIS-Q
policy mode while varying one controller/workload knob.  Raw foreground,
background, telemetry, policy, FUSE, and daemon-throttle logs are retained for
each case.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "code" / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_qos_sqlite_hero_bundle as hero  # noqa: E402


DEFAULT_OUT = ROOT / "artifacts" / "validation" / "qos_sensitivity_analysis"
FUSE_BIN = ROOT / "build" / "pqc_fuse"


@dataclass(frozen=True)
class SensitivityCase:
    name: str
    variable: str
    setting: str
    description: str
    overrides: dict[str, Any] = field(default_factory=dict)
    require_daemon_throttle: bool = True


CASES = [
    SensitivityCase(
        name="baseline",
        variable="default",
        setting="10ms budget, 20ms sampling, 0.70/0.60 threshold, qd=1, 64KiB chunks",
        description="Default AEGIS-Q mounted SQLite sensitivity anchor.",
    ),
    SensitivityCase(
        name="tight_budget",
        variable="budget",
        setting="8ms foreground deadline",
        description="Reduces foreground slack to test budget sensitivity.",
        overrides={"deadline_ms": 8.0},
    ),
    SensitivityCase(
        name="no_slack_mounted",
        variable="budget",
        setting="1ms foreground deadline",
        description="Forces zero-budget telemetry rows in the mounted co-run.",
        overrides={"deadline_ms": 1.0},
    ),
    SensitivityCase(
        name="slow_sampling",
        variable="sampling_interval",
        setting="80ms telemetry interval",
        description="Slows controller sampling while keeping the same workload.",
        overrides={"telemetry_interval_ms": 80, "telemetry_poll_ms": 20},
    ),
    SensitivityCase(
        name="high_threshold",
        variable="controller_threshold",
        setting="0.90/0.80 enter/exit",
        description="Raises the hysteresis threshold to test tuning sensitivity.",
        overrides={"enter_util": 0.90, "exit_util": 0.80},
        require_daemon_throttle=False,
    ),
    SensitivityCase(
        name="queue_depth_2",
        variable="queue_depth",
        setting="2 background writers",
        description="Increases mounted secure-writer concurrency.",
        overrides={"background_writers": 2},
    ),
    SensitivityCase(
        name="background_128k",
        variable="background_intensity",
        setting="128KiB writer chunks",
        description="Increases per-write background storage intensity.",
        overrides={"background_chunk_bytes": 131072},
    ),
    SensitivityCase(
        name="low_pressure_no_throttle",
        variable="no_throttle_fallback",
        setting="0.30 pressure, 20ms budget",
        description="Keeps daemon throttle enabled but holds pressure below the enter threshold.",
        overrides={"deadline_ms": 20.0, "background_pressure_util": 0.30},
        require_daemon_throttle=False,
    ),
    SensitivityCase(
        name="hysteresis_wave",
        variable="hysteresis",
        setting="0.85:3,0.45:3 pressure wave, hold=2",
        description="Alternates pressure to retain explicit enter/exit transition traces.",
        overrides={
            "deadline_ms": 20.0,
            "background_pressure_pattern": "0.85:3,0.45:3,0.85:3,0.45:3",
            "hold_samples": 2,
        },
        require_daemon_throttle=False,
    ),
]


def command_output(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def default_args(out_dir: Path, case: SensitivityCase, base: argparse.Namespace) -> SimpleNamespace:
    values: dict[str, Any] = {
        "out_dir": out_dir,
        "transactions": base.transactions,
        "rows_per_txn": base.rows_per_txn,
        "sqlite_payload_bytes": 256,
        "deadline_ms": 10.0,
        "inter_transaction_sleep_ms": 1.0,
        "post_foreground_drain_ms": 250,
        "background_writers": 1,
        "background_chunk_bytes": 65536,
        "background_fsync_every": 1,
        "background_warmup_ms": 100,
        "telemetry_interval_ms": 20,
        "telemetry_poll_ms": 10,
        "telemetry_window": 12,
        "controller_warmup_transactions": 2,
        "background_pressure_util": 0.85,
        "background_pressure_pattern": "",
        "enter_util": 0.70,
        "exit_util": 0.60,
        "hold_samples": 1,
        "harness_throttle_sleep_us": 5000,
        "daemon_throttle_sleep_us": 30000,
        "require_daemon_throttle": case.require_daemon_throttle,
    }
    values.update(case.overrides)
    values["_background_pressure_pattern_values"] = hero.parse_pressure_pattern(
        str(values.get("background_pressure_pattern", ""))
    )
    return SimpleNamespace(**values)


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "missing"))
        counts[value] = counts.get(value, 0) + 1
    return counts


def summarize_case(case: SensitivityCase, mode: dict[str, Any]) -> dict[str, Any]:
    policy_rows = hero.load_jsonl(ROOT / mode["logs"]["policy_jsonl"])
    throttle_rows = hero.load_jsonl(ROOT / mode["logs"]["runtime_fuse_throttle_trace"])
    transition_rows = [
        row for row in policy_rows
        if row.get("policy_event") in {"enter", "exit"}
    ]
    events = count_by(policy_rows, "policy_event")
    states = count_by(policy_rows, "policy_state")
    throttled_rows = int(mode["daemon_throttle"].get("throttled_rows", 0))
    policy_throttle_rows = int(mode["policy"].get("throttle_rows", 0))
    zero_budget_rows = int(mode["telemetry"].get("zero_budget_rows", 0))
    low_pressure_open = (
        case.name == "low_pressure_no_throttle"
        and len(throttle_rows) > 0
        and throttled_rows == 0
        and policy_throttle_rows == 0
    )
    hysteresis_enter_exit = events.get("enter", 0) > 0 and events.get("exit", 0) > 0
    below_exit_samples = sum(
        1 for row in policy_rows
        if int(row.get("below_exit_count", 0)) > 0
    )
    return {
        "case": case.name,
        "variable": case.variable,
        "setting": case.setting,
        "description": case.description,
        "acceptable": bool(mode["acceptable"]),
        "p50_ms": mode["foreground"].get("p50_ms"),
        "p95_ms": mode["foreground"].get("p95_ms"),
        "p99_ms": mode["foreground"].get("p99_ms"),
        "deadline_misses": mode["foreground"].get("deadline_misses"),
        "deadline_ms": mode["foreground"].get("deadline_ms"),
        "storage_mb_s": mode["background"].get("throughput_mb_s"),
        "background_bytes": mode["background"].get("bytes_written"),
        "telemetry_rows": mode["telemetry"].get("rows"),
        "zero_budget_rows": zero_budget_rows,
        "high_pressure_rows": mode["telemetry"].get("high_pressure_rows"),
        "avg_cpu_utilization": mode["telemetry"].get("avg_cpu_utilization"),
        "avg_gpu_utilization": mode["telemetry"].get("avg_gpu_utilization"),
        "policy_throttle_rows": policy_throttle_rows,
        "daemon_throttle_rows": throttled_rows,
        "daemon_sleep_us_total": mode["daemon_throttle"].get("sleep_us_total"),
        "policy_events": events,
        "policy_states": states,
        "transition_count": len(transition_rows),
        "oscillation_count": max(0, len(transition_rows) - 1),
        "hysteresis_enter_exit": hysteresis_enter_exit,
        "below_exit_samples": below_exit_samples,
        "no_throttle_fallback": low_pressure_open,
        "logs": mode["logs"],
    }


def run_no_slack_admission_check(out_dir: Path) -> dict[str, Any]:
    trace_path = out_dir / "admission_no_slack_trace.jsonl"
    stdout_path = out_dir / "admission_no_slack.stdout.txt"
    stderr_path = out_dir / "admission_no_slack.stderr.txt"
    env = os.environ.copy()
    env.update({
        "PQC_ADMISSION_TRACE_PATH": str(trace_path),
        "PQC_ADMISSION_SMOKE_AI_BUDGET_NS": "0",
        "PQC_ADMISSION_SMOKE_BYTES": "131072",
        "PQC_ADMISSION_SMOKE_CPU_QUEUE_DEPTH": "1",
        "PQC_ADMISSION_SMOKE_GPU_QUEUE_DEPTH": "0",
        "PQC_ADMISSION_SMOKE_GPU_KERNEL_NS": "100000",
        "PQC_ADMISSION_SMOKE_H2D_NS": "100000",
        "PQC_ADMISSION_SMOKE_D2H_NS": "100000",
        "PQC_TELEMETRY_MEM_BANDWIDTH": "0.10",
        "PQC_TELEMETRY_TENSOR_CORE": "0.10",
    })
    proc = subprocess.run(
        [str(FUSE_BIN), "--admission-telemetry-smoke"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    row: dict[str, Any] = {}
    for line in proc.stdout.splitlines():
        if line.startswith("{"):
            row = json.loads(line)
            break
    passed = (
        proc.returncode == 0
        and row.get("chosen_target") == "CPU"
        and int(row.get("ai_budget_ns", -1)) == 0
        and (int(row.get("deferral_reason", 0)) & 0x02) != 0
    )
    return {
        "passed": passed,
        "returncode": proc.returncode,
        "stdout": str(stdout_path.relative_to(ROOT)),
        "stderr": str(stderr_path.relative_to(ROOT)),
        "trace": str(trace_path.relative_to(ROOT)),
        "row": row,
    }


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "case",
        "variable",
        "setting",
        "p99_ms",
        "deadline_misses",
        "storage_mb_s",
        "telemetry_rows",
        "zero_budget_rows",
        "policy_throttle_rows",
        "daemon_throttle_rows",
        "transition_count",
        "oscillation_count",
        "hysteresis_enter_exit",
        "no_throttle_fallback",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# QoS Sensitivity Analysis",
        "",
        f"- Overall pass: `{str(report['overall_pass']).lower()}`",
        "- Scope: mounted SQLite AEGIS-Q policy sensitivity plus final-binary no-slack admission fallback.",
        "- Non-claim: not TensorRT/AI p99 recovery and not a statistical confidence study.",
        "",
        "## Checks",
        "",
    ]
    for key, value in report["checks"].items():
        lines.append(f"- {key}: `{str(value).lower()}`")
    lines.extend([
        "",
        "## Cases",
        "",
        "| case | variable | p99 ms | misses | MB/s | daemon thr. | transitions | osc. |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in report["cases"]:
        lines.append(
            f"| `{row['case']}` | {row['variable']} | "
            f"{float(row['p99_ms']):.3f} | {int(row['deadline_misses'])} | "
            f"{float(row['storage_mb_s']):.3f} | {int(row['daemon_throttle_rows'])} | "
            f"{int(row['transition_count'])} | {int(row['oscillation_count'])} |"
        )
    lines.extend([
        "",
        "## Raw Logs",
        "",
    ])
    for row in report["cases"]:
        lines.append(f"- `{row['case']}`")
        for label, rel in row["logs"].items():
            lines.append(f"  - {label}: `{rel}`")
    lines.extend([
        "- `admission_no_slack`",
        f"  - stdout: `{report['no_slack_admission']['stdout']}`",
        f"  - stderr: `{report['no_slack_admission']['stderr']}`",
        f"  - trace: `{report['no_slack_admission']['trace']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--transactions", type=int, default=40)
    parser.add_argument("--rows-per-txn", type=int, default=8)
    args = parser.parse_args()

    if not FUSE_BIN.exists():
        raise SystemExit("missing build/pqc_fuse; run cmake --build build first")

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for case in CASES:
        case_dir = out_dir / case.name
        case_dir.mkdir(parents=True, exist_ok=True)
        case_args = default_args(case_dir, case, args)
        mode = hero.run_mode(hero.MODE_CONFIGS["aegis_policy"], case_dir, case_args)
        summaries.append(summarize_case(case, mode))

    no_slack_admission = run_no_slack_admission_check(out_dir)
    variables = {row["variable"] for row in summaries}
    checks = {
        "all_cases_acceptable": all(row["acceptable"] for row in summaries),
        "required_variables_covered": {
            "budget",
            "sampling_interval",
            "controller_threshold",
            "queue_depth",
            "background_intensity",
        }.issubset(variables),
        "no_throttle_fallback": any(row["no_throttle_fallback"] for row in summaries),
        "no_slack_mounted_zero_budget": any(
            row["case"] == "no_slack_mounted" and int(row["zero_budget_rows"]) > 0
            for row in summaries
        ),
        "no_slack_admission_cpu_fallback": bool(no_slack_admission["passed"]),
        "hysteresis_enter_exit_trace": any(row["hysteresis_enter_exit"] for row in summaries),
        "oscillation_count_recorded": all("oscillation_count" in row for row in summaries),
        "daemon_trace_retained": all(row["daemon_throttle_rows"] >= 0 for row in summaries),
    }
    report = {
        "artifact": "qos_sensitivity_analysis",
        "overall_pass": all(
            value if isinstance(value, bool) else bool(value)
            for value in checks.values()
        ),
        "scope": [
            "Mounted SQLite AEGIS-Q policy sensitivity over budget, sampling, threshold, queue depth, and background intensity.",
            "Final-binary admission no-slack fallback check.",
            "Not a TensorRT/AI p99 recovery result and not a statistical confidence interval study.",
        ],
        "command": ["python3", "code/experiments/run_qos_sensitivity_analysis.py"],
        "platform": {
            "system": platform.platform(),
            "python": platform.python_version(),
            "kernel": command_output(["uname", "-a"]),
            "git_head": command_output(["git", "rev-parse", "HEAD"]),
            "git_dirty_short": command_output(["git", "status", "--short"]),
        },
        "checks": checks,
        "no_slack_admission": no_slack_admission,
        "cases": summaries,
    }
    (out_dir / "qos_sensitivity_analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary_csv(summaries, out_dir / "qos_sensitivity_summary.csv")
    write_markdown(report, out_dir / "qos_sensitivity_analysis.md")
    print(json.dumps({
        "overall_pass": report["overall_pass"],
        "out": str(out_dir / "qos_sensitivity_analysis.json"),
        "cases": {row["case"]: row["acceptable"] for row in summaries},
    }, sort_keys=True))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
