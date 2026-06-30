#!/usr/bin/env python3
"""Build the E3 second-macrobenchmark closeout.

The closeout runs the production mounted secure-inference-log macro smoke in
strict and epoch-redo-log modes, retains the raw runner JSON, records coarse
process resource usage, links Jetson thermal metadata, and checks that the paper
describes the result as a second scoped macrobenchmark rather than broad workload
generality or SOSP/OSDI readiness.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "code" / "experiments" / "run_secure_inference_log_macro_smoke.py"
FUSE_BIN = ROOT / "build" / "pqc_fuse"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "second_macrobenchmark_closeout"
THERMAL_CONTRACT = ROOT / "artifacts" / "validation" / "jetson_power_thermal_contract" / "jetson_power_thermal_contract.json"
EVAL_TEX = ROOT / "Paper" / "4_Evaluation.tex"
SECURITY_TEX = ROOT / "Paper" / "8_Security_Analysis.tex"
DISCUSSION_TEX = ROOT / "Paper" / "10_Discussion_and_Limitations.tex"
MAIN_TEX = ROOT / "Paper" / "main.tex"

RUN_SPECS = [
    {
        "name": "strict",
        "publication_mode": "strict",
        "records": 4,
        "payload_bytes": 16384,
    },
    {
        "name": "epoch_redo_log",
        "publication_mode": "epoch-redo-log",
        "records": 2,
        "payload_bytes": 16384,
    },
]

REQUIRED_PAPER_PHRASES = {
    "scoped_mounted_application_rows":
        "The scoped mounted-application rows are not synthetic fio rows",
    "secure_append_log_modes":
        "The secure append-log macro appends ordered 16~KiB records in strict and epoch modes",
    "cache_manifest_third_scoped_workload":
        "A cache-manifest workload publishes 24 hashed 16~KiB objects",
    "not_broad_workload_suite":
        "not a broad workload suite or non-storage application QoS claim",
    "broader_workloads_still_missing":
        "but still lacks physical power-loss, kernel-crash, drive-cache, broad workload-suite",
}

FORBIDDEN_UNGUARDED_PATTERNS = [
    ("broad_workload_diversity_closed", re.compile(r"broad workload diversity (?:is )?(?:closed|proven|complete)", re.I)),
    ("sosp_osdi_ready", re.compile(r"(?:SOSP|OSDI)[-/ ]ready|ready for (?:SOSP|OSDI)", re.I)),
    ("generalizes_to_workloads", re.compile(r"generalizes to .*workloads", re.I)),
    ("second_macro_proves_generality", re.compile(r"second macrobenchmark proves", re.I)),
]

NEGATION_TERMS = (
    "not ",
    "no ",
    "without ",
    "unless ",
    "before ",
    "missing",
    "future",
    "scope",
    "scoped",
    "claim",
    "readiness",
)


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return data


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def path_meta(path: Path) -> dict[str, Any]:
    return {
        "path": relpath(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
    }


def rusage_snapshot() -> resource.struct_rusage:
    return resource.getrusage(resource.RUSAGE_CHILDREN)


def rusage_delta(before: resource.struct_rusage,
                 after: resource.struct_rusage,
                 wall_s: float) -> dict[str, Any]:
    user_s = after.ru_utime - before.ru_utime
    sys_s = after.ru_stime - before.ru_stime
    cpu_s = user_s + sys_s
    return {
        "wall_s": wall_s,
        "user_cpu_s": user_s,
        "system_cpu_s": sys_s,
        "total_cpu_s": cpu_s,
        "cpu_utilization_percent_one_core": (cpu_s / wall_s * 100.0) if wall_s > 0 else 0.0,
        "max_rss_kb_delta": max(0, after.ru_maxrss - before.ru_maxrss),
        "minor_faults_delta": after.ru_minflt - before.ru_minflt,
        "major_faults_delta": after.ru_majflt - before.ru_majflt,
        "voluntary_context_switches_delta": after.ru_nvcsw - before.ru_nvcsw,
        "involuntary_context_switches_delta": after.ru_nivcsw - before.ru_nivcsw,
    }


def run_macro(spec: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    raw_json = out_dir / f"{spec['name']}_raw.json"
    stderr_path = out_dir / f"{spec['name']}.stderr.txt"
    cmd = [
        "python3",
        str(RUNNER),
        "--publication-mode",
        str(spec["publication_mode"]),
        "--records",
        str(spec["records"]),
        "--payload-bytes",
        str(spec["payload_bytes"]),
    ]
    env = os.environ.copy()
    env["PQC_FUSE_BIN"] = str(FUSE_BIN)
    before = rusage_snapshot()
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    wall_s = time.perf_counter() - start
    after = rusage_snapshot()
    raw_json.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")

    parsed: dict[str, Any]
    parse_error = None
    try:
        parsed = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        parsed = {}
        parse_error = repr(exc)

    append = parsed.get("append", {}) if isinstance(parsed, dict) else {}
    summary = append.get("summary", {}) if isinstance(append, dict) else {}
    mounted = parsed.get("mounted_readback", {}) if isinstance(parsed, dict) else {}
    remount = parsed.get("remount_readback", {}) if isinstance(parsed, dict) else {}
    unmount = parsed.get("unmount", {}) if isinstance(parsed, dict) else {}
    return {
        "name": spec["name"],
        "command": cmd,
        "returncode": proc.returncode,
        "stdout_json": relpath(raw_json),
        "stderr": relpath(stderr_path),
        "parse_error": parse_error,
        "overall_pass": bool(parsed.get("overall_pass")) if isinstance(parsed, dict) else False,
        "publication_mode": parsed.get("publication_mode"),
        "workload": parsed.get("workload"),
        "retained_repository_outputs": parsed.get("retained_repository_outputs"),
        "records_requested": parsed.get("records_requested"),
        "payload_bytes": parsed.get("payload_bytes"),
        "latency_throughput_summary": {
            "durable_append_latency": summary.get("durable_append_latency"),
            "write_latency": summary.get("write_latency"),
            "sync_latency": summary.get("sync_latency"),
            "payload_mib_s": summary.get("payload_mib_s"),
            "record_mib_s": summary.get("record_mib_s"),
            "records_s": summary.get("records_s"),
            "total_payload_bytes": summary.get("total_payload_bytes"),
            "total_record_bytes": summary.get("total_record_bytes"),
        },
        "durability_recovery": {
            "mounted_pass": mounted.get("pass"),
            "remount_pass": remount.get("pass"),
            "mounted_record_count": (mounted.get("parse") or {}).get("record_count") if isinstance(mounted.get("parse"), dict) else None,
            "remount_record_count": (remount.get("parse") or {}).get("record_count") if isinstance(remount.get("parse"), dict) else None,
            "mounted_malformed": (mounted.get("parse") or {}).get("malformed") if isinstance(mounted.get("parse"), dict) else None,
            "remount_malformed": (remount.get("parse") or {}).get("malformed") if isinstance(remount.get("parse"), dict) else None,
            "mounted_hash_mismatches": (mounted.get("parse") or {}).get("hash_mismatches") if isinstance(mounted.get("parse"), dict) else None,
            "remount_hash_mismatches": (remount.get("parse") or {}).get("hash_mismatches") if isinstance(remount.get("parse"), dict) else None,
            "visible_sidecars_mounted": mounted.get("visible_sidecars"),
            "visible_sidecars_remount": remount.get("visible_sidecars"),
            "unmount_pass": unmount.get("pass"),
        },
        "process_resource_usage": rusage_delta(before, after, wall_s),
    }


def thermal_metadata() -> dict[str, Any]:
    meta = path_meta(THERMAL_CONTRACT)
    if not THERMAL_CONTRACT.exists():
        return {
            "contract": meta,
            "available": False,
            "complete": False,
            "reason": "missing jetson power/thermal contract",
        }
    data = read_json(THERMAL_CONTRACT)
    tegra = data.get("tegrastats", {})
    nvpmodel = data.get("nvpmodel", {})
    platform = data.get("platform_manifest", {})
    return {
        "contract": meta,
        "available": True,
        "tegrastats_available": bool(tegra.get("available")),
        "tegrastats_samples": len((tegra.get("parsed") or {}).get("raw_lines", []))
            if isinstance(tegra.get("parsed"), dict) else None,
        "nvpmodel_available": bool(nvpmodel.get("available")),
        "platform_manifest_present": bool(platform),
        "scope": "linked platform-state metadata; macro runner also records per-run process resource usage",
        "complete": bool(tegra.get("available")) and bool(nvpmodel.get("available")) and bool(platform),
    }


def paper_guard() -> dict[str, Any]:
    files = [
        path for path in [MAIN_TEX, EVAL_TEX, SECURITY_TEX, DISCUSSION_TEX]
        if path.exists()
    ]
    by_file = {relpath(path): read_text(path) for path in files}
    combined = "\n".join(by_file.values())
    required = {name: phrase in combined for name, phrase in REQUIRED_PAPER_PHRASES.items()}
    hits: list[dict[str, Any]] = []
    unguarded: list[dict[str, Any]] = []
    for rel, text in by_file.items():
        lines = text.splitlines()
        for idx, line in enumerate(lines, start=1):
            for name, pattern in FORBIDDEN_UNGUARDED_PATTERNS:
                if not pattern.search(line):
                    continue
                context = " ".join(lines[max(0, idx - 3): min(len(lines), idx + 3)]).lower()
                guarded = any(term in context for term in NEGATION_TERMS)
                item = {
                    "kind": name,
                    "path": rel,
                    "line": idx,
                    "text": line.strip(),
                    "guarded": guarded,
                }
                hits.append(item)
                if not guarded:
                    unguarded.append(item)
    return {
        "paper_files": sorted(by_file),
        "required_phrases": required,
        "forbidden_hits": hits,
        "unguarded_forbidden_hits": unguarded,
        "complete": all(required.values()) and not unguarded,
    }


def build_report(out_dir: Path, skip_run: bool) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    if skip_run:
        for spec in RUN_SPECS:
            raw = out_dir / f"{spec['name']}_raw.json"
            if raw.exists():
                parsed = read_json(raw)
                runs.append({
                    "name": spec["name"],
                    "stdout_json": relpath(raw),
                    "overall_pass": bool(parsed.get("overall_pass")),
                    "publication_mode": parsed.get("publication_mode"),
                    "workload": parsed.get("workload"),
                    "records_requested": parsed.get("records_requested"),
                    "payload_bytes": parsed.get("payload_bytes"),
                    "latency_throughput_summary": (parsed.get("append") or {}).get("summary"),
                    "durability_recovery": {
                        "mounted_pass": (parsed.get("mounted_readback") or {}).get("pass"),
                        "remount_pass": (parsed.get("remount_readback") or {}).get("pass"),
                        "unmount_pass": (parsed.get("unmount") or {}).get("pass"),
                    },
                    "process_resource_usage": None,
                })
    else:
        for spec in RUN_SPECS:
            runs.append(run_macro(spec, out_dir))

    thermal = thermal_metadata()
    paper = paper_guard()
    mode_names = {run.get("publication_mode") for run in runs}
    close_conditions = {
        "runner_exists": RUNNER.exists() and RUNNER.stat().st_size > 0,
        "fuse_binary_exists": FUSE_BIN.exists(),
        "strict_and_epoch_runs_present": {"strict", "epoch-redo-log"}.issubset(mode_names),
        "all_macro_runs_pass": bool(runs) and all(run.get("overall_pass") for run in runs),
        "raw_latency_throughput_retained": all(
            bool(run.get("latency_throughput_summary", {}).get("durable_append_latency"))
            and (ROOT / str(run.get("stdout_json", ""))).exists()
            for run in runs
        ),
        "durability_recovery_retained": all(
            (run.get("durability_recovery") or {}).get("mounted_pass") is True
            and (run.get("durability_recovery") or {}).get("remount_pass") is True
            and (run.get("durability_recovery") or {}).get("unmount_pass") is True
            for run in runs
        ),
        "process_resource_usage_retained": all(run.get("process_resource_usage") is not None for run in runs),
        "thermal_metadata_linked": bool(thermal.get("complete")),
        "paper_scope_guard_passes": bool(paper.get("complete")),
    }
    return {
        "artifact": "second_macrobenchmark_closeout",
        "timestamp_utc": now_utc(),
        "source": {
            "runner": relpath(RUNNER),
            "fuse_binary": relpath(FUSE_BIN),
            "thermal_contract": relpath(THERMAL_CONTRACT),
        },
        "runs": runs,
        "thermal_metadata": thermal,
        "paper_guard": paper,
        "claim_boundary": {
            "allowed": [
                "secure append-log exists as a second mounted macrobenchmark smoke",
                "strict and epoch modes append, sync, read back, remount, and verify ordered records",
                "latency, throughput, process resource usage, and linked Jetson thermal metadata are retained",
            ],
            "forbidden": [
                "broad workload diversity",
                "SOSP/OSDI readiness from the second macrobenchmark alone",
                "non-storage application QoS recovery",
                "general application compatibility or database certification",
            ],
        },
        "close_conditions": close_conditions,
        "overall_pass": all(close_conditions.values()),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Second Macrobenchmark Closeout",
        "",
        f"- Overall pass: `{str(report['overall_pass']).lower()}`",
        f"- Generated: `{report['timestamp_utc']}`",
        "",
        "## Close Conditions",
        "",
    ]
    for key, value in report["close_conditions"].items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")
    lines.extend(["", "## Runs", ""])
    for run in report["runs"]:
        summary = run.get("latency_throughput_summary") or {}
        durable = summary.get("durable_append_latency") or {}
        lines.append(
            f"- `{run.get('publication_mode')}`: pass={run.get('overall_pass')}, "
            f"records={run.get('records_requested')}, "
            f"p99_us={durable.get('p99_us')}, "
            f"payload_mib_s={summary.get('payload_mib_s')}, "
            f"raw=`{run.get('stdout_json')}`"
        )
    lines.extend(["", "## Claim Boundary", ""])
    for item in report["claim_boundary"]["allowed"]:
        lines.append(f"- Allowed: {item}")
    for item in report["claim_boundary"]["forbidden"]:
        lines.append(f"- Forbidden: {item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args.out_dir, args.skip_run)
    json_path = args.out_dir / "second_macrobenchmark_closeout.json"
    md_path = args.out_dir / "second_macrobenchmark_closeout.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)

    print(f"wrote {relpath(json_path)}")
    print(f"wrote {relpath(md_path)}")
    print(f"overall_pass={str(report['overall_pass']).lower()}")
    for key, value in report["close_conditions"].items():
        print(f"{key}={str(value).lower()}")
    if args.require_complete and not report["overall_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
