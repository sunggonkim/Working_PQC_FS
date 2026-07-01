#!/usr/bin/env python3
"""Build a scoped workload-diversity matrix from retained evidence.

This is an evidence aggregation step, not a new benchmark.  The matrix records
which retained workloads are synthetic filesystem microbenchmarks, which are
application-level storage workloads, and which are trace-only interference
evidence.  It deliberately keeps TensorRT/YOLO separate from the SQLite QoS
claim and keeps WAL/FULL evidence separate from the DELETE/FULL QoS bundle.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "workload_diversity_matrix"

FROZEN_AEGISQ = ROOT / "artifacts" / "validation" / "frozen_aegisq_contract" / "frozen_aegisq_contract.json"
SQLITE_ORACLE = ROOT / "artifacts" / "validation" / "sqlite_recovery_oracle" / "sqlite_recovery_oracle.json"
SQLITE_QOS = ROOT / "artifacts" / "validation" / "qos_sqlite_hero_bundle" / "qos_sqlite_hero_bundle.json"
COMBINED_DURABILITY = ROOT / "artifacts" / "validation" / "combined_durability_bundle" / "combined_durability_bundle.json"
TENSORRT_CI = ROOT / "artifacts" / "reports" / "tensorrt_ci_report" / "tensorrt_ci_report.json"

FROZEN_SCRIPT = ROOT / "code" / "experiments" / "run_frozen_aegisq_contract.py"
SQLITE_ORACLE_SCRIPT = ROOT / "code" / "experiments" / "build_sqlite_recovery_oracle.py"
MOTIVATION_SCRIPT = ROOT / "code" / "experiments" / "run_motivation_bench.py"
QOS_SCRIPT = ROOT / "code" / "experiments" / "run_qos_sqlite_hero_bundle.py"
COMBINED_SCRIPT = ROOT / "code" / "experiments" / "run_combined_durability_bundle.py"
TENSORRT_SCRIPT = ROOT / "code" / "experiments" / "benchmark_tensorrt_interference.py"

PAPER_FILES = [
    ROOT / "Paper" / "4_Evaluation.tex",
    ROOT / "Paper" / "6_Conclusion.tex",
]


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(relpath(path))
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def resolve_artifact_path(path_text: str | None) -> dict[str, Any]:
    if not path_text:
        return {"declared": None, "resolved": None, "exists": False, "taxonomy_remap": False}
    declared = ROOT / path_text
    if declared.exists():
        return {
            "declared": path_text,
            "resolved": relpath(declared),
            "exists": True,
            "taxonomy_remap": False,
        }
    if path_text.startswith("artifacts/motivation/"):
        remapped = ROOT / path_text.replace("artifacts/motivation/", "artifacts/results/motivation/", 1)
        if remapped.exists():
            return {
                "declared": path_text,
                "resolved": relpath(remapped),
                "exists": True,
                "taxonomy_remap": True,
            }
    return {"declared": path_text, "resolved": path_text, "exists": False, "taxonomy_remap": False}


def all_required_substrings(text: str, needles: list[str]) -> bool:
    return all(needle in text for needle in needles)


def qos_mode(qos: dict[str, Any], mode: str) -> dict[str, Any]:
    for row in qos.get("modes", []):
        if row.get("mode") == mode:
            return row
    raise KeyError(mode)


def sqlite_wal_rows(oracle: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for section_name in ("workload", "contention"):
        section = oracle.get(section_name) or {}
        source = resolve_artifact_path(section.get("path"))
        for row in section.get("rows") or []:
            rows.append({**row, "section": section_name, "source": source})
    return rows


def build_fio_row(frozen: dict[str, Any], frozen_script: str) -> dict[str, Any]:
    fio_cmd = frozen.get("fio") or frozen.get("fio_command") or []
    return {
        "row_id": "synthetic_fio_frozen_aegisq",
        "required_category": "synthetic_fio_style_filesystem_workload",
        "workload": "fio 4 KiB 70/30 random read/write, psync, per-write fdatasync",
        "evidence_type": "filesystem microbenchmark",
        "paper_claim_scope": "AEGIS-Q-only warm-cache frozen-contract row; not a cross-system superiority claim",
        "artifact_paths": [relpath(FROZEN_AEGISQ)],
        "script_paths": [relpath(FROZEN_SCRIPT)],
        "filesystem_modes": {
            "fuse_cache_setting": "final pqc_fuse mount with no writeback_cache option in the runner",
            "sync_fsync_behavior": "fio psync with --fdatasync=1, queue depth 1, one client",
            "journal_or_wal_mode": "AEGIS-Q encrypted record plus journal/checkpoint path, not an application WAL",
            "cache_state": "warm-cache row valid after one warmup pass; cold-cache row invalid when privileged cache dropping is unavailable",
        },
        "metrics": {
            "contract_id": frozen.get("contract_id"),
            "workload_profile": frozen.get("workload_profile"),
            "warm_valid_repetitions": (frozen.get("warm_cache_summary") or {}).get("valid_repetitions"),
            "cold_cache_status": (frozen.get("cold_cache") or {}).get("status"),
            "fio_command": fio_cmd,
        },
        "checks": {
            "artifact_pass": frozen.get("overall_pass") is True,
            "aegisq_mode": frozen.get("filesystem_mode") == "aegis_q",
            "warm_repetitions_5": (frozen.get("warm_cache_summary") or {}).get("valid_repetitions") == 5,
            "cold_cache_invalid_recorded": (frozen.get("cold_cache") or {}).get("status") == "invalid_not_run",
            "fdatasync_contract": "--fdatasync=1" in fio_cmd or "fdatasync" in " ".join(fio_cmd),
            "direct_buffered_contract": "--direct=0" in fio_cmd,
            "script_mounts_final_fuse": "str(FUSE_BIN), str(storage_dir), str(mount_dir), \"-f\"" in frozen_script,
        },
    }


def build_sqlite_wal_row(oracle: dict[str, Any], motivation_script: str) -> dict[str, Any]:
    rows = sqlite_wal_rows(oracle)
    wal_rows = [
        row
        for row in rows
        if row.get("requested_mode") == "WAL"
        and row.get("actual_mode") == "WAL"
        and row.get("sync_mode") == "FULL"
        and row.get("integrity_check") == "ok"
        and int(row.get("sample_count") or 0) > 0
    ]
    source_paths = sorted(
        {
            str((row.get("source") or {}).get("resolved"))
            for row in rows
            if (row.get("source") or {}).get("exists")
        }
    )
    taxonomy_remaps = [
        row.get("source")
        for row in rows
        if (row.get("source") or {}).get("taxonomy_remap")
    ]
    return {
        "row_id": "sqlite_wal_full_retained_samples",
        "required_category": "sqlite_wal_full",
        "workload": "SQLite transactions and contention samples using WAL/FULL",
        "evidence_type": "application-level WAL/FULL samples and recovery-oracle definition",
        "paper_claim_scope": "WAL/FULL workload evidence only; not the SQLite QoS recovery table and not crash certification",
        "artifact_paths": [relpath(SQLITE_ORACLE), *source_paths],
        "script_paths": [relpath(MOTIVATION_SCRIPT), relpath(SQLITE_ORACLE_SCRIPT)],
        "filesystem_modes": {
            "fuse_cache_setting": "motivation runner does not rely on a SQLite mmap redirect; WAL may fall back to DELETE when the mounted stack rejects it",
            "sync_fsync_behavior": "SQLite commit path uses PRAGMA synchronous=FULL and retained strace observes fsync/fdatasync boundaries",
            "journal_or_wal_mode": "PRAGMA journal_mode=WAL, actual_mode=WAL, PRAGMA integrity_check=ok",
            "cache_state": "no cold/warm filesystem claim; retained samples are application workload observations",
        },
        "metrics": {
            "wal_full_rows": len(wal_rows),
            "retained_sample_rows": (oracle.get("oracle") or {}).get("retained_sample_rows"),
            "source_path_taxonomy_remaps": taxonomy_remaps,
            "cut_points_observed": [
                cut.get("id")
                for cut in oracle.get("cut_points") or []
                if str(cut.get("observed_in_strace")) == "True"
            ],
        },
        "checks": {
            "oracle_exists": SQLITE_ORACLE.exists(),
            "wal_full_rows_present": len(wal_rows) >= 2,
            "raw_sources_exist": bool(source_paths),
            "integrity_ok": len(wal_rows) == len(rows) and len(rows) > 0,
            "script_requests_wal_full": all_required_substrings(
                motivation_script,
                ["PRAGMA journal_mode=WAL", "PRAGMA synchronous=FULL"],
            ),
            "script_records_writeback_cache_mode": '"-o", "writeback_cache"' in motivation_script,
            "script_avoids_sqlite_mmap_redirect_env": "PQC_ALLOW_SQLITE_MMAP" not in motivation_script,
        },
    }


def build_sqlite_qos_row(qos: dict[str, Any], qos_script: str) -> dict[str, Any]:
    policy = qos_mode(qos, "aegis_policy")
    unthrottled = qos_mode(qos, "unthrottled_storage")
    simple = qos_mode(qos, "simple_controller")
    return {
        "row_id": "sqlite_delete_full_qos_recovery",
        "required_category": "additional_sqlite_application_qos_row",
        "workload": "SQLite foreground transactions under mounted secure-storage pressure",
        "evidence_type": "application-level QoS recovery",
        "paper_claim_scope": "SQLite DELETE/FULL recovery under mounted AEGIS-Q pressure, not SQLite WAL/FULL and not TensorRT/AI recovery",
        "artifact_paths": [relpath(SQLITE_QOS)],
        "script_paths": [relpath(QOS_SCRIPT)],
        "filesystem_modes": {
            "fuse_cache_setting": "final pqc_fuse mount; foreground file marked latency, background file marked elastic with user.pqc_qos_class",
            "sync_fsync_behavior": "SQLite PRAGMA synchronous=FULL; background writer uses secure-storage writes and daemon throttle traces",
            "journal_or_wal_mode": "PRAGMA journal_mode=DELETE for the QoS bundle",
            "cache_state": "fresh temporary lower directory per mode; not a cold-cache filesystem baseline",
        },
        "metrics": {
            "unthrottled_p99_ms": float(unthrottled["foreground"]["p99_ms"]),
            "simple_p99_ms": float(simple["foreground"]["p99_ms"]),
            "aegis_policy_p99_ms": float(policy["foreground"]["p99_ms"]),
            "aegis_background_mb_s": float(policy["background"]["throughput_mb_s"]),
            "aegis_deadline_misses": int(policy["foreground"]["deadline_misses"]),
            "component_coverage": qos.get("component_coverage"),
        },
        "checks": {
            "artifact_pass": qos.get("overall_pass") is True,
            "required_modes_present": (qos.get("component_coverage") or {}).get("required_modes_present") is True,
            "aegis_mode_acceptable": policy.get("acceptable") is True,
            "qos_script_delete_full": all_required_substrings(
                qos_script,
                ["PRAGMA journal_mode=DELETE", "PRAGMA synchronous=FULL"],
            ),
            "qos_script_classifies_files": "user.pqc_qos_class" in qos_script,
        },
    }


def build_dbm_row(combined: dict[str, Any], combined_script: str) -> dict[str, Any]:
    campaign = combined.get("unified_dbm_campaign") or {}
    replay = campaign.get("replay") or {}
    baseline = ((campaign.get("baseline") or {}).get("state") or {})
    advanced = ((campaign.get("advanced") or {}).get("state") or {})
    return {
        "row_id": "dbm_dumb_tpm_stale_snapshot",
        "required_category": "key_value_or_append_log_workload",
        "workload": "Python dbm.dumb key-value store on TPM-backed AEGIS-Q FUSE backing store",
        "evidence_type": "application-level key-value stale-snapshot recovery",
        "paper_claim_scope": "dbm.dumb stale-snapshot fail-closed evidence; not RocksDB, LMDB, or arbitrary append-log certification",
        "artifact_paths": [relpath(COMBINED_DURABILITY)],
        "script_paths": [relpath(COMBINED_SCRIPT)],
        "filesystem_modes": {
            "fuse_cache_setting": "TPM-backed pqc_fuse mounted without writeback_cache option in the combined durability runner",
            "sync_fsync_behavior": "dbm.dumb sync is invoked when available; replay verdict is oracle-based rather than a throughput result",
            "journal_or_wal_mode": "dbm.dumb persistent .dat/.dir key-value files, no SQLite WAL",
            "cache_state": "same-backing-store stale snapshot replay, not cold/warm cache benchmarking",
        },
        "metrics": {
            "baseline_row_count": baseline.get("row_count"),
            "advanced_row_count": advanced.get("row_count"),
            "replay_verdict": replay.get("verdict"),
            "replay_detail": replay.get("detail"),
        },
        "checks": {
            "artifact_exists": COMBINED_DURABILITY.exists(),
            "dbm_baseline_one_row": baseline.get("row_count") == 1,
            "dbm_advanced_three_rows": advanced.get("row_count") == 3,
            "dbm_replay_fail_closed": replay.get("verdict") == "fail_closed",
            "dbm_replay_acceptable": replay.get("acceptable") is True,
            "script_uses_dbm_dumb": "dbm.dumb.open" in combined_script,
            "script_mounts_hardware_anchor": "PQC_FRESHNESS_ANCHOR_BACKEND=hardware" in combined_script,
        },
    }


def build_tensorrt_row(report: dict[str, Any], tensorrt_script: str) -> dict[str, Any]:
    modes = {row.get("mode_label"): row for row in report.get("rows") or []}
    source = resolve_artifact_path(report.get("source"))
    p99_by_mode = {
        mode: [
            float(trial["p99_ms"])
            for trial in (row.get("trial_summaries") or [])
            if trial.get("p99_ms") is not None
        ]
        for mode, row in modes.items()
    }
    return {
        "row_id": "tensorrt_yolo_secure_io_corun",
        "required_category": "foreground_inference_co_run_workload",
        "workload": "TensorRT YOLOv8 foreground inference co-run with mounted AEGIS-Q secure writer",
        "evidence_type": "foreground-inference interference trace",
        "paper_claim_scope": "co-run interference evidence only; not a validated closed-loop foreground AI p99 recovery claim",
        "artifact_paths": [relpath(TENSORRT_CI), str(source.get("resolved"))],
        "script_paths": [relpath(TENSORRT_SCRIPT)],
        "filesystem_modes": {
            "fuse_cache_setting": "benchmark_tensorrt_interference.py mounts final pqc_fuse without writeback_cache option",
            "sync_fsync_behavior": "background secure_writer writes mounted stream.bin and calls os.fsync every sync cycle",
            "journal_or_wal_mode": "secure-storage stream workload, no application WAL",
            "cache_state": "temporary mount per trial; no cold/warm cache distinction or filesystem-comparison claim",
        },
        "metrics": {
            "modes": sorted(modes),
            "trials_by_mode": {mode: row.get("trials") for mode, row in modes.items()},
            "sample_count_total_by_mode": {mode: row.get("sample_count_total") for mode, row in modes.items()},
            "p99_trial_values_ms": p99_by_mode,
            "source_path": source,
        },
        "checks": {
            "report_exists": TENSORRT_CI.exists(),
            "raw_source_exists": source.get("exists") is True,
            "required_modes_present": {"inference_only", "cpu_only", "gpu_only", "adaptive"}.issubset(set(modes)),
            "three_trials_each_mode": all((row.get("trials") == 3) for row in modes.values()) and bool(modes),
            "script_mounts_final_fuse": "str(FUSE), str(storage_dir), str(mount_dir), \"-f\"" in tensorrt_script,
            "script_fsyncs_writer": "os.fsync(handle.fileno())" in tensorrt_script,
            "script_refuses_cpu_fallback": "GPU-resident TensorRT inference was not verified" in tensorrt_script,
        },
    }


def row_pass(row: dict[str, Any]) -> bool:
    return all(bool(value) for value in (row.get("checks") or {}).values())


def paper_scope_gate() -> dict[str, bool]:
    paper = "\n".join(read_text(path) for path in PAPER_FILES)
    return {
        "mentions_workload_diversity_manifest": "workload-diversity manifest" in paper,
        "separates_fio_and_application_evidence": "fio and primitive benchmarks explain placement/overhead" in paper,
        "mentions_sqlite_dbm_application_scope": "SQLite and \\texttt{dbm.dumb} are the application-level storage evidence" in paper,
        "keeps_tensorrt_trace_only": "TensorRT/YOLO remains co-run interference trace evidence" in paper,
        "keeps_no_ai_recovery_claim": "foreground AI p99 recovery" in paper and "not" in paper,
    }


def build_manifest() -> dict[str, Any]:
    frozen = load_json(FROZEN_AEGISQ)
    oracle = load_json(SQLITE_ORACLE)
    qos = load_json(SQLITE_QOS)
    combined = load_json(COMBINED_DURABILITY)
    tensorrt = load_json(TENSORRT_CI)

    scripts = {
        "frozen": read_text(FROZEN_SCRIPT),
        "motivation": read_text(MOTIVATION_SCRIPT),
        "qos": read_text(QOS_SCRIPT),
        "combined": read_text(COMBINED_SCRIPT),
        "tensorrt": read_text(TENSORRT_SCRIPT),
    }

    rows = [
        build_fio_row(frozen, scripts["frozen"]),
        build_sqlite_wal_row(oracle, scripts["motivation"]),
        build_sqlite_qos_row(qos, scripts["qos"]),
        build_dbm_row(combined, scripts["combined"]),
        build_tensorrt_row(tensorrt, scripts["tensorrt"]),
    ]
    required_categories = {
        "synthetic_fio_style_filesystem_workload",
        "sqlite_wal_full",
        "key_value_or_append_log_workload",
        "foreground_inference_co_run_workload",
    }
    passed_required_categories = {
        row["required_category"]
        for row in rows
        if row["required_category"] in required_categories and row_pass(row)
    }
    paper_gate = paper_scope_gate()
    checks = {
        "required_workloads_present": required_categories.issubset(passed_required_categories),
        "all_rows_have_filesystem_modes": all(
            set((row.get("filesystem_modes") or {}).keys())
            == {"fuse_cache_setting", "sync_fsync_behavior", "journal_or_wal_mode", "cache_state"}
            for row in rows
        ),
        "sqlite_qos_extra_row_pass": row_pass(rows[2]),
        "paper_scope_gate_pass": all(paper_gate.values()),
    }
    for row in rows:
        checks[f"{row['row_id']}_pass"] = row_pass(row)

    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": all(checks.values()),
        "scope": (
            "Scoped workload-diversity matrix over retained artifacts.  It "
            "separates synthetic fio evidence, SQLite WAL/FULL samples, "
            "SQLite DELETE/FULL QoS recovery, dbm.dumb stale-snapshot recovery, "
            "and TensorRT/YOLO co-run traces.  It does not create a broad "
            "filesystem baseline matrix or a foreground-AI recovery claim."
        ),
        "checks": checks,
        "paper_scope_gate": paper_gate,
        "required_categories": sorted(required_categories),
        "passed_required_categories": sorted(passed_required_categories),
        "rows": rows,
        "non_claims": [
            "no fscrypt/dm-crypt frozen-contract execution",
            "no cold-cache filesystem result outside the invalid recorded row",
            "no broad application-workload generalization beyond SQLite/dbm.dumb and TensorRT trace scope",
            "no foreground TensorRT/AI p99 recovery claim",
            "no full crash or power-loss certification",
        ],
    }


def write_csv(report: dict[str, Any], path: Path) -> None:
    fields = [
        "row_id",
        "required_category",
        "workload",
        "evidence_type",
        "paper_claim_scope",
        "row_pass",
        "fuse_cache_setting",
        "sync_fsync_behavior",
        "journal_or_wal_mode",
        "cache_state",
        "artifact_paths",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in report["rows"]:
            fs_modes = row["filesystem_modes"]
            writer.writerow(
                {
                    "row_id": row["row_id"],
                    "required_category": row["required_category"],
                    "workload": row["workload"],
                    "evidence_type": row["evidence_type"],
                    "paper_claim_scope": row["paper_claim_scope"],
                    "row_pass": row_pass(row),
                    "fuse_cache_setting": fs_modes["fuse_cache_setting"],
                    "sync_fsync_behavior": fs_modes["sync_fsync_behavior"],
                    "journal_or_wal_mode": fs_modes["journal_or_wal_mode"],
                    "cache_state": fs_modes["cache_state"],
                    "artifact_paths": "; ".join(row["artifact_paths"]),
                }
            )


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Workload Diversity Matrix",
        "",
        f"- Overall pass: `{str(report['overall_pass']).lower()}`",
        f"- Scope: {report['scope']}",
        "",
        "## Checks",
        "",
    ]
    for name, value in report["checks"].items():
        lines.append(f"- `{name}`: `{str(value).lower()}`")
    lines.extend(["", "## Paper Scope Gate", ""])
    for name, value in report["paper_scope_gate"].items():
        lines.append(f"- `{name}`: `{str(value).lower()}`")
    lines.extend(["", "## Rows", ""])
    for row in report["rows"]:
        lines.extend(
            [
                f"### {row['row_id']}",
                "",
                f"- Pass: `{str(row_pass(row)).lower()}`",
                f"- Category: `{row['required_category']}`",
                f"- Workload: {row['workload']}",
                f"- Evidence type: {row['evidence_type']}",
                f"- Claim scope: {row['paper_claim_scope']}",
                f"- Artifacts: `{', '.join(row['artifact_paths'])}`",
                f"- Scripts: `{', '.join(row['script_paths'])}`",
                "",
                "Filesystem modes:",
            ]
        )
        for key, value in row["filesystem_modes"].items():
            lines.append(f"- `{key}`: {value}")
        lines.extend(["", "Row checks:", ""])
        for name, value in row["checks"].items():
            lines.append(f"- `{name}`: `{str(value).lower()}`")
        lines.append("")
    lines.extend(["## Non-Claims", ""])
    for item in report["non_claims"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    report = build_manifest()
    json_path = out / "workload_diversity_matrix.json"
    csv_path = out / "workload_diversity_matrix.csv"
    md_path = out / "workload_diversity_matrix.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(report, csv_path)
    md_path.write_text(markdown(report), encoding="utf-8")

    print(
        json.dumps(
            {
                "overall_pass": report["overall_pass"],
                "json": relpath(json_path),
                "csv": relpath(csv_path),
                "markdown": relpath(md_path),
                "checks": report["checks"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.require_complete and not report["overall_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
