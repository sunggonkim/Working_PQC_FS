#!/usr/bin/env python3
"""Build a retained mechanism-ablation manifest from existing evidence.

The manifest deliberately aggregates already-retained validation bundles rather
than inventing new comparisons.  It supports only scoped mechanism attribution:
lowerfs-vs-AEGIS-Q isolates the total mounted encrypted-format cost relative to
raw ext4, SQLite QoS rows isolate controller behavior, the key-plane workflow
isolates CPU/GPU placement for envelope refresh, and the anchor rows isolate
file-backed negative-control behavior from TPM replay-after-advance behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "mechanism_ablation_manifest"

FROZEN_PLAINTEXT = ROOT / "artifacts" / "validation" / "frozen_plaintext_contract" / "frozen_plaintext_contract.json"
FROZEN_AEGISQ = ROOT / "artifacts" / "validation" / "frozen_aegisq_contract" / "frozen_aegisq_contract.json"
QOS_HERO = ROOT / "artifacts" / "validation" / "qos_sqlite_hero_bundle" / "qos_sqlite_hero_bundle.json"
KEYPLANE = ROOT / "artifacts" / "validation" / "keyplane_rekey_methodology" / "keyplane_rekey_workflow.json"
GENERATION = ROOT / "artifacts" / "validation" / "generation_fault_matrix" / "generation_fault_matrix.json"
TPM_REPLAY = ROOT / "artifacts" / "validation" / "tpm_monotonic_replay" / "tpm_monotonic_replay.json"
PAPER_EVAL = ROOT / "Paper" / "4_Evaluation.tex"


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(relpath(path))
    return json.loads(path.read_text(encoding="utf-8"))


def metric(summary: dict[str, Any], name: str) -> dict[str, float]:
    metrics = summary["warm_cache_summary"]["metrics"][name]
    return {
        "median": float(metrics["median"]),
        "ci95_low": float(metrics["ci95_low"]),
        "ci95_high": float(metrics["ci95_high"]),
    }


def qos_mode(qos: dict[str, Any], mode: str) -> dict[str, Any]:
    for row in qos.get("modes", []):
        if row.get("mode") == mode:
            return row
    raise KeyError(mode)


def keyplane_mode(keyplane: dict[str, Any], mode: str) -> dict[str, Any]:
    for row in keyplane.get("mode_summaries", []):
        if row.get("mode") == mode:
            return row
    raise KeyError(mode)


def generation_case(generation: dict[str, Any], case: str) -> dict[str, Any]:
    for row in generation.get("rows", []):
        if row.get("case") == case:
            return row
    raise KeyError(case)


def fmt_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def build_manifest() -> dict[str, Any]:
    plaintext = load_json(FROZEN_PLAINTEXT)
    aegisq = load_json(FROZEN_AEGISQ)
    qos = load_json(QOS_HERO)
    keyplane = load_json(KEYPLANE)
    generation = load_json(GENERATION)
    tpm = load_json(TPM_REPLAY)

    plain_tp = metric(plaintext, "throughput_mib_s")
    plain_p99 = metric(plaintext, "latency_p99_us")
    aegis_tp = metric(aegisq, "throughput_mib_s")
    aegis_p99 = metric(aegisq, "latency_p99_us")

    unthrottled = qos_mode(qos, "unthrottled_storage")
    simple = qos_mode(qos, "simple_controller")
    policy = qos_mode(qos, "aegis_policy")
    app_only = qos_mode(qos, "app_only")

    cpu = keyplane_mode(keyplane, "cpu_only")
    gpu = keyplane_mode(keyplane, "gpu_batch")
    fallback = keyplane_mode(keyplane, "policy_fallback")

    file_anchor = generation_case(generation, "stale_snapshot_replay_file_anchor_negative_control")
    tpm_anchor = generation_case(generation, "stale_snapshot_replay_tpm_anchor_existing_artifact")
    tpm_result = tpm.get("result") or {}

    filesystem_overhead = {
        "id": "fs_format_total_cost",
        "mechanism": "encrypted mounted format and journal/checkpoint path",
        "variants": ["plaintext_lowerfs", "aegis_q"],
        "artifact_paths": [
            relpath(FROZEN_PLAINTEXT),
            relpath(FROZEN_AEGISQ),
        ],
        "raw_log_paths": [
            "artifacts/validation/frozen_plaintext_contract/fio_raw",
            "artifacts/validation/frozen_aegisq_contract/fio_raw",
            "artifacts/validation/frozen_aegisq_contract/mount_logs",
        ],
        "metrics": {
            "plaintext_throughput_mib_s": plain_tp,
            "aegisq_throughput_mib_s": aegis_tp,
            "aegisq_vs_plaintext_throughput_ratio": aegis_tp["median"] / plain_tp["median"],
            "plaintext_conservative_p99_ms": {k: v / 1000.0 for k, v in plain_p99.items()},
            "aegisq_conservative_p99_ms": {k: v / 1000.0 for k, v in aegis_p99.items()},
            "aegisq_vs_plaintext_p99_ratio": aegis_p99["median"] / plain_p99["median"],
        },
        "interpretation": (
            "The warm-cache frozen-contract gap attributes the fio overhead to "
            "the mounted encrypted format as a whole, including FUSE, AEAD, "
            "journal, checkpoint, and file-anchor work.  It does not isolate a "
            "plaintext-FUSE row or kernel encryption baselines."
        ),
        "scope_boundary": "No fscrypt/dm-crypt, cold-cache, or component-only filesystem decomposition.",
    }

    qos_controller = {
        "id": "qos_controller_variants",
        "mechanism": "SQLite foreground recovery under secure-storage pressure",
        "variants": ["app_only", "unthrottled_storage", "simple_controller", "aegis_policy"],
        "artifact_paths": [relpath(QOS_HERO)],
        "raw_log_paths": [
            "artifacts/validation/qos_sqlite_hero_bundle/*/foreground_sqlite_latency.jsonl",
            "artifacts/validation/qos_sqlite_hero_bundle/*/background_writer.jsonl",
            "artifacts/validation/qos_sqlite_hero_bundle/*/policy_trace.jsonl",
            "artifacts/validation/qos_sqlite_hero_bundle/*/runtime_fuse_throttle_trace.jsonl",
        ],
        "metrics": {
            "app_only_p99_ms": float(app_only["foreground"]["p99_ms"]),
            "unthrottled_p99_ms": float(unthrottled["foreground"]["p99_ms"]),
            "simple_controller_p99_ms": float(simple["foreground"]["p99_ms"]),
            "aegis_policy_p99_ms": float(policy["foreground"]["p99_ms"]),
            "unthrottled_deadline_misses": int(unthrottled["foreground"]["deadline_misses"]),
            "simple_controller_deadline_misses": int(simple["foreground"]["deadline_misses"]),
            "aegis_policy_deadline_misses": int(policy["foreground"]["deadline_misses"]),
            "simple_background_mb_s": float(simple["background"]["throughput_mb_s"]),
            "aegis_background_mb_s": float(policy["background"]["throughput_mb_s"]),
            "simple_policy_throttle_rows": int(simple["policy"]["throttle_rows"]),
            "aegis_policy_throttle_rows": int(policy["policy"]["throttle_rows"]),
            "aegis_daemon_throttled_rows": int(policy["daemon_throttle"]["throttled_rows"]),
        },
        "interpretation": (
            "The controller rows attribute the SQLite recovery to throttling "
            "elastic mounted-FUSE writes.  The simple controller has slightly "
            "lower p99 in this run, while AEGIS-Q retains more background "
            "throughput and records daemon-side throttle evidence."
        ),
        "scope_boundary": "Single retained SQLite workflow; not foreground TensorRT/AI p99 recovery.",
    }

    keyplane_ablation = {
        "id": "keyplane_cpu_gpu_policy",
        "mechanism": "open-file envelope-refresh placement",
        "variants": ["cpu_only", "gpu_batch", "policy_fallback"],
        "artifact_paths": [relpath(KEYPLANE)],
        "raw_log_paths": [
            "artifacts/validation/keyplane_rekey_methodology/rep_*/cpu_only/mount_logs/pqc_fuse.stderr.txt",
            "artifacts/validation/keyplane_rekey_methodology/rep_*/gpu_batch/admission_trace.jsonl",
            "artifacts/validation/keyplane_rekey_methodology/rep_*/policy_fallback/admission_trace.jsonl",
        ],
        "metrics": {
            "files_per_mode": int(keyplane["files_per_mode"]),
            "repetitions": int(keyplane["repetitions_measured"]),
            "cpu_only_files_per_s": {
                "median": float(cpu["throughput_files_per_s_median"]),
                "ci95_low": float(cpu["throughput_files_per_s_ci95_low"]),
                "ci95_high": float(cpu["throughput_files_per_s_ci95_high"]),
            },
            "gpu_batch_files_per_s": {
                "median": float(gpu["throughput_files_per_s_median"]),
                "ci95_low": float(gpu["throughput_files_per_s_ci95_low"]),
                "ci95_high": float(gpu["throughput_files_per_s_ci95_high"]),
            },
            "policy_fallback_files_per_s": {
                "median": float(fallback["throughput_files_per_s_median"]),
                "ci95_low": float(fallback["throughput_files_per_s_ci95_low"]),
                "ci95_high": float(fallback["throughput_files_per_s_ci95_high"]),
            },
            "gpu_vs_cpu_speedup": keyplane["gpu_vs_cpu_speedup_summary"],
        },
        "interpretation": (
            "The optional batch lane improves this maintenance workflow when "
            "slack is present and falls back to CPU under zero slack/high "
            "pressure without changing the storage format."
        ),
        "scope_boundary": "Open-file envelope refresh only; not deployed credential lifecycle or data-plane acceleration.",
    }

    anchor_ablation = {
        "id": "file_vs_tpm_anchor",
        "mechanism": "external freshness boundary",
        "variants": ["file_anchor_negative_control", "tpm_replay_after_advance"],
        "artifact_paths": [
            relpath(GENERATION),
            relpath(TPM_REPLAY),
        ],
        "raw_log_paths": [
            "artifacts/validation/generation_fault_matrix",
            "artifacts/validation/tpm_monotonic_replay/live_mount",
            "artifacts/validation/tpm_monotonic_replay/replay_mount",
        ],
        "metrics": {
            "file_anchor_oracle_verdict": file_anchor["oracle_verdict"],
            "file_anchor_acceptable": bool(file_anchor["acceptable"]),
            "tpm_anchor_oracle_verdict": tpm_anchor["oracle_verdict"],
            "tpm_anchor_acceptable": bool(tpm_anchor["acceptable"]),
            "tpm_replay_fail_closed": bool(tpm_result.get("fail_closed")),
            "tpm_replay_visible": bool(tpm_result.get("replay_visible")),
        },
        "interpretation": (
            "The file-backed anchor is replayable with the backing directory "
            "and is therefore a negative control.  The retained TPM "
            "replay-after-advance artifact fails closed against stale disk "
            "state."
        ),
        "scope_boundary": "No persistent PCR-bound filesystem freshness, NV authorization lifecycle, or power-loss certification.",
    }

    entries = [
        filesystem_overhead,
        qos_controller,
        keyplane_ablation,
        anchor_ablation,
    ]

    checks = {
        "plaintext_frozen_pass": plaintext.get("overall_pass") is True
        and plaintext.get("filesystem_mode") == "plaintext_lowerfs"
        and plaintext.get("warm_cache_summary", {}).get("valid_repetitions") == 5,
        "aegisq_frozen_pass": aegisq.get("overall_pass") is True
        and aegisq.get("filesystem_mode") == "aegis_q"
        and aegisq.get("warm_cache_summary", {}).get("valid_repetitions") == 5,
        "qos_required_modes_pass": qos.get("overall_pass") is True
        and all(qos_mode(qos, mode).get("acceptable") is True for mode in ("app_only", "unthrottled_storage", "simple_controller", "aegis_policy")),
        "keyplane_required_modes_pass": keyplane.get("overall_pass") is True
        and all(keyplane_mode(keyplane, mode).get("all_acceptable") is True for mode in ("cpu_only", "gpu_batch", "policy_fallback")),
        "anchor_required_modes_pass": generation.get("overall_pass") is True
        and file_anchor.get("oracle_verdict") == "previous_committed"
        and tpm_anchor.get("oracle_verdict") == "fail_closed"
        and tpm_result.get("fail_closed") is True,
    }

    paper_text = PAPER_EVAL.read_text(encoding="utf-8") if PAPER_EVAL.exists() else ""
    paper_snippets = {
        "mechanism_attribution_mentions_manifest": "mechanism-ablation manifest" in paper_text,
        "filesystem_gap_reported": "32.88~MiB/s to 0.359~MiB/s" in paper_text,
        "qos_controller_variants_reported": "unthrottled, simple-controller, and AEGIS-Q policy SQLite rows" in paper_text,
        "keyplane_speedup_reported": "1.186$\\times$ median speedup" in paper_text,
        "anchor_negative_control_reported": "file-backed anchor replay is a negative control" in paper_text,
    }
    checks["paper_scope_gate_pass"] = all(paper_snippets.values())

    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": all(checks.values()),
        "scope": (
            "Mechanism attribution over existing retained evidence.  This is "
            "not a new broad filesystem comparison and does not close "
            "fscrypt/dm-crypt, cold-cache, PCR-bound freshness, or AI QoS gaps."
        ),
        "checks": checks,
        "paper_snippets": paper_snippets,
        "entries": entries,
        "non_claims": [
            "no fscrypt/dm-crypt frozen-contract rows",
            "no cold-cache filesystem result",
            "no plaintext-FUSE decomposition row",
            "no persistent PCR-bound freshness lifecycle",
            "no foreground TensorRT/AI p99 recovery",
            "no data-plane GPU acceleration claim",
        ],
    }


def write_csv(report: dict[str, Any], path: Path) -> None:
    fields = [
        "id",
        "mechanism",
        "variants",
        "primary_metric",
        "interpretation",
        "scope_boundary",
        "artifact_paths",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for entry in report["entries"]:
            metrics = entry["metrics"]
            if entry["id"] == "fs_format_total_cost":
                primary = (
                    f"{metrics['plaintext_throughput_mib_s']['median']:.3f}->"
                    f"{metrics['aegisq_throughput_mib_s']['median']:.3f} MiB/s; "
                    f"p99 {metrics['plaintext_conservative_p99_ms']['median']:.3f}->"
                    f"{metrics['aegisq_conservative_p99_ms']['median']:.3f} ms"
                )
            elif entry["id"] == "qos_controller_variants":
                primary = (
                    f"p99 unthrottled/simple/aegis "
                    f"{metrics['unthrottled_p99_ms']:.3f}/"
                    f"{metrics['simple_controller_p99_ms']:.3f}/"
                    f"{metrics['aegis_policy_p99_ms']:.3f} ms"
                )
            elif entry["id"] == "keyplane_cpu_gpu_policy":
                primary = (
                    f"GPU/CPU speedup "
                    f"{metrics['gpu_vs_cpu_speedup']['median']:.3f}x"
                )
            else:
                primary = (
                    f"file={metrics['file_anchor_oracle_verdict']}; "
                    f"tpm={metrics['tpm_anchor_oracle_verdict']}"
                )
            writer.writerow(
                {
                    "id": entry["id"],
                    "mechanism": entry["mechanism"],
                    "variants": "; ".join(entry["variants"]),
                    "primary_metric": primary,
                    "interpretation": entry["interpretation"],
                    "scope_boundary": entry["scope_boundary"],
                    "artifact_paths": "; ".join(entry["artifact_paths"]),
                }
            )


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Mechanism Ablation Manifest",
        "",
        f"- Overall pass: `{str(report['overall_pass']).lower()}`",
        f"- Scope: {report['scope']}",
        "",
        "## Checks",
        "",
    ]
    for name, value in report["checks"].items():
        lines.append(f"- `{name}`: `{str(value).lower()}`")
    lines.extend(["", "## Entries", ""])
    for entry in report["entries"]:
        lines.extend(
            [
                f"### {entry['id']}",
                "",
                f"- Mechanism: {entry['mechanism']}",
                f"- Variants: `{', '.join(entry['variants'])}`",
                f"- Artifacts: `{', '.join(entry['artifact_paths'])}`",
                f"- Interpretation: {entry['interpretation']}",
                f"- Scope boundary: {entry['scope_boundary']}",
                "",
            ]
        )
        metrics = entry["metrics"]
        if entry["id"] == "fs_format_total_cost":
            lines.append(
                f"- Throughput medians: plaintext `{metrics['plaintext_throughput_mib_s']['median']:.6g}` MiB/s, "
                f"AEGIS-Q `{metrics['aegisq_throughput_mib_s']['median']:.6g}` MiB/s"
            )
            lines.append(
                f"- Conservative p99 medians: plaintext `{metrics['plaintext_conservative_p99_ms']['median']:.6g}` ms, "
                f"AEGIS-Q `{metrics['aegisq_conservative_p99_ms']['median']:.6g}` ms"
            )
        elif entry["id"] == "qos_controller_variants":
            lines.append(
                "- SQLite p99 medians: "
                f"unthrottled `{metrics['unthrottled_p99_ms']:.6g}` ms, "
                f"simple `{metrics['simple_controller_p99_ms']:.6g}` ms, "
                f"AEGIS-Q `{metrics['aegis_policy_p99_ms']:.6g}` ms"
            )
            lines.append(
                "- Background throughput: "
                f"simple `{metrics['simple_background_mb_s']:.6g}` MB/s, "
                f"AEGIS-Q `{metrics['aegis_background_mb_s']:.6g}` MB/s"
            )
        elif entry["id"] == "keyplane_cpu_gpu_policy":
            speedup = metrics["gpu_vs_cpu_speedup"]
            lines.append(
                f"- GPU-vs-CPU speedup median `{speedup['median']:.6g}`x "
                f"(95% CI `{speedup['ci95_low']:.6g}`--`{speedup['ci95_high']:.6g}`)"
            )
        elif entry["id"] == "file_vs_tpm_anchor":
            lines.append(
                f"- Oracle verdicts: file `{metrics['file_anchor_oracle_verdict']}`, "
                f"TPM `{metrics['tpm_anchor_oracle_verdict']}`"
            )
        lines.append("")
    lines.extend(
        [
            "## Non-Claims",
            "",
        ]
    )
    for item in report["non_claims"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    args.out = args.out if args.out.is_absolute() else ROOT / args.out
    args.out.mkdir(parents=True, exist_ok=True)

    report = build_manifest()
    json_path = args.out / "mechanism_ablation_manifest.json"
    csv_path = args.out / "mechanism_ablation_manifest.csv"
    md_path = args.out / "mechanism_ablation_manifest.md"
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
