#!/usr/bin/env python3
"""Build the D2 crypto-plane paper claim guard.

The guard is intentionally narrow: it verifies that paper-facing text keeps
bulk block encryption on AES-GCM, limits ML-KEM/Kyber to the key/session/envelope
plane, and does not imply that key-plane rekey work is on the ordinary mounted
read/write critical path.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "crypto_plane_separation"
TRACE_SMOKE = DEFAULT_OUT / "crypto_plane_trace_smoke.json"
SOURCE_EXTS = {".c", ".h", ".cu", ".cpp", ".hpp", ".py", ".sh"}


@dataclass
class Finding:
    path: str
    line: int
    kind: str
    pattern: str
    text: str
    guarded: bool


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def iter_document_paths() -> list[Path]:
    paths = sorted((ROOT / "Paper").glob("*.tex"))
    for path in (ROOT / "README.md",):
        if path.exists():
            paths.append(path)
    docs = ROOT / "docs"
    if docs.exists():
        paths.extend(sorted(docs.rglob("*.md")))
    return [path for path in paths if path.exists()]


def iter_source_comment_lines(path: Path) -> list[tuple[int, str]]:
    if path.suffix in {".py", ".sh"}:
        rows: list[tuple[int, str]] = []
        for line_no, raw in enumerate(read(path).splitlines(), 1):
            stripped = raw.lstrip()
            if stripped.startswith("#!") or not stripped.startswith("#"):
                continue
            rows.append((line_no, stripped[1:].strip()))
        return [(line_no, line) for line_no, line in rows if line]

    rows: list[tuple[int, str]] = []
    in_block = False
    for line_no, raw in enumerate(read(path).splitlines(), 1):
        text = raw
        while text:
            if in_block:
                end = text.find("*/")
                if end < 0:
                    rows.append((line_no, text.strip()))
                    text = ""
                else:
                    rows.append((line_no, text[:end].strip()))
                    text = text[end + 2:]
                    in_block = False
                continue

            slash = text.find("//")
            block = text.find("/*")
            if slash < 0 and block < 0:
                break
            if slash >= 0 and (block < 0 or slash < block):
                rows.append((line_no, text[slash + 2:].strip()))
                break

            end = text.find("*/", block + 2)
            if end < 0:
                rows.append((line_no, text[block + 2:].strip()))
                in_block = True
                break
            rows.append((line_no, text[block + 2:end].strip()))
            text = text[end + 2:]
    return [(line_no, line) for line_no, line in rows if line]


def iter_claim_lines() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in iter_document_paths():
        for line_no, line in enumerate(read(path).splitlines(), 1):
            rows.append({
                "path": relpath(path),
                "line": line_no,
                "kind": "document",
                "text": line.strip(),
            })

    code_root = ROOT / "code"
    for path in sorted(code_root.rglob("*")):
        if not path.is_file() or path.suffix not in SOURCE_EXTS:
            continue
        if "experiments" in path.parts:
            continue
        for line_no, line in iter_source_comment_lines(path):
            rows.append({
                "path": relpath(path),
                "line": line_no,
                "kind": "source_comment",
                "text": line.strip(),
            })
    return rows


NEGATION_RE = re.compile(
    r"\b(no|not|never|without|cannot|does not|do not|is not|are not|"
    r"rather than|only|optional|prototype|diagnostic|future|requires|"
    r"before|until|unless|boundary|scope|scoped|non-claim|limitation|"
    r"anti-pattern|fallback|not claimed|doesn't)\b|주장하지|아니|제한",
    re.IGNORECASE,
)


DANGEROUS_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "bulk_data_direct_pqc",
        re.compile(
            r"\b(?:bulk|ordinary|file|block|data)[-\w ]{0,45}"
            r"(?:encrypt|encrypted|encryption)[-\w ]{0,45}"
            r"(?:PQC|ML-KEM|Kyber|ML-DSA)\b|"
            r"\b(?:PQC|ML-KEM|Kyber|ML-DSA)[-\w ]{0,45}"
            r"(?:encrypts|encrypted|encryption)[-\w ]{0,45}"
            r"(?:bulk|ordinary|file data|file blocks|blocks|data)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "gpu_crypto_mandatory_data_blocks",
        re.compile(
            r"\b(?:GPU|CUDA)[-\w ]{0,40}(?:mandatory|required|must|only|fixed)"
            r"[-\w ]{0,40}(?:AES-GCM|data plane|data blocks|file blocks)\b|"
            r"\b(?:data plane|data blocks|file blocks)[-\w ]{0,40}"
            r"(?:mandatory|required|must|only|fixed)[-\w ]{0,40}(?:GPU|CUDA)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "keyplane_rekey_rw_critical_path",
        re.compile(
            r"\b(?:rekey|key-plane|key plane|ML-KEM|Kyber)[-\w ]{0,50}"
            r"(?:ordinary|normal|read/write|write path|read path|critical path)"
            r"|"
            r"\b(?:ordinary|normal|read/write|write path|read path|critical path)"
            r"[-\w ]{0,50}(?:rekey|key-plane|key plane|ML-KEM|Kyber)\b",
            re.IGNORECASE,
        ),
    ),
]


def scan_claims() -> list[Finding]:
    findings: list[Finding] = []
    for row in iter_claim_lines():
        text = row["text"]
        if not text:
            continue
        for name, pattern in DANGEROUS_PATTERNS:
            if pattern.search(text):
                findings.append(Finding(
                    path=row["path"],
                    line=int(row["line"]),
                    kind=row["kind"],
                    pattern=name,
                    text=text,
                    guarded=bool(NEGATION_RE.search(text)),
                ))
    return findings


def load_trace_smoke() -> dict[str, Any]:
    if not TRACE_SMOKE.exists():
        return {"exists": False}
    payload = json.loads(read(TRACE_SMOKE))
    verdict = payload.get("verdict", {})
    ordinary_run = payload.get("runs", {}).get("ordinary", {})
    forced_run = payload.get("runs", {}).get("forced_keyplane", {})
    ordinary = payload.get("runs", {}).get("ordinary", {}).get("trace", {})
    forced = payload.get("runs", {}).get("forced_keyplane", {}).get("trace", {})
    ordinary_stderr = ROOT / str(ordinary_run.get("stderr", ""))
    forced_stderr = ROOT / str(forced_run.get("stderr", ""))
    ordinary_stderr_text = read(ordinary_stderr) if ordinary_stderr.exists() else ""
    forced_stderr_text = read(forced_stderr) if forced_stderr.exists() else ""
    return {
        "exists": True,
        "path": relpath(TRACE_SMOKE),
        "overall_pass": verdict.get("overall_pass") is True,
        "source_checks": payload.get("source_checks", {}),
        "ordinary_stderr": ordinary_run.get("stderr"),
        "forced_stderr": forced_run.get("stderr"),
        "ordinary_keyplane_startup_disabled":
            "PQC key-plane rekey disabled for this mount" in ordinary_stderr_text,
        "ordinary_admission_startup_disabled":
            "Elastic admission controller disabled for this mount" in ordinary_stderr_text and
            "Admission controller initialized" not in ordinary_stderr_text,
        "ordinary_scheduler_policy_skipped":
            "Elastic scheduler policy reload skipped for this mount" in ordinary_stderr_text and
            "Scheduler policy:" not in ordinary_stderr_text,
        "ordinary_qos_monitor_skipped":
            "QoS monitor startup skipped for this mount" in ordinary_stderr_text and
            "GPU load monitor started" not in ordinary_stderr_text and
            "Admission telemetry file monitor started" not in ordinary_stderr_text,
        "ordinary_freshness_anchor_disabled":
            ordinary.get("freshness_anchor_events", 0) == 0 and
            ordinary.get("freshness_anchor_file_backend", 0) == 0 and
            ordinary.get("freshness_anchor_hardware_backend", 0) == 0,
        "ordinary_kem_keypair_absent":
            "KEM algorithm" not in ordinary_stderr_text and
            "Keypair generated" not in ordinary_stderr_text and
            "PQC background rekey worker started" not in ordinary_stderr_text,
        "forced_keyplane_worker_started":
            forced_run.get("rekey_log_seen") is True and
            forced.get("keyplane_batches", 0) > 0 and
            forced.get("keyplane_refreshed_files", 0) > 0,
        "source_plane_boundary_present":
            verdict.get("source_plane_boundary_present") is True,
        "ordinary_aes_encrypt_blocks":
            ordinary.get("data_aes_gcm_encrypt_blocks", 0),
        "ordinary_aes_decrypt_blocks":
            ordinary.get("data_aes_gcm_decrypt_blocks", 0),
        "ordinary_keyplane_batches": ordinary.get("keyplane_batches", 0),
        "ordinary_keyplane_refreshed_files":
            ordinary.get("keyplane_refreshed_files", 0),
        "forced_keyplane_batches": forced.get("keyplane_batches", 0),
        "forced_keyplane_refreshed_files":
            forced.get("keyplane_refreshed_files", 0),
        "ordinary_read_write_aes_gcm_only":
            verdict.get("ordinary_read_write_aes_gcm_only") is True,
        "forced_rekey_is_keyplane_only":
            verdict.get("forced_rekey_is_keyplane_only") is True,
    }


def paper_requirements() -> dict[str, Any]:
    paper = "\n".join(read(path) for path in sorted((ROOT / "Paper").glob("*.tex")))
    requirements = {
        "paper_says_data_plane_aes_gcm":
            bool(re.search(r"AES-GCM encrypts and authenticates each data block", paper)),
        "paper_says_mlkem_limited_to_keyplane_envelope":
            bool(re.search(r"ML-KEM only for the optional batched key-plane experiment", paper)),
        "paper_says_mlkem_does_not_encrypt_blocks":
            bool(re.search(r"ML-KEM establishes shared secrets, whereas AES-GCM remains the appropriate mechanism for bulk data records", paper)),
        "paper_says_aes_data_writes_cpu_first":
            bool(re.search(r"keeps bulk data encryption on the CPU path", paper)),
        "paper_says_rekey_not_hardware_lifecycle":
            bool(re.search(r"not a persistent KEM hierarchy or hardware-backed credential lifecycle", paper)),
        "paper_says_default_mount_skips_keyplane_startup":
            bool(re.search(r"ordinary mounts with no rekey trigger (?:do not allocate|allocate no) (?:an )?ML-KEM object", paper)),
        "paper_says_default_mount_skips_cuda_executor":
            bool(re.search(r"(?:do not preallocate|allocate no)[-\w\s,;]*CUDA AES executor", paper)),
        "paper_says_default_mount_skips_unused_admission":
            (
                "elastic admission state" in paper
                and "scheduler accounting" in paper
                and "QoS monitor" in paper
            ),
        "paper_says_external_anchor_is_optional":
            bool(re.search(r"(?:external freshness anchor remains disabled unless configured|external anchor)", paper)),
    }
    return {
        "requirements": requirements,
        "all_present": all(requirements.values()),
    }


def build_report() -> dict[str, Any]:
    findings = scan_claims()
    unguarded = [finding for finding in findings if not finding.guarded]
    trace = load_trace_smoke()
    paper = paper_requirements()
    checks = {
        "trace_smoke_exists": trace.get("exists") is True,
        "trace_smoke_overall_pass": trace.get("overall_pass") is True,
        "ordinary_io_has_aes_blocks":
            int(trace.get("ordinary_aes_encrypt_blocks") or 0) > 0 and
            int(trace.get("ordinary_aes_decrypt_blocks") or 0) > 0,
        "ordinary_io_has_no_keyplane_batches":
            int(trace.get("ordinary_keyplane_batches") or 0) == 0 and
            int(trace.get("ordinary_keyplane_refreshed_files") or 0) == 0,
        "ordinary_mount_skips_keyplane_startup":
            trace.get("ordinary_keyplane_startup_disabled") is True and
            trace.get("ordinary_kem_keypair_absent") is True,
        "ordinary_mount_skips_admission_startup":
            trace.get("ordinary_admission_startup_disabled") is True and
            trace.get("ordinary_scheduler_policy_skipped") is True and
            trace.get("ordinary_qos_monitor_skipped") is True,
        "ordinary_mount_skips_freshness_anchor":
            trace.get("ordinary_freshness_anchor_disabled") is True,
        "forced_mount_starts_keyplane_worker":
            trace.get("forced_keyplane_worker_started") is True,
        "forced_rekey_records_keyplane_work":
            int(trace.get("forced_keyplane_batches") or 0) > 0 and
            int(trace.get("forced_keyplane_refreshed_files") or 0) > 0,
        "source_ordinary_writeback_cpu_first":
            trace.get("source_checks", {}).get("ordinary_writeback_disables_gpu_batch") is True and
            trace.get("source_checks", {}).get("gpu_batch_requires_explicit_request") is True and
            trace.get("source_checks", {}).get("gpu_batch_tags_not_recomputed_in_flush_wrapper") is True and
            trace.get("source_checks", {}).get("gpu_batch_fallback_owned_by_crypto_layer") is True,
        "source_rekey_gpu_lane_reachable_by_default":
            trace.get("source_checks", {}).get("rekey_batch_can_reach_gpu_byte_gate") is True and
            trace.get("source_checks", {}).get("rekey_success_logs_are_verbose_gated") is True,
        "source_runtime_no_cuda_aes_prealloc":
            trace.get("source_checks", {}).get("runtime_does_not_preallocate_cuda_aes_executor") is True,
        "source_runtime_disables_unused_admission":
            trace.get("source_checks", {}).get("runtime_disables_admission_when_unused") is True and
            trace.get("source_checks", {}).get("runtime_skips_scheduler_policy_when_unused") is True and
            trace.get("source_checks", {}).get("scheduler_data_accounting_is_gated") is True and
            trace.get("source_checks", {}).get("fault_cutpoints_are_fast_gated") is True and
            trace.get("source_checks", {}).get("rekey_policy_snapshot_is_trigger_gated") is True and
            trace.get("source_checks", {}).get("qos_open_xattr_load_is_throttle_gated") is True and
            trace.get("source_checks", {}).get("qos_writeback_uses_single_throttle_gate") is True and
            trace.get("source_checks", {}).get("read_visible_size_snapshot_is_single_pass") is True and
            trace.get("source_checks", {}).get("read_eof_skips_scratch_acquire") is True and
            trace.get("source_checks", {}).get("read_marker_path_copy_is_epoch_gated") is True and
            trace.get("source_checks", {}).get("writeback_reuses_published_logical_size") is True and
            trace.get("source_checks", {}).get("writeback_snapshots_secret_and_epoch_path_conditionally") is True and
            trace.get("source_checks", {}).get("strict_open_allocates_epoch_cache_only_when_needed") is True and
            trace.get("source_checks", {}).get("release_hidden_cleanup_is_conditional") is True and
            trace.get("source_checks", {}).get("runtime_skips_qos_monitors_when_unused") is True,
        "source_anchor_backend_cached":
            trace.get("source_checks", {}).get("anchor_backend_is_cached") is True and
            trace.get("source_checks", {}).get("anchor_disabled_skips_window_policy") is True and
            trace.get("source_checks", {}).get("runtime_skips_anchor_probe_when_disabled") is True,
        "source_trace_byte_aggregation_gated":
            trace.get("source_checks", {}).get("trace_byte_aggregation_is_gated") is True,
        "paper_crypto_plane_boundary_present": paper["all_present"],
        "no_unguarded_d2_overclaims": len(unguarded) == 0,
    }
    parent_d2_closed = all(checks.values())
    return {
        "generated_at": now_utc(),
        "generated_by": "code/experiments/build_crypto_plane_claim_guard.py",
        "trace_smoke": trace,
        "paper": paper,
        "checks": checks,
        "findings": [finding.__dict__ for finding in findings],
        "unguarded_findings": [finding.__dict__ for finding in unguarded],
        "unguarded_count": len(unguarded),
        "overall_pass": parent_d2_closed,
        "parent_d2_closed": parent_d2_closed,
        "negative_claim_guard": (
            "Bulk file data must remain described as AES-GCM block data. "
            "ML-KEM/Kyber may be described only as key/session/envelope-plane "
            "work, and ordinary read/write claims must not put rekey on the "
            "critical path or start key-plane machinery unless mounted trace "
            "evidence changes."
        ),
    }


def write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "crypto_plane_claim_guard.json"
    md_path = out_dir / "crypto_plane_claim_guard.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")

    lines = [
        "# Crypto Plane Claim Guard",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Overall pass: `{str(report['overall_pass']).lower()}`",
        f"- Parent D2 closed: `{str(report['parent_d2_closed']).lower()}`",
        f"- Unguarded D2 overclaims: `{report['unguarded_count']}`",
        "",
        "## Checks",
        "",
    ]
    for key, value in report["checks"].items():
        lines.append(f"- {key}: `{str(value).lower()}`")
    lines.extend(["", "## Negative Claim Guard", "", report["negative_claim_guard"], ""])
    if report["unguarded_findings"]:
        lines.extend(["", "## Unguarded Findings", ""])
        for finding in report["unguarded_findings"]:
            lines.append(
                f"- `{finding['path']}:{finding['line']}` "
                f"{finding['pattern']}: {finding['text']}"
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    report = build_report()
    write_report(report, args.out_dir)
    print(json.dumps({
        "overall_pass": report["overall_pass"],
        "parent_d2_closed": report["parent_d2_closed"],
        "unguarded_count": report["unguarded_count"],
        "json": relpath(args.out_dir / "crypto_plane_claim_guard.json"),
        "md": relpath(args.out_dir / "crypto_plane_claim_guard.md"),
    }, indent=2, sort_keys=True))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
