#!/usr/bin/env python3
"""Build Gate 0 refactor inventories from the current source tree.

The output is intentionally conservative.  It records where the monolith owns
state and mechanisms today, so later refactors can prove behavior equivalence
instead of relying on memory or paper text.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
OUT = ROOT / "artifacts" / "validation" / "refactor_inventory"
DOCS = ROOT / "docs" / "architecture"

SOURCE_FILES = [
    "pqc_fuse.c",
    "pqc_anchor.c",
    "pqc_anchor_worker.c",
    "pqc_admission.c",
    "pqc_checkpoint.c",
    "pqc_config.c",
    "pqc_crypto.c",
    "pqc_epoch_log.c",
    "pqc_epoch_publish.c",
    "pqc_fd_context.c",
    "pqc_file_lock.c",
    "pqc_file_io.c",
    "pqc_flush_batch.c",
    "pqc_flush_crypto.c",
    "pqc_journal.c",
    "pqc_keyring.c",
    "pqc_lifecycle.c",
    "pqc_lock_profile.c",
    "pqc_main.c",
    "pqc_metrics.c",
    "pqc_namespace.c",
    "pqc_parallel_commit.c",
    "pqc_posix.c",
    "pqc_publish.c",
    "pqc_qos.c",
    "pqc_recovery.c",
    "pqc_rekey.c",
    "pqc_runtime.c",
    "pqc_scheduler.c",
    "pqc_selftest.c",
    "pqc_state.c",
    "pqc_storage_path.c",
    "pqc_strict_publish.c",
    "pqc_writeback.c",
    "pqc_xattr.c",
    "pqc_format.h",
    "pqc_test_hooks.c",
    "pqc_anchor.h",
    "pqc_admission.h",
    "pqc_anchor_worker.h",
    "pqc_block_job.h",
    "pqc_checkpoint.h",
    "pqc_config.h",
    "pqc_crypto.h",
    "pqc_epoch_log.h",
    "pqc_epoch_publish.h",
    "pqc_fd_context.h",
    "pqc_file_lock.h",
    "pqc_file_io.h",
    "pqc_flush_batch.h",
    "pqc_flush_crypto.h",
    "pqc_journal.h",
    "pqc_keyring.h",
    "pqc_lifecycle.h",
    "pqc_lock_profile.h",
    "pqc_fuse.h",
    "pqc_metrics.h",
    "pqc_namespace.h",
    "pqc_parallel_commit.h",
    "pqc_posix.h",
    "pqc_publish.h",
    "pqc_qos.h",
    "pqc_recovery.h",
    "pqc_rekey.h",
    "pqc_runtime.h",
    "pqc_scheduler.h",
    "pqc_selftest.h",
    "pqc_state.h",
    "pqc_storage_path.h",
    "pqc_strict_publish.h",
    "pqc_writeback.h",
    "pqc_test_hooks.h",
    "pqc_xattr.h",
    "cuda_aead.cu",
    "cuda_aead.h",
    "cuda_pqc.cu",
    "cuda_pqc.h",
    "cuda_integrity.cu",
    "cuda_integrity.h",
]

TARGET_MODULES = [
    "config",
    "error",
    "entrypoint",
    "format",
    "data crypto",
    "keyring",
    "publish",
    "recovery",
    "anchor",
    "merkle",
    "posix",
    "metrics",
    "test hooks",
    "FUSE adapter",
    "runtime",
    "admission",
    "CUDA backend",
]

FORBIDDEN_CONTROL_WORDS = {
    "if",
    "for",
    "while",
    "switch",
    "return",
    "sizeof",
    "catch",
}


def discover_source_files(seed: list[str]) -> list[str]:
    """Return every production source/header name currently under code/."""
    names = set(seed)
    for pattern in ("pqc_*.c", "pqc_*.h", "cuda_*.cu", "cuda_*.h"):
        for path in CODE.rglob(pattern):
            names.add(path.name)
    ordered = [name for name in seed if name in names]
    ordered.extend(sorted(names - set(ordered)))
    return ordered


SOURCE_FILES = discover_source_files(SOURCE_FILES)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def source_path(name: str) -> Path:
    code_path = CODE / name
    if code_path.exists():
        return code_path
    matches = sorted(CODE.rglob(name))
    if matches:
        return matches[0]
    return ROOT / name


def source_id(path: Path) -> str:
    try:
        rel_path = path.relative_to(CODE)
        return str(rel_path)
    except ValueError:
        return str(path.relative_to(ROOT))


def source_basename(path_or_name: str) -> str:
    return Path(path_or_name).name


def normalize_worktree_path(path: str) -> str:
    return path[5:] if path.startswith("code/") else path


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_json_optional(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def artifact_pass(path: str, key: str = "overall_pass") -> bool:
    payload = load_json_optional(ROOT / path)
    return bool(payload and payload.get(key) is True)


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_git(args: list[str]) -> list[str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        return [f"<git {' '.join(args)} failed: {completed.stderr.strip()}>"]
    return completed.stdout.splitlines()


def first_or_unknown(lines: list[str]) -> str:
    return lines[0] if lines else "unknown"


def classify_worktree_path(path: str) -> dict[str, str]:
    logical_path = normalize_worktree_path(path)
    module_files = {
        "pqc_anchor_worker.c",
        "pqc_anchor_worker.h",
        "pqc_checkpoint.c",
        "pqc_checkpoint.h",
        "pqc_config.c",
        "pqc_config.h",
        "pqc_crypto.c",
        "pqc_crypto.h",
        "pqc_fd_context.c",
        "pqc_fd_context.h",
        "pqc_file_lock.c",
        "pqc_file_lock.h",
        "pqc_file_io.c",
        "pqc_file_io.h",
        "pqc_flush_batch.c",
        "pqc_flush_batch.h",
        "pqc_flush_crypto.c",
        "pqc_flush_crypto.h",
        "pqc_format.h",
        "pqc_journal.c",
        "pqc_journal.h",
        "pqc_keyring.c",
        "pqc_keyring.h",
        "pqc_lifecycle.c",
        "pqc_lifecycle.h",
        "pqc_lock_profile.c",
        "pqc_lock_profile.h",
        "pqc_main.c",
        "pqc_fuse.h",
        "pqc_metrics.c",
        "pqc_metrics.h",
        "pqc_namespace.c",
        "pqc_namespace.h",
        "pqc_parallel_commit.c",
        "pqc_parallel_commit.h",
        "pqc_posix.c",
        "pqc_posix.h",
        "pqc_publish.c",
        "pqc_publish.h",
        "pqc_qos.c",
        "pqc_qos.h",
        "pqc_recovery.c",
        "pqc_recovery.h",
        "pqc_rekey.c",
        "pqc_rekey.h",
        "pqc_runtime.c",
        "pqc_runtime.h",
        "pqc_scheduler.c",
        "pqc_scheduler.h",
        "pqc_selftest.c",
        "pqc_selftest.h",
        "pqc_state.c",
        "pqc_state.h",
        "pqc_storage_path.c",
        "pqc_storage_path.h",
        "pqc_strict_publish.c",
        "pqc_strict_publish.h",
        "pqc_writeback.c",
        "pqc_writeback.h",
        "pqc_test_hooks.c",
        "pqc_test_hooks.h",
        "pqc_xattr.c",
        "pqc_xattr.h",
    }
    if path == "SUBMISSION_CHECKLIST.md":
        return {
            "owner": "codex",
            "change_class": "authoritative-checklist-rewrite",
            "status": "pending-review",
            "claim_status": "checklist-only",
            "notes": "Authoritative Gate 0-first checklist; boxes remain unchecked until code/script, artifact, paper text, and negative-claim guard exist.",
        }
    if logical_path == "pqc_fuse.c":
        return {
            "owner": "codex",
            "change_class": "mechanical-decomposition-plus-dirty-sidecar-patch",
            "status": "pending-unclaimed",
            "claim_status": "no-performance-or-correctness-claim",
            "notes": "Still carries the dirty-sidecar sync dedup patch while also routing extracted config, crypto, journal, keyring, POSIX/path, publish, checkpoint, anchor-worker, recovery, state, test-hook, and strict-publish helpers.",
        }
    if logical_path in {"pqc_anchor.c", "pqc_admission.c"}:
        return {
            "owner": "codex",
            "change_class": "gate0-mechanical-extraction-callsite-update",
            "status": "pending-behavior-equivalence",
            "claim_status": "no-new-paper-claim",
            "notes": "Routes existing behavior through extracted Gate 0 helper modules without claiming a new mechanism.",
        }
    if path == "CMakeLists.txt":
        return {
            "owner": "codex",
            "change_class": "build-graph-update",
            "status": "required-for-extracted-modules",
            "claim_status": "no-new-paper-claim",
            "notes": "Adds extracted Gate 0 modules to the pqc_fuse build.",
        }
    if path == "experiments":
        return {
            "owner": "codex",
            "change_class": "legacy-source-root-removal",
            "status": "removed-in-favor-of-code-experiments",
            "claim_status": "no-new-paper-claim",
            "notes": "Legacy root experiments path is removed; experiment, benchmark, report, and probe code lives under code/experiments.",
        }
    if logical_path == "experiments/scheduler_trace.jsonl":
        return {
            "owner": "codex",
            "change_class": "generated-trace-removal",
            "status": "removed-from-source-tree",
            "claim_status": "no-new-paper-claim",
            "notes": "Default scheduler/admission trace output now goes under artifacts/validation instead of the source tree.",
        }
    if logical_path == "experiments/build_refactor_inventory.py":
        return {
            "owner": "codex",
            "change_class": "gate0-evidence-generator",
            "status": "retained-script",
            "claim_status": "artifact-generation-only",
            "notes": "Generates refactor, ownership, lock, fault, durability, format, config, POSIX/path, and worktree-freeze inventories.",
        }
    if logical_path.startswith("experiments/"):
        return {
            "owner": "codex",
            "change_class": "experiment-source-relocation",
            "status": "moved-under-code-root",
            "claim_status": "no-new-paper-claim",
            "notes": "Experiment, report, benchmark, shell, and trace-helper source now lives under code/experiments; root experiments is removed.",
        }
    if path.startswith("artifacts/validation/refactor_inventory/"):
        return {
            "owner": "codex",
            "change_class": "retained-gate0-artifact",
            "status": "generated-evidence",
            "claim_status": "does-not-close-paper-gate-alone",
            "notes": "Retained inventory evidence; paper closure still requires corresponding paper text and negative-claim guard audit.",
        }
    if path.startswith("docs/architecture/"):
        return {
            "owner": "codex",
            "change_class": "generated-architecture-inventory",
            "status": "generated-evidence",
            "claim_status": "does-not-claim-clean-architecture",
            "notes": "Human-readable architecture inventory generated from current sources.",
        }
    if logical_path in module_files:
        return {
            "owner": "codex",
            "change_class": "gate0-mechanical-module-extraction",
            "status": "pending-full-gate-0.14-closure",
            "claim_status": "behavior-preserving-only",
            "notes": "Extracted helper module for Gate 0 decomposition; does not by itself claim epoch publication, fine-grained locking, or production-clean architecture.",
        }
    return {
        "owner": "unknown",
        "change_class": "unclassified",
        "status": "requires-triage",
        "claim_status": "no-claim-allowed",
        "notes": "This path is not classified by the Gate 0 freeze script and must be triaged before any claim uses it.",
    }


def build_worktree_freeze() -> dict[str, Any]:
    status_sb = run_git(["status", "-sb"])
    porcelain = run_git(["status", "--porcelain=v1", "-uall"])
    diff_name_only = run_git(["diff", "--name-only"])
    untracked_files = run_git(["ls-files", "--others", "--exclude-standard"])
    changed_paths = sorted(set(diff_name_only) | set(untracked_files))

    fuse_text = source_path("pqc_fuse.c").read_text(encoding="utf-8", errors="replace")
    strict_publish_text = source_path("pqc_strict_publish.c").read_text(encoding="utf-8", errors="replace") if source_path("pqc_strict_publish.c").exists() else ""
    fd_context_text = ""
    for name in ("pqc_fd_context.h", "pqc_fd_context.c"):
        path = source_path(name)
        if path.exists():
            fd_context_text += "\n" + path.read_text(encoding="utf-8", errors="replace")
    dirty_sidecar_presence = {
        "pqc_fd_ctx_has_dirty_flags": "data_sidecar_dirty" in fd_context_text and "journal_sidecar_dirty" in fd_context_text,
        "sidecar_sync_uses_dup_snapshot": "pqc_fd_context_prepare_dirty_sidecar_sync_locked" in fd_context_text and "dup(ctx->data_fd)" in fd_context_text,
        "sidecar_sync_runs_outside_fd_context_module": "pqc_fd_context_run_dirty_sidecar_sync" in fd_context_text,
        "sidecar_sync_finish_checks_epoch": "pqc_fd_context_finish_dirty_sidecar_sync_locked" in fd_context_text and "dirty_epoch == sync->" in fd_context_text,
        "strict_publish_marks_dirty": "*req->data_sidecar_dirty = 1" in strict_publish_text and "*req->journal_sidecar_dirty = 1" in strict_publish_text,
        "strict_publish_advances_dirty_epochs": "++*req->data_sidecar_dirty_epoch" in strict_publish_text and "++*req->journal_sidecar_dirty_epoch" in strict_publish_text,
        "strict_publish_clears_after_sync": "*req->data_sidecar_dirty = 0" in strict_publish_text and "*req->journal_sidecar_dirty = 0" in strict_publish_text,
    }

    return {
        "schema_version": 2,
        "captured_utc": now_utc(),
        "generated_by": "experiments/build_refactor_inventory.py",
        "repository": {
            "branch": first_or_unknown(run_git(["branch", "--show-current"])),
            "head": first_or_unknown(run_git(["rev-parse", "HEAD"])),
            "upstream_status": first_or_unknown(status_sb),
        },
        "freeze_scope": {
            "authoritative_checklist": "SUBMISSION_CHECKLIST.md",
            "obsolete_strategy_file_expected_absent": "CODE_REFACTOR_STRATEGY.md",
            "obsolete_strategy_file_present": (ROOT / "CODE_REFACTOR_STRATEGY.md").exists(),
            "paper_files_touched_in_freeze": [path for path in changed_paths if path.startswith("Paper/")],
            "paper_claims_upgraded_in_freeze": False,
            "checklist_items_marked_complete_in_freeze": False,
        },
        "captured_worktree_at_generation": {
            "git_status_sb": status_sb,
            "git_status_porcelain_v1_uall": porcelain,
            "git_diff_name_only": diff_name_only,
            "untracked_files": untracked_files,
        },
        "changed_file_ownership": [
            {"path": path, **classify_worktree_path(path)}
            for path in changed_paths
        ],
        "dirty_sidecar_sync_dedup_patch": {
            "classification": "pending-unclaimed",
            "retention_decision": "carry-forward-for-benchmark-or-revert",
            "reason": "The patch is still present and plausible as a strict-mode cleanup.  A post-decomposition strict frozen-contract run exists, but no before/after benchmark, syscall/fdatasync attribution, full dirty-flag fault matrix, or paper negative-claim audit has isolated this patch.",
            "evidence_presence": dirty_sidecar_presence,
            "modified_areas": [
                "pqc_fd_ctx_t carries data_sidecar_dirty and journal_sidecar_dirty flags",
                "ctx_set, ctx_clear, and pqc_subsystem_init initialize dirty flags",
                "ctx_wait_pending_locked waits for pending jobs before sidecar sync/release paths",
                "dirty sidecar sync snapshots duplicated sidecar descriptors under fd_lock, performs fdatasync outside fd_lock, and clears dirty flags only when the captured dirty epoch still matches",
                "pqc_strict_publish_commit marks data and journal sidecars dirty after writes/appends and clean after successful fdatasync",
                "pqc_fsync, pqc_flush, and pqc_release propagate flush/sync errors instead of ignoring sidecar durability failures",
            ],
            "known_risks_to_verify": [
                "No before/after benchmark yet proves reduced sync count or improved p99 under the frozen workload contract.",
                "No syscall trace yet proves sidecar fdatasync count changed in the mounted production path.",
                "No fault-injection evidence yet proves dirty flags are correct across partial data write, data fdatasync failure, journal append failure, journal fdatasync failure, daemon kill, flush, fsync, or release error paths.",
                "The patch no longer holds fd_lock for residual dirty-sidecar sync.  Strict full-tier writeback now snapshots the fd write buffer, reserves generations, runs prepare/crypto outside fd_lock and commit_lock, performs strict durable data/journal/checkpoint publication outside commit_lock, authenticated read snapshots state before recovery/authentication outside fd_lock, truncate/fallocate metadata publication uses publish-turn ordering outside hot locks, and release detaches fds/buffers/file-state references before close/free/release work.  The concurrency smoke now includes same-file and disjoint-file repeated open/write/fdatasync/read/close lifecycle phases for both in-process threads and external OS processes, plus a strace profile of client-visible mounted-path fdatasync/pwrite/pread blocking time.  Gate 0.15 remains open because reserved-generation fault expansion, long-duration client-count sweeps, and scheduler off-CPU evidence are still missing.",
                "The patch is not an epoch publication protocol, group commit, sharded queue, or checkpoint compaction mechanism.",
            ],
            "required_before_claim": [
                "strict frozen-contract benchmark before/after this patch",
                "syscall or trace evidence showing sidecar fdatasync count change",
                "generation/fault matrix covering data and journal dirty flag transitions",
                "lock-hold evidence showing whether fd_lock covers blocking sync",
                "build, ctest, and git diff whitespace checks",
                "paper negative-claim guard preventing any throughput or crash-safety claim from this patch until retained evidence exists",
            ],
        },
        "negative_claim_guards": [
            "Do not claim that the dirty-sidecar patch fixes the AEGIS-Q throughput bottleneck.",
            "Do not claim that strict mode is optimized or production-clean from this patch alone.",
            "Do not claim batching, epoch publication, group commit, sharded commit, or fine-grained locking from this patch.",
            "Do not claim crash certification or power-loss safety from dirty flag bookkeeping without the retained fault matrix.",
            "Do not claim codebase production-clean status from mechanical extraction until Gate 0.14, 0.15, 0.16, and 0.9 evidence closes.",
        ],
        "next_gate0_steps": [
            "Keep dirty-sidecar sync dedup classified as pending-unclaimed until benchmarked/retained or reverted.",
            "Continue Gate 0.14 by shrinking the FUSE adapter boundary without changing publication semantics.",
            "Before Gate 0.15, instrument lock hold time and move blocking work out of hot locks.",
            "Before Gate 0.9, preserve strict mode while designing epoch redo-log, group commit, checkpoint compaction, replay, and fault matrix.",
        ],
    }


def target_module_for_symbol(name: str, body: str = "") -> str:
    lower = f"{name}\n{body}".lower()
    name_lower = name.lower()
    if name_lower in {"main", "print_usage"}:
        return "entrypoint"
    if name_lower in {"pqc_lock", "pqc_flock"}:
        return "posix"
    if name_lower in {"pqc_destroy", "pqc_fuse_init"}:
        return "runtime"
    if name_lower in {"pqc_fuse_init", "pqc_fuse_operations"}:
        return "FUSE adapter"
    if name_lower.startswith("pqc_runtime_"):
        return "runtime"
    if name_lower in {"pqc_open", "pqc_create"}:
        return "keyring"
    if name_lower == "pqc_read":
        return "recovery"
    if name_lower in {
        "pqc_write",
        "pqc_fsync",
        "pqc_flush",
        "pqc_truncate",
        "pqc_fallocate",
        "pqc_release",
    }:
        return "publish"
    if name_lower == "restore_qos_class_for_fd":
        return "admission"
    if (
        name_lower == "pqc_cleanup"
        or name_lower.startswith("ctx_")
        or name_lower.startswith("pqc_fd_context_")
    ):
        return "FUSE adapter"
    if (
        name_lower == "pqc_log"
        or name_lower.startswith("pqc_metrics_")
        or name_lower.startswith("pqc_lock_profile_")
        or name_lower.startswith("pqc_profiled_mutex_")
    ):
        return "metrics"
    if name_lower.startswith("pqc_selftest_"):
        return "test hooks"
    if (
        name_lower == "pqc_admit"
        or name_lower.startswith("pqc_admission_")
        or name_lower.startswith("pqc_qos_")
        or name_lower.startswith("pqc_scheduler_")
        or name_lower == "default_policy"
        or name_lower.startswith("qos_")
        or "qos_class" in name_lower
    ):
        return "admission"
    if name_lower.endswith("_self_test") or "smoke_report" in name_lower:
        return "test hooks"
    if name_lower.startswith("pqc_anchor_worker_") or name_lower == "anchor_worker_main":
        return "anchor"
    if (
        name_lower.startswith("pqc_strict_publish_")
        or name_lower.startswith("pqc_writeback_")
        or name_lower.startswith("pqc_parallel_commit_")
    ):
        return "publish"
    if name_lower.startswith("pqc_storage_path_"):
        return "posix"
    if (
        name_lower.startswith("pqc_xattr_")
        or name_lower in {"pqc_setxattr", "pqc_getxattr", "pqc_listxattr"}
    ):
        return "posix"
    if (
        name_lower.startswith("pqc_namespace_")
        or name_lower in {
            "pqc_getattr",
            "pqc_readdir",
            "pqc_unlink",
            "pqc_mkdir",
            "pqc_rmdir",
            "pqc_rename",
            "pqc_fsyncdir",
            "pqc_utimens",
        }
    ):
        return "posix"
    if name_lower == "g_storage_dir":
        return "posix"
    if name_lower in {"pqc_hash_path", "pqc_path_has_suffix"} or "sidecar" in name_lower:
        return "posix"
    if "recovery" in name_lower or "load_authenticated_block" in name_lower:
        return "recovery"
    if (
        name_lower.startswith("pqc_flush_batch_")
        or name_lower.startswith("pqc_flush_crypto_")
        or name_lower == "block_plaintext_length"
        or name_lower in {"flush_crypto_begin_job", "flush_crypto_end_job"}
    ):
        return "data crypto"
    if name_lower.startswith("pqc_crypto_") or any(token in name_lower for token in (
        "crypto",
        "gcm",
        "nonce",
        "aad",
        "aes",
        "authenticated_block",
    )) or name_lower == "store_u64_be":
        return "data crypto"
    if "file_state" in name_lower or "commit_lock" in name_lower:
        return "publish"
    if any(token in name_lower for token in ("committed", "pending_anchor", "global_sequence", "freshness")):
        return "anchor"
    if name_lower in {"g_kem", "g_public_key", "g_secret_key", "g_master_key", "g_has_master_key"}:
        return "keyring"
    if name_lower.startswith("g_profile_"):
        return "metrics"
    if any(token in name_lower for token in ("telemetry", "sched", "qos", "gpu_load", "rekey", "admission", "gpu_inflight")):
        return "admission"
    if "fault" in name_lower:
        return "test hooks"
    if name_lower.startswith("pqc_") and any(op in name_lower for op in (
        "open",
        "read",
        "write",
        "fsync",
        "flush",
        "release",
        "unlink",
        "mkdir",
        "rmdir",
        "rename",
        "truncate",
        "fallocate",
        "create",
        "lock",
        "flock",
        "setxattr",
        "getxattr",
        "listxattr",
        "getattr",
        "readdir",
        "destroy",
        "utimens",
        "fsyncdir",
    )):
        return "FUSE adapter"
    if name_lower.endswith("_self_test") or "smoke_report" in name_lower:
        return "test hooks"
    if "journal" in name_lower or "checkpoint" in name_lower or "logical_size" in name_lower or "flush_wbuf" in name_lower or "ctx_sync" in name_lower:
        return "publish"
    if "anchor" in name_lower or "tpm" in name_lower or "freshness" in name_lower:
        return "anchor"
    if "qos" in name_lower or "sched" in name_lower or "admission" in name_lower or "gpu_load" in name_lower or "rekey" in name_lower:
        return "admission"
    if "metadata" in name_lower or "master_key" in name_lower or "shared_secret" in name_lower or "kdf" in name_lower:
        return "keyring"
    if "crypto" in name_lower or "gcm" in name_lower or "nonce" in name_lower or "aad" in name_lower or "aes" in name_lower or "authenticated_block" in name_lower:
        return "data crypto"
    if "getenv" in lower or name.startswith("parse_") or "config" in lower or "policy_from_env" in lower:
        return "config"
    if "merkle" in lower or "prefix" in lower or "sha256" in lower:
        return "merkle"
    if "cuda" in lower or "gpu" in lower or "skim_" in lower:
        return "CUDA backend"
    if "xattr" in lower or "mmap" in lower or "sqlite" in lower or "sidecar_path" in lower:
        return "posix"
    if "log" in lower or "time" in lower or "stats" in lower or "trace" in lower:
        return "metrics"
    if "error" in lower or "errno" in lower:
        return "error"
    if "format" in lower or "magic" in lower or "version" in lower:
        return "format"
    return "FUSE adapter"


def source_module_hint(path: Path) -> str:
    rel_path = source_id(path)
    base = source_basename(rel_path)
    if rel_path.startswith("crypto/"):
        return "keyring" if base in {"pqc_keyring.c", "pqc_keyring.h"} else "data crypto"
    if rel_path.startswith("fs/"):
        if base in {"pqc_file_io.c", "pqc_file_io.h"}:
            return "publish"
        if base in {"pqc_recovery.c", "pqc_recovery.h"}:
            return "recovery"
        if base in {"pqc_namespace.c", "pqc_namespace.h", "pqc_posix.c", "pqc_posix.h", "pqc_file_lock.c", "pqc_file_lock.h"}:
            return "posix"
        if base in {"pqc_parallel_commit.c", "pqc_parallel_commit.h"}:
            return "publish"
        return "FUSE adapter"
    if rel_path.startswith("storage/"):
        if base in {"pqc_anchor.c", "pqc_anchor.h", "pqc_anchor_worker.c", "pqc_anchor_worker.h"}:
            return "anchor"
        if base in {"pqc_state.c", "pqc_state.h", "pqc_strict_publish.c", "pqc_strict_publish.h", "pqc_writeback.c", "pqc_writeback.h", "pqc_epoch_publish.c", "pqc_epoch_publish.h", "pqc_epoch_log.c", "pqc_epoch_log.h", "pqc_journal.c", "pqc_journal.h", "pqc_checkpoint.c", "pqc_checkpoint.h", "pqc_flush_batch.c", "pqc_flush_batch.h"}:
            return "publish"
        if base in {"pqc_storage_path.c", "pqc_storage_path.h", "pqc_xattr.c", "pqc_xattr.h"}:
            return "posix"
        if base in {"pqc_durability.c", "pqc_durability.h"}:
            return "publish"
    if rel_path.startswith("runtime/"):
        if base in {"pqc_config.c", "pqc_config.h"}:
            return "config"
        if base in {"pqc_metrics.c", "pqc_metrics.h", "pqc_plane_trace.c", "pqc_plane_trace.h"}:
            return "metrics"
        if base in {"pqc_lifecycle.c", "pqc_lifecycle.h", "pqc_runtime.c", "pqc_runtime.h"}:
            return "runtime"
        return "admission"
    if rel_path.startswith("support/"):
        if base in {"pqc_lock_profile.c", "pqc_lock_profile.h", "pqc_trace_sink.c", "pqc_trace_sink.h"}:
            return "metrics"
        return "test hooks"
    if rel_path.startswith("frontend/"):
        return "entrypoint" if base == "pqc_main.c" else "FUSE adapter"
    if rel_path.startswith("gpu/") or rel_path.endswith(".cu") or base.startswith("cuda_"):
        return "CUDA backend"
    if rel_path.startswith("common/"):
        return "format"
    return "FUSE adapter"


def extract_functions(path: Path) -> list[dict[str, Any]]:
    lines = read_lines(path)
    functions: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if (
            not stripped
            or stripped.startswith("#")
            or stripped.startswith("*")
            or stripped.startswith("/*")
            or stripped.startswith("//")
            or stripped.startswith("typedef")
            or stripped.startswith("struct ")
            or stripped.startswith("enum ")
        ):
            i += 1
            continue
        if "(" not in stripped:
            i += 1
            continue
        first = stripped.split("(", 1)[0].strip().split()
        if not first or first[-1] in FORBIDDEN_CONTROL_WORDS:
            i += 1
            continue

        signature_parts = [stripped]
        j = i
        while "{" not in " ".join(signature_parts) and ";" not in " ".join(signature_parts):
            j += 1
            if j >= len(lines):
                break
            signature_parts.append(lines[j].strip())
        signature = " ".join(signature_parts)
        if ";" in signature.split("{", 1)[0]:
            i += 1
            continue
        if "=" in signature.split("{", 1)[0]:
            i += 1
            continue

        matches = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*", signature.split("{", 1)[0])
        if not matches:
            i += 1
            continue
        name = matches[-1]
        if name == "__attribute__":
            attr_name = re.search(r"\)\)\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", signature)
            if attr_name:
                name = attr_name.group(1)
            else:
                i += 1
                continue
        if name in FORBIDDEN_CONTROL_WORDS:
            i += 1
            continue

        brace_depth = 0
        body_lines: list[str] = []
        end = j
        saw_open = False
        for k in range(i, len(lines)):
            line = lines[k]
            body_lines.append(line)
            brace_depth += line.count("{")
            if "{" in line:
                saw_open = True
            brace_depth -= line.count("}")
            if saw_open and brace_depth == 0:
                end = k
                break

        body = "\n".join(body_lines)
        target_module = target_module_for_symbol(name, body)
        if target_module == "FUSE adapter":
            target_module = source_module_hint(path)
        if source_id(path).endswith(".cu"):
            target_module = "CUDA backend"
        functions.append(
            {
                "file": source_basename(source_id(path)),
                "path": source_id(path),
                "name": name,
                "line": i + 1,
                "end_line": end + 1,
                "signature": signature.split("{", 1)[0].strip(),
                "target_module": target_module,
                "calls_getenv": re.search(r"\bgetenv\s*\(", body) is not None,
                "calls_config_env": "pqc_config_" in body,
                "calls_fdatasync_or_fsync": "fdatasync(" in body or "fsync(" in body,
                "uses_mutex": "pthread_mutex_" in body,
                "uses_xattr": "xattr(" in body or "PQC_XATTR" in body,
                "uses_fault_cutpoint": "pqc_fault_cutpoint(" in body and name != "pqc_fault_cutpoint",
            }
        )
        i = end + 1
    return functions


def function_at_line(functions: list[dict[str, Any]], file: str, line: int) -> str | None:
    for fn in functions:
        if source_basename(fn["file"]) == source_basename(file) and fn["line"] <= line <= fn["end_line"]:
            return fn["name"]
    return None


def module_at_line(functions: list[dict[str, Any]], file: str, line: int, fallback_name: str = "") -> str:
    fn_name = function_at_line(functions, file, line)
    if fn_name:
        for fn in functions:
            if source_basename(fn["file"]) == source_basename(file) and fn["name"] == fn_name:
                return fn["target_module"]
    return target_module_for_symbol(fallback_name)


def line_inside_function(functions: list[dict[str, Any]], file: str, line: int) -> bool:
    return any(
        source_basename(fn["file"]) == source_basename(file) and
        fn["line"] <= line <= fn["end_line"]
        for fn in functions
    )


def extract_globals(path: Path, functions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    globals_out: list[dict[str, Any]] = []
    lines = read_lines(path)
    for idx, line in enumerate(lines, start=1):
        if line_inside_function(functions, source_id(path), idx):
            continue
        stripped = line.strip()
        if not stripped.endswith(";"):
            continue
        if "(" in stripped and ")" in stripped:
            continue
        match = re.search(r"\b([sg]_[A-Za-z0-9_]+|LOG_FILENAME)\b", stripped)
        if not match:
            continue
        name = match.group(1)
        target_module = target_module_for_symbol(name, stripped)
        if target_module == "FUSE adapter":
            target_module = source_module_hint(path)
        if source_id(path).endswith(".cu"):
            target_module = "CUDA backend"
        globals_out.append(
            {
                "file": source_basename(source_id(path)),
                "path": source_id(path),
                "line": idx,
                "name": name,
                "declaration": stripped,
                "target_module": target_module,
            }
        )
    return globals_out


def extract_environment_variables(functions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: dict[tuple[str, str, int], dict[str, Any]] = {}
    raw_getenv_re = re.compile(r'\bgetenv\(\s*([^)]+?)\s*\)')
    patterns = [
        ("config-wrapper", re.compile(r'\bpqc_config_getenv\(\s*"([^"]+)"\s*\)')),
        ("string-config", re.compile(r'\bpqc_config_get_nonempty\(\s*"([^"]+)"\s*\)')),
        ("config-presence", re.compile(r'\bpqc_config_present\(\s*"([^"]+)"\s*\)')),
        ("config-flag", re.compile(r'\bpqc_config_enabled\(\s*"([^"]+)"\s*\)')),
        ("typed-config", re.compile(r'\bpqc_config_[A-Za-z0-9_]*or_default\(\s*"([^"]+)"')),
        ("raw-env-helper", re.compile(r'\bparse_[A-Za-z0-9_]*env[A-Za-z0-9_]*\(\s*"([^"]+)"')),
    ]
    for source in SOURCE_FILES:
        path = source_path(source)
        if not path.exists():
            continue
        for line_no, line in enumerate(read_lines(path), start=1):
            for match in raw_getenv_re.finditer(line):
                expr = match.group(1).strip()
                if expr.startswith('"') and expr.endswith('"'):
                    name = expr.strip('"')
                else:
                    name = f"<dynamic:{expr}>"
                key = (source, name, line_no)
                fn_name = function_at_line(functions, source, line_no)
                entries[key] = {
                    "name": name,
                    "file": source,
                    "line": line_no,
                    "function": fn_name,
                    "target_module": module_at_line(functions, source, line_no, name),
                    "status": "raw-getenv",
                    "migration_target": "pqc_config.[ch]",
                }
            for status, pattern in patterns:
                for match in pattern.finditer(line):
                    name = match.group(1)
                    key = (source, name, line_no)
                    fn_name = function_at_line(functions, source, line_no)
                    entries[key] = {
                        "name": name,
                        "file": source,
                        "line": line_no,
                        "function": fn_name,
                        "target_module": module_at_line(functions, source, line_no, name),
                        "status": status,
                        "migration_target": "pqc_config.[ch]",
                    }
    return sorted(entries.values(), key=lambda item: (item["name"], item["file"], item["line"]))


def env_accessor_family(status: str) -> str:
    if status == "raw-getenv":
        return "raw"
    if status in {"config-wrapper", "string-config"}:
        return "string"
    if status in {"config-presence", "config-flag"}:
        return "boolean-or-presence"
    if status == "typed-config":
        return "typed"
    if status == "raw-env-helper":
        return "legacy-helper"
    return "unknown"


def gate05_env_status(item: dict[str, Any], runtime_dump_support: bool) -> tuple[str, str]:
    status = item["status"]
    file = item["file"]
    if status == "raw-getenv" and file not in {"pqc_config.c", "pqc_config.h"}:
        return (
            "blocked-raw-getenv-outside-config",
            "raw getenv is outside pqc_config.[ch]",
        )
    if status == "raw-getenv":
        return (
            "allowed-config-internal-raw",
            "raw getenv is isolated inside pqc_config.[ch]",
        )
    if not runtime_dump_support:
        return (
            "blocked-no-runtime-dump-support",
            "pqc_config_dump_file/pqc_config_dump_if_requested support is missing",
        )
    if status == "config-wrapper":
        return (
            "open-wrapper-only",
            "string passthrough is centralized but not typed; keep as string only if the knob is path/text semantics",
        )
    if status == "raw-env-helper":
        return (
            "open-legacy-helper",
            "legacy parse helper should migrate to typed pqc_config accessors",
        )
    return (
        "centralized-with-runtime-dump",
        "site is routed through pqc_config.[ch] and can be captured by runtime config dump",
    )


def enrich_environment_variables(env_vars: list[dict[str, Any]],
                                 runtime_dump_support: bool) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for item in env_vars:
        gate_status, reason = gate05_env_status(item, runtime_dump_support)
        enriched.append(
            {
                **item,
                "accessor_family": env_accessor_family(item["status"]),
                "gate05_blocker_status": gate_status,
                "gate05_blocker_reason": reason,
                "paper_knob_status": (
                    "paper-eligible-only-with-runtime-dump"
                    if gate_status == "centralized-with-runtime-dump"
                    else "not-paper-eligible"
                ),
            }
        )
    return enriched


def config_dump_support_present() -> bool:
    config_c = source_path("pqc_config.c")
    config_h = source_path("pqc_config.h")
    if not config_c.exists() or not config_h.exists():
        return False
    body = config_c.read_text(encoding="utf-8", errors="replace")
    header = config_h.read_text(encoding="utf-8", errors="replace")
    return (
        "pqc_config_dump_file" in body and
        "pqc_config_dump_if_requested" in body and
        "PQC_CONFIG_DUMP_PATH" in body and
        "pqc_config_dump_file" in header and
        "pqc_config_dump_if_requested" in header
    )


def extract_defines_and_structs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    defines: list[dict[str, Any]] = []
    structs: list[dict[str, Any]] = []
    define_re = re.compile(
        r"^\s*#define\s+([A-Za-z0-9_]*(?:MAGIC|VERSION|XATTR|TIER|QOS|ALGO|TPM_NV|LOGICAL_BLOCK|AEAD)[A-Za-z0-9_]*)\s+(.+)$"
    )
    for source in SOURCE_FILES:
        path = source_path(source)
        if not path.exists():
            continue
        lines = read_lines(path)
        for line_no, line in enumerate(lines, start=1):
            m = define_re.match(line)
            if m:
                if m.group(1) == "FUSE_USE_VERSION":
                    continue
                defines.append(
                    {
                        "file": source,
                        "line": line_no,
                        "name": m.group(1),
                        "value": m.group(2).strip(),
                        "target_module": target_module_for_symbol(m.group(1), m.group(2)),
                    }
                )

        i = 0
        while i < len(lines):
            if "typedef struct" not in lines[i]:
                i += 1
                continue
            block: list[str] = [lines[i]]
            j = i
            while j + 1 < len(lines) and "}" not in lines[j]:
                j += 1
                block.append(lines[j])
            trailer = lines[j] if j < len(lines) else ""
            name_match = re.search(r"}\s*([A-Za-z_][A-Za-z0-9_]*)\s*;", trailer)
            if name_match:
                name = name_match.group(1)
                fields = []
                for field_line in block[1:-1]:
                    stripped = field_line.strip().rstrip(";")
                    if stripped and not stripped.startswith("/*") and not stripped.startswith("*"):
                        fields.append(stripped)
                structs.append(
                    {
                        "file": source,
                        "line": i + 1,
                        "end_line": j + 1,
                        "name": name,
                        "fields": fields,
                        "target_module": target_module_for_symbol(name, "\n".join(block)),
                    }
                )
            i = j + 1
    return defines, structs


def format_kind_for_define(item: dict[str, Any]) -> str:
    name = item["name"]
    if name.endswith("_STACK_CAP") or name.endswith("_CAPACITY"):
        return "runtime-capacity"
    if "MAGIC" in name:
        return "magic"
    if "VERSION" in name:
        return "version"
    if "XATTR" in name:
        return "xattr-name"
    if "TPM_NV" in name:
        return "tpm-nv-index"
    if "AEAD" in name or "ALGO" in name:
        return "crypto-format-constant"
    if "LOGICAL_BLOCK" in name:
        return "block-size"
    if "TIER" in name or "QOS" in name:
        return "policy-xattr-value"
    return "format-constant"


def format_kind_for_struct(item: dict[str, Any]) -> str:
    name = item["name"]
    if name in {"pqc_metadata_t"}:
        return "metadata-xattr-record"
    if name in {"pqc_checkpoint_t"}:
        return "checkpoint-xattr-record"
    if name in {"block_mapping_t"}:
        return "journal-mapping-record"
    if name in {"journal_record_t"}:
        return "journal-record"
    if name in {"pqc_prefix_anchor_t", "pqc_freshness_anchor_t"}:
        return "freshness-anchor-record"
    if "epoch_log" in name:
        return "epoch-log-helper"
    return "runtime-struct"


def gate06_define_status(item: dict[str, Any]) -> tuple[str, str]:
    kind = format_kind_for_define(item)
    name = item["name"]
    file = item["file"]
    if kind in {"magic", "version", "xattr-name", "tpm-nv-index", "block-size"}:
        if file == "pqc_format.h":
            return (
                "format-owned-by-pqc-format-h",
                "stable on-disk/on-xattr constant is centralized in pqc_format.h",
            )
        return (
            "blocked-format-constant-outside-pqc-format-h",
            "stable magic/version/xattr/TPM/block-size constant must move to pqc_format.h or be downgraded to a private implementation detail",
        )
    if name.startswith("PQC_QOS_") or name.startswith("PQC_TIER_"):
        return (
            "policy-constant-in-format-inventory",
            "policy xattr value is inventoried but does not by itself define a stable compatibility guarantee",
        )
    if kind == "runtime-capacity":
        return (
            "runtime-capacity-not-persistent-format",
            "runtime buffer/capacity constant is inventoried but is not a stable on-disk or xattr format claim",
        )
    return (
        "format-constant-inventory-only",
        "constant is inventoried; compatibility and upgrade behavior remain unclaimed",
    )


def gate06_struct_status(item: dict[str, Any]) -> tuple[str, str]:
    kind = format_kind_for_struct(item)
    if kind in {
        "metadata-xattr-record",
        "checkpoint-xattr-record",
        "journal-mapping-record",
        "journal-record",
        "freshness-anchor-record",
    }:
        if item["file"] == "pqc_format.h":
            return (
                "persistent-record-owned-by-pqc-format-h",
                "persistent record layout is centralized in pqc_format.h",
            )
        return (
            "blocked-persistent-record-outside-pqc-format-h",
            "persistent record layout must move to pqc_format.h or be downgraded to runtime-only state",
        )
    return (
        "runtime-struct-not-persistent-format",
        "runtime-only struct is inventoried but is not a stable on-disk format claim",
    )


def enrich_format_defines(defines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in defines:
        status, reason = gate06_define_status(item)
        out.append(
            {
                **item,
                "format_kind": format_kind_for_define(item),
                "gate06_blocker_status": status,
                "gate06_blocker_reason": reason,
            }
        )
    return out


def enrich_format_structs(structs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in structs:
        status, reason = gate06_struct_status(item)
        out.append(
            {
                **item,
                "format_kind": format_kind_for_struct(item),
                "gate06_blocker_status": status,
                "gate06_blocker_reason": reason,
            }
        )
    return out


def sidecar_object_status(obj: dict[str, Any]) -> tuple[str, str]:
    if obj["name"] == "epoch redo log":
        return (
            "epoch-log-inventory-only",
            "epoch redo-log object exists, but full Gate 0.9 publication/replay closure is separate",
        )
    if obj["name"] == "external freshness anchor":
        return (
            "freshness-anchor-inventory-only",
            "file/TPM freshness target is inventoried but does not claim PCR-bound rollback resistance",
        )
    return (
        "sidecar-inventory-only",
        "sidecar suffix and owner are inventoried; compatibility and upgrade behavior remain unclaimed",
    )


def enrich_sidecar_objects(sidecar_objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for obj in sidecar_objects:
        status, reason = sidecar_object_status(obj)
        out.append(
            {
                **obj,
                "gate06_blocker_status": status,
                "gate06_blocker_reason": reason,
            }
        )
    return out


def extract_durability_barriers(functions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ops = [
        "fdatasync",
        "fsync",
        "syncfs",
        "syscall",
        "setxattr",
        "fsetxattr",
        "removexattr",
        "fremovexattr",
        "getxattr",
        "fgetxattr",
        "write",
        "pwrite",
        "read",
        "pread",
        "rename",
        "open",
        "mkstemp",
        "fchmod",
        "unlink",
        "close",
    ]
    pattern = re.compile(r"\b(" + "|".join(re.escape(op) for op in ops) + r")\s*\(")
    entries: list[dict[str, Any]] = []
    for source in [name for name in SOURCE_FILES if name.endswith((".c", ".cu"))]:
        path = source_path(source)
        if not path.exists():
            continue
        for line_no, line in enumerate(read_lines(path), start=1):
            stripped = line.strip()
            if (
                not stripped
                or stripped.startswith("*")
                or stripped.startswith("/*")
                or stripped.startswith("//")
                or stripped.startswith("#")
            ):
                continue
            for match in pattern.finditer(line):
                op = match.group(1)
                fn_name = function_at_line(functions, source, line_no)
                lower = line.lower()
                if op in {"write", "open", "close"} and "fprintf" in lower:
                    continue
                if op == "syscall" and "sys_syncfs" not in lower:
                    continue
                entries.append(
                    {
                        "file": source,
                        "line": line_no,
                        "operation": op,
                        "function": fn_name,
                        "target_module": module_at_line(functions, source, line_no, op),
                        "source_line": line.strip(),
                        "durability_role": durability_role(op, line, fn_name or ""),
                        "gate0_note": durability_note(op, line, fn_name or ""),
                    }
                )
    return entries


def durability_role(op: str, line: str, fn_name: str) -> str:
    text = f"{op} {line} {fn_name}".lower()
    if op == "syscall" and "syncfs" in text:
        return "epoch publication"
    if op == "syncfs":
        return "epoch publication"
    if "journal" in text:
        return "journal publication"
    if "checkpoint" in text or "xattr" in text:
        return "metadata/checkpoint persistence"
    if "anchor" in text or "tpm" in text:
        return "freshness anchor persistence"
    if "data_fd" in text or "pwrite" in text or "storage_fd" in text:
        return "data persistence"
    if op in {"getxattr", "fgetxattr", "read", "pread"}:
        return "persistent state read"
    if "trace" in text or "log" in text:
        return "diagnostic trace persistence"
    return "filesystem operation"


def durability_note(op: str, line: str, fn_name: str) -> str:
    text = f"{line} {fn_name}".lower()
    if op in {"fdatasync", "fsync"}:
        return "blocking durability barrier; must not remain under a hot lock after Gate 0.15"
    if op == "setxattr":
        return "metadata mutation; Gate 0.6 must define format and ordering"
    if op == "rename":
        return "atomic namespace publication; Gate C1 must bound POSIX semantics"
    if "sidecar" in text or "journal" in text or "checkpoint" in text:
        return "publication path component; Gate 0.9 must assign strict/epoch semantics"
    return "inventory item for ordering and error propagation"


def durability_blocking_class(op: str) -> str:
    if op in {"fdatasync", "fsync", "syncfs", "syscall"}:
        return "blocking-sync"
    if op in {"setxattr", "fsetxattr", "removexattr", "fremovexattr", "rename"}:
        return "metadata-publication"
    if op in {"write", "pwrite"}:
        return "data-or-record-write"
    if op in {"open", "mkstemp", "fchmod", "close", "unlink"}:
        return "namespace-or-fd-lifecycle"
    if op in {"getxattr", "fgetxattr", "read", "pread"}:
        return "persistent-state-read"
    return "persistence-operation"


def gate06_durability_status(item: dict[str, Any]) -> tuple[str, str]:
    op = item["operation"]
    role = item["durability_role"]
    source = item["source_line"].lower()
    if op in {"fdatasync", "fsync", "syncfs"} or (op == "syscall" and "syncfs" in source):
        return (
            "blocking-sync-inventory",
            "blocking sync point is inventoried; Gate 0.15 must prove it is not under a hot lock and Gate 0.9 must classify strict/epoch semantics",
        )
    if role in {
        "journal publication",
        "metadata/checkpoint persistence",
        "freshness anchor persistence",
        "data persistence",
        "epoch publication",
    }:
        return (
            "publication-operation-inventory",
            "persistent publication operation is inventoried; ordering and crash semantics remain scoped to strict/epoch evidence",
        )
    if role == "persistent state read":
        return (
            "persistent-read-inventory",
            "persistent read is inventoried for recovery/validation but does not prove crash safety by itself",
        )
    return (
        "supporting-filesystem-operation",
        "operation is inventoried for ownership/error propagation and is not by itself a durable publication claim",
    )


def enrich_durability_barriers(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for item in entries:
        status, reason = gate06_durability_status(item)
        enriched.append(
            {
                **item,
                "blocking_class": durability_blocking_class(item["operation"]),
                "gate06_blocker_status": status,
                "gate06_blocker_reason": reason,
            }
        )
    return enriched


LOCK_RULES: dict[str, dict[str, Any]] = {
    "epoch_group_barrier.lock": {
        "owner_module": "publish",
        "protected_state": "epoch group barrier open/completed epoch state and group statistics",
        "hot_path_status": "epoch-redo-log publication path",
        "deadlock_order": "must not nest under fd_lock or commit_lock while waiting for group completion",
    },
    "g_config_lock": {
        "owner_module": "config",
        "protected_state": "runtime configuration access registry and dump state",
        "hot_path_status": "runtime config lookup and artifact dump path",
        "deadlock_order": "must not call back into mounted-path locks while held",
    },
    "g_anchor_lock": {
        "owner_module": "anchor",
        "protected_state": "anchor worker state, dirty flag, last commit timestamp",
        "hot_path_status": "write/fsync path via checkpoint_store and pqc_anchor_worker_flush_now",
        "deadlock_order": "fd_lock -> commit_lock -> g_anchor_lock is currently possible and must be redesigned",
    },
    "g_anchor_lifecycle_lock": {
        "owner_module": "anchor",
        "protected_state": "anchor worker lifecycle state: started/joining/stop flags and worker thread handle",
        "hot_path_status": "mount/unmount and anchor-worker lifecycle path",
        "deadlock_order": "lifecycle lock must not be held while waiting on g_anchor_lock or durable anchor I/O",
    },
    "g_fault_lock": {
        "owner_module": "test hooks",
        "protected_state": "single fault-cutpoint trigger flag",
        "hot_path_status": "only active when fault injection is enabled",
        "deadlock_order": "must not be acquired while holding production locks after Gate 0.8",
    },
    "g_file_state_table_lock": {
        "owner_module": "publish",
        "protected_state": "global file_state_t list and references",
        "hot_path_status": "open/release path",
        "deadlock_order": "must precede per-file commit_lock only during lookup/refcount operations",
    },
    "commit_lock": {
        "owner_module": "publish",
        "protected_state": "per-file generation high-water, publish tickets, logical size, mappings, checkpoint publication",
        "hot_path_status": "write/read/truncate/fallocate path; full-tier prepare/crypto, strict durable publication, authenticated-read recovery, metadata resize publication, and release resource teardown no longer run under fd_lock or this lock on their hot paths, but broad stress and off-CPU evidence are still missing",
        "deadlock_order": "must not be held across crypto, CUDA, fdatasync, or anchor I/O after Gate 0.15",
    },
    "fd_lock": {
        "owner_module": "FUSE adapter",
        "protected_state": "per-open fd context, write buffer, pending jobs, sidecar descriptors",
        "hot_path_status": "read/write/fsync/flush/release path",
        "deadlock_order": "must protect context only and must not cover blocking publication after Gate 0.15",
    },
    "g_sched_pressure_lock": {
        "owner_module": "admission",
        "protected_state": "GPU inflight job and byte counters",
        "hot_path_status": "rekey and scheduler path",
        "deadlock_order": "must not nest inside fd_lock or commit_lock",
    },
    "g_gpu_load_lock": {
        "owner_module": "admission",
        "protected_state": "GPU load EWMA",
        "hot_path_status": "scheduler decision path",
        "deadlock_order": "leaf lock only",
    },
    "g_qos_throttle_lock": {
        "owner_module": "admission",
        "protected_state": "runtime QoS throttle hysteresis state",
        "hot_path_status": "write/flush throttle path",
        "deadlock_order": "leaf lock; sleep must occur outside this lock",
    },
    "g_rekey_queue.lock": {
        "owner_module": "admission",
        "protected_state": "background rekey queue",
        "hot_path_status": "write enqueue and rekey worker",
        "deadlock_order": "must not nest with fd_lock during GPU work",
    },
    "g_rekey_lifecycle_lock": {
        "owner_module": "admission",
        "protected_state": "background rekey worker lifecycle state and worker thread handle",
        "hot_path_status": "mount/unmount and background rekey lifecycle path",
        "deadlock_order": "must not be held while waiting on g_rekey_queue.lock or while running GPU/KEM work",
    },
    "g_committed_lock": {
        "owner_module": "anchor",
        "protected_state": "committed-prefix map and sequence",
        "hot_path_status": "checkpoint and anchor load/store path",
        "deadlock_order": "profiled as committed_map_lock; must not nest with fd_lock or commit_lock after Gate C6",
    },
    "g_epoch_record_lock": {
        "owner_module": "anchor",
        "protected_state": "async freshness epoch record status, generation range, and durable commit markers",
        "hot_path_status": "async Merkle/TPM epoch publication path when Gate C6 is enabled",
        "deadlock_order": "must not nest under fd_lock or commit_lock while performing TPM, file-anchor, or xattr I/O",
    },
    "g_file_anchor_commit_lock": {
        "owner_module": "anchor",
        "protected_state": "file-anchor commit serialization and last committed anchor snapshot",
        "hot_path_status": "file freshness-anchor commit path",
        "deadlock_order": "must not nest under fd_lock or commit_lock; anchor publication must be outside hot file locks",
    },
    "g_pending_anchor_lock": {
        "owner_module": "anchor",
        "protected_state": "pending TPM/file anchor record",
        "hot_path_status": "anchor store/flush path",
        "deadlock_order": "leaf lock inside anchor backend",
    },
    "g_profile_lock": {
        "owner_module": "metrics",
        "protected_state": "lock-profile trace file, initialization state, and sequence counter",
        "hot_path_status": "only active when PQC_LOCK_PROFILE_PATH enables profiling",
        "deadlock_order": "profile leaf lock; must not be held before acquiring profiled production locks",
    },
    "g_admission.state_lock": {
        "owner_module": "admission",
        "protected_state": "admission policy, telemetry, and counters",
        "hot_path_status": "every admission decision",
        "deadlock_order": "must not perform file I/O while held",
    },
    "g_admission.trace_lock": {
        "owner_module": "metrics",
        "protected_state": "admission JSONL trace file",
        "hot_path_status": "every traced admission decision",
        "deadlock_order": "must not nest under state_lock when blocking I/O is possible after Gate 0.15",
    },
    "g_runtime_lock": {
        "owner_module": "publish",
        "protected_state": "process-wide parallel-commit coordinator pointer, config, trace file, and enabled flag",
        "hot_path_status": "parallel-commit runtime setup/snapshot path",
        "deadlock_order": "must not nest with shard locks while writing traces or initializing runtime state",
    },
    "g_sched_lock": {
        "owner_module": "admission",
        "protected_state": "scheduler runtime policy, statistics, and GPU in-flight counters",
        "hot_path_status": "scheduler decision and telemetry path",
        "deadlock_order": "must not nest inside fd_lock or commit_lock",
    },
    "parallel_commit_shard.lock": {
        "owner_module": "publish",
        "protected_state": "per-shard epoch queue, leader/follower state, completed result, and batching counters",
        "hot_path_status": "parallel commit request path",
        "deadlock_order": "leaf shard lock; waiters may block on shard cv, but lock must not cover durable publication",
    },
    "selftest_gate.lock": {
        "owner_module": "test hooks",
        "protected_state": "self-test thread gate state",
        "hot_path_status": "self-test only",
        "deadlock_order": "not a production lock",
    },
    "sink.lock": {
        "owner_module": "metrics",
        "protected_state": "trace sink file descriptor and path buffer",
        "hot_path_status": "trace emission path when a trace sink is enabled",
        "deadlock_order": "leaf metrics lock; must not call back into mounted-path locks while held",
    },
    "cond_mutex": {
        "owner_module": "metrics",
        "protected_state": "lock-profile self-test condition state",
        "hot_path_status": "self-test only",
        "deadlock_order": "not a production lock",
    },
    "test.mutex": {
        "owner_module": "metrics",
        "protected_state": "lock-profile self-test signal mutex",
        "hot_path_status": "self-test only",
        "deadlock_order": "not a production lock",
    },
    "high_lock": {
        "owner_module": "metrics",
        "protected_state": "lock-profile self-test high-rank lock",
        "hot_path_status": "self-test only",
        "deadlock_order": "not a production lock",
    },
    "low_lock": {
        "owner_module": "metrics",
        "protected_state": "lock-profile self-test low-rank lock",
        "hot_path_status": "self-test only",
        "deadlock_order": "not a production lock",
    },
    "locks": {
        "owner_module": "metrics",
        "protected_state": "lock-profile self-test lock array",
        "hot_path_status": "self-test only",
        "deadlock_order": "not a production lock",
    },
    "locks[i]": {
        "owner_module": "metrics",
        "protected_state": "lock-profile self-test lock array element",
        "hot_path_status": "self-test only",
        "deadlock_order": "not a production lock",
    },
}


def normalize_lock_expr(expr: str, source: str = "", line: str = "") -> str:
    expr = expr.strip()
    expr = expr.lstrip("&")
    expr = expr.split(",", 1)[0].strip()
    expr = expr.replace("->", ".")
    if "g_epoch_group_barrier.lock" in expr or "barrier.lock" in expr:
        return "epoch_group_barrier.lock"
    if "shard.lock" in expr:
        return "parallel_commit_shard.lock"
    if "gate.lock" in expr and source == "pqc_selftest.c":
        return "selftest_gate.lock"
    if expr.endswith(".lock") and "g_rekey_queue" in expr:
        return "g_rekey_queue.lock"
    if expr.endswith(".trace_lock") and "g_admission" in expr:
        return "g_admission.trace_lock"
    if expr.endswith(".state_lock") and "g_admission" in expr:
        return "g_admission.state_lock"
    if expr.endswith(".fd_lock"):
        return "fd_lock"
    if expr.endswith(".commit_lock"):
        return "commit_lock"
    return expr


def lock_definition_name(raw_name: str, source: str, line: str) -> str:
    name = normalize_lock_expr(raw_name, source, line)
    if name == "lock":
        if source == "pqc_rekey.c":
            return "g_rekey_queue.lock"
        if source == "pqc_parallel_commit.c":
            return "parallel_commit_shard.lock"
        if source == "pqc_epoch_publish.c":
            return "epoch_group_barrier.lock"
        if source == "pqc_selftest.c":
            return "selftest_gate.lock"
    if name == "state_lock" and source == "pqc_admission.c":
        return "g_admission.state_lock"
    if name == "trace_lock" and source == "pqc_admission.c":
        return "g_admission.trace_lock"
    if name == "fd_lock":
        return "fd_lock"
    if name == "commit_lock":
        return "commit_lock"
    return name


def source_lock_scope(source: str) -> str:
    if source in {"pqc_selftest.c", "pqc_test_hooks.c"}:
        return "test"
    if source.startswith("test_") or source.endswith("_test.c"):
        return "test"
    return "production"


def function_body_for_site(functions: list[dict[str, Any]],
                           source: str,
                           line: int) -> str:
    fn_name = function_at_line(functions, source, line)
    if not fn_name:
        return ""
    fn = next(
        (
            item
            for item in functions
            if item["file"] == source and item["name"] == fn_name
        ),
        None,
    )
    if not fn:
        return ""
    lines = read_lines(source_path(source))
    return "\n".join(lines[fn["line"] - 1:fn["end_line"]])


def lock_call_blocking_risk(lock_name: str,
                            source: str,
                            line: int,
                            functions: list[dict[str, Any]]) -> tuple[str, list[str]]:
    body = function_body_for_site(functions, source, line).lower()
    risk_terms: list[str] = []
    blocking_patterns = {
        "fdatasync": r"\bfdatasync\s*\(",
        "fsync": r"\bfsync\s*\(",
        "syncfs": r"\bsyscall\s*\(\s*sys_syncfs|\bsyncfs\s*\(",
        "pwrite": r"\bpwrite\s*\(",
        "write": r"\bwrite\s*\(",
        "pread": r"\bpread\s*\(",
        "read": r"\bread\s*\(",
        "xattr": r"\b[fl]?setxattr\s*\(|\b[fl]?getxattr\s*\(",
        "rename": r"\brename\s*\(",
        "open-close": r"\bopen\s*\(|\bclose\s*\(",
        "cuda": r"\bcuda[a-z0-9_]*\s*\(",
        "tpm-or-anchor-io": r"\btpm\b|pqc_anchor_(flush|store|load)",
        "condition-wait": r"\bpthread_cond_(timed)?wait\s*\(",
    }
    for label, pattern in blocking_patterns.items():
        if re.search(pattern, body):
            risk_terms.append(label)
    if not risk_terms:
        return (
            "no-blocking-call-detected-in-function",
            [],
        )
    if lock_name in {
        "fd_lock",
        "commit_lock",
        "epoch_group_barrier.lock",
        "parallel_commit_shard.lock",
    }:
        return (
            "hot-lock-function-contains-blocking-or-wait-call",
            risk_terms,
        )
    if "condition-wait" in risk_terms:
        return (
            "condition-wait-under-lock",
            risk_terms,
        )
    return (
        "function-contains-blocking-or-io-call",
        risk_terms,
    )


def gate07_callsite_status(site: dict[str, Any]) -> tuple[str, str]:
    if site.get("scope") == "test":
        return (
            "test-callsite-inventory-only",
            "test-only lock call site is inventoried but not a production scalability claim",
        )
    if site.get("blocking_call_risk") in {
        "hot-lock-function-contains-blocking-or-wait-call",
        "condition-wait-under-lock",
        "function-contains-blocking-or-io-call",
    }:
        return (
            "open-callsite-risk-needs-hold-time-evidence",
            "the enclosing function contains blocking, wait, I/O, CUDA, or xattr calls; retained lock-profile evidence is required before scalability claims",
        )
    return (
        "callsite-inventory-complete",
        "call site is mapped to lock, owner module, function, source line, and blocking-risk classification",
    )


def gate07_lock_status(lock: dict[str, Any]) -> tuple[str, str]:
    if lock.get("scope") == "test":
        return (
            "test-lock-inventory-only",
            "test-only lock is inventoried but does not contribute to production lock closure",
        )
    missing: list[str] = []
    for key in (
        "owner_module",
        "protected_state",
        "hot_path_status",
        "deadlock_order",
        "measured_hold_time_plan",
    ):
        value = str(lock.get(key, ""))
        if not value or value.startswith("needs "):
            missing.append(key)
    if missing:
        return (
            "blocked-missing-lock-contract-fields",
            "missing explicit " + ", ".join(missing),
        )
    if lock.get("lock_call_count", 0) == 0:
        return (
            "definition-only-lock",
            "lock definition is inventoried but no lock acquisition site was found",
        )
    if lock.get("blocking_call_risk_count", 0) > 0:
        return (
            "open-blocking-risk-needs-measurement",
            "one or more functions that acquire this lock contain blocking, wait, I/O, CUDA, or xattr calls; Gate 0.15 must prove hold-time safety",
        )
    return (
        "contract-inventory-complete",
        "owner, protected state, hot-path status, measured hold-time plan, and deadlock ordering rule are present",
    )


def extract_locks(functions: list[dict[str, Any]]) -> dict[str, Any]:
    defs: dict[str, dict[str, Any]] = {}
    call_sites: list[dict[str, Any]] = []
    lock_call_re = re.compile(
        r"(?:pthread_mutex_(lock|unlock)|pqc_profiled_mutex_(lock|unlock))\s*\(\s*([^,)]+)"
    )
    cond_wait_re = re.compile(r"pthread_cond_(timed)?wait\s*\(\s*[^,]+,\s*([^,)]+)")
    definition_re = re.compile(r"\b(pthread_mutex_t)\s+([A-Za-z0-9_\.]+)")

    for source in [name for name in SOURCE_FILES if name.endswith(".c")]:
        path = source_path(source)
        if not path.exists():
            continue
        for line_no, line in enumerate(read_lines(path), start=1):
            stripped = line.strip()
            if "pthread_mutex_t" in line and not re.search(
                r"\bpthread_mutex_t\s+\*?\s*[A-Za-z_][A-Za-z0-9_]*\s*[,)]",
                line,
            ):
                m = definition_re.search(line)
                if m:
                    name = lock_definition_name(m.group(2).strip(), source, line)
                    rule = LOCK_RULES.get(name, {})
                    defs[name] = {
                        "name": name,
                        "file": source,
                        "line": line_no,
                        "declaration": line.strip(),
                        "scope": source_lock_scope(source),
                        "owner_module": rule.get("owner_module", module_at_line(functions, source, line_no, name)),
                        "protected_state": rule.get("protected_state", "needs manual ownership refinement"),
                        "hot_path_status": rule.get("hot_path_status", "needs hot-path classification"),
                        "deadlock_order": rule.get("deadlock_order", "needs explicit ordering rule"),
                        "measured_hold_time_plan": "add p50/p95/p99 hold-time instrumentation before Gate 0.15 closes",
                    }
            if re.match(r"^(?:int|void)\s+pqc_profiled_mutex_(?:lock|unlock)\s*\(", stripped):
                continue
            for m in lock_call_re.finditer(line):
                action = m.group(1) or m.group(2)
                lock_name = normalize_lock_expr(m.group(3), source, line)
                if source == "pqc_lock_profile.c" and lock_name == "mutex":
                    continue
                blocking_risk, risk_terms = lock_call_blocking_risk(
                    lock_name, source, line_no, functions
                )
                site = {
                    "file": source,
                    "line": line_no,
                    "source_line": line.strip(),
                    "scope": source_lock_scope(source),
                    "action": action,
                    "lock": lock_name,
                    "function": function_at_line(functions, source, line_no),
                    "target_module": module_at_line(functions, source, line_no, lock_name),
                    "blocking_call_risk": blocking_risk,
                    "blocking_call_terms": risk_terms,
                }
                status, reason = gate07_callsite_status(site)
                site.update(
                    {
                        "gate07_blocker_status": status,
                        "gate07_blocker_reason": reason,
                    }
                )
                call_sites.append(site)
                if lock_name not in defs:
                    rule = LOCK_RULES.get(lock_name, {})
                    defs[lock_name] = {
                        "name": lock_name,
                        "file": source,
                        "line": None,
                        "declaration": "implicit/member lock",
                        "scope": source_lock_scope(source),
                        "owner_module": rule.get("owner_module", module_at_line(functions, source, line_no, lock_name)),
                        "protected_state": rule.get("protected_state", "needs manual ownership refinement"),
                        "hot_path_status": rule.get("hot_path_status", "needs hot-path classification"),
                        "deadlock_order": rule.get("deadlock_order", "needs explicit ordering rule"),
                        "measured_hold_time_plan": "add p50/p95/p99 hold-time instrumentation before Gate 0.15 closes",
                    }
            for m in cond_wait_re.finditer(line):
                lock_name = normalize_lock_expr(m.group(2), source, line)
                if source == "pqc_lock_profile.c" and lock_name == "mutex":
                    continue
                blocking_risk, risk_terms = lock_call_blocking_risk(
                    lock_name, source, line_no, functions
                )
                if "condition-wait" not in risk_terms:
                    risk_terms = [*risk_terms, "condition-wait"]
                site = {
                    "file": source,
                    "line": line_no,
                    "source_line": line.strip(),
                    "scope": source_lock_scope(source),
                    "action": "cond_timedwait" if m.group(1) else "cond_wait",
                    "lock": lock_name,
                    "function": function_at_line(functions, source, line_no),
                    "target_module": module_at_line(functions, source, line_no, lock_name),
                    "blocking_call_risk": "condition-wait-under-lock",
                    "blocking_call_terms": risk_terms,
                }
                status, reason = gate07_callsite_status(site)
                site.update(
                    {
                        "gate07_blocker_status": status,
                        "gate07_blocker_reason": reason,
                    }
                )
                call_sites.append(site)
                if lock_name not in defs:
                    rule = LOCK_RULES.get(lock_name, {})
                    defs[lock_name] = {
                        "name": lock_name,
                        "file": source,
                        "line": None,
                        "declaration": "implicit/member lock",
                        "scope": source_lock_scope(source),
                        "owner_module": rule.get("owner_module", module_at_line(functions, source, line_no, lock_name)),
                        "protected_state": rule.get("protected_state", "needs manual ownership refinement"),
                        "hot_path_status": rule.get("hot_path_status", "needs hot-path classification"),
                        "deadlock_order": rule.get("deadlock_order", "needs explicit ordering rule"),
                        "measured_hold_time_plan": "add p50/p95/p99 hold-time instrumentation before Gate 0.15 closes",
                    }

    counts = Counter(site["lock"] for site in call_sites if site["action"] == "lock")
    locks = []
    for name, entry in sorted(defs.items()):
        entry["lock_call_count"] = counts.get(name, 0)
        entry["blocking_call_risk_count"] = sum(
            1
            for site in call_sites
            if site["lock"] == name
            and site["gate07_blocker_status"] == "open-callsite-risk-needs-hold-time-evidence"
        )
        status, reason = gate07_lock_status(entry)
        entry["gate07_blocker_status"] = status
        entry["gate07_blocker_reason"] = reason
        locks.append(entry)
    return {"locks": locks, "call_sites": call_sites}


def extract_fault_cutpoints(functions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cutpoints: list[dict[str, Any]] = []
    pattern = re.compile(r'pqc_fault_cutpoint\(\s*"([^"]+)"\s*\)')
    implementation_owner = "missing"
    for fn in functions:
        if fn["name"] == "pqc_fault_cutpoint":
            implementation_owner = f"{fn['file']}:{fn['name']}"
            break
    for source in SOURCE_FILES:
        if not source.endswith(".c"):
            continue
        path = source_path(source)
        if not path.exists():
            continue
        for line_no, line in enumerate(read_lines(path), start=1):
            for match in pattern.finditer(line):
                name = match.group(1)
                invariant = fault_invariant(name)
                item = {
                    "name": name,
                    "file": source,
                    "line": line_no,
                    "source_line": line.strip(),
                    "function": function_at_line(functions, source, line_no),
                    "target_module": module_at_line(functions, source, line_no, name),
                    "scope": "test" if source in {"pqc_selftest.c", "pqc_test_hooks.c"} else "production",
                    "current_owner": implementation_owner,
                    "implementation_owner": implementation_owner,
                    "implementation_interface": "pqc_test_hooks.[ch]:pqc_fault_cutpoint",
                    "production_call_site": source not in {"pqc_selftest.c", "pqc_test_hooks.c"},
                    "gate0_status": (
                        "single-test-hook-interface"
                        if implementation_owner.startswith("pqc_test_hooks.c:")
                        else "unowned-monolith-cutpoint"
                    ),
                    "invariant": invariant,
                    "fault_model_scope": fault_model_scope(name),
                    "activation_env": {
                        "fault_target": "PQC_FAULT_CUTPOINT",
                        "fault_marker": "PQC_FAULT_MARKER_PATH",
                        "pause_target": "PQC_PAUSE_CUTPOINT",
                        "pause_marker": "PQC_PAUSE_MARKER_PATH",
                        "pause_duration": "PQC_PAUSE_US",
                    },
                    "failure_action": "optional pause followed by SIGKILL when selected through PQC_FAULT_CUTPOINT",
                    "migration_target": "pqc_test_hooks.[ch]",
                }
                status, reason = gate08_cutpoint_status(item)
                item["gate08_blocker_status"] = status
                item["gate08_blocker_reason"] = reason
                cutpoints.append(item)
    return cutpoints


def fault_invariant(name: str) -> str:
    if "generation_reservation" in name:
        return "reserved generation high-water must survive remount or fail closed before any later write can reuse that generation"
    if "epoch_checkpoint_compaction" in name:
        return "epoch-log compaction must preserve the latest committed mapping and reject stale or torn epoch prefixes after remount"
    if "checkpoint" in name or "logical_size" in name:
        return "checkpoint/logical-size xattr must not expose duplicate or stale generation after remount"
    if "anchor" in name:
        return "anchor update must fail closed if external freshness is ahead of local state"
    if "data_write" in name or "data_fsync" in name:
        return "ciphertext data sidecar must be durable before journal publication is trusted"
    if "journal" in name:
        return "journal records must publish only authenticated data mappings"
    if "read_after_auth" in name:
        return "plaintext is returned only after authentication succeeds"
    if "fsync" in name:
        return "fsync return must imply all strict-mode sidecar/checkpoint/anchor obligations completed"
    if "remount" in name:
        return "remount must reconstruct checkpoint and anchor state before exposing file contents"
    return "needs explicit invariant before Gate 0.8 closes"


def fault_model_scope(name: str) -> str:
    if "read_after_auth" in name:
        return "authentication-before-exposure process cutpoint"
    if "generation_reservation" in name:
        return "reserved-generation SIGKILL/remount cutpoint"
    if "epoch" in name:
        return "epoch-redo-log SIGKILL/remount cutpoint"
    if "anchor" in name:
        return "freshness-anchor SIGKILL/remount cutpoint"
    if "data" in name or "journal" in name or "checkpoint" in name or "logical_size" in name:
        return "strict-publication SIGKILL/remount cutpoint"
    if "fsync" in name:
        return "fsync-return SIGKILL cutpoint"
    if "remount" in name:
        return "remount-recovery cutpoint"
    return "unspecified process-fault cutpoint"


def gate08_cutpoint_status(item: dict[str, Any]) -> tuple[str, str]:
    if not str(item.get("implementation_owner", "")).startswith("pqc_test_hooks.c:"):
        return (
            "blocked-unowned-cutpoint-interface",
            "fault cutpoint implementation is not owned by pqc_test_hooks.[ch]",
        )
    if str(item.get("invariant", "")).startswith("needs explicit invariant"):
        return (
            "blocked-missing-cutpoint-invariant",
            "cutpoint does not name the crash/recovery invariant it is meant to test",
        )
    if item.get("scope") == "test":
        return (
            "test-cutpoint-interface-inventory",
            "test-only hook implementation is inventoried but is not a production crash boundary",
        )
    return (
        "production-cutpoint-owned-by-test-hooks",
        "production cutpoint call site uses the single pqc_test_hooks.[ch] interface and names a crash/recovery invariant",
    )


def production_source_inventory() -> dict[str, Any]:
    discovered = sorted(
        str(path.relative_to(CODE))
        for pattern in ("pqc_*.c", "pqc_*.h", "cuda_*.cu", "cuda_*.h")
        for path in CODE.rglob(pattern)
    )
    listed = sorted(SOURCE_FILES)
    listed_basenames = {source_basename(name) for name in listed}
    return {
        "discovered_source_files": discovered,
        "listed_source_files": listed,
        "missing_from_source_files": sorted(
            path for path in discovered
            if source_basename(path) not in listed_basenames
        ),
        "listed_but_missing": sorted(name for name in listed if not source_path(name).exists()),
    }


def ownership_ambiguity_reason(item: dict[str, Any]) -> str | None:
    file = item["file"]
    base_file = source_basename(file)
    target = item["target_module"]
    name = item["name"]
    if target not in TARGET_MODULES:
        return "target module is not in allowed Gate 0 target module set"
    if file.endswith(".cu") and target != "CUDA backend":
        return "CUDA source symbol is not owned by CUDA backend"
    if target == "FUSE adapter" and base_file not in {
        "pqc_fuse.c",
        "pqc_fuse.h",
        "pqc_fd_context.c",
        "pqc_fd_context.h",
        "pqc_writeback.c",
        "pqc_writeback.h",
        "pqc_selftest.c",
        "pqc_selftest.h",
    }:
        return "FUSE-adapter fallback outside FUSE-facing files; requires manual ownership review"
    if name in {"helper", "init", "cleanup", "worker", "run"}:
        return "generic symbol name requires manual ownership review"
    return None


def compact_symbol(item: dict[str, Any], reason: str | None = None) -> dict[str, Any]:
    out = {
        "file": item["file"],
        "line": item["line"],
        "name": item["name"],
        "target_module": item["target_module"],
    }
    if reason:
        out["reason"] = reason
    return out


def build_source_ownership(functions: list[dict[str, Any]], globals_out: list[dict[str, Any]]) -> dict[str, Any]:
    fuse_functions = [fn for fn in functions if fn["file"] == "pqc_fuse.c"]
    admission_functions = [fn for fn in functions if fn["file"] == "pqc_admission.c"]
    anchor_worker_functions = [fn for fn in functions if fn["file"] == "pqc_anchor_worker.c"]
    crypto_functions = [fn for fn in functions if fn["file"] == "pqc_crypto.c"]
    fd_context_functions = [fn for fn in functions if fn["file"] == "pqc_fd_context.c"]
    file_lock_functions = [fn for fn in functions if fn["file"] == "pqc_file_lock.c"]
    file_io_functions = [fn for fn in functions if fn["file"] == "pqc_file_io.c"]
    flush_batch_functions = [fn for fn in functions if fn["file"] == "pqc_flush_batch.c"]
    flush_crypto_functions = [fn for fn in functions if fn["file"] == "pqc_flush_crypto.c"]
    journal_functions = [fn for fn in functions if fn["file"] == "pqc_journal.c"]
    keyring_functions = [fn for fn in functions if fn["file"] == "pqc_keyring.c"]
    lifecycle_functions = [fn for fn in functions if fn["file"] == "pqc_lifecycle.c"]
    lock_profile_functions = [fn for fn in functions if fn["file"] == "pqc_lock_profile.c"]
    main_functions = [fn for fn in functions if fn["file"] == "pqc_main.c"]
    metrics_functions = [fn for fn in functions if fn["file"] == "pqc_metrics.c"]
    namespace_functions = [fn for fn in functions if fn["file"] == "pqc_namespace.c"]
    posix_functions = [fn for fn in functions if fn["file"] == "pqc_posix.c"]
    publish_functions = [fn for fn in functions if fn["file"] == "pqc_publish.c"]
    qos_functions = [fn for fn in functions if fn["file"] == "pqc_qos.c"]
    checkpoint_functions = [fn for fn in functions if fn["file"] == "pqc_checkpoint.c"]
    recovery_functions = [fn for fn in functions if fn["file"] == "pqc_recovery.c"]
    rekey_functions = [fn for fn in functions if fn["file"] == "pqc_rekey.c"]
    runtime_functions = [fn for fn in functions if fn["file"] == "pqc_runtime.c"]
    scheduler_functions = [fn for fn in functions if fn["file"] == "pqc_scheduler.c"]
    selftest_functions = [fn for fn in functions if fn["file"] == "pqc_selftest.c"]
    state_functions = [fn for fn in functions if fn["file"] == "pqc_state.c"]
    storage_path_functions = [fn for fn in functions if fn["file"] == "pqc_storage_path.c"]
    strict_publish_functions = [fn for fn in functions if fn["file"] == "pqc_strict_publish.c"]
    writeback_functions = [fn for fn in functions if fn["file"] == "pqc_writeback.c"]
    xattr_functions = [fn for fn in functions if fn["file"] == "pqc_xattr.c"]
    source_inventory = production_source_inventory()
    function_counts_by_source_file = Counter(fn["file"] for fn in functions)
    all_function_counts = Counter(fn["target_module"] for fn in functions)
    unmapped_functions = [
        compact_symbol(fn)
        for fn in functions
        if fn["target_module"] not in TARGET_MODULES
    ]
    unmapped_global_state = [
        compact_symbol(glob)
        for glob in globals_out
        if glob["target_module"] not in TARGET_MODULES
    ]
    ambiguous_functions = [
        compact_symbol(fn, reason)
        for fn in functions
        if (reason := ownership_ambiguity_reason(fn)) is not None
    ]
    ambiguous_global_state = [
        compact_symbol(glob, reason)
        for glob in globals_out
        if (reason := ownership_ambiguity_reason(glob)) is not None
    ]
    source_file_module_counts: dict[str, dict[str, int]] = {}
    for file in sorted(function_counts_by_source_file):
        source_file_module_counts[file] = dict(
            sorted(Counter(fn["target_module"] for fn in functions if fn["file"] == file).items())
        )
    ownership_blockers = []
    if source_inventory["missing_from_source_files"]:
        ownership_blockers.append(
            {
                "type": "source-files-missing-from-inventory",
                "items": source_inventory["missing_from_source_files"],
                "required_action": "Add these production files to SOURCE_FILES or explain why they are not production owned.",
            }
        )
    if unmapped_functions or unmapped_global_state:
        ownership_blockers.append(
            {
                "type": "unmapped-symbols",
                "functions": unmapped_functions,
                "global_state": unmapped_global_state,
                "required_action": "Assign every symbol to one allowed Gate 0 target module before claiming clean ownership.",
            }
        )
    if ambiguous_functions or ambiguous_global_state:
        ownership_blockers.append(
            {
                "type": "ambiguous-symbols",
                "functions": ambiguous_functions,
                "global_state": ambiguous_global_state,
                "required_action": "Review fallback/generic ownership before Gate 0.14 behavior-equivalence closure.",
            }
        )
    module_counts = Counter(fn["target_module"] for fn in fuse_functions)
    admission_module_counts = Counter(fn["target_module"] for fn in admission_functions)
    anchor_worker_module_counts = Counter(fn["target_module"] for fn in anchor_worker_functions)
    crypto_module_counts = Counter(fn["target_module"] for fn in crypto_functions)
    fd_context_module_counts = Counter(fn["target_module"] for fn in fd_context_functions)
    file_lock_module_counts = Counter(fn["target_module"] for fn in file_lock_functions)
    file_io_module_counts = Counter(fn["target_module"] for fn in file_io_functions)
    flush_batch_module_counts = Counter(fn["target_module"] for fn in flush_batch_functions)
    flush_crypto_module_counts = Counter(fn["target_module"] for fn in flush_crypto_functions)
    journal_module_counts = Counter(fn["target_module"] for fn in journal_functions)
    keyring_module_counts = Counter(fn["target_module"] for fn in keyring_functions)
    lifecycle_module_counts = Counter(fn["target_module"] for fn in lifecycle_functions)
    lock_profile_module_counts = Counter(fn["target_module"] for fn in lock_profile_functions)
    main_module_counts = Counter(fn["target_module"] for fn in main_functions)
    metrics_module_counts = Counter(fn["target_module"] for fn in metrics_functions)
    namespace_module_counts = Counter(fn["target_module"] for fn in namespace_functions)
    posix_module_counts = Counter(fn["target_module"] for fn in posix_functions)
    publish_module_counts = Counter(fn["target_module"] for fn in publish_functions)
    qos_module_counts = Counter(fn["target_module"] for fn in qos_functions)
    checkpoint_module_counts = Counter(fn["target_module"] for fn in checkpoint_functions)
    recovery_module_counts = Counter(fn["target_module"] for fn in recovery_functions)
    rekey_module_counts = Counter(fn["target_module"] for fn in rekey_functions)
    runtime_module_counts = Counter(fn["target_module"] for fn in runtime_functions)
    scheduler_module_counts = Counter(fn["target_module"] for fn in scheduler_functions)
    selftest_module_counts = Counter(fn["target_module"] for fn in selftest_functions)
    state_module_counts = Counter(fn["target_module"] for fn in state_functions)
    storage_path_module_counts = Counter(fn["target_module"] for fn in storage_path_functions)
    strict_publish_module_counts = Counter(fn["target_module"] for fn in strict_publish_functions)
    writeback_module_counts = Counter(fn["target_module"] for fn in writeback_functions)
    xattr_module_counts = Counter(fn["target_module"] for fn in xattr_functions)
    global_counts = Counter(glob["target_module"] for glob in globals_out)
    return {
        "schema_version": 1,
        "generated_utc": now_utc(),
        "generated_by": "code/experiments/build_refactor_inventory.py",
        "target_modules": TARGET_MODULES,
        "summary": {
            "production_source_file_count": len(source_inventory["discovered_source_files"]),
            "source_files_in_inventory_count": len(SOURCE_FILES),
            "missing_production_source_files_count": len(source_inventory["missing_from_source_files"]),
            "listed_but_missing_source_files_count": len(source_inventory["listed_but_missing"]),
            "all_current_production_function_count": len(functions),
            "all_current_production_functions_mapped": not unmapped_functions,
            "all_current_global_state_mapped": not unmapped_global_state,
            "ambiguous_function_count": len(ambiguous_functions),
            "ambiguous_global_state_count": len(ambiguous_global_state),
            "ownership_blocker_count": len(ownership_blockers),
            "all_function_counts_by_target_module": dict(sorted(all_function_counts.items())),
            "function_counts_by_source_file": dict(sorted(function_counts_by_source_file.items())),
            "source_file_module_counts": source_file_module_counts,
            "pqc_fuse_function_count": len(fuse_functions),
            "all_pqc_fuse_functions_mapped": all(fn["target_module"] in TARGET_MODULES for fn in fuse_functions),
            "pqc_fuse_function_counts_by_target_module": dict(sorted(module_counts.items())),
            "pqc_admission_function_count": len(admission_functions),
            "pqc_admission_function_counts_by_target_module": dict(sorted(admission_module_counts.items())),
            "pqc_anchor_worker_function_count": len(anchor_worker_functions),
            "pqc_anchor_worker_function_counts_by_target_module": dict(sorted(anchor_worker_module_counts.items())),
            "pqc_crypto_function_count": len(crypto_functions),
            "pqc_crypto_function_counts_by_target_module": dict(sorted(crypto_module_counts.items())),
            "pqc_fd_context_function_count": len(fd_context_functions),
            "pqc_fd_context_function_counts_by_target_module": dict(sorted(fd_context_module_counts.items())),
            "pqc_file_lock_function_count": len(file_lock_functions),
            "pqc_file_lock_function_counts_by_target_module": dict(sorted(file_lock_module_counts.items())),
            "pqc_file_io_function_count": len(file_io_functions),
            "pqc_file_io_function_counts_by_target_module": dict(sorted(file_io_module_counts.items())),
            "pqc_flush_batch_function_count": len(flush_batch_functions),
            "pqc_flush_batch_function_counts_by_target_module": dict(sorted(flush_batch_module_counts.items())),
            "pqc_flush_crypto_function_count": len(flush_crypto_functions),
            "pqc_flush_crypto_function_counts_by_target_module": dict(sorted(flush_crypto_module_counts.items())),
            "pqc_journal_function_count": len(journal_functions),
            "pqc_journal_function_counts_by_target_module": dict(sorted(journal_module_counts.items())),
            "pqc_keyring_function_count": len(keyring_functions),
            "pqc_keyring_function_counts_by_target_module": dict(sorted(keyring_module_counts.items())),
            "pqc_lifecycle_function_count": len(lifecycle_functions),
            "pqc_lifecycle_function_counts_by_target_module": dict(sorted(lifecycle_module_counts.items())),
            "pqc_lock_profile_function_count": len(lock_profile_functions),
            "pqc_lock_profile_function_counts_by_target_module": dict(sorted(lock_profile_module_counts.items())),
            "pqc_main_function_count": len(main_functions),
            "pqc_main_function_counts_by_target_module": dict(sorted(main_module_counts.items())),
            "pqc_metrics_function_count": len(metrics_functions),
            "pqc_metrics_function_counts_by_target_module": dict(sorted(metrics_module_counts.items())),
            "pqc_namespace_function_count": len(namespace_functions),
            "pqc_namespace_function_counts_by_target_module": dict(sorted(namespace_module_counts.items())),
            "pqc_posix_function_count": len(posix_functions),
            "pqc_posix_function_counts_by_target_module": dict(sorted(posix_module_counts.items())),
            "pqc_publish_function_count": len(publish_functions),
            "pqc_publish_function_counts_by_target_module": dict(sorted(publish_module_counts.items())),
            "pqc_qos_function_count": len(qos_functions),
            "pqc_qos_function_counts_by_target_module": dict(sorted(qos_module_counts.items())),
            "pqc_checkpoint_function_count": len(checkpoint_functions),
            "pqc_checkpoint_function_counts_by_target_module": dict(sorted(checkpoint_module_counts.items())),
            "pqc_recovery_function_count": len(recovery_functions),
            "pqc_recovery_function_counts_by_target_module": dict(sorted(recovery_module_counts.items())),
            "pqc_rekey_function_count": len(rekey_functions),
            "pqc_rekey_function_counts_by_target_module": dict(sorted(rekey_module_counts.items())),
            "pqc_runtime_function_count": len(runtime_functions),
            "pqc_runtime_function_counts_by_target_module": dict(sorted(runtime_module_counts.items())),
            "pqc_scheduler_function_count": len(scheduler_functions),
            "pqc_scheduler_function_counts_by_target_module": dict(sorted(scheduler_module_counts.items())),
            "pqc_selftest_function_count": len(selftest_functions),
            "pqc_selftest_function_counts_by_target_module": dict(sorted(selftest_module_counts.items())),
            "pqc_state_function_count": len(state_functions),
            "pqc_state_function_counts_by_target_module": dict(sorted(state_module_counts.items())),
            "pqc_storage_path_function_count": len(storage_path_functions),
            "pqc_storage_path_function_counts_by_target_module": dict(sorted(storage_path_module_counts.items())),
            "pqc_strict_publish_function_count": len(strict_publish_functions),
            "pqc_strict_publish_function_counts_by_target_module": dict(sorted(strict_publish_module_counts.items())),
            "pqc_writeback_function_count": len(writeback_functions),
            "pqc_writeback_function_counts_by_target_module": dict(sorted(writeback_module_counts.items())),
            "pqc_xattr_function_count": len(xattr_functions),
            "pqc_xattr_function_counts_by_target_module": dict(sorted(xattr_module_counts.items())),
            "global_state_count": len(globals_out),
            "global_state_counts_by_target_module": dict(sorted(global_counts.items())),
            "gate0_status": (
                "inventory-blocked; ownership blockers require review before clean module boundary is claimed"
                if ownership_blockers else
                "inventory-only; no clean module boundary is claimed"
            ),
        },
        "source_inventory": source_inventory,
        "production_functions": functions,
        "unmapped_functions": unmapped_functions,
        "unmapped_global_state": unmapped_global_state,
        "ambiguous_functions": ambiguous_functions,
        "ambiguous_global_state": ambiguous_global_state,
        "ownership_blockers": ownership_blockers,
        "functions": fuse_functions,
        "pqc_admission_functions": admission_functions,
        "pqc_anchor_worker_functions": anchor_worker_functions,
        "pqc_crypto_functions": crypto_functions,
        "pqc_fd_context_functions": fd_context_functions,
        "pqc_file_lock_functions": file_lock_functions,
        "pqc_file_io_functions": file_io_functions,
        "pqc_flush_batch_functions": flush_batch_functions,
        "pqc_flush_crypto_functions": flush_crypto_functions,
        "pqc_journal_functions": journal_functions,
        "pqc_keyring_functions": keyring_functions,
        "pqc_lifecycle_functions": lifecycle_functions,
        "pqc_lock_profile_functions": lock_profile_functions,
        "pqc_main_functions": main_functions,
        "pqc_metrics_functions": metrics_functions,
        "pqc_namespace_functions": namespace_functions,
        "pqc_posix_functions": posix_functions,
        "pqc_publish_functions": publish_functions,
        "pqc_qos_functions": qos_functions,
        "pqc_checkpoint_functions": checkpoint_functions,
        "pqc_recovery_functions": recovery_functions,
        "pqc_rekey_functions": rekey_functions,
        "pqc_runtime_functions": runtime_functions,
        "pqc_scheduler_functions": scheduler_functions,
        "pqc_selftest_functions": selftest_functions,
        "pqc_state_functions": state_functions,
        "pqc_storage_path_functions": storage_path_functions,
        "pqc_strict_publish_functions": strict_publish_functions,
        "pqc_writeback_functions": writeback_functions,
        "pqc_xattr_functions": xattr_functions,
        "global_state": globals_out,
        "negative_claim_guard": "Implementation text must not claim clean module boundaries until mechanical decomposition and behavior-equivalence evidence exist.",
    }


def render_source_map_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Source Ownership Map",
        "",
        "This file is generated by `code/experiments/build_refactor_inventory.py`.",
        "It maps the current `pqc_fuse.c` monolith to target modules for Gate 0",
        "decomposition.  It is an inventory, not a claim that those boundaries",
        "already exist in code.",
        "",
        "## Summary",
        "",
        f"- Production source files discovered: `{payload['summary']['production_source_file_count']}`",
        f"- Source files in inventory: `{payload['summary']['source_files_in_inventory_count']}`",
        f"- Missing production source files: `{payload['summary']['missing_production_source_files_count']}`",
        f"- Current production functions mapped: `{payload['summary']['all_current_production_function_count']}`",
        f"- All current production functions mapped: `{str(payload['summary']['all_current_production_functions_mapped']).lower()}`",
        f"- All current global state mapped: `{str(payload['summary']['all_current_global_state_mapped']).lower()}`",
        f"- Ambiguous function ownership blockers: `{payload['summary']['ambiguous_function_count']}`",
        f"- Ownership blocker groups: `{payload['summary']['ownership_blocker_count']}`",
        f"- `pqc_fuse.c` functions mapped: `{payload['summary']['pqc_fuse_function_count']}`",
        f"- `pqc_admission.c` functions mapped: `{payload['summary']['pqc_admission_function_count']}`",
        f"- `pqc_anchor_worker.c` functions mapped: `{payload['summary']['pqc_anchor_worker_function_count']}`",
        f"- `pqc_crypto.c` functions mapped: `{payload['summary']['pqc_crypto_function_count']}`",
        f"- `pqc_fd_context.c` functions mapped: `{payload['summary']['pqc_fd_context_function_count']}`",
        f"- `pqc_file_lock.c` functions mapped: `{payload['summary']['pqc_file_lock_function_count']}`",
        f"- `pqc_file_io.c` functions mapped: `{payload['summary']['pqc_file_io_function_count']}`",
        f"- `pqc_flush_batch.c` functions mapped: `{payload['summary']['pqc_flush_batch_function_count']}`",
        f"- `pqc_flush_crypto.c` functions mapped: `{payload['summary']['pqc_flush_crypto_function_count']}`",
        f"- `pqc_journal.c` functions mapped: `{payload['summary']['pqc_journal_function_count']}`",
        f"- `pqc_keyring.c` functions mapped: `{payload['summary']['pqc_keyring_function_count']}`",
        f"- `pqc_lifecycle.c` functions mapped: `{payload['summary']['pqc_lifecycle_function_count']}`",
        f"- `pqc_lock_profile.c` functions mapped: `{payload['summary']['pqc_lock_profile_function_count']}`",
        f"- `pqc_main.c` functions mapped: `{payload['summary']['pqc_main_function_count']}`",
        f"- `pqc_metrics.c` functions mapped: `{payload['summary']['pqc_metrics_function_count']}`",
        f"- `pqc_namespace.c` functions mapped: `{payload['summary']['pqc_namespace_function_count']}`",
        f"- `pqc_posix.c` functions mapped: `{payload['summary']['pqc_posix_function_count']}`",
        f"- `pqc_publish.c` functions mapped: `{payload['summary']['pqc_publish_function_count']}`",
        f"- `pqc_qos.c` functions mapped: `{payload['summary']['pqc_qos_function_count']}`",
        f"- `pqc_checkpoint.c` functions mapped: `{payload['summary']['pqc_checkpoint_function_count']}`",
        f"- `pqc_recovery.c` functions mapped: `{payload['summary']['pqc_recovery_function_count']}`",
        f"- `pqc_rekey.c` functions mapped: `{payload['summary']['pqc_rekey_function_count']}`",
        f"- `pqc_runtime.c` functions mapped: `{payload['summary']['pqc_runtime_function_count']}`",
        f"- `pqc_scheduler.c` functions mapped: `{payload['summary']['pqc_scheduler_function_count']}`",
        f"- `pqc_selftest.c` functions mapped: `{payload['summary']['pqc_selftest_function_count']}`",
        f"- `pqc_state.c` functions mapped: `{payload['summary']['pqc_state_function_count']}`",
        f"- `pqc_storage_path.c` functions mapped: `{payload['summary']['pqc_storage_path_function_count']}`",
        f"- `pqc_writeback.c` functions mapped: `{payload['summary']['pqc_writeback_function_count']}`",
        f"- `pqc_strict_publish.c` functions mapped: `{payload['summary']['pqc_strict_publish_function_count']}`",
        f"- `pqc_xattr.c` functions mapped: `{payload['summary']['pqc_xattr_function_count']}`",
        f"- Global state objects mapped: `{payload['summary']['global_state_count']}`",
        f"- All `pqc_fuse.c` functions mapped to allowed target modules: `{str(payload['summary']['all_pqc_fuse_functions_mapped']).lower()}`",
        f"- Gate 0 status: `{payload['summary']['gate0_status']}`",
        "",
        "## Ownership Blockers",
        "",
    ]
    if payload["ownership_blockers"]:
        for blocker in payload["ownership_blockers"]:
            lines.append(f"- `{blocker['type']}`: {blocker['required_action']}")
            for key in ("items", "functions", "global_state"):
                entries = blocker.get(key, [])
                if not entries:
                    continue
                for entry in entries[:12]:
                    lines.append(f"  - `{entry}`" if isinstance(entry, str) else f"  - `{entry}`")
                if len(entries) > 12:
                    lines.append(f"  - `+{len(entries) - 12} more`")
    else:
        lines.append("- none")
    lines.extend([
        "",
        "## Function Counts",
        "",
        "| Target module | Functions |",
        "| --- | ---: |",
    ])
    for module, count in payload["summary"]["all_function_counts_by_target_module"].items():
        lines.append(f"| `{module}` | {count} |")
    lines.extend(["", "## Source File Coverage", "", "| Source file | Functions | Target modules |", "| --- | ---: | --- |"])
    for file, count in payload["summary"]["function_counts_by_source_file"].items():
        module_counts = ", ".join(
            f"`{module}`={module_count}"
            for module, module_count in payload["summary"]["source_file_module_counts"][file].items()
        )
        lines.append(f"| `{file}` | {count} | {module_counts} |")
    lines.extend(["", "## `pqc_fuse.c` Function Map", "", "| Function | Lines | Target module | Hot markers |", "| --- | ---: | --- | --- |"])
    for fn in payload["functions"]:
        markers = []
        if fn["calls_getenv"]:
            markers.append("raw-env")
        if fn.get("calls_config_env"):
            markers.append("config-env")
        if fn["calls_fdatasync_or_fsync"]:
            markers.append("sync")
        if fn["uses_mutex"]:
            markers.append("lock")
        if fn["uses_xattr"]:
            markers.append("xattr")
        if fn["uses_fault_cutpoint"]:
            markers.append("fault")
        marker_text = ", ".join(markers) if markers else "-"
        lines.append(f"| `{fn['name']}` | {fn['line']}-{fn['end_line']} | `{fn['target_module']}` | {marker_text} |")
    lines.extend(["", "## `pqc_admission.c` Function Map", "", "| Function | Lines | Target module |", "| --- | ---: | --- |"])
    for fn in payload.get("pqc_admission_functions", []):
        lines.append(f"| `{fn['name']}` | {fn['line']}-{fn['end_line']} | `{fn['target_module']}` |")
    lines.extend(["", "## `pqc_anchor_worker.c` Function Map", "", "| Function | Lines | Target module |", "| --- | ---: | --- |"])
    for fn in payload.get("pqc_anchor_worker_functions", []):
        lines.append(f"| `{fn['name']}` | {fn['line']}-{fn['end_line']} | `{fn['target_module']}` |")
    lines.extend(["", "## `pqc_crypto.c` Function Map", "", "| Function | Lines | Target module |", "| --- | ---: | --- |"])
    for fn in payload.get("pqc_crypto_functions", []):
        lines.append(f"| `{fn['name']}` | {fn['line']}-{fn['end_line']} | `{fn['target_module']}` |")
    lines.extend(["", "## `pqc_fd_context.c` Function Map", "", "| Function | Lines | Target module |", "| --- | ---: | --- |"])
    for fn in payload.get("pqc_fd_context_functions", []):
        lines.append(f"| `{fn['name']}` | {fn['line']}-{fn['end_line']} | `{fn['target_module']}` |")
    lines.extend(["", "## `pqc_file_lock.c` Function Map", "", "| Function | Lines | Target module |", "| --- | ---: | --- |"])
    for fn in payload.get("pqc_file_lock_functions", []):
        lines.append(f"| `{fn['name']}` | {fn['line']}-{fn['end_line']} | `{fn['target_module']}` |")
    lines.extend(["", "## `pqc_file_io.c` Function Map", "", "| Function | Lines | Target module |", "| --- | ---: | --- |"])
    for fn in payload.get("pqc_file_io_functions", []):
        lines.append(f"| `{fn['name']}` | {fn['line']}-{fn['end_line']} | `{fn['target_module']}` |")
    lines.extend(["", "## `pqc_flush_batch.c` Function Map", "", "| Function | Lines | Target module |", "| --- | ---: | --- |"])
    for fn in payload.get("pqc_flush_batch_functions", []):
        lines.append(f"| `{fn['name']}` | {fn['line']}-{fn['end_line']} | `{fn['target_module']}` |")
    lines.extend(["", "## `pqc_flush_crypto.c` Function Map", "", "| Function | Lines | Target module |", "| --- | ---: | --- |"])
    for fn in payload.get("pqc_flush_crypto_functions", []):
        lines.append(f"| `{fn['name']}` | {fn['line']}-{fn['end_line']} | `{fn['target_module']}` |")
    lines.extend(["", "## `pqc_journal.c` Function Map", "", "| Function | Lines | Target module |", "| --- | ---: | --- |"])
    for fn in payload.get("pqc_journal_functions", []):
        lines.append(f"| `{fn['name']}` | {fn['line']}-{fn['end_line']} | `{fn['target_module']}` |")
    lines.extend(["", "## `pqc_keyring.c` Function Map", "", "| Function | Lines | Target module |", "| --- | ---: | --- |"])
    for fn in payload.get("pqc_keyring_functions", []):
        lines.append(f"| `{fn['name']}` | {fn['line']}-{fn['end_line']} | `{fn['target_module']}` |")
    lines.extend(["", "## `pqc_lifecycle.c` Function Map", "", "| Function | Lines | Target module |", "| --- | ---: | --- |"])
    for fn in payload.get("pqc_lifecycle_functions", []):
        lines.append(f"| `{fn['name']}` | {fn['line']}-{fn['end_line']} | `{fn['target_module']}` |")
    lines.extend(["", "## `pqc_lock_profile.c` Function Map", "", "| Function | Lines | Target module |", "| --- | ---: | --- |"])
    for fn in payload.get("pqc_lock_profile_functions", []):
        lines.append(f"| `{fn['name']}` | {fn['line']}-{fn['end_line']} | `{fn['target_module']}` |")
    lines.extend(["", "## `pqc_main.c` Function Map", "", "| Function | Lines | Target module |", "| --- | ---: | --- |"])
    for fn in payload.get("pqc_main_functions", []):
        lines.append(f"| `{fn['name']}` | {fn['line']}-{fn['end_line']} | `{fn['target_module']}` |")
    lines.extend(["", "## `pqc_metrics.c` Function Map", "", "| Function | Lines | Target module |", "| --- | ---: | --- |"])
    for fn in payload.get("pqc_metrics_functions", []):
        lines.append(f"| `{fn['name']}` | {fn['line']}-{fn['end_line']} | `{fn['target_module']}` |")
    lines.extend(["", "## `pqc_namespace.c` Function Map", "", "| Function | Lines | Target module |", "| --- | ---: | --- |"])
    for fn in payload.get("pqc_namespace_functions", []):
        lines.append(f"| `{fn['name']}` | {fn['line']}-{fn['end_line']} | `{fn['target_module']}` |")
    lines.extend(["", "## `pqc_posix.c` Function Map", "", "| Function | Lines | Target module |", "| --- | ---: | --- |"])
    for fn in payload.get("pqc_posix_functions", []):
        lines.append(f"| `{fn['name']}` | {fn['line']}-{fn['end_line']} | `{fn['target_module']}` |")
    lines.extend(["", "## `pqc_publish.c` Function Map", "", "| Function | Lines | Target module |", "| --- | ---: | --- |"])
    for fn in payload.get("pqc_publish_functions", []):
        lines.append(f"| `{fn['name']}` | {fn['line']}-{fn['end_line']} | `{fn['target_module']}` |")
    lines.extend(["", "## `pqc_qos.c` Function Map", "", "| Function | Lines | Target module |", "| --- | ---: | --- |"])
    for fn in payload.get("pqc_qos_functions", []):
        lines.append(f"| `{fn['name']}` | {fn['line']}-{fn['end_line']} | `{fn['target_module']}` |")
    lines.extend(["", "## `pqc_checkpoint.c` Function Map", "", "| Function | Lines | Target module |", "| --- | ---: | --- |"])
    for fn in payload.get("pqc_checkpoint_functions", []):
        lines.append(f"| `{fn['name']}` | {fn['line']}-{fn['end_line']} | `{fn['target_module']}` |")
    lines.extend(["", "## `pqc_recovery.c` Function Map", "", "| Function | Lines | Target module |", "| --- | ---: | --- |"])
    for fn in payload.get("pqc_recovery_functions", []):
        lines.append(f"| `{fn['name']}` | {fn['line']}-{fn['end_line']} | `{fn['target_module']}` |")
    lines.extend(["", "## `pqc_rekey.c` Function Map", "", "| Function | Lines | Target module |", "| --- | ---: | --- |"])
    for fn in payload.get("pqc_rekey_functions", []):
        lines.append(f"| `{fn['name']}` | {fn['line']}-{fn['end_line']} | `{fn['target_module']}` |")
    lines.extend(["", "## `pqc_runtime.c` Function Map", "", "| Function | Lines | Target module |", "| --- | ---: | --- |"])
    for fn in payload.get("pqc_runtime_functions", []):
        lines.append(f"| `{fn['name']}` | {fn['line']}-{fn['end_line']} | `{fn['target_module']}` |")
    lines.extend(["", "## `pqc_scheduler.c` Function Map", "", "| Function | Lines | Target module |", "| --- | ---: | --- |"])
    for fn in payload.get("pqc_scheduler_functions", []):
        lines.append(f"| `{fn['name']}` | {fn['line']}-{fn['end_line']} | `{fn['target_module']}` |")
    lines.extend(["", "## `pqc_selftest.c` Function Map", "", "| Function | Lines | Target module |", "| --- | ---: | --- |"])
    for fn in payload.get("pqc_selftest_functions", []):
        lines.append(f"| `{fn['name']}` | {fn['line']}-{fn['end_line']} | `{fn['target_module']}` |")
    lines.extend(["", "## `pqc_state.c` Function Map", "", "| Function | Lines | Target module |", "| --- | ---: | --- |"])
    for fn in payload.get("pqc_state_functions", []):
        lines.append(f"| `{fn['name']}` | {fn['line']}-{fn['end_line']} | `{fn['target_module']}` |")
    lines.extend(["", "## `pqc_storage_path.c` Function Map", "", "| Function | Lines | Target module |", "| --- | ---: | --- |"])
    for fn in payload.get("pqc_storage_path_functions", []):
        lines.append(f"| `{fn['name']}` | {fn['line']}-{fn['end_line']} | `{fn['target_module']}` |")
    lines.extend(["", "## `pqc_strict_publish.c` Function Map", "", "| Function | Lines | Target module |", "| --- | ---: | --- |"])
    for fn in payload.get("pqc_strict_publish_functions", []):
        lines.append(f"| `{fn['name']}` | {fn['line']}-{fn['end_line']} | `{fn['target_module']}` |")
    lines.extend(["", "## `pqc_writeback.c` Function Map", "", "| Function | Lines | Target module |", "| --- | ---: | --- |"])
    for fn in payload.get("pqc_writeback_functions", []):
        lines.append(f"| `{fn['name']}` | {fn['line']}-{fn['end_line']} | `{fn['target_module']}` |")
    lines.extend(["", "## `pqc_xattr.c` Function Map", "", "| Function | Lines | Target module |", "| --- | ---: | --- |"])
    for fn in payload.get("pqc_xattr_functions", []):
        lines.append(f"| `{fn['name']}` | {fn['line']}-{fn['end_line']} | `{fn['target_module']}` |")
    lines.extend(["", "## Full Production Function Map", "", "| Source file | Function | Lines | Target module |", "| --- | --- | ---: | --- |"])
    for fn in payload.get("production_functions", []):
        lines.append(
            f"| `{fn['file']}` | `{fn['name']}` | {fn['line']}-{fn['end_line']} | `{fn['target_module']}` |"
        )
    lines.extend(["", "## Global State Map", "", "| Name | Location | Target module | Declaration |", "| --- | --- | --- | --- |"])
    for glob in payload["global_state"]:
        decl = glob["declaration"].replace("|", "\\|")
        lines.append(f"| `{glob['name']}` | `{glob['file']}:{glob['line']}` | `{glob['target_module']}` | `{decl}` |")
    lines.extend(["", "## Negative Claim Guard", "", payload["negative_claim_guard"], ""])
    return "\n".join(lines)


def build_phase1_behavior_equivalence(source_ownership: dict[str, Any]) -> dict[str, Any]:
    summary = source_ownership["summary"]
    frozen = load_json_optional(
        ROOT / "artifacts" / "validation" / "frozen_aegisq_contract" / "frozen_aegisq_contract.json"
    )
    behavior_evidence_status = {
        "posix_scope_audit_pass": artifact_pass(
            "artifacts/validation/posix_scope_audit/posix_scope_audit.json"
        ),
        "generation_fault_matrix_pass": artifact_pass(
            "artifacts/validation/generation_fault_matrix/generation_fault_matrix.json"
        ),
        "daemon_power_fault_campaign_pass": artifact_pass(
            "artifacts/validation/daemon_power_fault_campaign/daemon_power_fault_campaign.json"
        ),
        "frozen_workload_contract_pass": artifact_pass(
            "artifacts/validation/frozen_workload_contract/frozen_workload_contract.json"
        ),
        "strict_frozen_aegisq_warm_cache_pass": bool(
            frozen
            and frozen.get("overall_pass") is True
            and frozen.get("contract_compliant_warm_cache") is True
        ),
        "strict_frozen_aegisq_comparison_ready": bool(
            frozen and frozen.get("comparison_ready") is True
        ),
        "concurrency_contract_smoke_pass": artifact_pass(
            "artifacts/validation/concurrency_contract/lock_profile_summary.json"
        ),
        "paper_text_audit_done": False,
        "new_mechanism_claims_added": False,
    }
    missing_gate014 = [
        label for label, present in {
            "POSIX audit": behavior_evidence_status["posix_scope_audit_pass"],
            "generation matrix": behavior_evidence_status["generation_fault_matrix_pass"],
            "daemon matrix": behavior_evidence_status["daemon_power_fault_campaign_pass"],
            "frozen workload contract": behavior_evidence_status["frozen_workload_contract_pass"],
            "strict frozen AEGIS-Q warm-cache benchmark": behavior_evidence_status[
                "strict_frozen_aegisq_warm_cache_pass"
            ],
        }.items()
        if not present
    ]
    if missing_gate014:
        gate014_status = (
            "Gate 0.14 is not fully closed because these code/artifact checks "
            f"are still missing: {', '.join(missing_gate014)}; paper text and "
            "negative-claim audit also remain unclosed."
        )
    else:
        gate014_status = (
            "Gate 0.14 code/artifact evidence has advanced: POSIX audit, "
            "generation matrix, daemon matrix, frozen workload contract, and "
            "strict warm-cache AEGIS-Q frozen benchmark pass after decomposition. "
            "The checklist remains open until paper text and the final negative-claim "
            "audit prove no new mechanism or broad filesystem claim was added."
        )
    extracted_modules = {
        "config": ["pqc_config.c", "pqc_config.h"],
        "test_hooks": ["pqc_test_hooks.c", "pqc_test_hooks.h"],
        "format": ["pqc_format.h"],
        "data_crypto": ["pqc_crypto.c", "pqc_crypto.h"],
        "fd_context": ["pqc_fd_context.c", "pqc_fd_context.h"],
        "file_lock": ["pqc_file_lock.c", "pqc_file_lock.h"],
        "file_io": ["pqc_file_io.c", "pqc_file_io.h"],
        "flush_batch": ["pqc_flush_batch.c", "pqc_flush_batch.h"],
        "flush_crypto": ["pqc_flush_crypto.c", "pqc_flush_crypto.h"],
        "journal": ["pqc_journal.c", "pqc_journal.h"],
        "keyring": ["pqc_keyring.c", "pqc_keyring.h"],
        "lifecycle": ["pqc_lifecycle.c", "pqc_lifecycle.h"],
        "lock_profile": ["pqc_lock_profile.c", "pqc_lock_profile.h"],
        "entrypoint": ["pqc_main.c", "pqc_fuse.h"],
        "metrics": ["pqc_metrics.c", "pqc_metrics.h"],
        "namespace": ["pqc_namespace.c", "pqc_namespace.h"],
        "parallel_commit": ["pqc_parallel_commit.c", "pqc_parallel_commit.h"],
        "posix": ["pqc_posix.c", "pqc_posix.h"],
        "publish_xattr": ["pqc_publish.c", "pqc_publish.h"],
        "checkpoint": ["pqc_checkpoint.c", "pqc_checkpoint.h"],
        "anchor_worker": ["pqc_anchor_worker.c", "pqc_anchor_worker.h"],
        "recovery": ["pqc_recovery.c", "pqc_recovery.h"],
        "rekey": ["pqc_rekey.c", "pqc_rekey.h"],
        "runtime": ["pqc_runtime.c", "pqc_runtime.h"],
        "scheduler": ["pqc_scheduler.c", "pqc_scheduler.h"],
        "qos_runtime": ["pqc_qos.c", "pqc_qos.h"],
        "selftest": ["pqc_selftest.c", "pqc_selftest.h"],
        "state_table": ["pqc_state.c", "pqc_state.h"],
        "storage_path": ["pqc_storage_path.c", "pqc_storage_path.h"],
        "strict_publish": ["pqc_strict_publish.c", "pqc_strict_publish.h"],
        "writeback": ["pqc_writeback.c", "pqc_writeback.h"],
        "xattr": ["pqc_xattr.c", "pqc_xattr.h"],
    }
    module_roles = {
        "config": "centralized runtime configuration and artifact dump boundary",
        "test_hooks": "single fault/pause cutpoint interface",
        "format": "central persistent-format constant/struct boundary",
        "data_crypto": "AES-GCM data-block crypto boundary",
        "fd_context": "per-open context and sidecar descriptor ownership",
        "file_lock": "fcntl/flock passthrough boundary",
        "file_io": "mounted read/write/fsync/flush/release callback body",
        "flush_batch": "write-buffer block batching helper",
        "flush_crypto": "flush-time crypto job wrapper",
        "journal": "strict mapping journal append/read helper",
        "keyring": "mount/per-file key envelope boundary",
        "lifecycle": "FUSE init/destroy lifecycle boundary",
        "lock_profile": "lock hold-time trace hook boundary",
        "entrypoint": "process entrypoint and FUSE operation table declaration",
        "metrics": "trace/log/metrics helper boundary",
        "namespace": "namespace and metadata callback body",
        "parallel_commit": "parallel commit coordinator boundary; not a paper claim by itself",
        "posix": "lower filesystem path/syscall helper boundary",
        "publish_xattr": "logical-size/checkpoint xattr publication helper",
        "checkpoint": "checkpoint/generation reservation helper boundary",
        "anchor_worker": "background freshness-anchor worker boundary",
        "recovery": "authenticated mapping recovery helper boundary",
        "rekey": "background rekey queue boundary",
        "runtime": "process runtime initialization/shutdown boundary",
        "scheduler": "CPU/GPU routing policy boundary",
        "qos_runtime": "QoS throttle/runtime telemetry boundary",
        "selftest": "binary self-test boundary",
        "state_table": "shared per-file publish-state table boundary",
        "storage_path": "storage-root path resolution and sidecar naming boundary",
        "strict_publish": "strict durable publication ordering boundary",
        "writeback": "writeback snapshot/prepare/publish orchestration boundary",
        "xattr": "xattr callback policy boundary",
    }
    behavior_evidence_checks = [
        {
            "name": "POSIX scope audit",
            "artifact": "artifacts/validation/posix_scope_audit/posix_scope_audit.json",
            "status": "pass" if behavior_evidence_status["posix_scope_audit_pass"] else "missing-or-fail",
            "required_for_submilestone": True,
        },
        {
            "name": "generation fault matrix",
            "artifact": "artifacts/validation/generation_fault_matrix/generation_fault_matrix.json",
            "status": "pass" if behavior_evidence_status["generation_fault_matrix_pass"] else "missing-or-fail",
            "required_for_submilestone": True,
        },
        {
            "name": "daemon power fault campaign",
            "artifact": "artifacts/validation/daemon_power_fault_campaign/daemon_power_fault_campaign.json",
            "status": "pass" if behavior_evidence_status["daemon_power_fault_campaign_pass"] else "missing-or-fail",
            "required_for_submilestone": True,
        },
        {
            "name": "frozen workload contract",
            "artifact": "artifacts/validation/frozen_workload_contract/frozen_workload_contract.json",
            "status": "pass" if behavior_evidence_status["frozen_workload_contract_pass"] else "missing-or-fail",
            "required_for_submilestone": True,
        },
        {
            "name": "strict frozen AEGIS-Q warm-cache benchmark",
            "artifact": "artifacts/validation/frozen_aegisq_contract/frozen_aegisq_contract.json",
            "status": "pass" if behavior_evidence_status["strict_frozen_aegisq_warm_cache_pass"] else "missing-or-fail",
            "required_for_submilestone": True,
            "comparison_ready": behavior_evidence_status["strict_frozen_aegisq_comparison_ready"],
        },
        {
            "name": "concurrency contract smoke",
            "artifact": "artifacts/validation/concurrency_contract/lock_profile_summary.json",
            "status": "pass" if behavior_evidence_status["concurrency_contract_smoke_pass"] else "missing-or-fail",
            "required_for_submilestone": False,
        },
    ]
    missing_parent_gate_evidence = []
    if behavior_evidence_status["strict_frozen_aegisq_warm_cache_pass"] and not behavior_evidence_status["strict_frozen_aegisq_comparison_ready"]:
        missing_parent_gate_evidence.append(
            {
                "category": "baseline-comparison",
                "item": "strict_frozen_aegisq_comparison_ready",
                "reason": "strict warm-cache AEGIS-Q run exists, but matched comparison rows are not complete enough for a broad performance claim",
            }
        )
    if not behavior_evidence_status["paper_text_audit_done"]:
        missing_parent_gate_evidence.append(
            {
                "category": "paper",
                "item": "paper_text_audit",
                "reason": "paper text has not been audited to prove mechanical decomposition introduced no new mechanism claim",
            }
        )
    if summary.get("ownership_blocker_count", 0):
        missing_parent_gate_evidence.append(
            {
                "category": "ownership-review",
                "item": "ambiguous ownership blockers",
                "reason": "source ownership map still reports ambiguous functions/global state that must be reviewed before claiming clean architecture",
                "ambiguous_function_count": summary.get("ambiguous_function_count", 0),
                "ambiguous_global_state_count": summary.get("ambiguous_global_state_count", 0),
            }
        )
    gate014_code_artifact_pass = not missing_gate014
    return {
        "schema_version": 3,
        "generated_utc": now_utc(),
        "generated_by": "code/experiments/build_refactor_inventory.py",
        "phase": "Gate 0.14 mechanical decomposition",
        "summary": {
            "extracted_module_count": len(extracted_modules),
            "extracted_module_file_count": sum(len(files) for files in extracted_modules.values()),
            "pqc_fuse_adapter_function_count": summary["pqc_fuse_function_count"],
            "production_source_file_count": summary.get("production_source_file_count"),
            "production_function_count": summary.get("all_current_production_function_count"),
            "all_production_functions_mapped": summary.get("all_current_production_functions_mapped"),
            "all_global_state_mapped": summary.get("all_current_global_state_mapped"),
            "ownership_blocker_count": summary.get("ownership_blocker_count", 0),
            "ambiguous_function_count": summary.get("ambiguous_function_count", 0),
            "ambiguous_global_state_count": summary.get("ambiguous_global_state_count", 0),
            "required_behavior_evidence_pass_count": sum(
                1
                for item in behavior_evidence_checks
                if item["required_for_submilestone"] and item["status"] == "pass"
            ),
            "required_behavior_evidence_count": sum(
                1 for item in behavior_evidence_checks if item["required_for_submilestone"]
            ),
            "missing_required_behavior_evidence": missing_gate014,
            "missing_parent_gate_evidence_count": len(missing_parent_gate_evidence),
            "gate014_code_artifact_evidence_pass": gate014_code_artifact_pass,
            "submilestone_status": "mechanical-decomposition-evidence-refreshed",
            "parent_checklist_closed": False,
        },
        "source_layout": {
            "authoritative_source_root": "code/",
            "production_native_source_root": "code/",
            "experiment_source_root": "code/experiments/",
            "legacy_experiments_path": "removed; use code/experiments/",
            "root_source_files_allowed": False,
            "physical_experiments_source_files_allowed": False,
        },
        "scope": (
            "Records current behavior-preserving ownership extraction before "
            "epoch mode, fine-grained lock changes, or new paper claims. This "
            "manifest is generated from the current source ownership map and "
            "does not claim POSIX, crash, performance, or publication-protocol "
            "closure."
        ),
        "current_extracted_modules": extracted_modules,
        "module_classification": [
            {
                "module": module,
                "files": files,
                "file_count": len(files),
                "gate014_role": module_roles[module],
                "status": "extracted-module-inventory",
            }
            for module, files in sorted(extracted_modules.items())
        ],
        "function_counts": {
            "pqc_fuse.c": summary["pqc_fuse_function_count"],
            "pqc_fd_context.c": summary.get("pqc_fd_context_function_count", 0),
            "pqc_file_lock.c": summary.get("pqc_file_lock_function_count", 0),
            "pqc_file_io.c": summary.get("pqc_file_io_function_count", 0),
            "pqc_writeback.c": summary.get("pqc_writeback_function_count", 0),
            "pqc_strict_publish.c": summary.get("pqc_strict_publish_function_count", 0),
            "pqc_flush_batch.c": summary.get("pqc_flush_batch_function_count", 0),
            "pqc_flush_crypto.c": summary.get("pqc_flush_crypto_function_count", 0),
            "pqc_lifecycle.c": summary.get("pqc_lifecycle_function_count", 0),
            "pqc_lock_profile.c": summary.get("pqc_lock_profile_function_count", 0),
            "pqc_main.c": summary.get("pqc_main_function_count", 0),
            "pqc_namespace.c": summary.get("pqc_namespace_function_count", 0),
            "pqc_qos.c": summary.get("pqc_qos_function_count", 0),
            "pqc_rekey.c": summary.get("pqc_rekey_function_count", 0),
            "pqc_runtime.c": summary.get("pqc_runtime_function_count", 0),
            "pqc_scheduler.c": summary.get("pqc_scheduler_function_count", 0),
            "pqc_storage_path.c": summary.get("pqc_storage_path_function_count", 0),
            "pqc_xattr.c": summary.get("pqc_xattr_function_count", 0),
        },
        "preserved_behavior_contract": [
            "No epoch redo log, group commit, sharded queue, eBPF/io_uring path, GPUDirect path, zero-copy path, or KDF upgrade is introduced by mechanical decomposition.",
            "Strict writeback snapshots the live fd write buffer under fd_lock, clears it, marks a pending job, and runs full-tier prepare/crypto outside fd_lock and commit_lock after a persistent generation reservation and publish-ticket turn acquisition.",
            "The current strict durable publication ordering remains owned by pqc_strict_publish_commit and is invoked from pqc_writeback_flush_locked while the publish turn is current.",
            "Existing data_write_after_pwrite, data_fsync_after, journal_append_after, journal_fsync_after, logical_size_xattr_after, checkpoint_xattr_after, and fsync_before_return cutpoint names remain production-visible; generation_reservation_xattr_after is added for the reservation high-water boundary.",
            "AES-GCM block nonce, AAD, tag, journal digest, metadata HMAC, checkpoint HMAC, logical-size xattr, storage-root path resolution, hidden sidecar, internal xattr filtering, and QoS xattr parsing behavior are preserved by this extraction.",
            "xattr callback behavior is preserved while moving tier, QoS, freshness-window, and internal-xattr filtering policy into pqc_xattr.[ch].",
            "Namespace and metadata callbacks preserve hidden sidecar filtering, logical-size getattr, sidecar unlink cleanup, explicit rename/fsyncdir rejection, and utimens passthrough while moving into pqc_namespace.[ch].",
            "Mounted file I/O callbacks preserve open/create key-envelope restoration, read authentication, write coalescing, strict fsync/flush/release dirty-sidecar behavior, truncate/fallocate logical-size updates, and shared file-state logical-size updates while moving into pqc_file_io.[ch].",
            "File lock callbacks preserve fcntl/flock passthrough behavior while moving into pqc_file_lock.[ch].",
            "FUSE init/destroy lifecycle preserves cache capability choices and shutdown ordering while moving into pqc_lifecycle.[ch].",
            "pqc_fuse.c remains the FUSE callback adapter and still owns high-level callback sequencing; extracted modules do not imply a clean architecture claim until behavior and workload gates close.",
        ],
        "verification_contract": [
            "cat SUBMISSION_CHECKLIST.md",
            "python3 -m py_compile code/experiments/build_refactor_inventory.py",
            "cmake -S . -B build",
            "cmake --build build --parallel 2",
            "ctest --test-dir build --output-on-failure",
            "build/pqc_fuse --scheduler-smoke",
            "build/pqc_fuse --admission-telemetry-smoke",
            "python3 code/experiments/run_posix_scope_audit.py --out-dir artifacts/validation/posix_scope_audit",
            "python3 code/experiments/run_generation_fault_matrix.py --out-dir artifacts/validation/generation_fault_matrix",
            "python3 code/experiments/run_daemon_power_fault_matrix.py --out-dir artifacts/validation/daemon_power_fault_campaign",
            "python3 code/experiments/build_frozen_workload_contract.py --out-dir artifacts/validation/frozen_workload_contract",
            "python3 code/experiments/run_frozen_aegisq_contract.py --out artifacts/validation/frozen_aegisq_contract --warmup-runs 1 --repetitions 5 --fio-timeout-s 180 --overwrite",
            "python3 code/experiments/run_concurrency_contract_smoke.py --out artifacts/validation/concurrency_contract --iterations 4 --block-size 4096 --thread-counts 1,2,4 --phase-timeout-s 30 --overwrite",
            "python3 code/experiments/build_parallel_commit_contract.py",
            "python3 code/experiments/build_refactor_inventory.py",
            "jq -e . artifacts/validation/refactor_inventory/*.json artifacts/validation/parallel_commit_contract/*.json >/dev/null",
            "git diff --check --",
        ],
        "inventory_evidence": {
            "source_ownership_json": "artifacts/validation/refactor_inventory/source_ownership_map.json",
            "source_ownership_md": "docs/architecture/source_ownership_map.md",
            "worktree_freeze_json": "artifacts/validation/refactor_inventory/worktree_freeze.json",
            "environment_variables_json": "artifacts/validation/refactor_inventory/environment_variables.json",
            "on_disk_state_json": "artifacts/validation/refactor_inventory/on_disk_state.json",
            "durability_barriers_json": "artifacts/validation/refactor_inventory/durability_barriers.json",
            "lock_inventory_json": "artifacts/validation/refactor_inventory/lock_inventory.json",
            "fault_cutpoints_json": "artifacts/validation/refactor_inventory/fault_cutpoints.json",
            "posix_scope_audit_json": "artifacts/validation/posix_scope_audit/posix_scope_audit.json",
            "generation_fault_matrix_json": "artifacts/validation/generation_fault_matrix/generation_fault_matrix.json",
            "daemon_power_fault_campaign_json": "artifacts/validation/daemon_power_fault_campaign/daemon_power_fault_campaign.json",
            "frozen_workload_contract_json": "artifacts/validation/frozen_workload_contract/frozen_workload_contract.json",
            "strict_frozen_aegisq_json": "artifacts/validation/frozen_aegisq_contract/frozen_aegisq_contract.json",
            "parallel_commit_contract_json": "artifacts/validation/parallel_commit_contract/parallel_commit_contract.json",
        },
        "behavior_evidence_checks": behavior_evidence_checks,
        "behavior_evidence_status": behavior_evidence_status,
        "missing_parent_gate_evidence": missing_parent_gate_evidence,
        "artifact_verdict": {
            "overall_pass": (
                gate014_code_artifact_pass
                and bool(extracted_modules)
                and len(behavior_evidence_checks) > 0
            ),
            "submilestone_done": (
                gate014_code_artifact_pass
                and bool(extracted_modules)
                and len(behavior_evidence_checks) > 0
            ),
            "gate014_code_artifact_evidence_pass": gate014_code_artifact_pass,
            "required_behavior_evidence_missing_count": len(missing_gate014),
            "parent_gate_missing_evidence_count": len(missing_parent_gate_evidence),
            "paper_text_audit_done": behavior_evidence_status["paper_text_audit_done"],
            "parent_checklist_closed": False,
        },
        "not_closed": [
            gate014_status,
            "Gate 0.15 is not closed: a narrow production-mounted thread/process-count lock-profile sweep, repeated same-file/disjoint-file lifecycle phases, external OS-process lifecycle phases, a strace profile of client-visible blocking syscalls, and one paused-publish reader-visibility probe now exist, and strict full-tier writeback prepare/crypto/durable publication, authenticated-read recovery, truncate/fallocate metadata publication, and release resource teardown no longer run under fd_lock or commit_lock on their hot paths, but long-duration client-count sweep, scheduler off-CPU profile, and reserved-generation fault expansion are still missing.",
            "Gate 0.16 parent closure is still blocked: sharded leader/follower coordinator, epoch-path integration, telemetry, fairness/starvation, and replay-order evidence exist, but broad closure still requires deliberate paper text and negative-claim guard updates.",
            "Gate 0.9 parent closure is still blocked: strict/epoch publication code, redo-log, group sync, replay, fault matrix, measurement, and closeout artifacts exist, but broad closure still requires paper text and negative-claim guard updates.",
            "The dirty-sidecar sync dedup patch remains pending-unclaimed because it is not isolated by before/after benchmark, syscall/fdatasync attribution, dirty-flag fault matrix, or paper negative-claim audit.",
        ],
        "negative_claim_guard": (
            "This artifact does not justify KDF-strength, crash-safety, "
            "power-loss, performance, epoch-mode, group-commit, fine-grained "
            "locking, parallel-commit, eBPF/io_uring bypass, GPUDirect, "
            "zero-copy, broad POSIX, deployment, QoS superiority, or clean "
            "architecture claims."
        ),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    write_json(OUT / "worktree_freeze.json", build_worktree_freeze())

    source_paths = [source_path(name) for name in SOURCE_FILES if (source_path(name)).exists()]
    functions: list[dict[str, Any]] = []
    for path in source_paths:
        functions.extend(extract_functions(path))
    globals_out: list[dict[str, Any]] = []
    for path in source_paths:
        globals_out.extend(extract_globals(path, functions))

    source_ownership = build_source_ownership(functions, globals_out)
    write_json(OUT / "source_ownership_map.json", source_ownership)
    write_text(DOCS / "source_ownership_map.md", render_source_map_md(source_ownership))
    write_json(OUT / "phase1_behavior_equivalence.json",
               build_phase1_behavior_equivalence(source_ownership))

    env_vars = extract_environment_variables(functions)
    runtime_dump_support = config_dump_support_present()
    env_vars = enrich_environment_variables(env_vars, runtime_dump_support)
    raw_getenv_outside_config = [
        item
        for item in env_vars
        if item["status"] == "raw-getenv" and item["file"] not in {"pqc_config.c", "pqc_config.h"}
    ]
    raw_getenv_inside_config = [
        item
        for item in env_vars
        if item["status"] == "raw-getenv" and item["file"] in {"pqc_config.c", "pqc_config.h"}
    ]
    status_counts = Counter(item["status"] for item in env_vars)
    gate05_counts = Counter(item["gate05_blocker_status"] for item in env_vars)
    wrapper_only_count = status_counts.get("config-wrapper", 0)
    blocking_statuses = {
        "blocked-raw-getenv-outside-config",
        "blocked-no-runtime-dump-support",
    }
    open_statuses = {
        "open-wrapper-only",
        "open-legacy-helper",
    }
    gate05_blockers = [
        item for item in env_vars
        if item["gate05_blocker_status"] in blocking_statuses
    ]
    gate05_open_items = [
        item for item in env_vars
        if item["gate05_blocker_status"] in open_statuses
    ]
    if raw_getenv_outside_config:
        config_gate_status = "raw-getenv-outside-config; migration incomplete"
    elif not runtime_dump_support:
        config_gate_status = "central-config-access-in-code; runtime dump support missing"
    elif wrapper_only_count:
        config_gate_status = "central-wrapper-and-dump-support-in-code; typed/string contract for wrapper-only sites still pending"
    else:
        config_gate_status = "runtime-config-contract-in-code; paper-facing knob audit still pending"
    write_json(
        OUT / "environment_variables.json",
        {
            "schema_version": 2,
            "generated_utc": now_utc(),
            "generated_by": "code/experiments/build_refactor_inventory.py",
            "summary": {
                "environment_variable_site_count": len(env_vars),
                "unique_environment_variable_count": len({item["name"] for item in env_vars}),
                "status_counts": dict(sorted(status_counts.items())),
                "gate05_blocker_status_counts": dict(sorted(gate05_counts.items())),
                "raw_getenv_outside_pqc_config_count": len(raw_getenv_outside_config),
                "raw_getenv_inside_pqc_config_count": len(raw_getenv_inside_config),
                "config_wrapper_only_count": wrapper_only_count,
                "gate05_hard_blocker_count": len(gate05_blockers),
                "gate05_open_review_item_count": len(gate05_open_items),
                "runtime_config_dump_support": runtime_dump_support,
                "runtime_config_dump_env": "PQC_CONFIG_DUMP_PATH",
                "migration_target": "pqc_config.[ch]",
                "gate0_status": config_gate_status,
            },
            "runtime_config_dump_support": {
                "supported": runtime_dump_support,
                "request_env": "PQC_CONFIG_DUMP_PATH",
                "implementation": {
                    "source": "pqc_config.c",
                    "dump_file_function": "pqc_config_dump_file",
                    "dump_if_requested_function": "pqc_config_dump_if_requested",
                    "header": "pqc_config.h",
                },
                "negative_claim_guard": (
                    "The runtime dump records observed knob accesses for a run; "
                    "it does not prove that unexecuted code paths did not read a knob."
                ),
            },
            "variables": env_vars,
            "raw_getenv_outside_pqc_config": raw_getenv_outside_config,
            "raw_getenv_inside_pqc_config": raw_getenv_inside_config,
            "gate05_hard_blockers": gate05_blockers,
            "gate05_open_review_items": gate05_open_items,
            "artifact_verdict": {
                "overall_pass": (
                    runtime_dump_support
                    and not raw_getenv_outside_config
                    and all("gate05_blocker_status" in item for item in env_vars)
                ),
                "runtime_dump_support": runtime_dump_support,
                "raw_getenv_outside_pqc_config_count": len(raw_getenv_outside_config),
                "every_site_has_gate05_status": all(
                    "gate05_blocker_status" in item for item in env_vars
                ),
                "parent_checklist_closed": (
                    runtime_dump_support
                    and not raw_getenv_outside_config
                    and all("gate05_blocker_status" in item for item in env_vars)
                    and len(gate05_blockers) == 0
                    and len(gate05_open_items) == 0
                ),
            },
            "negative_claim_guard": (
                "Every paper-facing knob must be emitted in run artifacts before it "
                "is treated as an evaluated mechanism. A centralized config site is "
                "not a performance, correctness, security, or deployment claim."
            ),
        },
    )

    defines, structs = extract_defines_and_structs()
    defines = enrich_format_defines(defines)
    structs = enrich_format_structs(structs)
    format_define_names = {
        item["name"]
        for item in defines
        if any(token in item["name"] for token in ("MAGIC", "VERSION", "XATTR", "ALGO", "TIER", "QOS", "AEAD", "LOGICAL_BLOCK", "TPM_NV"))
    }
    format_defs_outside_header = [
        item
        for item in defines
        if item["name"] in format_define_names
        and item["file"] != "pqc_format.h"
        and str(item.get("gate06_blocker_status", "")).startswith("blocked-")
    ]
    persistent_struct_names = {
        "pqc_metadata_t",
        "pqc_checkpoint_t",
        "block_mapping_t",
        "journal_record_t",
        "pqc_prefix_anchor_t",
        "pqc_freshness_anchor_t",
    }
    persistent_structs_outside_header = [
        item
        for item in structs
        if item["name"] in persistent_struct_names and item["file"] != "pqc_format.h"
    ]
    format_core_owned_by_header = (
        len(format_defs_outside_header) == 0 and
        len(persistent_structs_outside_header) == 0 and
        len(format_define_names) > 0
    )
    sidecar_objects = [
        {
            "name": "marker file",
            "path_rule": "plain marker path under storage root",
            "current_owner": "pqc_storage_path.c:pqc_storage_path_resolve / FUSE callbacks",
            "durability_role": "namespace-visible file object used to derive sidecar paths and xattrs",
            "format_status": "no stable magic; ordinary lower-filesystem file path semantics",
            "endian_rule": "not applicable",
            "digest_or_hmac_rule": "metadata and checkpoint integrity live in xattrs/sidecars, not marker file contents",
            "upgrade_downgrade_behavior": "not versioned as a standalone format",
        },
        {
            "name": "data sidecar",
            "suffix": ".pqcdata",
            "current_owner": "pqc_fd_context.c:pqc_fd_context_set / pqc_writeback.c:pqc_writeback_flush_locked",
            "durability_role": "ciphertext storage for authenticated blocks",
            "format_status": "suffix literal remains in fd-context setup; magic/version structs live in pqc_format.h",
            "endian_rule": "block offsets are journal/checkpoint metadata; data sidecar stores ciphertext bytes",
            "digest_or_hmac_rule": "AES-GCM tag is stored in block_mapping_t journal records",
            "upgrade_downgrade_behavior": "suffix is inventoried; compatibility policy remains unclaimed",
        },
        {
            "name": "journal sidecar",
            "suffix": ".pqcmeta",
            "current_owner": "pqc_fd_context.c:pqc_fd_context_set / pqc_journal.c:pqc_journal_append_mapping_unsynced",
            "durability_role": "mapping journal for committed logical blocks",
            "format_status": "suffix literal remains in fd-context setup; magic/version structs live in pqc_format.h",
            "endian_rule": "journal_record_t currently uses host struct layout; portable endian compatibility is not claimed",
            "digest_or_hmac_rule": "journal_record_t digest authenticates record fields",
            "upgrade_downgrade_behavior": "PQC_JOURNAL_VERSION is inventoried; migration behavior remains unclaimed",
        },
        {
            "name": "epoch redo log",
            "suffix": ".pqcepoch",
            "current_owner": "pqc_epoch_log.c / pqc_epoch_publish.c",
            "durability_role": "append-only epoch record prefix used for grouped publication and journal repair",
            "format_status": "magic/version/record layout constants live in pqc_format.h",
            "endian_rule": "explicit little-endian encode/decode helpers in pqc_epoch_log.c",
            "digest_or_hmac_rule": "record digest covers encoded bytes with digest field zeroed",
            "upgrade_downgrade_behavior": "PQC_EPOCH_LOG_VERSION is inventoried; cross-version migration remains unclaimed",
        },
        {
            "name": "metadata xattr",
            "xattr": "user.pqc_metadata",
            "current_owner": "pqc_keyring.c",
            "durability_role": "per-file wrapped shared secret and file id",
            "format_status": "pqc_metadata_t and PQC_METADATA_VERSION are in pqc_format.h; legacy duplicate pqc_file_key definitions have been removed",
            "endian_rule": "pqc_metadata_t currently uses host struct layout; portable endian compatibility is not claimed",
            "digest_or_hmac_rule": "metadata digest field is inventoried through pqc_metadata_t",
            "upgrade_downgrade_behavior": "PQC_METADATA_VERSION is checked; upgrade path remains unclaimed",
        },
        {
            "name": "logical-size xattr",
            "xattr": "user.pqc_logical_size",
            "current_owner": "pqc_publish.c",
            "durability_role": "logical-size publication for sparse encrypted data sidecar",
            "format_status": "xattr name lives in pqc_format.h",
            "endian_rule": "uint64_t xattr currently uses host layout; portable endian compatibility is not claimed",
            "digest_or_hmac_rule": "covered indirectly by checkpoint/anchor flow, not a standalone authenticated record",
            "upgrade_downgrade_behavior": "not versioned as a standalone format",
        },
        {
            "name": "checkpoint xattr",
            "xattr": "user.pqc_checkpoint",
            "current_owner": "pqc_publish.c / pqc_checkpoint.c",
            "durability_role": "file generation high-water, sequence, logical size, and anchor staging state",
            "format_status": "pqc_checkpoint_t and PQC_CHECKPOINT_VERSION live in pqc_format.h",
            "endian_rule": "pqc_checkpoint_t currently uses host struct layout; portable endian compatibility is not claimed",
            "digest_or_hmac_rule": "checkpoint digest field is inventoried through pqc_checkpoint_t",
            "upgrade_downgrade_behavior": "PQC_CHECKPOINT_VERSION is checked; upgrade path remains unclaimed",
        },
        {
            "name": "external freshness anchor",
            "path_env": "PQC_FRESHNESS_ANCHOR_PATH",
            "current_owner": "pqc_anchor.c / pqc_anchor_worker.c",
            "durability_role": "external file anchor for prefix root and global sequence",
            "format_status": "pqc_prefix_anchor_t and PQC_PREFIX_ANCHOR_VERSION live in pqc_format.h",
            "endian_rule": "pqc_prefix_anchor_t currently uses host struct layout; portable endian compatibility is not claimed",
            "digest_or_hmac_rule": "prefix-anchor digest field is inventoried through pqc_prefix_anchor_t",
            "upgrade_downgrade_behavior": "PQC_PREFIX_ANCHOR_VERSION is checked; rollback resistance is not claimed",
        },
        {
            "name": "TPM NV freshness anchor",
            "index_env": "PQC_TPM_NV_INDEX",
            "current_owner": "pqc_anchor.c",
            "durability_role": "optional TPM NV backend for prefix anchor bytes",
            "format_status": "PQC_TPM_NV_DEFAULT_INDEX lives in pqc_format.h; provisioning is explicit admin state",
            "endian_rule": "TPM payload stores pqc_prefix_anchor_t bytes; portable endian compatibility is not claimed",
            "digest_or_hmac_rule": "prefix-anchor digest is retained; PCR-bound policy is not implemented",
            "upgrade_downgrade_behavior": "unprovisioned TPM fails closed; PCR-bound rollback resistance remains unclaimed",
        },
    ]
    sidecar_objects = enrich_sidecar_objects(sidecar_objects)
    on_disk_objects = [
        *[
            {
                "object_type": "format-constant",
                "name": item["name"],
                "file": item["file"],
                "line": item["line"],
                "target_module": item["target_module"],
                "format_kind": item["format_kind"],
                "gate06_blocker_status": item["gate06_blocker_status"],
                "gate06_blocker_reason": item["gate06_blocker_reason"],
            }
            for item in defines
            if item["format_kind"] in {"magic", "version", "xattr-name", "tpm-nv-index", "block-size"}
        ],
        *[
            {
                "object_type": "persistent-record",
                "name": item["name"],
                "file": item["file"],
                "line": item["line"],
                "target_module": item["target_module"],
                "format_kind": item["format_kind"],
                "gate06_blocker_status": item["gate06_blocker_status"],
                "gate06_blocker_reason": item["gate06_blocker_reason"],
            }
            for item in structs
            if item["format_kind"] != "runtime-struct"
        ],
        *[
            {
                "object_type": "storage-object",
                **obj,
            }
            for obj in sidecar_objects
        ],
    ]
    gate06_format_status_counts = Counter(
        item["gate06_blocker_status"] for item in on_disk_objects
    )
    gate06_format_blockers = [
        item for item in on_disk_objects
        if str(item["gate06_blocker_status"]).startswith("blocked-")
    ]
    write_json(
        OUT / "on_disk_state.json",
        {
            "schema_version": 2,
            "generated_utc": now_utc(),
            "generated_by": "code/experiments/build_refactor_inventory.py",
            "summary": {
                "define_count": len(defines),
                "struct_count": len(structs),
                "sidecar_object_count": len(sidecar_objects),
                "on_disk_object_count": len(on_disk_objects),
                "gate06_format_blocker_count": len(gate06_format_blockers),
                "gate06_blocker_status_counts": dict(sorted(gate06_format_status_counts.items())),
                "format_core_owned_by_pqc_format_h": format_core_owned_by_header,
                "format_defs_outside_header_count": len(format_defs_outside_header),
                "persistent_structs_outside_header_count": len(persistent_structs_outside_header),
                "gate0_status": "format-core-owned-by-header; sidecar naming and upgrade policy still pending",
            },
            "on_disk_objects": on_disk_objects,
            "defines": defines,
            "structs": structs,
            "format_defs_outside_header": format_defs_outside_header,
            "persistent_structs_outside_header": persistent_structs_outside_header,
            "sidecar_objects": sidecar_objects,
            "gate06_format_blockers": gate06_format_blockers,
            "artifact_verdict": {
                "overall_pass": (
                    all("gate06_blocker_status" in item for item in on_disk_objects)
                    and not gate06_format_blockers
                    and format_core_owned_by_header
                ),
                "every_on_disk_object_has_gate06_status": all(
                    "gate06_blocker_status" in item for item in on_disk_objects
                ),
                "format_core_owned_by_pqc_format_h": format_core_owned_by_header,
                "gate06_format_blocker_count": len(gate06_format_blockers),
                "parent_checklist_closed": (
                    all("gate06_blocker_status" in item for item in on_disk_objects)
                    and not gate06_format_blockers
                    and format_core_owned_by_header
                ),
            },
            "negative_claim_guard": "No on-disk format stability, endian portability, compatibility upgrade path, rollback resistance, or deployment-ready persistence claim is made until pqc_format.[ch] owns the stable format and retained compatibility/fault evidence exists.",
        },
    )

    durability = extract_durability_barriers(functions)
    durability = enrich_durability_barriers(durability)
    durability_operation_counts = Counter(item["operation"] for item in durability)
    durability_blocking_class_counts = Counter(
        item["blocking_class"] for item in durability
    )
    durability_gate06_counts = Counter(
        item["gate06_blocker_status"] for item in durability
    )
    blocking_sync_count = sum(
        1 for item in durability
        if item["blocking_class"] == "blocking-sync"
    )
    write_json(
        OUT / "durability_barriers.json",
        {
            "schema_version": 2,
            "generated_utc": now_utc(),
            "generated_by": "code/experiments/build_refactor_inventory.py",
            "summary": {
                "barrier_or_persistence_site_count": len(durability),
                "operation_counts": dict(sorted(durability_operation_counts.items())),
                "blocking_sync_count": blocking_sync_count,
                "blocking_class_counts": dict(
                    sorted(durability_blocking_class_counts.items())
                ),
                "gate06_blocker_status_counts": dict(
                    sorted(durability_gate06_counts.items())
                ),
                "every_barrier_has_gate06_status": all(
                    "gate06_blocker_status" in item for item in durability
                ),
                "gate0_status": (
                    "durability-barriers-inventoried; strict/epoch semantics "
                    "and hot-lock placement remain separate blockers"
                ),
            },
            "barriers": durability,
            "artifact_verdict": {
                "overall_pass": all(
                    "gate06_blocker_status" in item for item in durability
                ),
                "every_barrier_has_gate06_status": all(
                    "gate06_blocker_status" in item for item in durability
                ),
                "blocking_sync_count": blocking_sync_count,
                "parent_checklist_closed": all(
                    "gate06_blocker_status" in item for item in durability
                ),
            },
            "negative_claim_guard": (
                "No publication-protocol throughput, fsync-reduction, crash-safety, "
                "power-loss, epoch-mode, group-commit, or fine-grained-locking claim "
                "is upgraded until strict and epoch ordering evidence exists and "
                "hot-lock placement is measured."
            ),
        },
    )

    locks = extract_locks(functions)
    lock_status_counts = Counter(
        item["gate07_blocker_status"] for item in locks["locks"]
    )
    callsite_status_counts = Counter(
        item["gate07_blocker_status"] for item in locks["call_sites"]
    )
    lock_scope_counts = Counter(item["scope"] for item in locks["locks"])
    callsite_scope_counts = Counter(item["scope"] for item in locks["call_sites"])
    lock_blockers = [
        item
        for item in locks["locks"]
        if item["gate07_blocker_status"].startswith("blocked-")
    ]
    open_lock_risks = [
        item
        for item in locks["locks"]
        if item["gate07_blocker_status"] == "open-blocking-risk-needs-measurement"
    ]
    open_callsite_risks = [
        item
        for item in locks["call_sites"]
        if item["gate07_blocker_status"] == "open-callsite-risk-needs-hold-time-evidence"
    ]
    write_json(
        OUT / "lock_inventory.json",
        {
            "schema_version": 2,
            "generated_utc": now_utc(),
            "generated_by": "code/experiments/build_refactor_inventory.py",
            "summary": {
                "lock_count": len(locks["locks"]),
                "lock_call_site_count": len(locks["call_sites"]),
                "production_lock_count": lock_scope_counts.get("production", 0),
                "test_lock_count": lock_scope_counts.get("test", 0),
                "production_lock_call_site_count": callsite_scope_counts.get("production", 0),
                "test_lock_call_site_count": callsite_scope_counts.get("test", 0),
                "lock_gate07_status_counts": dict(sorted(lock_status_counts.items())),
                "callsite_gate07_status_counts": dict(sorted(callsite_status_counts.items())),
                "blocking_risk_lock_count": len(open_lock_risks),
                "blocking_risk_call_site_count": len(open_callsite_risks),
                "missing_contract_field_lock_count": len(lock_blockers),
                "every_lock_has_gate07_status": all(
                    "gate07_blocker_status" in item for item in locks["locks"]
                ),
                "every_callsite_has_gate07_status": all(
                    "gate07_blocker_status" in item for item in locks["call_sites"]
                ),
                "gate0_status": (
                    "lock-and-serialization-inventory-complete; measured hold-time "
                    "histograms and stress evidence remain Gate 0.15 blockers"
                ),
            },
            **locks,
            "lock_blockers": lock_blockers,
            "open_blocking_risk_locks": open_lock_risks,
            "open_blocking_risk_call_sites": open_callsite_risks,
            "artifact_verdict": {
                "overall_pass": (
                    all("gate07_blocker_status" in item for item in locks["locks"])
                    and all(
                        "gate07_blocker_status" in item
                        for item in locks["call_sites"]
                    )
                ),
                "every_lock_has_gate07_status": all(
                    "gate07_blocker_status" in item for item in locks["locks"]
                ),
                "every_callsite_has_gate07_status": all(
                    "gate07_blocker_status" in item for item in locks["call_sites"]
                ),
                "parent_checklist_closed": (
                    len(lock_blockers) == 0
                    and all("gate07_blocker_status" in item for item in locks["locks"])
                    and all(
                        "gate07_blocker_status" in item
                        for item in locks["call_sites"]
                    )
                ),
            },
            "negative_claim_guard": (
                "This inventory does not prove scalability, fine-grained locking, "
                "deadlock freedom, or absence of serialization.  No such claim is "
                "allowed until production-mounted lock hold-time histograms, "
                "contention profiles, off-CPU/blocking profiles, and stress evidence "
                "exist."
            ),
        },
    )

    fault_cutpoints = extract_fault_cutpoints(functions)
    fault_status_counts = Counter(
        item["gate08_blocker_status"] for item in fault_cutpoints
    )
    fault_scope_counts = Counter(item["scope"] for item in fault_cutpoints)
    fault_modules = Counter(item["target_module"] for item in fault_cutpoints)
    fault_blockers = [
        item
        for item in fault_cutpoints
        if item["gate08_blocker_status"].startswith("blocked-")
    ]
    production_fault_cutpoints = [
        item for item in fault_cutpoints if item["scope"] == "production"
    ]
    implementation_owner = next(
        (fn["file"] for fn in functions if fn["name"] == "pqc_fault_cutpoint"),
        "missing",
    )
    write_json(
        OUT / "fault_cutpoints.json",
        {
            "schema_version": 2,
            "generated_utc": now_utc(),
            "generated_by": "code/experiments/build_refactor_inventory.py",
            "summary": {
                "fault_cutpoint_count": len(fault_cutpoints),
                "production_fault_cutpoint_count": len(production_fault_cutpoints),
                "test_fault_cutpoint_count": fault_scope_counts.get("test", 0),
                "implementation_owner": implementation_owner,
                "implementation_interface": "pqc_test_hooks.[ch]:pqc_fault_cutpoint",
                "gate08_status_counts": dict(sorted(fault_status_counts.items())),
                "target_module_counts": dict(sorted(fault_modules.items())),
                "blocked_cutpoint_count": len(fault_blockers),
                "every_cutpoint_has_gate08_status": all(
                    "gate08_blocker_status" in item for item in fault_cutpoints
                ),
                "every_cutpoint_has_invariant": all(
                    not str(item["invariant"]).startswith("needs explicit invariant")
                    for item in fault_cutpoints
                ),
                "single_test_hook_interface": implementation_owner == "pqc_test_hooks.c",
                "gate0_status": (
                    "single-test-hook-interface-inventoried; crash model and "
                    "fault-matrix coverage remain separate paper/parent-gate blockers"
                ),
            },
            "cutpoints": fault_cutpoints,
            "fault_cutpoint_blockers": fault_blockers,
            "artifact_verdict": {
                "overall_pass": (
                    implementation_owner == "pqc_test_hooks.c"
                    and all(
                        "gate08_blocker_status" in item
                        for item in fault_cutpoints
                    )
                    and not fault_blockers
                ),
                "single_test_hook_interface": implementation_owner == "pqc_test_hooks.c",
                "every_cutpoint_has_gate08_status": all(
                    "gate08_blocker_status" in item for item in fault_cutpoints
                ),
                "every_cutpoint_has_invariant": all(
                    not str(item["invariant"]).startswith("needs explicit invariant")
                    for item in fault_cutpoints
                ),
                "parent_checklist_closed": (
                    implementation_owner == "pqc_test_hooks.c"
                    and all(
                        "gate08_blocker_status" in item
                        for item in fault_cutpoints
                    )
                    and all(
                        not str(item["invariant"]).startswith("needs explicit invariant")
                        for item in fault_cutpoints
                    )
                    and not fault_blockers
                ),
            },
            "negative_claim_guard": (
                "Crash claims must name the retained fault model and cutpoint matrix. "
                "This inventory proves only fault-hook ownership and invariant mapping; "
                "it does not prove power-loss safety, full crash certification, storage "
                "cache loss safety, kernel-crash safety, or physical power-fault safety."
            ),
        },
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
