#!/usr/bin/env python3
"""Gate 0.1-S0 worktree freeze and pending patch triage."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "validation" / "refactor_inventory"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def run_git(args: list[str], timeout: float = 20.0) -> dict[str, Any]:
    argv = ["git", *args]
    proc = subprocess.run(
        argv,
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    return {
        "argv": argv,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def split_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line]


def parse_porcelain(lines: list[str]) -> dict[str, Any]:
    entries: list[dict[str, str]] = []
    tracked_modified: list[str] = []
    tracked_deleted: list[str] = []
    tracked_added: list[str] = []
    untracked: list[str] = []
    renamed: list[str] = []
    conflicted: list[str] = []

    for line in lines:
        if len(line) < 4:
            continue
        status = line[:2]
        path = line[3:]
        entry = {"status": status, "path": path}
        entries.append(entry)
        if status == "??":
            untracked.append(path)
        elif "U" in status:
            conflicted.append(path)
        elif "D" in status:
            tracked_deleted.append(path)
        elif "A" in status:
            tracked_added.append(path)
        elif "R" in status:
            renamed.append(path)
        elif "M" in status:
            tracked_modified.append(path)

    return {
        "entries": entries,
        "tracked_modified": tracked_modified,
        "tracked_deleted": tracked_deleted,
        "tracked_added": tracked_added,
        "renamed": renamed,
        "untracked": untracked,
        "conflicted": conflicted,
        "counts": {
            "entries": len(entries),
            "tracked_modified": len(tracked_modified),
            "tracked_deleted": len(tracked_deleted),
            "tracked_added": len(tracked_added),
            "renamed": len(renamed),
            "untracked": len(untracked),
            "conflicted": len(conflicted),
        },
    }


def categorize(paths: list[str]) -> dict[str, list[str]]:
    categories = {
        "root_code_deleted_or_modified": [],
        "code_directory": [],
        "experiment_relocation": [],
        "validation_artifacts": [],
        "paper_or_readme": [],
        "checklist_or_build": [],
        "docs": [],
        "other": [],
    }
    for path in paths:
        target = path
        if path.startswith("code/"):
            categories["code_directory"].append(path)
        elif path.startswith("experiments/") or path == "experiments":
            categories["experiment_relocation"].append(path)
        elif path.startswith("artifacts/validation/"):
            categories["validation_artifacts"].append(path)
        elif path.startswith("Paper/") or path == "README.md":
            categories["paper_or_readme"].append(path)
        elif path in {"CMakeLists.txt", "SUBMISSION_CHECKLIST.md", "CODE_REFACTOR_STRATEGY.md"}:
            categories["checklist_or_build"].append(path)
        elif path.startswith("docs/"):
            categories["docs"].append(path)
        elif (
            target.endswith((".c", ".h", ".cu", ".cpp", ".hpp", ".py", ".sh", ".bt")) and
            "/" not in target
        ):
            categories["root_code_deleted_or_modified"].append(path)
        else:
            categories["other"].append(path)
    return categories


def term_present(path: Path, term: str) -> bool:
    return term in read_text(path)


def find_paths_containing(root: Path, terms: list[str]) -> list[str]:
    matches: list[str] = []
    if not root.exists():
        return matches
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in {
            ".c",
            ".h",
            ".py",
            ".md",
            ".json",
            ".txt",
            ".jsonl",
            ".csv",
        }:
            continue
        text = read_text(path)
        if all(term in text for term in terms):
            matches.append(relpath(path))
    return matches


def dirty_sidecar_patch_status() -> dict[str, Any]:
    fd_context = ROOT / "code" / "fs" / "pqc_fd_context.c"
    file_io = ROOT / "code" / "fs" / "pqc_file_io.c"
    evidence_presence = {
        "fd_context_exists": fd_context.exists(),
        "file_io_exists": file_io.exists(),
        "prepare_snapshot_function": term_present(
            fd_context, "pqc_fd_context_prepare_dirty_sidecar_sync_locked"),
        "run_sync_function": term_present(
            fd_context, "pqc_fd_context_run_dirty_sidecar_sync"),
        "finish_epoch_check_function": term_present(
            fd_context, "pqc_fd_context_finish_dirty_sidecar_sync_locked"),
        "duplicates_data_fd": term_present(fd_context, "dup(ctx->data_fd)"),
        "duplicates_journal_fd": term_present(fd_context, "dup(ctx->journal_fd)"),
        "checks_data_dirty_epoch": term_present(
            fd_context, "ctx->data_sidecar_dirty_epoch == sync->data_epoch"),
        "checks_journal_dirty_epoch": term_present(
            fd_context, "ctx->journal_sidecar_dirty_epoch == sync->journal_epoch"),
        "syncs_without_fd_lock_helper": term_present(
            file_io, "sync_dirty_sidecars_without_fd_lock"),
        "unlocks_fd_lock_before_fdatasync": term_present(
            file_io, "pqc_profiled_mutex_unlock(&ctx->fd_lock") and
            term_present(file_io, "pqc_fd_context_run_dirty_sidecar_sync(&sync)"),
        "used_by_fsync": term_present(file_io, "int pqc_fsync") and
        term_present(file_io, "sync_dirty_sidecars_without_fd_lock(ctx"),
        "used_by_flush": term_present(file_io, "int pqc_flush") and
        term_present(file_io, "sync_dirty_sidecars_without_fd_lock(ctx"),
        "used_by_release": term_present(file_io, "int pqc_release") and
        term_present(file_io, "sync_dirty_sidecars_without_fd_lock(ctx"),
    }
    implementation_present = all(evidence_presence.values())
    benchmark_like_paths = find_paths_containing(
        ROOT / "artifacts",
        ["dirty-sidecar", "fdatasync"],
    )
    source_mentions = find_paths_containing(
        ROOT / "code",
        ["dirty-sidecar", "pending-unclaimed"],
    )
    has_before_after_benchmark = any(
        "before" in path.lower() and "after" in path.lower()
        for path in benchmark_like_paths
    )
    if implementation_present and has_before_after_benchmark:
        classification = "retained+benchmarked"
    elif not implementation_present:
        classification = "reverted"
    else:
        classification = "pending-unclaimed"

    return {
        "name": "dirty-sidecar sync dedup patch",
        "classification": classification,
        "implementation_present": implementation_present,
        "evidence_presence": evidence_presence,
        "benchmark_or_fault_evidence_candidates": benchmark_like_paths,
        "source_or_inventory_mentions": source_mentions,
        "reason": (
            "The patch is implemented in the current code path, but no retained "
            "before/after benchmark or fault matrix isolates this patch as a "
            "claim. It must remain pending-unclaimed until benchmarked or reverted."
            if classification == "pending-unclaimed" else
            "Classification derived mechanically from implementation and benchmark evidence."
        ),
        "paper_claim_status": "not_upgraded",
        "negative_claim_guard": (
            "Do not claim dirty-sidecar fdatasync reduction, lock-safety, or "
            "performance benefit from this patch until a row-specific benchmark "
            "and fault matrix are retained."
        ),
    }


def build_payload() -> dict[str, Any]:
    status = run_git(["status", "--porcelain=v1"])
    status_lines = split_lines(status["stdout"])
    porcelain = parse_porcelain(status_lines)
    diff_name_only = split_lines(run_git(["diff", "--name-only", "--"])["stdout"])
    untracked_files = split_lines(
        run_git(["ls-files", "--others", "--exclude-standard"])["stdout"]
    )
    head = run_git(["rev-parse", "HEAD"])
    branch = run_git(["branch", "--show-current"])
    all_paths = sorted(set(diff_name_only + untracked_files + [
        entry["path"] for entry in porcelain["entries"]
    ]))
    dirty_patch = dirty_sidecar_patch_status()
    strategy_file_absent = not (ROOT / "CODE_REFACTOR_STRATEGY.md").exists()

    return {
        "schema_version": 1,
        "generated_by": "code/experiments/build_worktree_freeze.py",
        "generated_utc": now_utc(),
        "scope": "Gate 0.1-S0 worktree freeze and pending patch triage.",
        "git": {
            "head": head["stdout"].strip(),
            "branch": branch["stdout"].strip(),
            "status_returncode": status["returncode"],
            "status_stderr": status["stderr"],
            "status_porcelain": status_lines,
            "diff_name_only": diff_name_only,
            "untracked_files": untracked_files,
            "untracked_files_count": len(untracked_files),
            "porcelain_summary": porcelain,
            "path_categories": categorize(all_paths),
        },
        "authoritative_checklist": {
            "path": "SUBMISSION_CHECKLIST.md",
            "code_refactor_strategy_absent": strategy_file_absent,
        },
        "patch_ownership": {
            "source_tree_relocation_to_code_dir": {
                "status": "pending-unclaimed",
                "evidence": [
                    "CMakeLists.txt now builds from code/",
                    "root source files are deleted in git diff",
                    "code/ is currently untracked in this worktree",
                ],
                "guard": "Do not treat the file relocation as behavior-equivalent until Gate 0.14 behavior-equivalence evidence is retained.",
            },
            "dirty_sidecar_sync_dedup_patch": dirty_patch,
            "new_gate_artifacts_and_runners": {
                "status": "pending-review",
                "evidence": [
                    "NVIDIA/Jetson contract runners and artifacts are untracked or modified",
                    "publication/concurrency artifacts are present under artifacts/validation/",
                ],
                "guard": "Do not close broad checklist gates from these artifacts until paper text and negative-claim guards are updated deliberately.",
            },
        },
        "paper_text_status": "not_updated",
        "parent_checklist_closed": False,
        "artifact_verdict": {
            "overall_pass": (
                status["returncode"] == 0 and
                strategy_file_absent and
                dirty_patch["classification"] in {
                    "retained+benchmarked",
                    "reverted",
                    "pending-unclaimed",
                }
            ),
            "required_fields_present": {
                "git_status": bool(status_lines) or status["returncode"] == 0,
                "git_diff_name_only": isinstance(diff_name_only, list),
                "untracked_files": isinstance(untracked_files, list),
                "code_refactor_strategy_absence": strategy_file_absent,
                "dirty_sidecar_patch_classification": dirty_patch["classification"] in {
                    "retained+benchmarked",
                    "reverted",
                    "pending-unclaimed",
                },
            },
        },
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    git = payload["git"]
    porcelain = git["porcelain_summary"]
    dirty_patch = payload["patch_ownership"]["dirty_sidecar_sync_dedup_patch"]
    categories = git["path_categories"]
    lines = [
        "# Worktree Freeze",
        "",
        f"- Generated: `{payload['generated_utc']}`",
        f"- Generated by: `{payload['generated_by']}`",
        f"- HEAD: `{git['head']}`",
        f"- Branch: `{git['branch']}`",
        f"- Overall artifact pass: `{payload['artifact_verdict']['overall_pass']}`",
        f"- `CODE_REFACTOR_STRATEGY.md` absent: `{payload['authoritative_checklist']['code_refactor_strategy_absent']}`",
        f"- Parent checklist closed: `{payload['parent_checklist_closed']}`",
        "",
        "## Worktree Counts",
        "",
        f"- Porcelain entries: `{porcelain['counts']['entries']}`",
        f"- Tracked modified: `{porcelain['counts']['tracked_modified']}`",
        f"- Tracked deleted: `{porcelain['counts']['tracked_deleted']}`",
        f"- Tracked added: `{porcelain['counts']['tracked_added']}`",
        f"- Untracked porcelain entries: `{porcelain['counts']['untracked']}`",
        f"- Untracked files expanded by `git ls-files`: `{git['untracked_files_count']}`",
        "",
        "## Path Categories",
        "",
    ]
    for name, paths in categories.items():
        lines.append(f"- {name}: `{len(paths)}`")
    lines.extend([
        "",
        "## Dirty-Sidecar Sync Dedup Patch",
        "",
        f"- Classification: `{dirty_patch['classification']}`",
        f"- Implementation present: `{dirty_patch['implementation_present']}`",
        f"- Paper claim status: `{dirty_patch['paper_claim_status']}`",
        f"- Reason: {dirty_patch['reason']}",
        f"- Guard: {dirty_patch['negative_claim_guard']}",
        "",
        "## Patch Ownership",
        "",
    ])
    for name, item in payload["patch_ownership"].items():
        lines.append(f"- `{name}`: `{item['status'] if 'status' in item else item['classification']}`")
    lines.extend([
        "",
        "## Close Boundary",
        "",
        "- This artifact freezes the current worktree only.",
        "- It does not prove behavior equivalence for the code relocation.",
        "- It does not benchmark the dirty-sidecar sync dedup patch.",
        "- It does not close any broad checklist checkbox.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    json_path = OUT / "worktree_freeze.json"
    md_path = OUT / "worktree_freeze.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(payload, md_path)
    print(json.dumps({
        "overall_pass": payload["artifact_verdict"]["overall_pass"],
        "head": payload["git"]["head"],
        "porcelain_entries": payload["git"]["porcelain_summary"]["counts"]["entries"],
        "untracked_files_count": payload["git"]["untracked_files_count"],
        "dirty_sidecar_classification": payload["patch_ownership"]["dirty_sidecar_sync_dedup_patch"]["classification"],
        "code_refactor_strategy_absent": payload["authoritative_checklist"]["code_refactor_strategy_absent"],
        "json": relpath(json_path),
        "markdown": relpath(md_path),
    }, indent=2, sort_keys=True))
    return 0 if payload["artifact_verdict"]["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
