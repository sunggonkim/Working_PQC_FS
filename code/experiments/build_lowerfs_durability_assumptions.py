#!/usr/bin/env python3
"""Build the R6 lower-filesystem durability assumption matrix."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POSIX = ROOT / "artifacts" / "validation" / "posix_scope_audit" / "posix_scope_audit.json"
DEFAULT_DAEMON = ROOT / "artifacts" / "validation" / "daemon_power_fault_campaign" / "daemon_power_fault_campaign.json"
DEFAULT_CRASH = ROOT / "artifacts" / "reports" / "crash_audit_report" / "crash_audit_report.json"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "lowerfs_durability_assumptions"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_row(rows: list[dict[str, Any]], case: str) -> dict[str, Any]:
    for row in rows:
        if row.get("case") == case:
            return row
    return {}


def paper_contains_terms() -> dict[str, bool]:
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (ROOT / "Paper").glob("*.tex")
    ).lower()
    return {
        "xattr": "xattr" in text,
        "directory": "directory" in text,
        "ordering": "ordering" in text,
        "fdatasync": "fdatasync" in text,
        "lower_filesystem": "lower-filesystem" in text or "lower filesystem" in text,
        "not_certified": "not certified" in text or "not a" in text,
    }


def daemon_cutpoint(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    wanted = name.replace("/", " ")
    for row in rows:
        if row.get("cutpoint") == name or row.get("name") == name:
            return row
        if str(row.get("case") or "").replace("_", " ") == wanted:
            return row
    return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--posix", type=Path, default=DEFAULT_POSIX)
    parser.add_argument("--daemon", type=Path, default=DEFAULT_DAEMON)
    parser.add_argument("--crash", type=Path, default=DEFAULT_CRASH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    posix = load_json(args.posix)
    daemon = load_json(args.daemon)
    crash = load_json(args.crash)
    posix_rows = posix.get("rows") or []
    lower_row = find_row(posix_rows, "lower_filesystem_assumptions")
    rename_row = find_row(posix_rows, "closed_file_rename_supported_subset")
    overwrite_rename_row = find_row(
        posix_rows, "closed_file_overwrite_rename_supported_subset")
    open_source_rename_row = find_row(
        posix_rows, "open_source_rename_supported_subset")
    open_target_rename_row = find_row(
        posix_rows, "open_target_rename_supported_subset")
    empty_dir_rename_row = find_row(
        posix_rows, "empty_directory_rename_supported_subset")
    tree_rename_row = find_row(posix_rows, "directory_tree_rename_supported_subset")
    dir_row = find_row(posix_rows, "directory_fsync_supported_subset")
    hard_link_row = find_row(posix_rows, "hard_link_supported_subset")
    symlink_row = find_row(posix_rows, "symlink_supported_subset")
    xattr_row = find_row(posix_rows, "xattr_policy")
    daemon_rows = daemon.get("daemon_rows") or []
    coverage = daemon.get("coverage") or {}

    matrix = [
        {
            "assumption": "data sidecar reaches lower storage before mapping publication",
            "status": "fault-tested",
            "evidence": "daemon cutpoints `data_write`, `journal_append`, and `journal_fsync` all return acceptable committed-prefix verdicts",
            "claim_boundary": "requires lower `fdatasync` to order file data before the later journal mapping",
            "pass": all(
                bool(daemon_cutpoint(daemon_rows, name).get("acceptable"))
                for name in ("data write", "journal append", "journal fsync")
            ),
        },
        {
            "assumption": "checkpoint/logical-size xattr is exposed only after mapping publication",
            "status": "fault-tested-plus-scope-probed",
            "evidence": "`xattr_checkpoint_update` and `checkpoint_write` cutpoints are acceptable; POSIX audit records lower xattr availability and hidden internal xattrs",
            "claim_boundary": "xattr atomicity is delegated to the lower filesystem; physical power-loss certification is not claimed",
            "pass": (
                bool(daemon_cutpoint(daemon_rows, "xattr/checkpoint update").get("acceptable"))
                and bool(daemon_cutpoint(daemon_rows, "checkpoint write").get("acceptable"))
                and bool(lower_row.get("acceptable"))
                and bool(xattr_row.get("acceptable"))
            ),
        },
        {
            "assumption": "new-target/overwrite rename, directory-tree rename, directory fsync, hard-link, and symlink namespace support are bounded subsets",
            "status": "supported-subset",
            "evidence": "POSIX audit renames a closed regular file to a new target, supports closed-target overwrite rename, supports open-source rename to a new target, supports open-target overwrite rename with stale target fd isolation, supports empty-directory and non-empty directory-tree rename to a new target, fsyncs a lower directory, passes no-open regular-file hard-link read/write-through, and passes relative symlink readlink/read-through",
            "claim_boundary": "no open hard-link creation, open-subtree retargeting, full link-count lifecycle certification, or crash-atomic multi-file rename claim",
            "pass": (
                bool(rename_row.get("acceptable"))
                and bool(overwrite_rename_row.get("acceptable"))
                and bool(open_source_rename_row.get("acceptable"))
                and bool(open_target_rename_row.get("acceptable"))
                and bool(empty_dir_rename_row.get("acceptable"))
                and bool(tree_rename_row.get("acceptable"))
                and bool(dir_row.get("acceptable"))
                and bool(hard_link_row.get("acceptable"))
                and bool(symlink_row.get("acceptable"))
            ),
        },
        {
            "assumption": "recovery ordering is limited to retained D/J/C cutpoints",
            "status": "selected-boundary-fault-tested",
            "evidence": "daemon campaign covers all required daemon cutpoints and crash audit retains selected app-level evidence",
            "claim_boundary": "kernel crash, drive-cache flush loss, and physical power loss remain outside scope",
            "pass": (
                bool(daemon.get("overall_pass"))
                and not coverage.get("missing_daemon_cutpoints")
                and bool(crash.get("retained_evidence"))
            ),
        },
    ]
    terms = paper_contains_terms()
    overall_pass = (
        bool(posix.get("overall_pass"))
        and bool(posix.get("required_semantics_all_covered"))
        and bool(daemon.get("overall_pass"))
        and all(row["pass"] for row in matrix)
        and all(terms[key] for key in ("xattr", "directory", "ordering", "fdatasync"))
    )
    report = {
        "artifact": "lowerfs_durability_assumptions",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": overall_pass,
        "source_artifacts": {
            "posix_scope_audit": rel(args.posix),
            "daemon_power_fault_campaign": rel(args.daemon),
            "crash_audit_report": rel(args.crash),
        },
        "assumption_matrix": matrix,
        "paper_terms": terms,
        "lower_filesystem_probe": {
            "filesystem_type": lower_row.get("filesystem_type"),
            "marker_metadata_xattr_bytes": lower_row.get("marker_metadata_xattr_bytes"),
            "marker_checkpoint_xattr_bytes": lower_row.get("marker_checkpoint_xattr_bytes"),
            "storage_dir_fsync_ok": lower_row.get("storage_dir_fsync_ok"),
            "scope": lower_row.get("scope"),
        },
        "verdict": (
            "The retained evidence supports a scoped D/J/C ordering argument: "
            "data and journal cutpoints recover to an acceptable committed "
            "prefix, checkpoint/xattr cutpoints are covered, and closed-file "
            "new-target rename, closed-target overwrite rename, open-source rename to a new target, empty-"
            "directory and non-empty directory-tree rename to a new target, "
            "directory fsync, no-open regular-file hard-link read/write-through, "
            "open-target stale-fd isolation, and relative symlink read-through "
            "have bounded supported subsets. "
            "It does not certify lower-filesystem xattr atomicity, directory "
            "durability, drive-cache behavior, kernel crash, or physical "
            "power-loss semantics."
        ),
        "claim_guard": (
            "Paper text may use this artifact only as an assumption matrix for "
            "the implemented mounted prototype. It must not claim full POSIX "
            "crash consistency, crash-atomic rename, open hard-link creation, "
            "full link-count lifecycle certification, open-subtree retargeting, "
            "or power-loss certification."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "lowerfs_durability_assumptions.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    print(json.dumps({
        "overall_pass": overall_pass,
        "json": rel(json_path),
        "matrix_rows": len(matrix),
        "paper_terms": terms,
    }, indent=2, sort_keys=True))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
