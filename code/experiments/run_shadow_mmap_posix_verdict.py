#!/usr/bin/env python3
"""Verify the shared-mmap/shadow-paging decision boundary for Gate C2.

The production decision today is rejection, not shadow paging.  This runner
mounts the final binary and proves that encrypted files reject shared
``mmap``/``msync`` and that no SQLite WAL/SHM mmap redirect remains in the
runtime.  It also guards paper-facing text against accidentally claiming a
shadow-mmap or general POSIX implementation.
"""

from __future__ import annotations

import errno
import json
import mmap
import os
import re
import shutil
import signal
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FUSE_BIN = ROOT / "build" / "pqc_fuse"
OUT_DIR = ROOT / "artifacts" / "validation" / "shadow_mmap_posix"
POSIX_AUDIT = ROOT / "artifacts" / "validation" / "posix_scope_audit" / "posix_scope_audit.json"
PAPER_DIR = ROOT / "Paper"
README = ROOT / "README.md"

ACCEPTABLE_MMAP_ERRNOS = {
    errno.ENODEV,
    errno.EINVAL,
    errno.EOPNOTSUPP,
    getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
}


def errno_name(value: int | None) -> str | None:
    if value is None:
        return None
    return errno.errorcode.get(value, f"ERRNO_{value}")


def start_fuse(storage_dir: Path, mount_dir: Path,
               out_dir: Path) -> tuple[subprocess.Popen[bytes], Any, Any]:
    env = os.environ.copy()
    env["PQC_MASTER_PASSWORD"] = "shadow-mmap-boundary-password"
    env["PQC_FRESHNESS_ANCHOR_BACKEND"] = "file"
    env["PQC_FRESHNESS_ANCHOR_PATH"] = str(storage_dir / ".anchor")
    env.pop("PQC_ALLOW_SQLITE_MMAP", None)

    log_dir = out_dir / "mount_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    label = "default"
    stdout = (log_dir / f"{label}.stdout.txt").open("wb")
    stderr = (log_dir / f"{label}.stderr.txt").open("wb")
    proc = subprocess.Popen(
        [str(FUSE_BIN), str(storage_dir), str(mount_dir), "-f"],
        cwd=ROOT,
        env=env,
        stdout=stdout,
        stderr=stderr,
    )
    try:
        deadline = datetime.now().timestamp() + 15.0
        while datetime.now().timestamp() < deadline:
            if subprocess.run(
                ["mountpoint", "-q", str(mount_dir)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode == 0:
                return proc, stdout, stderr
            if proc.poll() is not None:
                raise RuntimeError(f"FUSE exited before mount: rc={proc.returncode}")
            subprocess.run(["sleep", "0.05"], check=False)
    except BaseException:
        stdout.close()
        stderr.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=3)
        raise
    stdout.close()
    stderr.close()
    if proc.poll() is None:
        proc.kill()
        proc.wait(timeout=3)
    raise TimeoutError("timed out waiting for FUSE mount")


def stop_fuse(proc: subprocess.Popen[bytes] | None,
              stdout: Any | None,
              stderr: Any | None,
              mount_dir: Path) -> None:
    subprocess.run(
        ["fusermount3", "-u", str(mount_dir)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if proc is not None and proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)
    if stdout is not None:
        stdout.close()
    if stderr is not None:
        stderr.close()


def write_fsync(path: Path, payload: bytes) -> None:
    with path.open("wb") as fp:
        fp.write(payload)
        fp.flush()
        os.fsync(fp.fileno())


def attempt_shared_mmap(path: Path) -> dict[str, Any]:
    fd = -1
    try:
        fd = os.open(path, os.O_RDWR)
        mapping = mmap.mmap(fd, 4096, access=mmap.ACCESS_WRITE)
        try:
            mapping[0] = (mapping[0] + 1) % 255
            mapping.flush()
        finally:
            mapping.close()
        return {
            "accepted": False,
            "unexpected_success": True,
            "observed_errno": None,
            "observed_errno_name": None,
            "detail": "shared mmap and msync unexpectedly succeeded",
        }
    except OSError as exc:
        return {
            "accepted": exc.errno in ACCEPTABLE_MMAP_ERRNOS,
            "unexpected_success": False,
            "observed_errno": exc.errno,
            "observed_errno_name": errno_name(exc.errno),
            "detail": str(exc),
        }
    except ValueError as exc:
        return {
            "accepted": True,
            "unexpected_success": False,
            "observed_errno": None,
            "observed_errno_name": None,
            "detail": f"ValueError: {exc}",
        }
    finally:
        if fd >= 0:
            os.close(fd)


def run_mount_probe(label: str) -> dict[str, Any]:
    storage_dir = Path(tempfile.mkdtemp(prefix=f"shadow_mmap_{label}_store_"))
    mount_dir = Path(tempfile.mkdtemp(prefix=f"shadow_mmap_{label}_mnt_"))
    proc = None
    stdout = None
    stderr = None
    try:
        proc, stdout, stderr = start_fuse(storage_dir, mount_dir, OUT_DIR)
        target = mount_dir / "encrypted.bin"
        write_fsync(target, b"M" * 8192)
        mmap_result = attempt_shared_mmap(target)
        size_after = target.stat().st_size
        retained_payload = target.read_bytes()[:16] == b"M" * 16
        accepted = bool(mmap_result["accepted"]) and size_after == 8192 and retained_payload
        return {
            "case": label,
            "sqlite_mmap_redirect": False,
            "mmap_msync_rejection": mmap_result,
            "size_after": size_after,
            "retained_payload_prefix": retained_payload,
            "acceptable": accepted,
            "scope": "Encrypted data files reject shared mmap/msync; the runtime no longer provides a SQLite WAL/SHM mmap redirect.",
        }
    except BaseException as exc:  # noqa: BLE001 - retained in verdict.
        return {
            "case": label,
            "sqlite_mmap_redirect": False,
            "acceptable": False,
            "detail": repr(exc),
        }
    finally:
        stop_fuse(proc, stdout, stderr, mount_dir)
        shutil.rmtree(storage_dir, ignore_errors=True)
        shutil.rmtree(mount_dir, ignore_errors=True)


def source_checks() -> dict[str, Any]:
    lifecycle = (ROOT / "code" / "runtime" / "pqc_lifecycle.c").read_text(encoding="utf-8")
    file_io = (ROOT / "code" / "fs" / "pqc_file_io.c").read_text(encoding="utf-8")
    checks = {
        "fuse_direct_io_mmap_capability_disabled": "conn->want &= ~FUSE_DIRECT_IO_ALLOW_MMAP" in lifecycle,
        "writeback_cache_disabled": "conn->want &= ~FUSE_CAP_WRITEBACK_CACHE" in lifecycle,
        "sqlite_mmap_redirect_removed": "PQC_ALLOW_SQLITE_MMAP" not in file_io and "sqlite_mmap_sidecar" not in file_io,
        "old_global_allow_mmap_pattern_absent": "allow_mmap ? 0 : 1" not in file_io,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
    }


def posix_audit_checks() -> dict[str, Any]:
    payload = json.loads(POSIX_AUDIT.read_text(encoding="utf-8"))
    shared = payload.get("semantic_status", {}).get("shared_mmap", {})
    msync = payload.get("semantic_status", {}).get("msync", {})
    checks = {
        "posix_audit_pass": payload.get("overall_pass") is True,
        "shared_mmap_formally_rejected": shared.get("status") == "formal rejection" and shared.get("acceptable") is True,
        "msync_formally_rejected": msync.get("status") == "formal rejection" and msync.get("acceptable") is True,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "source_artifact": str(POSIX_AUDIT.relative_to(ROOT)),
    }


def claim_guard() -> dict[str, Any]:
    scanned: list[dict[str, Any]] = []
    for path in sorted(PAPER_DIR.glob("*.tex")) + [README]:
        if not path.exists():
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if re.search(r"mmap|msync|shadow mmap|shadow paging|out-of-place|out of place|general-purpose POSIX|general-purpose filesystem|general purpose POSIX|general purpose filesystem", line, re.IGNORECASE):
                scanned.append({
                    "path": str(path.relative_to(ROOT)),
                    "line": line_no,
                    "text": line,
                })

    dangerous_regex = re.compile(
        r"shadow mmap|shadow paging|out-of-place update|out of place update|atomic pointer swap|general-purpose POSIX|general purpose POSIX|general-purpose filesystem|general purpose filesystem",
        re.IGNORECASE,
    )
    negation_regex = re.compile(
        r"not|no |without|unsupported|unvalidated|reject|rejected|fails|constraint|제약|피하면서|future|unless|outside",
        re.IGNORECASE,
    )
    dangerous = [
        item for item in scanned
        if dangerous_regex.search(item["text"]) and not negation_regex.search(item["text"])
    ]

    mmap_positive_regex = re.compile(r"validated shared .*mmap|shared .*mmap .*supported|mmap .*supported", re.IGNORECASE)
    positive_mmap_claims = [
        item for item in scanned
        if mmap_positive_regex.search(item["text"]) and not negation_regex.search(item["text"])
    ]
    paper_rejection_lines = [
        item for item in scanned
        if "Paper/" in item["path"]
        and re.search(r"mmap|msync", item["text"], re.IGNORECASE)
        and negation_regex.search(item["text"])
    ]
    checks = {
        "no_unbacked_shadow_mmap_terms": not dangerous,
        "no_positive_shared_mmap_claim": not positive_mmap_claims,
        "paper_explains_rejection_or_nonclaim": bool(paper_rejection_lines),
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "scanned_lines": scanned,
        "dangerous_lines": dangerous,
        "positive_mmap_claims": positive_mmap_claims,
        "paper_rejection_lines": paper_rejection_lines,
    }


def build_report() -> dict[str, Any]:
    probes = [
        run_mount_probe("default_env_encrypted_file"),
    ]
    source = source_checks()
    posix = posix_audit_checks()
    guard = claim_guard()
    shadow_followups = [
        {
            "case": "concurrent_dirty_page",
            "status": "not_applicable_mapping_rejected",
            "acceptable": True,
        },
        {
            "case": "mmap_truncate_interaction",
            "status": "not_applicable_mapping_rejected",
            "acceptable": True,
        },
        {
            "case": "torn_write_after_msync",
            "status": "not_applicable_mapping_rejected",
            "acceptable": True,
        },
        {
            "case": "remount_after_mmap_dirtying",
            "status": "not_applicable_mapping_rejected",
            "acceptable": True,
        },
        {
            "case": "rollback_after_mmap_dirtying",
            "status": "not_applicable_mapping_rejected",
            "acceptable": True,
        },
    ]
    checks = {
        "mounted_probes_pass": all(probe.get("acceptable") is True for probe in probes),
        "source_boundary_pass": source["pass"],
        "posix_audit_boundary_pass": posix["pass"],
        "claim_guard_pass": guard["pass"],
        "followup_cases_closed_by_rejection": all(row["acceptable"] for row in shadow_followups),
    }
    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "gate": "C2",
        "submilestone": "C2-S0",
        "decision": "formal_rejection_until_shadow_paging_is_implemented",
        "probes": probes,
        "shadow_followup_cases": shadow_followups,
        "source": source,
        "posix_audit": posix,
        "claim_guard": guard,
        "checks": checks,
        "overall_pass": all(checks.values()),
        "negative_claim_guard": {
            "shared_mmap_support_claim_allowed": False,
            "shadow_mmap_claim_allowed": False,
            "out_of_place_update_claim_allowed": False,
            "general_posix_claim_allowed": False,
        },
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Shared mmap / shadow-paging verdict",
        "",
        f"- Gate: `{report['gate']}`",
        f"- Submilestone: `{report['submilestone']}`",
        f"- Overall pass: `{str(report['overall_pass']).lower()}`",
        f"- Decision: `{report['decision']}`",
        "",
        "## Mounted probes",
        "",
    ]
    for probe in report["probes"]:
        rejection = probe.get("mmap_msync_rejection", {})
        lines.append(
            f"- `{probe['case']}`: acceptable=`{probe.get('acceptable')}`, "
            f"errno=`{rejection.get('observed_errno_name')}`, detail=`{rejection.get('detail')}`"
        )
    lines.extend(["", "## Shadow follow-up cases", ""])
    for row in report["shadow_followup_cases"]:
        lines.append(f"- `{row['case']}`: `{row['status']}`")
    lines.extend(["", "## Checks", ""])
    for key, value in report["checks"].items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    if not FUSE_BIN.exists():
        raise FileNotFoundError(f"missing final binary: {FUSE_BIN}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    json_path = OUT_DIR / "shadow_mmap_decision.json"
    md_path = OUT_DIR / "shadow_mmap_decision.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(json_path.relative_to(ROOT)),
                "markdown": str(md_path.relative_to(ROOT)),
                "overall_pass": report["overall_pass"],
                "decision": report["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
