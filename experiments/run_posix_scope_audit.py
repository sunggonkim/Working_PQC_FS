#!/usr/bin/env python3
"""Audit AEGIS-Q's intentionally narrow POSIX support boundary.

This harness records behavior for the POSIX modes that reviewers repeatedly
ask about: mmap, concurrent writers, rename/directory fsync, xattrs,
lower-filesystem assumptions, and FUSE writeback-cache interaction.  It is an
evidence artifact, not a broad POSIX certification campaign.
"""

from __future__ import annotations

import argparse
import errno
import json
import mmap
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
FUSE_BIN = ROOT / "build" / "pqc_fuse"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "posix_scope_audit"

INTERNAL_XATTRS = [
    "user.pqc_metadata",
    "user.pqc_logical_size",
    "user.pqc_checkpoint",
]


@dataclass
class FuseProc:
    proc: subprocess.Popen[bytes]
    stdout: Any
    stderr: Any


def errno_name(value: int | None) -> str | None:
    if value is None:
        return None
    return errno.errorcode.get(value, f"ERRNO_{value}")


def start_fuse(storage_dir: Path, mount_dir: Path, out_dir: Path, password: str) -> FuseProc:
    env = os.environ.copy()
    env["PQC_MASTER_PASSWORD"] = password
    env["PQC_FRESHNESS_ANCHOR_BACKEND"] = "file"
    env["PQC_FRESHNESS_ANCHOR_PATH"] = str(storage_dir / ".anchor")
    log_dir = out_dir / "mount_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout = (log_dir / "pqc_fuse.stdout.txt").open("wb")
    stderr = (log_dir / "pqc_fuse.stderr.txt").open("wb")
    proc = subprocess.Popen(
        [str(FUSE_BIN), str(storage_dir), str(mount_dir), "-f"],
        cwd=ROOT,
        env=env,
        stdout=stdout,
        stderr=stderr,
    )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if subprocess.run(["mountpoint", "-q", str(mount_dir)], check=False).returncode == 0:
            return FuseProc(proc=proc, stdout=stdout, stderr=stderr)
        if proc.poll() is not None:
            stdout.close()
            stderr.close()
            raise RuntimeError(f"FUSE exited before mount: rc={proc.returncode}")
        time.sleep(0.05)
    stdout.close()
    stderr.close()
    raise TimeoutError("timed out waiting for FUSE mount")


def stop_fuse(handle: FuseProc | None, mount_dir: Path) -> None:
    subprocess.run(
        ["fusermount3", "-u", str(mount_dir)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if handle is not None:
        if handle.proc.poll() is None:
            handle.proc.send_signal(signal.SIGINT)
            try:
                handle.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                handle.proc.kill()
                handle.proc.wait(timeout=3)
        handle.stdout.close()
        handle.stderr.close()


def write_fsync(path: Path, payload: bytes) -> None:
    with path.open("wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())


def expect_oserror(fn, accepted: set[int]) -> tuple[bool, int | None, str | None]:
    try:
        fn()
    except OSError as exc:
        return exc.errno in accepted, exc.errno, str(exc)
    except ValueError as exc:
        # Python's mmap wrapper can surface direct-I/O mmap rejection as
        # ValueError on some kernels.  Retain it explicitly rather than
        # pretending it is an errno.
        return True, None, f"ValueError: {exc}"
    return False, None, "operation unexpectedly succeeded"


def read_proc_mounts(mount_dir: Path) -> str | None:
    target = str(mount_dir)
    for line in Path("/proc/mounts").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[1] == target:
            return line
    return None


def audit_mmap(file_path: Path) -> dict[str, Any]:
    def attempt() -> None:
        fd = os.open(file_path, os.O_RDWR)
        try:
            mm = mmap.mmap(fd, 4096, access=mmap.ACCESS_WRITE)
            try:
                mm[0] = mm[0]
            finally:
                mm.close()
        finally:
            os.close(fd)

    ok, err, detail = expect_oserror(attempt, {errno.ENODEV, errno.EINVAL, errno.EOPNOTSUPP})
    return {
        "case": "default_mmap_on_encrypted_file",
        "expected_behavior": "rejected",
        "observed_errno": err,
        "observed_errno_name": errno_name(err),
        "detail": detail,
        "acceptable": ok,
        "scope": "Default encrypted files use FUSE direct_io and do not claim shared mmap semantics.",
    }


def audit_concurrent_disjoint_writers(file_path: Path) -> dict[str, Any]:
    payload_a = b"A" * 4096
    payload_b = b"B" * 4096
    write_fsync(file_path, b"\x00" * 8192)
    errors: list[str] = []

    def writer(offset: int, payload: bytes) -> None:
        try:
            fd = os.open(file_path, os.O_RDWR)
            try:
                os.pwrite(fd, payload, offset)
                os.fsync(fd)
            finally:
                os.close(fd)
        except BaseException as exc:  # retained below
            errors.append(repr(exc))

    t1 = threading.Thread(target=writer, args=(0, payload_a))
    t2 = threading.Thread(target=writer, args=(4096, payload_b))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    data = file_path.read_bytes()
    acceptable = not errors and data[:4096] == payload_a and data[4096:8192] == payload_b
    return {
        "case": "concurrent_disjoint_writers",
        "expected_behavior": "serialized_by_daemon_lock_for_disjoint_write_fsync_path",
        "errors": errors,
        "result_sha256_prefix": {
            "first_block_all_A": data[:4096] == payload_a,
            "second_block_all_B": data[4096:8192] == payload_b,
        },
        "acceptable": acceptable,
        "scope": "Narrow disjoint write/fsync path only; arbitrary conflicting app-level concurrency is not certified.",
    }


def audit_rename_and_dir_fsync(mount_dir: Path, file_path: Path) -> list[dict[str, Any]]:
    enotsup = {errno.EOPNOTSUPP, getattr(errno, "ENOTSUP", errno.EOPNOTSUPP)}

    def attempt_rename() -> None:
        os.rename(file_path, mount_dir / "renamed.bin")

    ok_rename, rename_errno, rename_detail = expect_oserror(attempt_rename, enotsup)

    subdir = mount_dir / "dirsync"
    subdir.mkdir(exist_ok=True)

    def attempt_fsyncdir() -> None:
        fd = os.open(subdir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    ok_fsyncdir, fsyncdir_errno, fsyncdir_detail = expect_oserror(attempt_fsyncdir, enotsup)
    return [
        {
            "case": "rename_rejection",
            "expected_behavior": "rejected_ENOTSUP",
            "observed_errno": rename_errno,
            "observed_errno_name": errno_name(rename_errno),
            "detail": rename_detail,
            "acceptable": ok_rename,
            "scope": "Rename would need atomic marker/data/journal/checkpoint/anchor transition; not implemented.",
        },
        {
            "case": "directory_fsync_rejection",
            "expected_behavior": "rejected_ENOTSUP",
            "observed_errno": fsyncdir_errno,
            "observed_errno_name": errno_name(fsyncdir_errno),
            "detail": fsyncdir_detail,
            "acceptable": ok_fsyncdir,
            "scope": "Directory-entry durability is not a supported boundary in this prototype.",
        },
    ]


def audit_xattrs(file_path: Path) -> dict[str, Any]:
    os.setxattr(file_path, "user.audit", b"ok")
    user_value = os.getxattr(file_path, "user.audit")
    listed = os.listxattr(file_path)

    internal_set_results = []
    internal_get_results = []
    for name in INTERNAL_XATTRS:
        def set_internal(n=name) -> None:
            os.setxattr(file_path, n, b"bad")

        ok_set, set_errno, set_detail = expect_oserror(set_internal, {errno.EPERM, errno.EACCES})
        internal_set_results.append({
            "name": name,
            "acceptable": ok_set,
            "observed_errno": set_errno,
            "observed_errno_name": errno_name(set_errno),
            "detail": set_detail,
        })

        def get_internal(n=name) -> None:
            os.getxattr(file_path, n)

        ok_get, get_errno, get_detail = expect_oserror(get_internal, {errno.ENODATA, errno.ENOATTR if hasattr(errno, "ENOATTR") else errno.ENODATA})
        internal_get_results.append({
            "name": name,
            "acceptable": ok_get,
            "observed_errno": get_errno,
            "observed_errno_name": errno_name(get_errno),
            "detail": get_detail,
        })

    def set_invalid_tier() -> None:
        os.setxattr(file_path, "user.pqc_tier", b"3")

    ok_invalid_tier, invalid_tier_errno, invalid_tier_detail = expect_oserror(set_invalid_tier, {errno.EINVAL})
    os.setxattr(file_path, "user.pqc_tier", b"1")
    tier_value = os.getxattr(file_path, "user.pqc_tier")
    listed_after = os.listxattr(file_path)

    hidden_internal = [name for name in INTERNAL_XATTRS if name in listed_after]
    acceptable = (
        user_value == b"ok"
        and tier_value == b"1"
        and ok_invalid_tier
        and not hidden_internal
        and all(row["acceptable"] for row in internal_set_results)
        and all(row["acceptable"] for row in internal_get_results)
    )
    return {
        "case": "xattr_policy",
        "expected_behavior": "user_xattr_roundtrip_internal_xattrs_hidden_invalid_tier_rejected",
        "user_audit_value": user_value.decode("utf-8", errors="replace"),
        "listxattr_before_tier": listed,
        "listxattr_after_tier": listed_after,
        "hidden_internal_xattrs_found": hidden_internal,
        "internal_set_results": internal_set_results,
        "internal_get_results": internal_get_results,
        "invalid_tier": {
            "acceptable": ok_invalid_tier,
            "observed_errno": invalid_tier_errno,
            "observed_errno_name": errno_name(invalid_tier_errno),
            "detail": invalid_tier_detail,
        },
        "tier_value": tier_value.decode("utf-8", errors="replace"),
        "acceptable": acceptable,
        "scope": "User xattrs and validated tier control are supported; internal metadata/checkpoint xattrs are not user-visible controls.",
    }


def audit_lower_fs(storage_dir: Path, marker_path: Path) -> dict[str, Any]:
    stat = subprocess.run(
        ["stat", "-f", "-c", "%T", str(storage_dir)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    marker_metadata_len = None
    marker_checkpoint_len = None
    marker_xattr_error = None
    try:
        marker_metadata_len = len(os.getxattr(marker_path, "user.pqc_metadata"))
    except OSError as exc:
        marker_xattr_error = f"metadata errno={exc.errno}"
    try:
        marker_checkpoint_len = len(os.getxattr(marker_path, "user.pqc_checkpoint"))
    except OSError as exc:
        marker_xattr_error = f"{marker_xattr_error or ''} checkpoint errno={exc.errno}".strip()

    dir_fsync_ok = None
    dir_fsync_errno = None
    try:
        fd = os.open(storage_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
            dir_fsync_ok = True
        finally:
            os.close(fd)
    except OSError as exc:
        dir_fsync_ok = False
        dir_fsync_errno = exc.errno

    acceptable = (
        stat.returncode == 0
        and marker_metadata_len is not None
        and marker_checkpoint_len is not None
    )
    return {
        "case": "lower_filesystem_assumptions",
        "expected_behavior": "recorded_not_certified",
        "filesystem_type": stat.stdout.strip() if stat.returncode == 0 else None,
        "stat_stderr": stat.stderr.strip(),
        "marker_metadata_xattr_bytes": marker_metadata_len,
        "marker_checkpoint_xattr_bytes": marker_checkpoint_len,
        "marker_xattr_error": marker_xattr_error,
        "storage_dir_fsync_ok": dir_fsync_ok,
        "storage_dir_fsync_errno": dir_fsync_errno,
        "storage_dir_fsync_errno_name": errno_name(dir_fsync_errno),
        "acceptable": acceptable,
        "scope": "The harness records lower-filesystem type and xattr availability; xattr atomicity and directory durability remain delegated, not certified.",
    }


def audit_writeback_cache(mount_dir: Path) -> dict[str, Any]:
    source = (ROOT / "pqc_fuse.c").read_text(encoding="utf-8")
    disables_writeback = "conn->want &= ~FUSE_CAP_WRITEBACK_CACHE" in source
    requests_writeback = "conn->want |= FUSE_CAP_WRITEBACK_CACHE" in source
    disables_direct_mmap = "conn->want &= ~FUSE_DIRECT_IO_ALLOW_MMAP" in source
    requests_direct_mmap = "conn->want |= FUSE_DIRECT_IO_ALLOW_MMAP" in source
    mount_line = read_proc_mounts(mount_dir)
    return {
        "case": "fuse_writeback_cache_and_direct_mmap_caps",
        "expected_behavior": "capabilities_not_requested",
        "source_disables_writeback_cache": disables_writeback,
        "source_requests_writeback_cache": requests_writeback,
        "source_disables_direct_io_mmap": disables_direct_mmap,
        "source_requests_direct_io_mmap": requests_direct_mmap,
        "proc_mounts_line": mount_line,
        "acceptable": (
            disables_writeback
            and not requests_writeback
            and disables_direct_mmap
            and not requests_direct_mmap
        ),
        "scope": "The daemon avoids FUSE writeback-cache and direct-IO mmap capabilities; no cache-coherence claim is made.",
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# POSIX scope audit",
        "",
        "Scope: final-binary audit for intentionally narrow POSIX semantics.",
        "",
        f"- Overall pass: `{result['overall_pass']}`",
        f"- Command: `{' '.join(result['command'])}`",
        "",
        "| Case | Expected behavior | Acceptable | Scope |",
        "|---|---|---:|---|",
    ]
    for row in result["rows"]:
        lines.append(
            f"| `{row['case']}` | `{row.get('expected_behavior')}` | `{row['acceptable']}` | {row.get('scope', '')} |"
        )
    lines.append("")
    lines.append("The JSON file retains errno values, mount logs, source assertions, and lower-filesystem metadata.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--password", default=os.environ.get("PQC_MASTER_PASSWORD", "posix-scope-password"))
    args = parser.parse_args()

    if not FUSE_BIN.exists():
        raise FileNotFoundError(f"missing final binary: {FUSE_BIN}")

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    storage_dir = Path(tempfile.mkdtemp(prefix="posix_scope_store_"))
    mount_dir = Path(tempfile.mkdtemp(prefix="posix_scope_mnt_"))
    handle: FuseProc | None = None
    rows: list[dict[str, Any]] = []
    try:
        handle = start_fuse(storage_dir, mount_dir, out_dir, args.password)
        file_path = mount_dir / "posix.bin"
        write_fsync(file_path, b"Z" * 8192)

        rows.append(audit_mmap(file_path))
        rows.append(audit_concurrent_disjoint_writers(file_path))
        rows.extend(audit_rename_and_dir_fsync(mount_dir, file_path))
        rows.append(audit_xattrs(file_path))
        rows.append(audit_lower_fs(storage_dir, storage_dir / "posix.bin"))
        rows.append(audit_writeback_cache(mount_dir))
    except Exception as exc:
        rows.append({
            "case": "harness_liveness",
            "expected_behavior": "audit_completes",
            "detail": repr(exc),
            "acceptable": False,
            "scope": "Harness failed before completing all rows.",
        })
    finally:
        stop_fuse(handle, mount_dir)
        shutil.rmtree(storage_dir, ignore_errors=True)
        shutil.rmtree(mount_dir, ignore_errors=True)

    result = {
        "command": ["experiments/run_posix_scope_audit.py", "--out-dir", str(out_dir.relative_to(ROOT))],
        "rows": rows,
        "overall_pass": bool(rows) and all(bool(row.get("acceptable")) for row in rows),
    }
    json_path = out_dir / "posix_scope_audit.json"
    md_path = out_dir / "posix_scope_audit.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_markdown(result, md_path)
    print(json.dumps(result, indent=2))
    return 0 if result["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
