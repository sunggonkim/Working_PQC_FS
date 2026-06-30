#!/usr/bin/env python3
"""Audit AEGIS-Q's intentionally narrow POSIX support boundary.

This harness records behavior for the POSIX modes that reviewers repeatedly
ask about: mmap/msync, truncate/fallocate, links, concurrent writers,
rename/directory fsync, xattrs, lower-filesystem assumptions, crash-time
visibility, and FUSE writeback-cache interaction.  It is an evidence artifact,
not a broad POSIX certification campaign.
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


ROOT = Path(__file__).resolve().parents[2]
FUSE_BIN = ROOT / "build" / "pqc_fuse"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "posix_scope_audit"

INTERNAL_XATTRS = [
    "user.pqc_metadata",
    "user.pqc_logical_size",
    "user.pqc_checkpoint",
]

REQUIRED_SEMANTICS = [
    "shared_mmap",
    "msync",
    "rename",
    "directory_fsync",
    "open_fd_truncate",
    "path_truncate",
    "fallocate",
    "hard_link",
    "symlink",
    "concurrent_disjoint_writes",
    "concurrent_same_block_writes",
    "crash_time_visibility",
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
                mm.flush()
            finally:
                mm.close()
        finally:
            os.close(fd)

    ok, err, detail = expect_oserror(attempt, {errno.ENODEV, errno.EINVAL, errno.EOPNOTSUPP})
    return {
        "case": "default_shared_mmap_msync_on_encrypted_file",
        "semantics": ["shared_mmap", "msync"],
        "status": "formal rejection",
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
        "semantics": ["concurrent_disjoint_writes"],
        "status": "support",
        "expected_behavior": "serialized_by_daemon_lock_for_disjoint_write_fsync_path",
        "errors": errors,
        "result_sha256_prefix": {
            "first_block_all_A": data[:4096] == payload_a,
            "second_block_all_B": data[4096:8192] == payload_b,
        },
        "acceptable": acceptable,
        "scope": "Narrow disjoint write/fsync path only; arbitrary conflicting app-level concurrency is not certified.",
    }


def audit_concurrent_same_block_writers(file_path: Path) -> dict[str, Any]:
    payload_a = b"A" * 4096
    payload_b = b"B" * 4096
    write_fsync(file_path, b"\x00" * 4096)
    errors: list[str] = []

    def writer(payload: bytes) -> None:
        try:
            fd = os.open(file_path, os.O_RDWR)
            try:
                os.pwrite(fd, payload, 0)
                os.fsync(fd)
            finally:
                os.close(fd)
        except BaseException as exc:  # retained below
            errors.append(repr(exc))

    t1 = threading.Thread(target=writer, args=(payload_a,))
    t2 = threading.Thread(target=writer, args=(payload_b,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    data = file_path.read_bytes()[:4096]
    acceptable = not errors and data in {payload_a, payload_b}
    return {
        "case": "concurrent_same_block_full_overwrite_writers",
        "semantics": ["concurrent_same_block_writes"],
        "status": "supported_subset",
        "expected_behavior": "same_block_full_overwrites_linearize_without_mixed_plaintext",
        "errors": errors,
        "final_all_A": data == payload_a,
        "final_all_B": data == payload_b,
        "mixed_or_partial": data not in {payload_a, payload_b},
        "acceptable": acceptable,
        "scope": "Same-block full-overwrite write/fsync races are serialized to one complete block; arbitrary overlapping byte ranges, mmap coherence, and application-level transactions are not certified.",
    }


def audit_concurrent_same_block_partial_overlap_writers(file_path: Path) -> dict[str, Any]:
    initial = b"\x00" * 4096
    payload_a = b"A" * 2048
    payload_b = b"B" * 2048
    write_fsync(file_path, initial)
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
    t2 = threading.Thread(target=writer, args=(1024, payload_b))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    data = file_path.read_bytes()[:4096]
    serial_a_then_b = b"A" * 1024 + b"B" * 2048 + b"\x00" * 1024
    serial_b_then_a = b"A" * 2048 + b"B" * 1024 + b"\x00" * 1024
    acceptable = not errors and data in {serial_a_then_b, serial_b_then_a}
    return {
        "case": "concurrent_same_block_partial_overlap_writers",
        "semantics": ["concurrent_same_block_writes"],
        "status": "supported_subset",
        "expected_behavior": "overlapping_same_block_partial_writes_match_one_serial_order",
        "errors": errors,
        "final_matches_a_then_b": data == serial_a_then_b,
        "final_matches_b_then_a": data == serial_b_then_a,
        "unexpected_mixed_or_corrupt": data not in {serial_a_then_b, serial_b_then_a},
        "acceptable": acceptable,
        "scope": "One same-block overlapping partial-write/fsync race is serializable to a complete-block recovery order; arbitrary byte-range transactions, mmap coherence, and application-level locking semantics are not certified.",
    }


def audit_resize_operations(file_path: Path) -> dict[str, Any]:
    payload_a = b"A" * 8192
    payload_c = b"C" * 4096
    errors: list[str] = []
    observations: dict[str, Any] = {}
    fd = -1
    try:
        fd = os.open(file_path, os.O_RDWR)
        os.pwrite(fd, payload_a, 0)
        os.fsync(fd)

        os.ftruncate(fd, 4096)
        os.fsync(fd)
        observations["after_truncate_4k_size"] = os.stat(file_path).st_size
        observations["after_truncate_4k_all_A"] = os.pread(fd, 4096, 0) == b"A" * 4096

        os.ftruncate(fd, 0)
        os.fsync(fd)
        observations["after_truncate_zero_size"] = os.stat(file_path).st_size
        observations["after_truncate_zero_empty_read"] = os.pread(fd, 4096, 0) == b""

        if hasattr(os, "posix_fallocate"):
            os.posix_fallocate(fd, 0, 8192)
            os.fsync(fd)
            observations["posix_fallocate_available"] = True
            observations["after_fallocate_size"] = os.stat(file_path).st_size
            observations["after_fallocate_zero_prefix"] = os.pread(fd, 4096, 0) == b"\x00" * 4096
            os.pwrite(fd, payload_c, 4096)
            os.fsync(fd)
            observations["after_post_fallocate_write_C"] = os.pread(fd, 4096, 4096) == payload_c
        else:
            observations["posix_fallocate_available"] = False
    except BaseException as exc:  # noqa: BLE001 - retained as audit evidence.
        errors.append(repr(exc))
    finally:
        if fd >= 0:
            os.close(fd)

    acceptable = (
        not errors
        and observations.get("after_truncate_4k_size") == 4096
        and observations.get("after_truncate_4k_all_A") is True
        and observations.get("after_truncate_zero_size") == 0
        and observations.get("after_truncate_zero_empty_read") is True
        and (
            observations.get("posix_fallocate_available") is False
            or (
                observations.get("after_fallocate_size") == 8192
                and observations.get("after_fallocate_zero_prefix") is True
                and observations.get("after_post_fallocate_write_C") is True
            )
        )
    )
    return {
        "case": "truncate_fallocate_visibility",
        "semantics": ["open_fd_truncate", "fallocate"],
        "status": "support",
        "expected_behavior": "open_fd_ftruncate_and_fallocate_preserve_visible_size_and_zero_fill",
        "observations": observations,
        "errors": errors,
        "acceptable": acceptable,
        "scope": "Open-file truncate/fallocate behavior only; path-only truncate, mmap coherence, and broad concurrent resize semantics are not certified.",
    }


def audit_path_truncate_rejection(file_path: Path) -> dict[str, Any]:
    write_fsync(file_path, b"T" * 4096)
    size_before = file_path.stat().st_size

    def attempt() -> None:
        os.truncate(file_path, 2048)

    enotsup = {errno.EOPNOTSUPP, getattr(errno, "ENOTSUP", errno.EOPNOTSUPP)}
    ok, err, detail = expect_oserror(attempt, enotsup)
    size_after = file_path.stat().st_size
    return {
        "case": "path_only_truncate_rejection",
        "semantics": ["path_truncate"],
        "status": "formal rejection",
        "expected_behavior": "rejected_ENOTSUP_without_open_file_context",
        "observed_errno": err,
        "observed_errno_name": errno_name(err),
        "detail": detail,
        "size_before": size_before,
        "size_after": size_after,
        "acceptable": ok and size_after == size_before,
        "scope": "Path-only truncate lacks an authenticated open-file context; open-fd ftruncate is tested separately.",
    }


def audit_rename_and_dir_fsync(mount_dir: Path, file_path: Path) -> list[dict[str, Any]]:
    del file_path
    rename_src = mount_dir / "rename_src.bin"
    rename_dst = mount_dir / "renamed.bin"
    rename_payload = b"rename-supported-subset" * 64
    write_fsync(rename_src, rename_payload)

    rename_errno = None
    rename_detail = None
    try:
        os.rename(rename_src, rename_dst)
        renamed_data = rename_dst.read_bytes()
        old_visible = rename_src.exists()
        ok_rename = renamed_data == rename_payload and not old_visible
        rename_detail = (
            "renamed file recovered expected payload"
            if ok_rename else
            f"payload_or_visibility_mismatch old_visible={old_visible}"
        )
    except OSError as exc:
        ok_rename = False
        rename_errno = exc.errno
        rename_detail = str(exc)

    overwrite_src = mount_dir / "overwrite_src.bin"
    overwrite_dst = mount_dir / "overwrite_dst.bin"
    overwrite_payload = b"overwrite-source" * 128
    write_fsync(overwrite_src, overwrite_payload)
    write_fsync(overwrite_dst, b"old-target" * 128)
    overwrite_errno = None
    overwrite_detail = None
    try:
        os.rename(overwrite_src, overwrite_dst)
        overwrite_data = overwrite_dst.read_bytes()
        old_src_visible = overwrite_src.exists()
        ok_overwrite = overwrite_data == overwrite_payload and not old_src_visible
        overwrite_detail = (
            "closed overwrite rename replaced target with source payload"
            if ok_overwrite else
            f"payload_or_visibility_mismatch old_src_visible={old_src_visible}"
        )
    except OSError as exc:
        ok_overwrite = False
        overwrite_errno = exc.errno
        overwrite_detail = str(exc)

    open_src = mount_dir / "open_rename_src.bin"
    open_dst = mount_dir / "open_rename_dst.bin"
    open_payload = b"open-source" * 128
    open_suffix = b"-after-rename-fd-write"
    write_fsync(open_src, open_payload)
    open_errno = None
    open_detail = None
    fd_open = os.open(open_src, os.O_RDWR)
    try:
        os.rename(open_src, open_dst)
        os.lseek(fd_open, 0, os.SEEK_SET)
        fd_before = os.read(fd_open, len(open_payload))
        os.lseek(fd_open, len(open_payload), os.SEEK_SET)
        os.write(fd_open, open_suffix)
        os.fsync(fd_open)
        final_payload = open_dst.read_bytes()
        old_visible = open_src.exists()
        ok_open_rename = (
            fd_before == open_payload
            and final_payload == open_payload + open_suffix
            and not old_visible
        )
        open_detail = (
            "open source fd survived rename and post-rename fsync"
            if ok_open_rename else
            f"payload_or_visibility_mismatch old_visible={old_visible}"
        )
    except OSError as exc:
        ok_open_rename = False
        open_errno = exc.errno
        open_detail = str(exc)
    finally:
        os.close(fd_open)

    open_target_src = mount_dir / "open_target_src.bin"
    open_target_dst = mount_dir / "open_target_dst.bin"
    open_target_src_payload = b"open-target-source" * 64
    open_target_dst_payload = b"open-target-destination" * 64
    write_fsync(open_target_src, open_target_src_payload)
    write_fsync(open_target_dst, open_target_dst_payload)
    fd_target = os.open(open_target_dst, os.O_RDWR)
    try:
        try:
            os.rename(open_target_src, open_target_dst)
            os.lseek(fd_target, 0, os.SEEK_SET)
            stale_before = os.read(fd_target, len(open_target_dst_payload))
            path_after_rename = open_target_dst.read_bytes()
            old_src_visible = open_target_src.exists()
            os.lseek(fd_target, 0, os.SEEK_SET)
            os.write(fd_target, b"stale-target-write")
            os.fsync(fd_target)
            path_after_stale_write = open_target_dst.read_bytes()
            ok_open_target_rename = (
                stale_before == open_target_dst_payload
                and path_after_rename == open_target_src_payload
                and path_after_stale_write == open_target_src_payload
                and not old_src_visible
            )
            open_target_errno = None
            open_target_detail = (
                "open target fd retained old file while path exposed source payload"
                if ok_open_target_rename else
                "open_target_mismatch "
                f"stale_before={stale_before == open_target_dst_payload} "
                f"path_after_rename={path_after_rename == open_target_src_payload} "
                f"path_after_stale_write={path_after_stale_write == open_target_src_payload} "
                f"old_src_visible={old_src_visible}"
            )
        except OSError as exc:
            ok_open_target_rename = False
            open_target_errno = exc.errno
            open_target_detail = str(exc)
    finally:
        os.close(fd_target)

    empty_dir_src = mount_dir / "empty_dir_src"
    empty_dir_dst = mount_dir / "empty_dir_dst"
    empty_dir_src.mkdir()
    empty_dir_errno = None
    empty_dir_detail = None
    try:
        os.rename(empty_dir_src, empty_dir_dst)
        ok_empty_dir_rename = empty_dir_dst.is_dir() and not empty_dir_src.exists()
        empty_dir_detail = (
            "empty directory renamed to new target"
            if ok_empty_dir_rename else
            "empty directory rename visibility mismatch"
        )
    except OSError as exc:
        ok_empty_dir_rename = False
        empty_dir_errno = exc.errno
        empty_dir_detail = str(exc)

    tree_dir_src = mount_dir / "tree_dir_src"
    tree_dir_dst = mount_dir / "tree_dir_dst"
    tree_dir_src.mkdir()
    tree_payload = b"directory-tree-rename-child"
    write_fsync(tree_dir_src / "child.bin", tree_payload)
    tree_rename_errno = None
    tree_rename_detail = None
    try:
        os.rename(tree_dir_src, tree_dir_dst)
        child_payload = (tree_dir_dst / "child.bin").read_bytes()
        old_tree_visible = tree_dir_src.exists()
        tree_rename_ok = (
            tree_dir_dst.is_dir()
            and child_payload == tree_payload
            and not old_tree_visible
        )
        tree_rename_detail = (
            "directory tree renamed and child payload remained readable"
            if tree_rename_ok else
            f"tree visibility/payload mismatch old_visible={old_tree_visible}"
        )
    except OSError as exc:
        tree_rename_ok = False
        tree_rename_errno = exc.errno
        tree_rename_detail = str(exc)

    subdir = mount_dir / "dirsync"
    subdir.mkdir(exist_ok=True)

    fsyncdir_errno = None
    fsyncdir_detail = None
    try:
        fd = os.open(subdir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        ok_fsyncdir = True
        fsyncdir_detail = "directory fsync completed"
    except OSError as exc:
        ok_fsyncdir = False
        fsyncdir_errno = exc.errno
        fsyncdir_detail = str(exc)
    return [
        {
            "case": "closed_file_rename_supported_subset",
            "semantics": ["rename"],
            "status": "supported_subset",
            "expected_behavior": "rename new closed regular file with sidecars",
            "observed_errno": rename_errno,
            "observed_errno_name": errno_name(rename_errno),
            "detail": rename_detail,
            "acceptable": ok_rename,
            "scope": "Closed regular-file rename to a non-existing target is supported; open-file and directory cases are tested separately, and crash-atomic rename remains outside this row.",
        },
        {
            "case": "closed_file_overwrite_rename_supported_subset",
            "semantics": ["rename"],
            "status": "supported_subset",
            "expected_behavior": "closed regular overwrite rename replaces closed target",
            "observed_errno": overwrite_errno,
            "observed_errno_name": errno_name(overwrite_errno),
            "detail": overwrite_detail,
            "acceptable": ok_overwrite,
            "scope": "Closed regular-file overwrite rename is supported only when the target is not open; open-target and crash-atomic multi-file rename remain outside this row.",
        },
        {
            "case": "open_source_rename_supported_subset",
            "semantics": ["rename"],
            "status": "supported_subset",
            "expected_behavior": "open source fd survives rename to new target",
            "observed_errno": open_errno,
            "observed_errno_name": errno_name(open_errno),
            "detail": open_detail,
            "acceptable": ok_open_rename,
            "scope": "Open-source regular-file rename to a non-existing target is supported after fd-context path retargeting; open-target overwrite is tested separately and crash-atomic rename remains outside this row.",
        },
        {
            "case": "open_target_rename_supported_subset",
            "semantics": ["rename"],
            "status": "supported_subset",
            "expected_behavior": "rename over open target preserves stale target fd and publishes source at target path",
            "observed_errno": open_target_errno,
            "observed_errno_name": errno_name(open_target_errno),
            "detail": open_target_detail,
            "acceptable": ok_open_target_rename,
            "scope": "Open-target regular-file overwrite rename is supported through FUSE hidden-file retargeting for the stale target fd; crash-atomic multi-file rename and open-subtree retargeting remain outside this row.",
        },
        {
            "case": "empty_directory_rename_supported_subset",
            "semantics": ["rename"],
            "status": "supported_subset",
            "expected_behavior": "empty directory rename to non-existing target succeeds",
            "observed_errno": empty_dir_errno,
            "observed_errno_name": errno_name(empty_dir_errno),
            "detail": empty_dir_detail,
            "acceptable": ok_empty_dir_rename,
            "scope": "Empty-directory rename to a non-existing target is supported only when no file contexts are open; open-subtree retargeting and crash-atomic namespace publication remain outside this row.",
        },
        {
            "case": "directory_tree_rename_supported_subset",
            "semantics": ["rename"],
            "status": "supported_subset",
            "expected_behavior": "non-empty directory tree rename to non-existing target succeeds",
            "observed_errno": tree_rename_errno,
            "observed_errno_name": errno_name(tree_rename_errno),
            "detail": tree_rename_detail,
            "acceptable": tree_rename_ok,
            "scope": "Directory-tree rename is supported only with no open file contexts and a non-existing target; open-subtree retargeting and crash-atomic multi-entry durability remain outside this row.",
        },
        {
            "case": "directory_fsync_supported_subset",
            "semantics": ["directory_fsync"],
            "status": "supported_subset",
            "expected_behavior": "lower directory fsync succeeds",
            "observed_errno": fsyncdir_errno,
            "observed_errno_name": errno_name(fsyncdir_errno),
            "detail": fsyncdir_detail,
            "acceptable": ok_fsyncdir,
            "scope": "Directory fsync is forwarded to the lower directory; this does not by itself certify crash-atomic multi-file rename.",
        },
    ]


def audit_link_rejections(mount_dir: Path, file_path: Path) -> list[dict[str, Any]]:
    hard_path = mount_dir / "hardlink.bin"
    hard_errno = None
    hard_detail = None
    try:
        before = file_path.read_bytes()
        os.link(file_path, hard_path)
        read_through_ok = hard_path.read_bytes() == before
        updated = (b"HARDLINK-UPDATED" * ((len(before) // 16) + 1))[:len(before)]
        with hard_path.open("r+b") as f:
            f.write(updated)
            f.flush()
            os.fsync(f.fileno())
        write_through_ok = file_path.read_bytes() == updated
        hard_path.unlink()
        unlink_preserves_original = file_path.read_bytes() == updated
        hard_ok = read_through_ok and write_through_ok and unlink_preserves_original
        hard_detail = (
            "hard link shared marker/sidecar state and unlink preserved original"
            if hard_ok else
            f"read_through={read_through_ok} write_through={write_through_ok} "
            f"unlink_preserves_original={unlink_preserves_original}"
        )
    except OSError as exc:
        hard_ok = False
        hard_errno = exc.errno
        hard_detail = str(exc)

    sym_path = mount_dir / "sym.bin"
    sym_errno = None
    sym_detail = None
    rejected_targets: list[dict[str, Any]] = []
    try:
        os.symlink("posix.bin", sym_path)
        link_target = os.readlink(sym_path)
        through_link = sym_path.read_bytes()
        for target, name in (
            ("/etc/passwd", "absolute_target"),
            ("../outside.bin", "parent_escape_target"),
            ("posix.bin.pqcdata", "internal_sidecar_target"),
        ):
            def attempt_bad_symlink(t=target, n=name) -> None:
                os.symlink(t, mount_dir / f"{n}.sym")

            ok_bad, bad_errno, bad_detail = expect_oserror(
                attempt_bad_symlink,
                {errno.EOPNOTSUPP, getattr(errno, "ENOTSUP", errno.EOPNOTSUPP), errno.ENOENT},
            )
            rejected_targets.append({
                "target": target,
                "acceptable": ok_bad,
                "observed_errno": bad_errno,
                "observed_errno_name": errno_name(bad_errno),
                "detail": bad_detail,
            })
        sym_ok = (
            link_target == "posix.bin"
            and through_link == file_path.read_bytes()
            and all(row["acceptable"] for row in rejected_targets)
        )
        sym_detail = (
            "symlink target and read-through payload matched"
            if sym_ok else
            f"target={link_target!r} payload_match={through_link == file_path.read_bytes()} rejected_targets={rejected_targets!r}"
        )
    except OSError as exc:
        sym_ok = False
        sym_errno = exc.errno
        sym_detail = str(exc)
    return [
        {
            "case": "hard_link_supported_subset",
            "semantics": ["hard_link"],
            "status": "supported_subset",
            "expected_behavior": "no-open regular-file hard link shares marker and sidecar state",
            "observed_errno": hard_errno,
            "observed_errno_name": errno_name(hard_errno),
            "detail": hard_detail,
            "acceptable": hard_ok,
            "scope": "Regular-file hard links are supported only with no open file contexts and linked marker/sidecar state; open hard-link creation, SQLite compatibility sidecars, crash-atomic multi-entry publication, and full link-count lifecycle certification remain outside this row.",
        },
        {
            "case": "symlink_supported_subset",
            "semantics": ["symlink"],
            "status": "supported_subset",
            "expected_behavior": "relative_symlink_readlink_and_read_through",
            "observed_errno": sym_errno,
            "observed_errno_name": errno_name(sym_errno),
            "detail": sym_detail,
            "rejected_escape_targets": rejected_targets,
            "acceptable": sym_ok,
            "scope": "Relative symlink namespace objects are supported when they do not target internal sidecar names or escape through absolute/parent paths; this is not hard-link support or crash-atomic directory-tree certification.",
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
        "semantics": ["user_xattr", "internal_xattr"],
        "status": "support",
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
        "semantics": ["lower_filesystem_xattr", "lower_directory_durability"],
        "status": "paper limitation",
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
    source_path = ROOT / "code" / "runtime" / "pqc_lifecycle.c"
    source = source_path.read_text(encoding="utf-8")
    disables_writeback = "conn->want &= ~FUSE_CAP_WRITEBACK_CACHE" in source
    requests_writeback = "conn->want |= FUSE_CAP_WRITEBACK_CACHE" in source
    disables_direct_mmap = "conn->want &= ~FUSE_DIRECT_IO_ALLOW_MMAP" in source
    requests_direct_mmap = "conn->want |= FUSE_DIRECT_IO_ALLOW_MMAP" in source
    mount_line = read_proc_mounts(mount_dir)
    return {
        "case": "fuse_writeback_cache_and_direct_mmap_caps",
        "semantics": ["fuse_writeback_cache", "direct_io_mmap_capability"],
        "status": "formal rejection",
        "expected_behavior": "capabilities_not_requested",
        "source_disables_writeback_cache": disables_writeback,
        "source_requests_writeback_cache": requests_writeback,
        "source_disables_direct_io_mmap": disables_direct_mmap,
        "source_requests_direct_io_mmap": requests_direct_mmap,
        "source_path": str(source_path.relative_to(ROOT)),
        "proc_mounts_line": mount_line,
        "acceptable": (
            disables_writeback
            and not requests_writeback
            and disables_direct_mmap
            and not requests_direct_mmap
        ),
        "scope": "The daemon avoids FUSE writeback-cache and direct-IO mmap capabilities; no cache-coherence claim is made.",
    }


def audit_crash_time_visibility_scope() -> dict[str, Any]:
    design_text = (ROOT / "Paper" / "3_Design.tex").read_text(encoding="utf-8")
    security_text = (ROOT / "Paper" / "8_Security_Analysis.tex").read_text(encoding="utf-8")
    discussion_text = (ROOT / "Paper" / "10_Discussion_and_Limitations.tex").read_text(encoding="utf-8")
    source_text = (ROOT / "code" / "fs" / "pqc_file_io.c").read_text(encoding="utf-8")
    has_fault_boundary = 'pqc_fault_cutpoint("fsync_before_return")' in source_text
    has_non_certification = (
        "not a formal proof of crash consistency" in security_text
        or "not certify" in discussion_text
        or "not certified" in design_text
    )
    return {
        "case": "crash_time_visibility_scope",
        "semantics": ["crash_time_visibility"],
        "status": "paper limitation",
        "expected_behavior": "bounded_to_existing_fault_models_not_full_posix_crash_visibility",
        "source_has_fsync_return_cutpoint": has_fault_boundary,
        "paper_has_non_certification_boundary": has_non_certification,
        "acceptable": has_fault_boundary and has_non_certification,
        "scope": "Crash-time visibility is delegated to C3/C4 fault models; C1 does not certify full POSIX crash visibility.",
    }


def build_semantic_status(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_rank = {
        "missing": 0,
        "paper limitation": 1,
        "formal rejection": 2,
        "supported_subset": 3,
        "support": 4,
    }
    status: dict[str, Any] = {
        name: {
            "covered": False,
            "status": "missing",
            "cases": [],
            "acceptable": False,
        }
        for name in REQUIRED_SEMANTICS
    }
    for row in rows:
        for semantic in row.get("semantics", []):
            if semantic not in status:
                continue
            status[semantic]["covered"] = True
            status[semantic]["cases"].append(row["case"])
            row_ok = bool(row.get("acceptable"))
            status[semantic]["acceptable"] = status[semantic]["acceptable"] or row_ok
            if row_ok:
                current = status[semantic]["status"]
                candidate = row.get("status", "covered")
                if status_rank.get(candidate, 0) >= status_rank.get(current, 0):
                    status[semantic]["status"] = candidate
    return status


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# POSIX scope audit",
        "",
        "Scope: final-binary audit for intentionally narrow POSIX semantics.",
        "",
        f"- Overall pass: `{result['overall_pass']}`",
        f"- Command: `{' '.join(result['command'])}`",
        f"- Required semantics covered: `{result['required_semantics_all_covered']}`",
        "",
        "## Required semantic status",
        "",
        "| Semantic | Status | Covered | Acceptable | Cases |",
        "|---|---|---:|---:|---|",
    ]
    for semantic in REQUIRED_SEMANTICS:
        row = result["semantic_status"][semantic]
        lines.append(
            f"| `{semantic}` | `{row['status']}` | `{row['covered']}` | `{row['acceptable']}` | `{', '.join(row['cases'])}` |"
        )
    lines.extend([
        "",
        "## Audit rows",
        "",
        "| Case | Status | Expected behavior | Acceptable | Scope |",
        "|---|---|---|---:|---|",
    ])
    for row in result["rows"]:
        lines.append(
            f"| `{row['case']}` | `{row.get('status', '')}` | `{row.get('expected_behavior')}` | `{row['acceptable']}` | {row.get('scope', '')} |"
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
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
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
        rows.append(audit_concurrent_same_block_writers(file_path))
        rows.append(audit_concurrent_same_block_partial_overlap_writers(file_path))
        rows.append(audit_resize_operations(file_path))
        rows.append(audit_path_truncate_rejection(file_path))
        rows.extend(audit_rename_and_dir_fsync(mount_dir, file_path))
        rows.extend(audit_link_rejections(mount_dir, file_path))
        rows.append(audit_xattrs(file_path))
        rows.append(audit_lower_fs(storage_dir, storage_dir / "posix.bin"))
        rows.append(audit_writeback_cache(mount_dir))
        rows.append(audit_crash_time_visibility_scope())
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

    semantic_status = build_semantic_status(rows)
    required_semantics_all_covered = all(
        row["covered"] and row["acceptable"]
        for row in semantic_status.values()
    )
    result = {
        "command": ["experiments/run_posix_scope_audit.py", "--out-dir", str(out_dir.relative_to(ROOT))],
        "required_semantics": REQUIRED_SEMANTICS,
        "semantic_status": semantic_status,
        "required_semantics_all_covered": required_semantics_all_covered,
        "rows": rows,
        "overall_pass": bool(rows)
        and all(bool(row.get("acceptable")) for row in rows)
        and required_semantics_all_covered,
    }
    json_path = out_dir / "posix_scope_audit.json"
    md_path = out_dir / "posix_scope_audit.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_markdown(result, md_path)
    print(json.dumps(result, indent=2))
    return 0 if result["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
