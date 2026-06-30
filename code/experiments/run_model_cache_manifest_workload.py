#!/usr/bin/env python3
"""Run a mounted model-cache manifest publish workload.

This X3 workload exercises the supported POSIX subset that reviewers asked for:
write/fsync a staged object, publish it with closed-file rename to a new target,
fsync the containing directory, publish a manifest the same way, remount, and
verify object hashes. It is intentionally scoped; it is not a broad workload
suite or a full POSIX certification campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FUSE_BIN = Path(os.environ.get("PQC_FUSE_BIN", ROOT / "build" / "pqc_fuse"))
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "model_cache_manifest_workload"


@dataclass
class FuseHandle:
    proc: subprocess.Popen[bytes]
    stdout: Any
    stderr: Any


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def deterministic_payload(index: int, payload_bytes: int) -> bytes:
    seed = hashlib.sha256(f"aegisq-model-cache-object:{index}".encode()).digest()
    out = bytearray()
    counter = 0
    while len(out) < payload_bytes:
        out.extend(hashlib.sha256(seed + counter.to_bytes(8, "little")).digest())
        counter += 1
    return bytes(out[:payload_bytes])


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * q)
    return ordered[index]


def summarize_us(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "avg_us": 0.0,
            "p50_us": 0.0,
            "p95_us": 0.0,
            "p99_us": 0.0,
            "max_us": 0.0,
        }
    return {
        "count": len(values),
        "avg_us": sum(values) / len(values),
        "p50_us": percentile(values, 0.50),
        "p95_us": percentile(values, 0.95),
        "p99_us": percentile(values, 0.99),
        "max_us": max(values),
    }


def mount_is_visible(mount_dir: Path) -> bool:
    return subprocess.run(
        ["mountpoint", "-q", str(mount_dir)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def fusermount_command() -> str:
    for name in ("fusermount3", "fusermount"):
        if shutil.which(name):
            return name
    return "fusermount3"


def start_fuse(storage_dir: Path, mount_dir: Path, out_dir: Path,
               password: str, label: str) -> FuseHandle:
    log_dir = out_dir / "daemon_logs" / label
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout = (log_dir / "pqc_fuse.stdout.txt").open("wb")
    stderr = (log_dir / "pqc_fuse.stderr.txt").open("wb")
    env = os.environ.copy()
    env.update({
        "PQC_MASTER_PASSWORD": password,
        "PQC_FRESHNESS_ANCHOR_BACKEND": "file",
        "PQC_FRESHNESS_ANCHOR_PATH": str(storage_dir / ".anchor"),
        "PQC_ENABLE_ADMISSION_ON_WRITE": "0",
        "PQC_ENABLE_QOS_THROTTLE_ON_WRITE": "0",
        "PQC_KEY_ROTATION_INTERVAL_S": "0",
    })
    proc = subprocess.Popen(
        [str(FUSE_BIN), str(storage_dir), str(mount_dir), "-f"],
        cwd=ROOT,
        env=env,
        stdout=stdout,
        stderr=stderr,
    )
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if mount_is_visible(mount_dir):
            return FuseHandle(proc=proc, stdout=stdout, stderr=stderr)
        if proc.poll() is not None:
            stdout.close()
            stderr.close()
            raise RuntimeError(f"FUSE exited before mount: rc={proc.returncode}")
        time.sleep(0.05)
    stdout.close()
    stderr.close()
    raise TimeoutError("timed out waiting for AEGIS-Q FUSE mount")


def stop_fuse(handle: FuseHandle | None, mount_dir: Path) -> dict[str, Any]:
    unmount_rc: int | None = None
    unmount_stderr = ""
    if mount_is_visible(mount_dir):
        unmount = subprocess.run(
            [fusermount_command(), "-u", str(mount_dir)],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        unmount_rc = unmount.returncode
        unmount_stderr = unmount.stderr.strip()
        if unmount_rc != 0:
            lazy = subprocess.run(
                [fusermount_command(), "-uz", str(mount_dir)],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            unmount_rc = lazy.returncode
            unmount_stderr = lazy.stderr.strip()
    fuse_rc: int | None = None
    if handle is not None:
        if handle.proc.poll() is None:
            handle.proc.send_signal(signal.SIGINT)
            try:
                handle.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                handle.proc.kill()
                handle.proc.wait(timeout=5)
        fuse_rc = handle.proc.returncode
        handle.stdout.close()
        handle.stderr.close()
    return {
        "unmount_rc": unmount_rc,
        "unmount_stderr": unmount_stderr,
        "fuse_rc": fuse_rc,
    }


def fsync_directory(path: Path) -> float:
    start = time.monotonic_ns()
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return (time.monotonic_ns() - start) / 1000.0


def write_fsync_close(path: Path, payload: bytes) -> float:
    start = time.monotonic_ns()
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        view = memoryview(payload)
        total = 0
        while total < len(payload):
            written = os.write(fd, view[total:])
            if written <= 0:
                raise OSError("short write while staging model-cache object")
            total += written
        if hasattr(os, "fdatasync"):
            os.fdatasync(fd)
        else:
            os.fsync(fd)
    finally:
        os.close(fd)
    return (time.monotonic_ns() - start) / 1000.0


def publish_file(temp_path: Path, final_path: Path, directory: Path) -> dict[str, float]:
    start = time.monotonic_ns()
    os.rename(temp_path, final_path)
    rename_us = (time.monotonic_ns() - start) / 1000.0
    dir_fsync_us = fsync_directory(directory)
    return {
        "rename_us": rename_us,
        "directory_fsync_us": dir_fsync_us,
    }


def write_object_set(mount_dir: Path, objects: int,
                     payload_bytes: int) -> dict[str, Any]:
    objects_dir = mount_dir / "objects"
    tmp_dir = mount_dir / "tmp"
    objects_dir.mkdir(exist_ok=True)
    tmp_dir.mkdir(exist_ok=True)
    fsync_directory(mount_dir)
    fsync_directory(objects_dir)
    fsync_directory(tmp_dir)

    expected: list[dict[str, Any]] = []
    latencies: list[dict[str, Any]] = []
    workload_start = time.monotonic_ns()
    for index in range(objects):
        name = f"obj-{index:04d}.bin"
        temp_path = tmp_dir / f"{name}.tmp"
        final_path = objects_dir / name
        payload = deterministic_payload(index, payload_bytes)
        digest = sha256_bytes(payload)
        write_us = write_fsync_close(temp_path, payload)
        publish = publish_file(temp_path, final_path, objects_dir)
        total_us = write_us + publish["rename_us"] + publish["directory_fsync_us"]
        expected.append({
            "name": name,
            "size": payload_bytes,
            "sha256": digest,
        })
        latencies.append({
            "name": name,
            "write_fsync_close_us": write_us,
            "rename_us": publish["rename_us"],
            "directory_fsync_us": publish["directory_fsync_us"],
            "publish_us": total_us,
        })
    workload_wall_us = (time.monotonic_ns() - workload_start) / 1000.0
    total_payload_bytes = objects * payload_bytes
    publish_us = [float(row["publish_us"]) for row in latencies]
    return {
        "expected": expected,
        "latencies": latencies,
        "summary": {
            "objects": objects,
            "payload_bytes_per_object": payload_bytes,
            "total_payload_bytes": total_payload_bytes,
            "workload_wall_us": workload_wall_us,
            "publish_latency": summarize_us(publish_us),
            "write_fsync_close_latency": summarize_us([
                float(row["write_fsync_close_us"]) for row in latencies
            ]),
            "rename_latency": summarize_us([
                float(row["rename_us"]) for row in latencies
            ]),
            "directory_fsync_latency": summarize_us([
                float(row["directory_fsync_us"]) for row in latencies
            ]),
            "payload_mib_s": (
                total_payload_bytes / (1024 * 1024) / (workload_wall_us / 1_000_000.0)
                if workload_wall_us > 0 else 0.0
            ),
            "objects_s": (
                objects / (workload_wall_us / 1_000_000.0)
                if workload_wall_us > 0 else 0.0
            ),
        },
    }


def manifest_bytes(expected: list[dict[str, Any]]) -> bytes:
    lines = [
        "# aegisq.model_cache_manifest.v1",
        "# name size sha256",
    ]
    for row in expected:
        lines.append(f"{row['name']} {row['size']} {row['sha256']}")
    return ("\n".join(lines) + "\n").encode("ascii")


def publish_manifest(mount_dir: Path, expected: list[dict[str, Any]]) -> dict[str, Any]:
    temp_path = mount_dir / "manifest.v1.tmp"
    final_path = mount_dir / "manifest.v1"
    payload = manifest_bytes(expected)
    write_us = write_fsync_close(temp_path, payload)
    publish = publish_file(temp_path, final_path, mount_dir)
    return {
        "path": "manifest.v1",
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "write_fsync_close_us": write_us,
        "rename_us": publish["rename_us"],
        "directory_fsync_us": publish["directory_fsync_us"],
        "publish_us": write_us + publish["rename_us"] + publish["directory_fsync_us"],
    }


def verify_manifest(mount_dir: Path) -> dict[str, Any]:
    manifest_path = mount_dir / "manifest.v1"
    objects_dir = mount_dir / "objects"
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    start = time.monotonic_ns()
    try:
        lines = manifest_path.read_text(encoding="ascii").splitlines()
    except OSError as exc:
        return {
            "pass": False,
            "errors": [repr(exc)],
            "objects_verified": 0,
            "verify_us": (time.monotonic_ns() - start) / 1000.0,
        }
    for line in lines:
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 3:
            errors.append(f"malformed manifest line: {line!r}")
            continue
        name, size_text, expected_hash = parts
        try:
            expected_size = int(size_text)
            data = (objects_dir / name).read_bytes()
        except (OSError, ValueError) as exc:
            errors.append(f"{name}: {exc!r}")
            continue
        observed_hash = sha256_bytes(data)
        row = {
            "name": name,
            "expected_size": expected_size,
            "observed_size": len(data),
            "expected_sha256": expected_hash,
            "observed_sha256": observed_hash,
            "match": len(data) == expected_size and observed_hash == expected_hash,
        }
        if not row["match"]:
            errors.append(f"{name}: hash_or_size_mismatch")
        rows.append(row)
    return {
        "pass": not errors and bool(rows),
        "errors": errors,
        "objects_verified": len(rows),
        "verify_us": (time.monotonic_ns() - start) / 1000.0,
        "rows": rows,
    }


def run_workload_in_dir(args: argparse.Namespace, base: Path) -> dict[str, Any]:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    storage_dir = base / "storage"
    mount_dir = base / "mnt"
    storage_dir.mkdir()
    mount_dir.mkdir()
    password = args.password

    handle: FuseHandle | None = None
    first_unmount: dict[str, Any] = {}
    try:
        handle = start_fuse(storage_dir, mount_dir, out_dir, password, "initial")
        objects = write_object_set(mount_dir, args.objects, args.payload_bytes)
        manifest = publish_manifest(mount_dir, objects["expected"])
        mounted_verify = verify_manifest(mount_dir)
    finally:
        first_unmount = stop_fuse(handle, mount_dir)

    remount_handle: FuseHandle | None = None
    remount_unmount: dict[str, Any] = {}
    try:
        remount_handle = start_fuse(storage_dir, mount_dir, out_dir, password, "remount")
        remount_verify = verify_manifest(mount_dir)
    finally:
        remount_unmount = stop_fuse(remount_handle, mount_dir)

    overall_pass = (
        mounted_verify.get("pass") is True
        and remount_verify.get("pass") is True
        and first_unmount.get("unmount_rc") in (0, None)
        and remount_unmount.get("unmount_rc") in (0, None)
        and objects["summary"]["objects"] == args.objects
    )
    return {
        "schema": "aegisq.model_cache_manifest_workload.v1",
        "created_utc": now_utc(),
        "workload": "model-cache manifest publish",
        "scope": [
            "Mounted application-style object cache publish workload.",
            "Uses write/fsync/close, closed-file rename to a new target, directory fsync, remount, and hash verification.",
            "Not a broad workload suite, full POSIX certification, AI-inference QoS claim, or crash-atomic multi-file rename proof.",
        ],
        "command": [
            "python3",
            relpath(Path(__file__)),
            "--objects",
            str(args.objects),
            "--payload-bytes",
            str(args.payload_bytes),
        ],
        "fuse_bin": relpath(FUSE_BIN),
        "objects_requested": args.objects,
        "payload_bytes": args.payload_bytes,
        "kept_work_dir": relpath(base) if args.keep_work_dir else None,
        "publish": objects,
        "manifest": manifest,
        "mounted_verify": mounted_verify,
        "remount_verify": remount_verify,
        "unmount": {
            "first": first_unmount,
            "remount": remount_unmount,
        },
        "supported_subset_exercised": {
            "write_fsync_close": True,
            "closed_file_rename_to_new_target": True,
            "directory_fsync": True,
            "remount_hash_verification": True,
        },
        "overall_pass": overall_pass,
    }


def run_workload(args: argparse.Namespace) -> dict[str, Any]:
    if args.keep_work_dir:
        base = Path(tempfile.mkdtemp(prefix="aegisq-model-cache-"))
        return run_workload_in_dir(args, base)
    with tempfile.TemporaryDirectory(prefix="aegisq-model-cache-") as tmp:
        return run_workload_in_dir(args, Path(tmp))


def write_markdown(result: dict[str, Any], path: Path) -> None:
    summary = result["publish"]["summary"]
    manifest = result["manifest"]
    lines = [
        "# Model-Cache Manifest Workload",
        "",
        f"- Overall pass: `{str(result['overall_pass']).lower()}`",
        f"- Objects: `{summary['objects']}`",
        f"- Payload bytes per object: `{summary['payload_bytes_per_object']}`",
        f"- Payload throughput MiB/s: `{summary['payload_mib_s']:.3f}`",
        f"- Publish p50/p99 us: `{summary['publish_latency']['p50_us']:.1f}` / `{summary['publish_latency']['p99_us']:.1f}`",
        f"- Manifest publish us: `{manifest['publish_us']:.1f}`",
        f"- Mounted verify pass: `{str(result['mounted_verify']['pass']).lower()}`",
        f"- Remount verify pass: `{str(result['remount_verify']['pass']).lower()}`",
        "",
        "Scope: mounted model-cache object staging with closed-file rename and directory fsync; not a broad workload suite or full POSIX proof.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", type=int, default=24)
    parser.add_argument("--payload-bytes", type=int, default=16 * 1024)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument(
        "--password",
        default=os.environ.get("PQC_MASTER_PASSWORD", "1234qwer"),
        help="mount password; defaults to PQC_MASTER_PASSWORD or the local test password",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.objects <= 0:
        raise SystemExit("--objects must be positive")
    if args.payload_bytes <= 0:
        raise SystemExit("--payload-bytes must be positive")
    result = run_workload(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "model_cache_manifest_workload.json"
    md_path = args.out_dir / "model_cache_manifest_workload.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(result, md_path)
    print(json.dumps({
        "overall_pass": result["overall_pass"],
        "json": relpath(json_path),
        "markdown": relpath(md_path),
        "objects": args.objects,
        "payload_bytes": args.payload_bytes,
        "publish_p99_us": result["publish"]["summary"]["publish_latency"]["p99_us"],
        "payload_mib_s": result["publish"]["summary"]["payload_mib_s"],
    }, indent=2, sort_keys=True))
    return 0 if result["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
