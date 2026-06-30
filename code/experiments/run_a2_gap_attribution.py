#!/usr/bin/env python3
"""Short AEGIS-Q versus gocryptfs gap attribution probe.

This is not the retained frozen-contract headline runner. It derives a smaller
profile from the frozen contract, runs both user-space encrypted filesystems
with the same fio workload, traces the fio client with ``strace -f -c``, and
prints a compact attribution summary for Gate A2.  The FUSE daemons are not
started under strace because ptraced fusermount can be rejected by the host.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import run_frozen_aegisq_contract as aegisq
import run_frozen_gocryptfs_contract as gocryptfs


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path("/tmp/aegisq_a2_gap_attribution")


def derive_profile(contract_path: Path, size: str, runtime_s: int) -> dict[str, Any]:
    payload = aegisq.load_contract(contract_path)
    profile = copy.deepcopy(aegisq.workload_profile(payload))
    fio_common = profile["mount_options"]["fio_common"]
    fio_common["size"] = size
    fio_common["runtime"] = f"{runtime_s}s"
    fio_common["ramp_time"] = "0s"
    profile["file_size_bytes"] = parse_size_bytes(size)
    profile["repetition_count"] = 1
    profile["profile_id"] = f"{profile['profile_id']}_a2_short_{size}_{runtime_s}s"
    return profile


def parse_size_bytes(size: str) -> int:
    match = re.fullmatch(r"(\d+)([kKmMgG]?)", size.strip())
    if not match:
        raise ValueError(f"unsupported fio size: {size}")
    value = int(match.group(1))
    suffix = match.group(2).lower()
    if suffix == "k":
        return value * 1024
    if suffix == "m":
        return value * 1024 * 1024
    if suffix == "g":
        return value * 1024 * 1024 * 1024
    return value


def parse_strace_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "path": str(path), "syscalls": {}}
    syscalls: dict[str, dict[str, Any]] = {}
    total: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%") or stripped.startswith("-"):
            continue
        parts = stripped.split()
        if len(parts) < 5:
            continue
        if parts[-1] == "total":
            total = {
                "seconds": parse_float(parts[1]),
                "calls": parse_int(parts[3]),
                "errors": parse_int(parts[4]) if len(parts) > 5 else 0,
            }
            continue
        name = parts[-1]
        errors = 0
        calls_index = 3
        if len(parts) >= 6:
            errors = parse_int(parts[4])
        syscalls[name] = {
            "pct_time": parse_float(parts[0]),
            "seconds": parse_float(parts[1]),
            "usecs_per_call": parse_float(parts[2]),
            "calls": parse_int(parts[calls_index]),
            "errors": errors,
        }
    return {
        "available": True,
        "path": str(path),
        "total": total,
        "syscalls": syscalls,
        "top_by_time": sorted(
            syscalls.items(), key=lambda item: item[1].get("seconds", 0.0), reverse=True
        )[:12],
    }


def parse_float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return 0.0


def parse_int(text: str) -> int:
    try:
        return int(text)
    except ValueError:
        return 0


def parse_durability_summary(stderr_path: Path) -> dict[str, int]:
    if not stderr_path.exists():
        return {}
    lines = stderr_path.read_text(encoding="utf-8", errors="replace").splitlines()
    target = ""
    for line in lines:
        if "Durability mounted-operation sync stats:" in line:
            target = line
    if not target:
        for line in lines:
            if "Durability sync stats:" in line:
                target = line
    return {key: int(value) for key, value in re.findall(r"([a-zA-Z_]+)=([0-9]+)", target)}


def stop_aegisq_graceful(
    handle: aegisq.FuseHandle | None,
    mount_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    log_dir = out_dir / "mount_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    unmount = subprocess.run(
        [aegisq.fusermount_command(), "-u", str(mount_dir)],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    (log_dir / "unmount.stdout.txt").write_text(unmount.stdout, encoding="utf-8")
    (log_dir / "unmount.stderr.txt").write_text(unmount.stderr, encoding="utf-8")
    if handle is None:
        return {"argv": [aegisq.fusermount_command(), "-u", str(mount_dir)], "returncode": unmount.returncode}
    if handle.proc.poll() is None:
        try:
            handle.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            handle.proc.send_signal(signal.SIGINT)
            try:
                handle.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                handle.proc.kill()
                handle.proc.wait(timeout=5)
    handle.stdout.close()
    handle.stderr.close()
    return {
        "argv": [aegisq.fusermount_command(), "-u", str(mount_dir)],
        "returncode": unmount.returncode,
        "fuse_returncode": handle.proc.returncode,
    }


def run_one_fio(
    profile: dict[str, Any],
    bench_dir: Path,
    out_dir: Path,
    label: str,
    trace: bool = False,
) -> dict[str, Any]:
    argv = aegisq.fio_command(profile, bench_dir)
    strace_path: Path | None = None
    if trace:
        strace_path = out_dir / f"{label}.strace_summary.txt"
        argv = ["strace", "-f", "-qq", "-c", "-o", str(strace_path)] + argv
    command = aegisq.run_fio_with_timeout(label, argv, out_dir, timeout_s=None)
    row = aegisq.parse_fio_result(command, "warm", 0)
    result = {"command": command, "row": row}
    if strace_path is not None:
        result["strace"] = parse_strace_summary(strace_path)
    return result


def run_aegisq(profile: dict[str, Any], out_dir: Path, password: str) -> dict[str, Any]:
    storage_dir = Path(tempfile.mkdtemp(prefix="a2_aegisq_storage_"))
    mount_dir = Path(tempfile.mkdtemp(prefix="a2_aegisq_mnt_"))
    handle: aegisq.FuseHandle | None = None
    try:
        handle = aegisq.start_fuse(storage_dir, mount_dir, out_dir / "aegisq", password)
        bench_dir = mount_dir / "contract"
        bench_dir.mkdir(parents=True, exist_ok=True)
        prep = aegisq.precreate_fio_file(profile, bench_dir, out_dir / "aegisq")
        warmup = run_one_fio(profile, bench_dir, out_dir / "aegisq", "warmup")
        measured = run_one_fio(profile, bench_dir, out_dir / "aegisq", "measured", trace=True)
        stop = stop_aegisq_graceful(handle, mount_dir, out_dir / "aegisq")
        handle = None
        stderr_path = out_dir / "aegisq" / "mount_logs" / "pqc_fuse.stderr.txt"
        return {
            "mode": "aegisq",
            "file_preparation": prep,
            "warmup": warmup["row"],
            "measured": measured["row"],
            "unmount": stop,
            "durability": parse_durability_summary(stderr_path),
            "fio_client_strace": measured.get("strace", {}),
        }
    finally:
        if handle is not None:
            stop_aegisq_graceful(handle, mount_dir, out_dir / "aegisq")
        shutil.rmtree(storage_dir, ignore_errors=True)
        shutil.rmtree(mount_dir, ignore_errors=True)


def run_gocryptfs(profile: dict[str, Any], out_dir: Path, password: str) -> dict[str, Any]:
    cipher_dir = Path(tempfile.mkdtemp(prefix="a2_gocryptfs_cipher_"))
    mount_dir = Path(tempfile.mkdtemp(prefix="a2_gocryptfs_mnt_"))
    secret_dir = Path(tempfile.mkdtemp(prefix="a2_gocryptfs_secret_"))
    passfile = secret_dir / "passfile"
    handle: gocryptfs.GocryptfsHandle | None = None
    try:
        gocryptfs.write_passfile(passfile, password)
        init = gocryptfs.init_cipher_dir(cipher_dir, passfile, out_dir / "gocryptfs")
        if init.get("returncode") != 0:
            raise RuntimeError("gocryptfs init failed")
        handle = gocryptfs.start_gocryptfs(cipher_dir, mount_dir, passfile, out_dir / "gocryptfs")
        bench_dir = mount_dir / "contract"
        bench_dir.mkdir(parents=True, exist_ok=True)
        prep = aegisq.precreate_fio_file(profile, bench_dir, out_dir / "gocryptfs")
        warmup = run_one_fio(profile, bench_dir, out_dir / "gocryptfs", "warmup")
        measured = run_one_fio(profile, bench_dir, out_dir / "gocryptfs", "measured", trace=True)
        stop = gocryptfs.stop_gocryptfs(handle, mount_dir, out_dir / "gocryptfs")
        handle = None
        return {
            "mode": "gocryptfs",
            "init": init,
            "file_preparation": prep,
            "warmup": warmup["row"],
            "measured": measured["row"],
            "unmount": stop,
            "fio_client_strace": measured.get("strace", {}),
        }
    finally:
        if handle is not None:
            gocryptfs.stop_gocryptfs(handle, mount_dir, out_dir / "gocryptfs")
        shutil.rmtree(cipher_dir, ignore_errors=True)
        shutil.rmtree(mount_dir, ignore_errors=True)
        shutil.rmtree(secret_dir, ignore_errors=True)


def throughput(row: dict[str, Any]) -> float:
    return float(row.get("throughput_mib_s") or 0.0)


def summarize(aegisq_result: dict[str, Any], gocryptfs_result: dict[str, Any]) -> dict[str, Any]:
    a_tput = throughput(aegisq_result["measured"])
    g_tput = throughput(gocryptfs_result["measured"])
    ratio = a_tput / g_tput if g_tput > 0 else 0.0
    durability = aegisq_result.get("durability", {})
    strace_a = aegisq_result["fio_client_strace"].get("syscalls", {})
    strace_g = gocryptfs_result["fio_client_strace"].get("syscalls", {})
    return {
        "aegisq_throughput_mib_s": a_tput,
        "gocryptfs_throughput_mib_s": g_tput,
        "aegisq_to_gocryptfs_ratio": ratio,
        "durability_calls": {
            key: durability.get(key, 0)
            for key in (
                "fdatasync",
                "fsync",
                "syncfs",
                "data_sidecar",
                "journal_sidecar",
                "epoch_log",
                "anchor_file",
                "marker_metadata",
                "parent_dir",
            )
        },
        "fio_client_syscall_calls": {
            "aegisq_total": aegisq_result["fio_client_strace"].get("total", {}).get("calls", 0),
            "gocryptfs_total": gocryptfs_result["fio_client_strace"].get("total", {}).get("calls", 0),
            "aegisq_fsync_family": sum(
                strace_a.get(name, {}).get("calls", 0)
                for name in ("fdatasync", "fsync", "syncfs")
            ),
            "gocryptfs_fsync_family": sum(
                strace_g.get(name, {}).get("calls", 0)
                for name in ("fdatasync", "fsync", "syncfs")
            ),
            "aegisq_futex": strace_a.get("futex", {}).get("calls", 0),
            "gocryptfs_futex": strace_g.get("futex", {}).get("calls", 0),
        },
        "attribution": [
            "AEGIS-Q uses extra authenticated-storage durability boundaries that gocryptfs does not expose in this trace.",
            "fio client syscall counts quantify the same mounted workload issued through each FUSE filesystem.",
            "Daemon-level strace is intentionally not used by default because ptraced fusermount can be rejected by the host.",
            "This short probe is gap attribution, not a replacement for retained full frozen-contract rows.",
        ],
    }


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    durability = summary["durability_calls"]
    syscalls = summary["fio_client_syscall_calls"]
    lines = [
        "# A2 Gap Attribution",
        "",
        "This is a short same-profile attribution probe derived from the frozen filesystem contract.",
        "It is not a replacement for the retained full frozen-contract rows.",
        "",
        "## Result",
        "",
        f"- AEGIS-Q throughput: `{summary['aegisq_throughput_mib_s']:.3f} MiB/s`",
        f"- gocryptfs throughput: `{summary['gocryptfs_throughput_mib_s']:.3f} MiB/s`",
        f"- AEGIS-Q/gocryptfs ratio: `{summary['aegisq_to_gocryptfs_ratio']:.3f}`",
        "",
        "## AEGIS-Q Durability Boundary",
        "",
        f"- fdatasync calls: `{durability['fdatasync']}`",
        f"- syncfs calls: `{durability['syncfs']}`",
        f"- data sidecar publications: `{durability['data_sidecar']}`",
        f"- journal sidecar publications: `{durability['journal_sidecar']}`",
        f"- marker metadata publications: `{durability['marker_metadata']}`",
        "",
        "## Same-Workload Client Syscalls",
        "",
        f"- AEGIS-Q fio-client syscall count: `{syscalls['aegisq_total']}`",
        f"- gocryptfs fio-client syscall count: `{syscalls['gocryptfs_total']}`",
        f"- AEGIS-Q fio-client fsync-family calls: `{syscalls['aegisq_fsync_family']}`",
        f"- gocryptfs fio-client fsync-family calls: `{syscalls['gocryptfs_fsync_family']}`",
        "",
        "## Interpretation",
        "",
        "- The gap is not presented as a surprise throughput bug or a GPU/PQC failure.",
        "- The current strict path pays an authenticated-publication boundary: data sidecar, journal sidecar, and marker/checkpoint publication.",
        "- Collapsing these barriers would require a different crash-ordering proof, not just replacing fdatasync calls with a later syncfs.",
        "- Daemon-level strace is not used by this probe because ptraced fusermount can be rejected by the host.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=aegisq.DEFAULT_CONTRACT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--size", default="64M")
    parser.add_argument("--runtime-s", type=int, default=5)
    parser.add_argument("--password", default="aegisq-a2-gap-password")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.contract = args.contract if args.contract.is_absolute() else ROOT / args.contract
    args.out = args.out if args.out.is_absolute() else ROOT / args.out
    if not args.contract.exists():
        raise SystemExit(f"missing frozen workload contract: {args.contract}")
    for tool in ("fio", "gocryptfs", "strace", aegisq.fusermount_command()):
        if shutil.which(tool) is None and not Path(tool).exists():
            raise SystemExit(f"{tool} is required")
    if not aegisq.FUSE_BIN.exists():
        raise SystemExit(f"missing AEGIS-Q binary: {aegisq.FUSE_BIN}")
    if args.out.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.out} exists; pass --overwrite")
        shutil.rmtree(args.out)
    (args.out / "aegisq").mkdir(parents=True, exist_ok=True)
    (args.out / "gocryptfs").mkdir(parents=True, exist_ok=True)

    profile = derive_profile(args.contract, args.size, args.runtime_s)
    aegisq_result = run_aegisq(profile, args.out, args.password)
    gocryptfs_result = run_gocryptfs(profile, args.out, args.password)
    payload = {
        "overall_pass": bool(aegisq_result["measured"].get("valid"))
        and bool(gocryptfs_result["measured"].get("valid")),
        "scope": "short same-profile A2 gap attribution probe",
        "profile": {
            "source_contract": str(args.contract),
            "profile_id": profile["profile_id"],
            "size": profile["mount_options"]["fio_common"]["size"],
            "runtime": profile["mount_options"]["fio_common"]["runtime"],
            "fdatasync": profile["mount_options"]["fio_common"]["fdatasync"],
            "rw": profile["mount_options"]["fio_common"]["rw"],
            "rwmixread": profile["mount_options"]["fio_common"]["rwmixread"],
        },
        "aegisq": aegisq_result,
        "gocryptfs": gocryptfs_result,
        "summary": summarize(aegisq_result, gocryptfs_result),
    }
    json_path = args.out / "a2_gap_attribution.json"
    md_path = args.out / "a2_gap_attribution.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(build_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "overall_pass": payload["overall_pass"],
                "json": str(json_path),
                "markdown": str(md_path),
                "summary": payload["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
