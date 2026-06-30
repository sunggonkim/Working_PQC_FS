#!/usr/bin/env python3
"""Syscall-exact SQLite crash timing on TPM-backed FUSE.

Each trial creates a fresh TPM-backed FUSE store, commits a one-row SQLite
baseline, then runs the advancing transaction under strace with
`inject=fdatasync:signal=KILL:when=N`.  After the writer is killed at that exact
fdatasync call, the harness unmounts/remounts the same FUSE store and applies a
SQLite recovery oracle.

Accepted outcomes are:
  * previous_committed: baseline row count/digest remains visible;
  * latest_committed: advanced row count/digest is visible;
  * fail_closed: mount or SQLite read rejects the state.

Any other readable digest is silent corruption.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from run_combined_durability_bundle import (
    ROOT,
    sqlite_exec,
    sqlite_try_read,
    start_fuse,
    stop_fuse,
    sudo_password,
    sudo_rmtree,
    wait_for_mount,
)


DEFAULT_OUT = ROOT / "artifacts" / "validation" / "sqlite_syscall_crash_tpm"


ADVANCE_SCRIPT = r"""
import os, pathlib, sqlite3
db_path = pathlib.Path(os.environ["DB_PATH"])
conn = sqlite3.connect(str(db_path), timeout=5.0)
try:
    conn.execute("PRAGMA mmap_size=0")
    conn.execute("PRAGMA locking_mode=EXCLUSIVE")
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA synchronous=EXTRA")
    conn.execute("CREATE TABLE IF NOT EXISTS kv(id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT OR REPLACE INTO kv(id, value) VALUES(2, 'latest')")
    conn.execute("INSERT OR REPLACE INTO kv(id, value) VALUES(3, 'syscall-crash-candidate')")
    conn.commit()
finally:
    conn.close()
"""


def run_sudo_strace_crash(db_path: Path, when: int, trial_dir: Path) -> dict[str, Any]:
    strace_prefix = trial_dir / f"strace_fdatasync_when_{when}"
    stdout_path = trial_dir / f"advance_when_{when}.stdout.txt"
    stderr_path = trial_dir / f"advance_when_{when}.stderr.txt"
    cmd = [
        "sudo",
        "-S",
        "-p",
        "",
        "env",
        f"DB_PATH={db_path}",
        "strace",
        "-ff",
        "-ttt",
        "-T",
        "-s",
        "128",
        "-e",
        "trace=openat,pwrite64,fdatasync,fsync,unlink,rename,close",
        "-e",
        f"inject=fdatasync:signal=KILL:when={when}",
        "-o",
        str(strace_prefix),
        "python3",
        "-c",
        ADVANCE_SCRIPT,
    ]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        input=sudo_password() + "\n",
        text=True,
        stdout=stdout_path.open("w", encoding="utf-8"),
        stderr=stderr_path.open("w", encoding="utf-8"),
        check=False,
    )
    strace_files = sorted(str(p.relative_to(ROOT)) for p in trial_dir.glob(f"{strace_prefix.name}*"))
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": str(stdout_path.relative_to(ROOT)),
        "stderr": str(stderr_path.relative_to(ROOT)),
        "strace_files": strace_files,
        "killed": proc.returncode < 0 or proc.returncode in (137, 159),
    }


def classify_replay(read_result: dict[str, Any] | None,
                    baseline: dict[str, Any],
                    latest: dict[str, Any],
                    mounted: bool) -> dict[str, Any]:
    if not mounted:
        return {"verdict": "fail_closed", "acceptable": True, "detail": "replay mount rejected state"}
    if read_result is None or read_result.get("returncode") != 0:
        return {
            "verdict": "fail_closed",
            "acceptable": True,
            "detail": f"SQLite read failed rc={(read_result or {}).get('returncode')}",
        }
    state = read_result.get("state") or {}
    digest = state.get("digest")
    if digest == (baseline.get("state") or {}).get("digest"):
        return {"verdict": "previous_committed", "acceptable": True, "detail": "baseline digest visible after crash"}
    if digest == (latest.get("state") or {}).get("digest"):
        return {"verdict": "latest_committed", "acceptable": True, "detail": "advanced digest visible after crash"}
    return {
        "verdict": "silent_corruption",
        "acceptable": False,
        "detail": "readable digest matched neither baseline nor advanced state",
    }


def run_trial(out_dir: Path, when: int) -> dict[str, Any]:
    trial_dir = out_dir / f"fdatasync_when_{when}"
    if trial_dir.exists():
        shutil.rmtree(trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)
    storage_dir = Path(tempfile.mkdtemp(prefix=f"syscall_tpm_store_{when}_", dir="/tmp"))
    mount_dir = Path(tempfile.mkdtemp(prefix=f"syscall_tpm_mnt_{when}_", dir="/tmp"))
    live_log = trial_dir / "live_mount"
    replay_log = trial_dir / "replay_mount"
    proc = None
    replay_proc = None

    try:
        proc = start_fuse(storage_dir, mount_dir, live_log)
        if not wait_for_mount(proc, mount_dir):
            raise RuntimeError("baseline TPM-backed FUSE mount did not come up")
        db_path = mount_dir / "syscall_crash.sqlite"
        baseline = sqlite_exec(db_path, "baseline", trial_dir)
        if baseline["returncode"] != 0:
            raise RuntimeError("baseline SQLite setup failed")
        latest = sqlite_exec(db_path, "advance", trial_dir)
        if latest["returncode"] != 0:
            raise RuntimeError("latest SQLite reference setup failed")
        # Reset to the one-row baseline on the same mounted store before the
        # crash-timed advance.  This keeps the latest digest as an oracle while
        # preserving a clean pre-crash state.
        baseline = sqlite_exec(db_path, "baseline", trial_dir)
        if baseline["returncode"] != 0:
            raise RuntimeError("baseline reset failed")

        crash = run_sudo_strace_crash(db_path, when, trial_dir)
        time.sleep(0.2)
        stop_fuse(proc, mount_dir, live_log)
        proc = None

        replay_proc = start_fuse(storage_dir, mount_dir, replay_log)
        mounted = wait_for_mount(replay_proc, mount_dir)
        read_result = sqlite_try_read(db_path, trial_dir) if mounted else None
        verdict = classify_replay(read_result, baseline, latest, mounted)
        stop_fuse(replay_proc, mount_dir, replay_log)
        replay_proc = None

        return {
            "when": when,
            "baseline": baseline,
            "latest_reference": latest,
            "crash": crash,
            "replay": {
                "mounted": mounted,
                "read": read_result,
                **verdict,
            },
        }
    finally:
        stop_fuse(proc, mount_dir, live_log)
        stop_fuse(replay_proc, mount_dir, replay_log)
        sudo_rmtree(storage_dir)
        shutil.rmtree(mount_dir, ignore_errors=True)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# SQLite syscall-exact TPM crash campaign",
        "",
        "Each trial runs SQLite on TPM-backed FUSE and kills the advancing transaction at an exact `fdatasync` call using `strace --inject`.",
        "",
        f"- Trials: `{len(report['trials'])}`",
        f"- Acceptable trials: `{report['summary']['acceptable_trials']}`",
        f"- Unacceptable trials: `{report['summary']['unacceptable_trials']}`",
        "",
        "## Per-trial verdicts",
        "",
    ]
    for trial in report["trials"]:
        replay = trial["replay"]
        crash = trial["crash"]
        lines.append(
            f"- fdatasync when={trial['when']}: verdict `{replay['verdict']}`, "
            f"acceptable `{replay['acceptable']}`, writer rc `{crash['returncode']}`, "
            f"strace files `{len(crash['strace_files'])}`"
        )
    lines.extend([
        "",
        "Conservative interpretation: this is syscall-exact app-crash timing for SQLite on the TPM-backed mounted FUSE path. It does not model power loss of the FUSE daemon or arbitrary kernel-level interruption.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--when", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trials = [run_trial(out_dir, when) for when in args.when]
    acceptable = sum(1 for trial in trials if trial["replay"]["acceptable"])
    report = {
        "note": "Syscall-exact SQLite app-crash timing on TPM-backed FUSE using strace fdatasync SIGKILL injection.",
        "trials": trials,
        "summary": {
            "trial_count": len(trials),
            "acceptable_trials": acceptable,
            "unacceptable_trials": len(trials) - acceptable,
            "all_acceptable": acceptable == len(trials),
        },
    }
    (out_dir / "sqlite_syscall_crash_tpm.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, out_dir / "sqlite_syscall_crash_tpm.md")
    print(json.dumps({"out_dir": str(out_dir), **report["summary"]}, indent=2))
    return 0 if report["summary"]["all_acceptable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
