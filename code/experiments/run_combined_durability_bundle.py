#!/usr/bin/env python3
"""Run combined TPM freshness and SQLite recovery evidence.

This script keeps the previous orchestration checks, then adds same-backing-store
campaigns:

  1. mount pqc_fuse with the hardware TPM anchor backend;
  2. create a SQLite database inside that mounted filesystem;
  3. unmount and snapshot the entire encrypted backing store;
  4. remount, advance the SQLite state, and therefore advance the TPM-backed
     freshness anchor;
  5. restore the stale backing-store snapshot and attempt to remount/read the
     SQLite database.

The supported claim is narrow.  A fail-closed result means the stale application
backing store was rejected after the hardware anchor advanced.  SQLite and
dbm.dumb cover two application formats; this is not RocksDB and not
syscall-exact crash timing.
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
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FUSE_BIN = ROOT / "build" / "pqc_fuse"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "combined_durability_bundle"


@dataclass
class SQLiteState:
    label: str
    row_count: int
    digest: str
    rows: list[list[Any]]


def sudo_password() -> str:
    password = os.environ.get("PQC_SUDO_PASSWORD")
    if not password:
        raise RuntimeError("PQC_SUDO_PASSWORD is required for the combined durability campaign")
    return password


def run_cmd(cmd: list[str], out_dir: Path, name: str) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout = out_dir / f"{name}.stdout.txt"
    stderr = out_dir / f"{name}.stderr.txt"
    stdout.write_text(proc.stdout, encoding="utf-8")
    stderr.write_text(proc.stderr, encoding="utf-8")
    return {
        "name": name,
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": str(stdout),
        "stderr": str(stderr),
    }


def run_sudo(cmd: list[str], *, cwd: Path, stdout_path: Path, stderr_path: Path) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        ["sudo", "-S", "-p", "", *cmd],
        cwd=cwd,
        input=sudo_password() + "\n",
        text=True,
        stdout=stdout_path.open("w", encoding="utf-8"),
        stderr=stderr_path.open("w", encoding="utf-8"),
        check=False,
    )
    return proc


def mount_is_visible(mount_dir: Path) -> bool:
    mount_path = mount_dir.resolve()
    try:
        with open("/proc/mounts", "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                fields = line.split()
                if len(fields) >= 3 and Path(fields[1]).resolve() == mount_path and fields[2].startswith("fuse"):
                    return True
    except FileNotFoundError:
        pass
    return subprocess.run(["mountpoint", "-q", str(mount_dir)], check=False).returncode == 0


def wait_for_mount(proc: subprocess.Popen[str], mount_dir: Path, timeout_s: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        if mount_is_visible(mount_dir):
            return True
        time.sleep(0.05)
    return False


def start_fuse(storage_dir: Path, mount_dir: Path, log_dir: Path) -> subprocess.Popen[str]:
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PQC_MASTER_PASSWORD"] = env.get("PQC_MASTER_PASSWORD", "benchmark-password")
    env["PQC_FRESHNESS_ANCHOR_BACKEND"] = "hardware"
    env.setdefault("PQC_FRESHNESS_ANCHOR_PATH", str(storage_dir / ".anchor"))
    env.setdefault("PQC_TPM_TCTI", "device:/dev/tpmrm0")
    env.setdefault("PQC_FRESHNESS_WINDOW_N", "1")
    env.setdefault("PQC_KEY_ROTATION_INTERVAL_S", "0")

    cmd = [
        "sudo",
        "-S",
        "-p",
        "",
        "env",
        f"PQC_MASTER_PASSWORD={env['PQC_MASTER_PASSWORD']}",
        "PQC_FRESHNESS_ANCHOR_BACKEND=hardware",
        f"PQC_FRESHNESS_ANCHOR_PATH={env['PQC_FRESHNESS_ANCHOR_PATH']}",
        f"PQC_TPM_TCTI={env['PQC_TPM_TCTI']}",
        f"PQC_FRESHNESS_WINDOW_N={env['PQC_FRESHNESS_WINDOW_N']}",
        f"PQC_KEY_ROTATION_INTERVAL_S={env['PQC_KEY_ROTATION_INTERVAL_S']}",
        str(FUSE_BIN),
        str(storage_dir),
        str(mount_dir),
        "-f",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdin=subprocess.PIPE,
        stdout=(log_dir / "pqc_fuse.stdout.txt").open("w", encoding="utf-8"),
        stderr=(log_dir / "pqc_fuse.stderr.txt").open("w", encoding="utf-8"),
        text=True,
    )
    assert proc.stdin is not None
    proc.stdin.write(sudo_password() + "\n")
    proc.stdin.flush()
    proc.stdin.close()
    return proc


def stop_fuse(proc: subprocess.Popen[str] | None, mount_dir: Path, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    if proc is None:
        return
    if proc.poll() is None:
        unmount = run_sudo(
            ["fusermount3", "-u", str(mount_dir)],
            cwd=ROOT,
            stdout_path=log_dir / "unmount.stdout.txt",
            stderr_path=log_dir / "unmount.stderr.txt",
        )
        if unmount.returncode != 0:
            run_sudo(
                ["umount", str(mount_dir)],
                cwd=ROOT,
                stdout_path=log_dir / "umount.stdout.txt",
                stderr_path=log_dir / "umount.stderr.txt",
            )
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def sudo_copytree(src: Path, dst: Path) -> None:
    subprocess.run(
        [
            "sudo",
            "-S",
            "-p",
            "",
            "python3",
            "-c",
            (
                "import pathlib, shutil; "
                f"src=pathlib.Path({str(src)!r}); dst=pathlib.Path({str(dst)!r}); "
                "shutil.rmtree(dst, ignore_errors=True); "
                "shutil.copytree(src, dst, symlinks=True)"
            ),
        ],
        input=sudo_password() + "\n",
        text=True,
        check=True,
    )


def sudo_rmtree(path: Path) -> None:
    subprocess.run(
        ["sudo", "-S", "-p", "", "rm", "-rf", str(path)],
        input=sudo_password() + "\n",
        text=True,
        check=True,
    )


def state_digest(rows: list[list[Any]]) -> str:
    h = hashlib.sha256()
    for row_id, value in rows:
        h.update(f"{row_id}\0{value}\n".encode("utf-8"))
    return h.hexdigest()


def sqlite_exec(db_path: Path, phase: str, out_dir: Path) -> dict[str, Any]:
    script = r"""
import hashlib, json, os, pathlib, sqlite3, sys
db_path = pathlib.Path(os.environ["DB_PATH"])
phase = os.environ["PHASE"]
conn = sqlite3.connect(str(db_path), timeout=5.0)
try:
    conn.execute("PRAGMA mmap_size=0")
    conn.execute("PRAGMA locking_mode=EXCLUSIVE")
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA synchronous=EXTRA")
    conn.execute("CREATE TABLE IF NOT EXISTS kv(id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
    if phase == "baseline":
        conn.execute("DELETE FROM kv")
        conn.execute("INSERT INTO kv(id, value) VALUES(1, 'baseline')")
    elif phase == "advance":
        conn.execute("INSERT OR REPLACE INTO kv(id, value) VALUES(2, 'latest')")
        conn.execute("INSERT OR REPLACE INTO kv(id, value) VALUES(3, 'anchor-advanced')")
    else:
        raise ValueError(f"unknown phase {phase}")
    conn.commit()
    rows = [[int(r[0]), str(r[1])] for r in conn.execute("SELECT id, value FROM kv ORDER BY id")]
    h = hashlib.sha256()
    for row_id, value in rows:
        h.update(f"{row_id}\0{value}\n".encode("utf-8"))
    print(json.dumps({
        "phase": phase,
        "row_count": len(rows),
        "digest": h.hexdigest(),
        "rows": rows,
        "integrity_check": conn.execute("PRAGMA integrity_check").fetchone()[0],
    }))
finally:
    conn.close()
"""
    proc = subprocess.run(
        ["sudo", "-S", "-p", "", "env", f"DB_PATH={db_path}", f"PHASE={phase}", "python3", "-c", script],
        input=sudo_password() + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
    (out_dir / f"sqlite_{phase}.stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (out_dir / f"sqlite_{phase}.stderr.txt").write_text(proc.stderr, encoding="utf-8")
    result: dict[str, Any] = {
        "phase": phase,
        "returncode": proc.returncode,
        "stdout": str(out_dir / f"sqlite_{phase}.stdout.txt"),
        "stderr": str(out_dir / f"sqlite_{phase}.stderr.txt"),
    }
    if proc.returncode == 0:
        result["state"] = json.loads(proc.stdout.strip().splitlines()[-1])
    return result


def sqlite_try_read(db_path: Path, out_dir: Path) -> dict[str, Any]:
    script = r"""
import hashlib, json, os, pathlib, sqlite3
db_path = pathlib.Path(os.environ["DB_PATH"])
conn = sqlite3.connect(str(db_path), timeout=5.0)
try:
    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    rows = [[int(r[0]), str(r[1])] for r in conn.execute("SELECT id, value FROM kv ORDER BY id")]
    h = hashlib.sha256()
    for row_id, value in rows:
        h.update(f"{row_id}\0{value}\n".encode("utf-8"))
    print(json.dumps({"integrity_check": integrity, "row_count": len(rows), "digest": h.hexdigest(), "rows": rows}))
finally:
    conn.close()
"""
    proc = subprocess.run(
        ["sudo", "-S", "-p", "", "env", f"DB_PATH={db_path}", "python3", "-c", script],
        input=sudo_password() + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
    (out_dir / "sqlite_replay_read.stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (out_dir / "sqlite_replay_read.stderr.txt").write_text(proc.stderr, encoding="utf-8")
    result: dict[str, Any] = {
        "returncode": proc.returncode,
        "stdout": str(out_dir / "sqlite_replay_read.stdout.txt"),
        "stderr": str(out_dir / "sqlite_replay_read.stderr.txt"),
        "stderr_tail": proc.stderr[-4000:],
    }
    if proc.returncode == 0 and proc.stdout.strip():
        result["state"] = json.loads(proc.stdout.strip().splitlines()[-1])
    return result


def dbm_exec(db_path: Path, phase: str, out_dir: Path) -> dict[str, Any]:
    script = r"""
import dbm.dumb, hashlib, json, os, pathlib
dbm.dumb._Database._chmod = lambda self, file: None
db_path = pathlib.Path(os.environ["DB_PATH"])
phase = os.environ["PHASE"]
with dbm.dumb.open(str(db_path), "c") as db:
    if phase == "baseline":
        for key in list(db.keys()):
            del db[key]
        db[b"1"] = b"baseline"
    elif phase == "advance":
        db[b"2"] = b"latest"
        db[b"3"] = b"anchor-advanced"
    else:
        raise ValueError(f"unknown phase {phase}")
    try:
        db.sync()
    except AttributeError:
        pass
with dbm.dumb.open(str(db_path), "r") as db:
    rows = [[key.decode("utf-8"), db[key].decode("utf-8")] for key in sorted(db.keys())]
h = hashlib.sha256()
for key, value in rows:
    h.update(f"{key}\0{value}\n".encode("utf-8"))
print(json.dumps({
    "phase": phase,
    "row_count": len(rows),
    "digest": h.hexdigest(),
    "rows": rows,
}))
"""
    proc = subprocess.run(
        ["sudo", "-S", "-p", "", "env", f"DB_PATH={db_path}", f"PHASE={phase}", "python3", "-c", script],
        input=sudo_password() + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
    (out_dir / f"dbm_{phase}.stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (out_dir / f"dbm_{phase}.stderr.txt").write_text(proc.stderr, encoding="utf-8")
    result: dict[str, Any] = {
        "phase": phase,
        "returncode": proc.returncode,
        "stdout": str(out_dir / f"dbm_{phase}.stdout.txt"),
        "stderr": str(out_dir / f"dbm_{phase}.stderr.txt"),
    }
    if proc.returncode == 0:
        result["state"] = json.loads(proc.stdout.strip().splitlines()[-1])
    return result


def dbm_try_read(db_path: Path, out_dir: Path) -> dict[str, Any]:
    script = r"""
import dbm.dumb, hashlib, json, os, pathlib
dbm.dumb._Database._chmod = lambda self, file: None
db_path = pathlib.Path(os.environ["DB_PATH"])
with dbm.dumb.open(str(db_path), "r") as db:
    rows = [[key.decode("utf-8"), db[key].decode("utf-8")] for key in sorted(db.keys())]
h = hashlib.sha256()
for key, value in rows:
    h.update(f"{key}\0{value}\n".encode("utf-8"))
print(json.dumps({"row_count": len(rows), "digest": h.hexdigest(), "rows": rows}))
"""
    proc = subprocess.run(
        ["sudo", "-S", "-p", "", "env", f"DB_PATH={db_path}", "python3", "-c", script],
        input=sudo_password() + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
    (out_dir / "dbm_replay_read.stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (out_dir / "dbm_replay_read.stderr.txt").write_text(proc.stderr, encoding="utf-8")
    result: dict[str, Any] = {
        "returncode": proc.returncode,
        "stdout": str(out_dir / "dbm_replay_read.stdout.txt"),
        "stderr": str(out_dir / "dbm_replay_read.stderr.txt"),
        "stderr_tail": proc.stderr[-4000:],
    }
    if proc.returncode == 0 and proc.stdout.strip():
        result["state"] = json.loads(proc.stdout.strip().splitlines()[-1])
    return result


def tail(path: Path, limit: int = 5000) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-limit:]


def run_unified_campaign(out_dir: Path) -> dict[str, Any]:
    storage_dir = Path(tempfile.mkdtemp(prefix="combined_dur_store_", dir="/tmp"))
    mount_dir = Path(tempfile.mkdtemp(prefix="combined_dur_mnt_", dir="/tmp"))
    snapshot_dir = Path(tempfile.mkdtemp(prefix="combined_dur_snapshot_", dir="/tmp"))
    live1_log = out_dir / "unified_live_mount"
    replay_log = out_dir / "unified_replay_mount"
    proc: subprocess.Popen[str] | None = None
    replay_proc: subprocess.Popen[str] | None = None

    try:
        proc = start_fuse(storage_dir, mount_dir, live1_log)
        if not wait_for_mount(proc, mount_dir):
            raise RuntimeError("baseline TPM-backed FUSE mount did not come up")
        db_path = mount_dir / "durability.sqlite"
        baseline = sqlite_exec(db_path, "baseline", out_dir)
        if baseline["returncode"] != 0:
            raise RuntimeError("baseline SQLite workload failed")

        run_sudo(
            ["sync"],
            cwd=ROOT,
            stdout_path=out_dir / "sync_after_baseline.stdout.txt",
            stderr_path=out_dir / "sync_after_baseline.stderr.txt",
        )
        sudo_copytree(storage_dir, snapshot_dir)

        advanced = sqlite_exec(db_path, "advance", out_dir)
        if advanced["returncode"] != 0:
            raise RuntimeError("advance SQLite workload failed")
        stop_fuse(proc, mount_dir, live1_log)
        proc = None

        sudo_rmtree(storage_dir)
        sudo_copytree(snapshot_dir, storage_dir)

        replay_proc = start_fuse(storage_dir, mount_dir, replay_log)
        mounted = wait_for_mount(replay_proc, mount_dir)
        replay_read: dict[str, Any] | None = None
        verdict = "fail_closed"
        detail = "stale SQLite backing-store snapshot rejected at mount"
        if mounted:
            replay_read = sqlite_try_read(db_path, out_dir)
            if replay_read["returncode"] != 0:
                verdict = "fail_closed"
                detail = f"stale snapshot mounted but SQLite read failed closed rc={replay_read['returncode']}"
            else:
                state = replay_read.get("state") or {}
                digest = state.get("digest")
                if digest == (baseline.get("state") or {}).get("digest"):
                    verdict = "rollback_visible"
                    detail = "stale baseline SQLite state remained readable after hardware anchor advanced"
                elif digest == (advanced.get("state") or {}).get("digest"):
                    verdict = "unexpected_latest_visible"
                    detail = "latest SQLite state visible after stale snapshot restore"
                else:
                    verdict = "silent_corruption"
                    detail = "readable SQLite state matched neither baseline nor advanced digest"
            stop_fuse(replay_proc, mount_dir, replay_log)
            replay_proc = None
        elif replay_proc.poll() is None:
            stop_fuse(replay_proc, mount_dir, replay_log)
            replay_proc = None

        return {
            "note": "Same-backing-store SQLite + TPM replay-after-advance campaign.",
            "storage_dir": str(storage_dir),
            "snapshot_dir": str(snapshot_dir),
            "mount_dir": str(mount_dir),
            "baseline": baseline,
            "advanced": advanced,
            "replay": {
                "mounted": mounted,
                "verdict": verdict,
                "acceptable": verdict == "fail_closed",
                "detail": detail,
                "read": replay_read,
                "stderr_tail": tail(replay_log / "pqc_fuse.stderr.txt"),
            },
            "logs": {
                "live_mount_stderr": str(live1_log / "pqc_fuse.stderr.txt"),
                "replay_mount_stderr": str(replay_log / "pqc_fuse.stderr.txt"),
            },
        }
    finally:
        stop_fuse(proc, mount_dir, live1_log)
        stop_fuse(replay_proc, mount_dir, replay_log)
        sudo_rmtree(storage_dir)
        sudo_rmtree(snapshot_dir)
        shutil.rmtree(mount_dir, ignore_errors=True)


def run_unified_dbm_campaign(out_dir: Path) -> dict[str, Any]:
    storage_dir = Path(tempfile.mkdtemp(prefix="combined_dbm_store_", dir="/tmp"))
    mount_dir = Path(tempfile.mkdtemp(prefix="combined_dbm_mnt_", dir="/tmp"))
    snapshot_dir = Path(tempfile.mkdtemp(prefix="combined_dbm_snapshot_", dir="/tmp"))
    live_log = out_dir / "unified_dbm_live_mount"
    replay_log = out_dir / "unified_dbm_replay_mount"
    proc: subprocess.Popen[str] | None = None
    replay_proc: subprocess.Popen[str] | None = None

    try:
        proc = start_fuse(storage_dir, mount_dir, live_log)
        if not wait_for_mount(proc, mount_dir):
            raise RuntimeError("baseline TPM-backed FUSE mount did not come up for dbm")
        db_path = mount_dir / "durability_ndbm"
        baseline = dbm_exec(db_path, "baseline", out_dir)
        if baseline["returncode"] != 0:
            raise RuntimeError("baseline dbm workload failed")

        run_sudo(
            ["sync"],
            cwd=ROOT,
            stdout_path=out_dir / "sync_after_dbm_baseline.stdout.txt",
            stderr_path=out_dir / "sync_after_dbm_baseline.stderr.txt",
        )
        sudo_copytree(storage_dir, snapshot_dir)

        advanced = dbm_exec(db_path, "advance", out_dir)
        if advanced["returncode"] != 0:
            raise RuntimeError("advance dbm workload failed")
        stop_fuse(proc, mount_dir, live_log)
        proc = None

        sudo_rmtree(storage_dir)
        sudo_copytree(snapshot_dir, storage_dir)

        replay_proc = start_fuse(storage_dir, mount_dir, replay_log)
        mounted = wait_for_mount(replay_proc, mount_dir)
        replay_read: dict[str, Any] | None = None
        verdict = "fail_closed"
        detail = "stale dbm backing-store snapshot rejected at mount"
        if mounted:
            replay_read = dbm_try_read(db_path, out_dir)
            if replay_read["returncode"] != 0:
                verdict = "fail_closed"
                detail = f"stale dbm snapshot mounted but dbm read failed closed rc={replay_read['returncode']}"
            else:
                state = replay_read.get("state") or {}
                digest = state.get("digest")
                if digest == (baseline.get("state") or {}).get("digest"):
                    verdict = "rollback_visible"
                    detail = "stale baseline dbm state remained readable after hardware anchor advanced"
                elif digest == (advanced.get("state") or {}).get("digest"):
                    verdict = "unexpected_latest_visible"
                    detail = "latest dbm state visible after stale snapshot restore"
                else:
                    verdict = "silent_corruption"
                    detail = "readable dbm state matched neither baseline nor advanced digest"
            stop_fuse(replay_proc, mount_dir, replay_log)
            replay_proc = None
        elif replay_proc.poll() is None:
            stop_fuse(replay_proc, mount_dir, replay_log)
            replay_proc = None

        return {
            "note": "Same-backing-store dbm.dumb + TPM replay-after-advance campaign.",
            "storage_dir": str(storage_dir),
            "snapshot_dir": str(snapshot_dir),
            "mount_dir": str(mount_dir),
            "baseline": baseline,
            "advanced": advanced,
            "replay": {
                "mounted": mounted,
                "verdict": verdict,
                "acceptable": verdict == "fail_closed",
                "detail": detail,
                "read": replay_read,
                "stderr_tail": tail(replay_log / "pqc_fuse.stderr.txt"),
            },
            "logs": {
                "live_mount_stderr": str(live_log / "pqc_fuse.stderr.txt"),
                "replay_mount_stderr": str(replay_log / "pqc_fuse.stderr.txt"),
            },
        }
    finally:
        stop_fuse(proc, mount_dir, live_log)
        stop_fuse(replay_proc, mount_dir, replay_log)
        sudo_rmtree(storage_dir)
        sudo_rmtree(snapshot_dir)
        shutil.rmtree(mount_dir, ignore_errors=True)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    unified = report.get("unified_campaign") or {}
    dbm_unified = report.get("unified_dbm_campaign") or {}
    replay = unified.get("replay") or {}
    dbm_replay = dbm_unified.get("replay") or {}
    lines = [
        "# Combined durability bundle",
        "",
        "This bundle retains the previous TPM-only and app-only orchestration checks and adds same-backing-store SQLite+TPM and dbm.dumb+TPM campaigns.",
        "",
        "## Orchestration checks",
        "",
    ]
    for check in report.get("checks", []):
        lines.append(f"- `{check['name']}`: returncode `{check['returncode']}`")
    lines.extend([
        "",
        "## Unified same-backing-store campaign",
        "",
        f"- Present: `{bool(unified)}`",
        f"- Verdict: `{replay.get('verdict')}`",
        f"- Acceptable: `{replay.get('acceptable')}`",
        f"- Detail: `{replay.get('detail')}`",
        f"- Replay mounted: `{replay.get('mounted')}`",
        f"- Baseline rows: `{((unified.get('baseline') or {}).get('state') or {}).get('row_count')}`",
        f"- Advanced rows: `{((unified.get('advanced') or {}).get('state') or {}).get('row_count')}`",
        "",
        "## Unified second-workload campaign",
        "",
        f"- Present: `{bool(dbm_unified)}`",
        f"- Workload: `dbm.dumb` key-value store",
        f"- Verdict: `{dbm_replay.get('verdict')}`",
        f"- Acceptable: `{dbm_replay.get('acceptable')}`",
        f"- Detail: `{dbm_replay.get('detail')}`",
        f"- Replay mounted: `{dbm_replay.get('mounted')}`",
        f"- Baseline rows: `{((dbm_unified.get('baseline') or {}).get('state') or {}).get('row_count')}`",
        f"- Advanced rows: `{((dbm_unified.get('advanced') or {}).get('state') or {}).get('row_count')}`",
        "",
        "Conservative interpretation: this supports fail-closed stale-snapshot results for SQLite and dbm.dumb on TPM-backed FUSE backing stores. It does not establish RocksDB coverage, syscall-exact crash timing, or arbitrary interruption safety.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--skip-orchestration", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, Any]] = []
    if not args.skip_orchestration:
        checks.append(run_cmd(["python3", "code/experiments/run_tpm_only_bundle.py"], out_dir, "tpm_only_bundle"))
        checks.append(run_cmd(["python3", "code/experiments/run_app_recovery_bundle.py"], out_dir, "app_recovery_bundle"))

    report: dict[str, Any] = {
        "note": "Combined durability bundle with same-backing-store SQLite+TPM and dbm.dumb+TPM campaigns.",
        "checks": checks,
        "unified_campaign": None,
        "unified_dbm_campaign": None,
    }
    try:
        report["unified_campaign"] = run_unified_campaign(out_dir)
    except Exception as exc:
        report["unified_error"] = str(exc)
    try:
        report["unified_dbm_campaign"] = run_unified_dbm_campaign(out_dir)
    except Exception as exc:
        report["unified_dbm_error"] = str(exc)

    json_path = out_dir / "combined_durability_bundle.json"
    md_path = out_dir / "combined_durability_bundle.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)

    unified = report.get("unified_campaign") or {}
    dbm_unified = report.get("unified_dbm_campaign") or {}
    acceptable = ((unified.get("replay") or {}).get("acceptable") is True)
    dbm_acceptable = ((dbm_unified.get("replay") or {}).get("acceptable") is True)
    orchestration_ok = all(check["returncode"] == 0 for check in checks)
    print(json.dumps({
        "out_dir": str(out_dir),
        "checks": len(checks),
        "orchestration_ok": orchestration_ok,
        "unified_acceptable": acceptable,
        "unified_dbm_acceptable": dbm_acceptable,
        "unified_error": report.get("unified_error"),
        "unified_dbm_error": report.get("unified_dbm_error"),
    }, indent=2))
    return 0 if acceptable and dbm_acceptable and orchestration_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
