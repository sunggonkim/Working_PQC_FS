#!/usr/bin/env python3
"""Run a deterministic SQLite fault-injection campaign.

This harness turns the SQLite durable-boundary oracle into executable evidence.
It deliberately stays conservative: faults are injected by mutating retained
SQLite database/WAL files at named durable-boundary states, not by claiming
cycle-accurate process-kill timing inside SQLite or the filesystem.

Acceptable oracle outcomes are:
  * the previous committed state is recovered,
  * the latest committed state is recovered, or
  * SQLite fails closed instead of returning silently corrupted content.
Any other readable state is classified as silent_corruption.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sqlite3
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "sqlite_fault_campaign"


@dataclass
class StateDigest:
    label: str
    row_count: int
    digest: str
    rows: list[tuple[int, str]]


@dataclass
class Trial:
    cut_point: str
    trial: int
    mutation: str
    verdict: str
    detail: str
    integrity_check: str | None
    row_count: int | None
    digest: str | None
    acceptable: bool


def sha_rows(rows: list[tuple[int, str]]) -> str:
    h = hashlib.sha256()
    for row_id, value in rows:
        h.update(f"{row_id}\0{value}\n".encode("utf-8"))
    return h.hexdigest()


def read_state(db_path: Path, label: str) -> StateDigest:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = [(int(row[0]), str(row[1])) for row in conn.execute("SELECT id, value FROM kv ORDER BY id")]
        return StateDigest(label=label, row_count=len(rows), digest=sha_rows(rows), rows=rows)
    finally:
        conn.close()


def init_sqlite_state(work_dir: Path) -> tuple[Path, Path, StateDigest, StateDigest]:
    db_path = work_dir / "campaign.db"
    previous_dir = work_dir / "previous"
    latest_dir = work_dir / "latest"
    previous_dir.mkdir()
    latest_dir.mkdir()

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.execute("CREATE TABLE kv(id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute("INSERT INTO kv(id, value) VALUES(1, 'baseline')")
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.commit()
        previous = read_state(db_path, "previous")
        copy_sqlite_files(db_path, previous_dir)

        conn.execute("INSERT INTO kv(id, value) VALUES(2, 'latest')")
        conn.commit()
        latest = read_state(db_path, "latest")
        copy_sqlite_files(db_path, latest_dir)
    finally:
        conn.close()

    return db_path, previous_dir, previous, latest


def copy_sqlite_files(db_path: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm", "-journal"):
        src = Path(str(db_path) + suffix)
        if src.exists():
            shutil.copy2(src, dst / src.name)


def materialize_case(src_dir: Path, case_dir: Path) -> Path:
    if case_dir.exists():
        shutil.rmtree(case_dir)
    shutil.copytree(src_dir, case_dir)
    return case_dir / "campaign.db"


def truncate_file(path: Path, size: int) -> None:
    if path.exists():
        with path.open("r+b") as f:
            f.truncate(max(0, min(size, path.stat().st_size)))


def flip_byte(path: Path, offset: int) -> bool:
    if not path.exists() or path.stat().st_size <= offset:
        return False
    with path.open("r+b") as f:
        f.seek(offset)
        b = f.read(1)
        if not b:
            return False
        f.seek(offset)
        f.write(bytes([b[0] ^ 0xFF]))
    return True


def inject_fault(db_path: Path, cut_point: str) -> str:
    wal_path = Path(str(db_path) + "-wal")
    shm_path = Path(str(db_path) + "-shm")
    journal_path = Path(str(db_path) + "-journal")
    if cut_point == "before_sqlite_journal_header_sync":
        # Simulate a crash before a durable WAL/journal record is usable.
        truncate_file(wal_path, 0)
        truncate_file(journal_path, 0)
        return "truncate_wal_and_journal_to_zero"
    if cut_point == "after_journal_header_before_db_sync":
        # Keep the WAL header but remove frame payload, matching a partial record.
        truncate_file(wal_path, 32)
        return "retain_wal_header_only"
    if cut_point == "after_db_fsync_before_journal_unlink":
        # Corrupt the main DB image after it could have been written but before
        # sidecar cleanup; SQLite should recover from WAL or fail closed.
        changed = flip_byte(db_path, 128)
        return "flip_main_db_byte_128" if changed else "main_db_too_small_noop"
    if cut_point == "wal_file_created_before_checkpoint":
        # Remove SHM and truncate the WAL mid-file to exercise WAL recovery.
        if shm_path.exists():
            shm_path.unlink()
        if wal_path.exists():
            truncate_file(wal_path, max(32, wal_path.stat().st_size // 2))
        return "drop_shm_and_truncate_wal_midpoint"
    raise ValueError(f"unknown cut point: {cut_point}")


def run_oracle(db_path: Path, previous: StateDigest, latest: StateDigest) -> tuple[str, str, str | None, int | None, str | None, bool]:
    try:
        conn = sqlite3.connect(db_path)
        try:
            integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
            rows = [(int(row[0]), str(row[1])) for row in conn.execute("SELECT id, value FROM kv ORDER BY id")]
            digest = sha_rows(rows)
            row_count = len(rows)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return "fail_closed", f"sqlite_error={exc}", None, None, None, True
    except OSError as exc:
        return "fail_closed", f"os_error={exc}", None, None, None, True

    if integrity != "ok":
        return "fail_closed", f"integrity_check={integrity}", integrity, row_count, digest, True
    if digest == previous.digest and row_count == previous.row_count:
        return "previous_committed", "digest_matches_previous", integrity, row_count, digest, True
    if digest == latest.digest and row_count == latest.row_count:
        return "latest_committed", "digest_matches_latest", integrity, row_count, digest, True
    return "silent_corruption", "readable_digest_matches_neither_previous_nor_latest", integrity, row_count, digest, False


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# SQLite fault-injection campaign",
        "",
        "This report executes deterministic SQLite file-state fault injection against the durable-boundary oracle.",
        "It does not claim syscall-exact crash timing or complete application crash certification.",
        "",
        "## Reference states",
        "",
    ]
    for state in (report["previous_state"], report["latest_state"]):
        lines.extend([
            f"- {state['label']}: rows={state['row_count']}, sha256={state['digest']}",
        ])
    lines.extend(["", "## Summary", ""])
    for row in report["summary"]:
        lines.append(
            f"- {row['cut_point']}: trials={row['trials']}, acceptable={row['acceptable']}, "
            f"silent_corruption={row['silent_corruption']}, verdicts={row['verdicts']}"
        )
    lines.extend(["", "## Conservative interpretation", ""])
    lines.extend([
        "- This closes a SQLite-only per-cut oracle execution for the selected durable-boundary states.",
        "- It does not close the broader two-workload app-recovery requirement.",
        "- It does not prove crash behavior for arbitrary interruption times inside the filesystem.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize(trials: list[Trial]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Trial]] = {}
    for trial in trials:
        grouped.setdefault(trial.cut_point, []).append(trial)
    summary = []
    for cut_point, rows in sorted(grouped.items()):
        verdicts: dict[str, int] = {}
        for row in rows:
            verdicts[row.verdict] = verdicts.get(row.verdict, 0) + 1
        summary.append({
            "cut_point": cut_point,
            "trials": len(rows),
            "acceptable": sum(1 for row in rows if row.acceptable),
            "silent_corruption": sum(1 for row in rows if row.verdict == "silent_corruption"),
            "verdicts": verdicts,
        })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--trials-per-cutpoint", type=int, default=5)
    parser.add_argument(
        "--cut-points",
        nargs="+",
        default=[
            "before_sqlite_journal_header_sync",
            "after_journal_header_before_db_sync",
            "after_db_fsync_before_journal_unlink",
            "wal_file_created_before_checkpoint",
        ],
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trials: list[Trial] = []
    with tempfile.TemporaryDirectory(prefix="aegis_sqlite_fault_") as tmp:
        work_dir = Path(tmp)
        _, latest_dir, previous, latest = init_sqlite_state(work_dir)
        for cut_point in args.cut_points:
            for trial_idx in range(args.trials_per_cutpoint):
                case_dir = work_dir / f"{cut_point}_{trial_idx}"
                case_db = materialize_case(latest_dir, case_dir)
                mutation = inject_fault(case_db, cut_point)
                verdict, detail, integrity, row_count, digest, acceptable = run_oracle(case_db, previous, latest)
                trials.append(Trial(
                    cut_point=cut_point,
                    trial=trial_idx,
                    mutation=mutation,
                    verdict=verdict,
                    detail=detail,
                    integrity_check=integrity,
                    row_count=row_count,
                    digest=digest,
                    acceptable=acceptable,
                ))

    report = {
        "note": "SQLite-only file-state fault campaign; not full app crash certification.",
        "trials_per_cutpoint": args.trials_per_cutpoint,
        "previous_state": asdict(previous),
        "latest_state": asdict(latest),
        "rows": [asdict(row) for row in trials],
        "summary": summarize(trials),
    }
    json_path = args.out_dir / "sqlite_fault_campaign.json"
    csv_path = args.out_dir / "sqlite_fault_campaign.csv"
    md_path = args.out_dir / "sqlite_fault_campaign.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(trials[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in trials)
    write_markdown(report, md_path)
    failed = sum(1 for row in trials if not row.acceptable)
    print(json.dumps({"out_dir": str(args.out_dir), "trials": len(trials), "unacceptable": failed}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
