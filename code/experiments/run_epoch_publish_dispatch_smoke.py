#!/usr/bin/env python3
"""Mounted-path smoke for Gate 0.9-S0 epoch-publication dispatch.

This runner proves only the skeleton contract: strict remains the default
mounted publication path and explicit strict mode still commits through the
dispatcher. Later runners cover any epoch redo-log behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_parallel_commit_epoch_path_smoke import (  # noqa: E402
    FUSE_BIN,
    ROOT,
    mounted_write_read,
    relpath,
    start_fuse,
    stop_fuse,
)


DEFAULT_OUT = ROOT / "artifacts" / "validation" / "publication_protocol_fault_matrix"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_jsonl(path: Path) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    malformed = 0
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(payload, dict):
                events.append(payload)
    return {
        "path": relpath(path),
        "exists": path.exists(),
        "event_count": len(events),
        "malformed_line_count": malformed,
        "events": events,
        "modes": sorted({str(event.get("mode", "")) for event in events
                         if event.get("mode")}),
    }


def source_evidence() -> dict[str, Any]:
    header = ROOT / "code" / "storage" / "pqc_epoch_publish.h"
    source = ROOT / "code" / "storage" / "pqc_epoch_publish.c"
    writeback = ROOT / "code" / "storage" / "pqc_writeback.c"
    source_text = source.read_text(encoding="utf-8", errors="replace")
    writeback_text = writeback.read_text(encoding="utf-8", errors="replace")
    return {
        "header": relpath(header),
        "source": relpath(source),
        "header_exists": header.exists(),
        "source_exists": source.exists(),
        "dispatch_call_visible": "pqc_publication_dispatch_commit" in writeback_text,
        "mode_env_visible": "PQC_PUBLICATION_MODE" in source_text,
        "strict_fallback_visible": "pqc_strict_publish_commit(req)" in source_text,
        "epoch_skeleton_unsupported_visible": "-ENOTSUP" in source_text,
        "epoch_redo_log_mode_visible": "PQC_PUBLICATION_MODE_EPOCH_REDO_LOG" in source_text,
    }


def run_strict_case(case_dir: Path, label: str,
                    extra_env: dict[str, str] | None = None) -> dict[str, Any]:
    storage_dir = Path(tempfile.mkdtemp(prefix=f"aegisq_{label}_storage_"))
    mount_dir = Path(tempfile.mkdtemp(prefix=f"aegisq_{label}_mnt_"))
    case_dir.mkdir(parents=True, exist_ok=True)
    fuse = None
    unmount: dict[str, Any] = {}
    client: dict[str, Any] = {}
    error: str | None = None
    try:
        fuse = start_fuse(storage_dir, mount_dir, case_dir,
                          f"epoch-publish-{label}", extra_env)
        payload = (f"epoch-publish-dispatch:{label}:".encode("ascii") +
                   bytes((i % 251 for i in range(16384))))
        client = mounted_write_read(mount_dir, "dispatch-smoke.dat", payload)
    except Exception as exc:  # noqa: BLE001 - retained as artifact evidence
        error = repr(exc)
    finally:
        try:
            unmount = stop_fuse(fuse, mount_dir, case_dir)
        finally:
            shutil.rmtree(mount_dir, ignore_errors=True)
            shutil.rmtree(storage_dir, ignore_errors=True)
    return {
        "label": label,
        "client": client,
        "unmount": unmount,
        "error": error,
        "pass": (
            error is None and
            client.get("matches") is True and
            unmount.get("returncode") == 0
        ),
    }


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "epoch_publish_dispatch_smoke.json"
    md_path = out_dir / "epoch_publish_dispatch_smoke.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    lines = [
        "# Epoch Publication Dispatch Smoke",
        "",
        f"- Generated: `{payload['generated_utc']}`",
        f"- Overall pass: `{str(payload['overall_pass']).lower()}`",
        f"- Header exists: `{str(payload['source_evidence']['header_exists']).lower()}`",
        f"- Source exists: `{str(payload['source_evidence']['source_exists']).lower()}`",
        f"- Dispatch call visible: `{str(payload['source_evidence']['dispatch_call_visible']).lower()}`",
        f"- Strict fallback visible: `{str(payload['source_evidence']['strict_fallback_visible']).lower()}`",
        f"- Epoch skeleton unsupported visible: `{str(payload['source_evidence']['epoch_skeleton_unsupported_visible']).lower()}`",
        f"- Epoch redo-log mode visible: `{str(payload['source_evidence']['epoch_redo_log_mode_visible']).lower()}`",
        "",
    ]
    for case in payload["cases"]:
        lines.append(f"## {case['label']}")
        lines.append(f"- Pass: `{str(case['pass']).lower()}`")
        lines.append(f"- Read matches write: `{str(case['client'].get('matches')).lower()}`")
        lines.append("")
    lines.extend([
        "## Negative Claim Guard",
        "",
        payload["negative_claim_guard"],
        "",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not FUSE_BIN.exists():
        raise SystemExit(f"missing build artifact: {relpath(FUSE_BIN)}")

    default_case = run_strict_case(args.out_dir / "default_strict",
                                   "default_strict")
    trace_path = args.out_dir / "explicit_strict" / "publication_trace.jsonl"
    explicit_case = run_strict_case(
        args.out_dir / "explicit_strict",
        "explicit_strict",
        {
            "PQC_PUBLICATION_MODE": "strict",
            "PQC_PUBLICATION_TRACE_PATH": str(trace_path),
        },
    )
    source = source_evidence()
    trace = parse_jsonl(trace_path)
    source_pass = all(bool(source[key]) for key in (
        "header_exists",
        "source_exists",
        "dispatch_call_visible",
        "mode_env_visible",
        "strict_fallback_visible",
        "epoch_skeleton_unsupported_visible",
        "epoch_redo_log_mode_visible",
    ))
    trace_pass = (
        trace["event_count"] >= 1 and
        "strict" in trace["modes"] and
        trace["malformed_line_count"] == 0
    )
    payload = {
        "schema_version": 1,
        "generated_by": "experiments/run_epoch_publish_dispatch_smoke.py",
        "generated_utc": now_utc(),
        "scope": "Gate 0.9-S0 skeleton dispatch proof only.",
        "cases": [default_case, explicit_case],
        "source_evidence": source,
        "explicit_strict_trace": trace,
        "source_pass": source_pass,
        "trace_pass": trace_pass,
        "overall_pass": (
            default_case["pass"] and
            explicit_case["pass"] and
            source_pass and
            trace_pass
        ),
        "negative_claim_guard": (
            "This artifact proves only that an epoch-publication module skeleton "
            "builds and strict mode dispatches through it. Epoch redo-log "
            "mounted-path behavior is covered by the Gate 0.9-S2 artifact; this "
            "S0 smoke is not evidence of checkpoint compaction, crash replay, "
            "fdatasync reduction, throughput improvement, or rollback resistance."
        ),
    }
    write_outputs(args.out_dir, payload)
    print(json.dumps({
        "overall_pass": payload["overall_pass"],
        "json": relpath(args.out_dir / "epoch_publish_dispatch_smoke.json"),
        "trace_pass": trace_pass,
    }, indent=2, sort_keys=True))
    return 0 if payload["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
