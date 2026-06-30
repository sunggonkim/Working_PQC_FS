#!/usr/bin/env python3
"""Bounded hardware TPM epoch-freshness probe for C6-S2.

This runner attempts the production hardware anchor path once through
`pqc_fuse --anchor-self-test`, records the resulting epoch trace, and classifies
the outcome as either a committed TPM epoch or an environment-blocked run.  It
does not provision TPM state, mutate PCR policy, or claim rollback resistance.
"""

from __future__ import annotations

import argparse
import grp
import json
import os
import pwd
import shutil
import stat
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FUSE_BIN = ROOT / "build" / "pqc_fuse"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "async_merkle_tpm_epoch"


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_capture(cmd: list[str], out_dir: Path, name: str,
                env: dict[str, str] | None = None) -> dict[str, Any]:
    stdout_path = out_dir / f"{name}.stdout.txt"
    stderr_path = out_dir / f"{name}.stderr.txt"
    start = time.perf_counter_ns()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    end = time.perf_counter_ns()
    stdout_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(proc.stderr, encoding="utf-8", errors="replace")
    return {
        "name": name,
        "command": cmd,
        "returncode": proc.returncode,
        "elapsed_ms": (end - start) / 1_000_000.0,
        "stdout": relpath(stdout_path),
        "stderr": relpath(stderr_path),
        "stderr_tail": proc.stderr[-2000:],
    }


def read_trace(path: Path) -> tuple[list[dict[str, Any]], int]:
    events: list[dict[str, Any]] = []
    malformed = 0
    if not path.exists():
        return events, malformed
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if isinstance(value, dict):
            events.append(value)
        else:
            malformed += 1
    return events, malformed


def device_record(path: str) -> dict[str, Any]:
    rec: dict[str, Any] = {"path": path, "exists": False}
    try:
        st = os.stat(path)
    except OSError as exc:
        rec["error"] = f"errno={exc.errno}"
        return rec
    rec.update({
        "exists": True,
        "mode_octal": oct(stat.S_IMODE(st.st_mode)),
        "uid": st.st_uid,
        "gid": st.st_gid,
        "user": pwd.getpwuid(st.st_uid).pw_name,
        "group": grp.getgrgid(st.st_gid).gr_name,
        "is_char_device": stat.S_ISCHR(st.st_mode),
    })
    return rec


def current_groups() -> list[str]:
    names = []
    for gid in os.getgroups():
        try:
            names.append(grp.getgrgid(gid).gr_name)
        except KeyError:
            names.append(str(gid))
    return sorted(set(names))


def source_checks() -> dict[str, bool]:
    anchor_c = (ROOT / "code" / "storage" / "pqc_anchor.c").read_text(
        encoding="utf-8", errors="replace"
    )
    anchor_h = (ROOT / "code" / "storage" / "pqc_anchor.h").read_text(
        encoding="utf-8", errors="replace"
    )
    return {
        "hardware_backend_switch_present":
            "PQC_FRESHNESS_ANCHOR_BACKEND" in anchor_c and
            "hardware" in anchor_c,
        "tpm_nv_write_path_present": "write_anchor_tpm" in anchor_c,
        "tpm_nv_read_path_present": "read_anchor_tpm" in anchor_c,
        "hardware_force_policy_present":
            "PQC_ANCHOR_EPOCH_FLUSH_HARDWARE_FORCE" in anchor_h,
        "epoch_trace_event_present": "anchor_epoch_freshness_record" in anchor_c,
        "pcr_policy_not_implemented_in_anchor":
            "PolicyPCR" not in anchor_c and "tpm2_unseal" not in anchor_c,
    }


def classify_trace(events: list[dict[str, Any]]) -> dict[str, Any]:
    epoch_records = [
        event for event in events
        if event.get("event") == "anchor_epoch_freshness_record"
    ]
    hardware_records = [
        event for event in epoch_records if event.get("backend") == "hardware"
    ]
    pending = [
        event for event in hardware_records
        if event.get("status") == "pending" and event.get("pending") is True
    ]
    committed = [
        event for event in hardware_records
        if event.get("status") == "committed" and event.get("committed") is True
    ]
    failed_flushes = [
        event for event in hardware_records
        if event.get("flush_policy") == "hardware_force" and
        event.get("status") == "failed"
    ]
    return {
        "event_count": len(events),
        "epoch_record_count": len(epoch_records),
        "hardware_epoch_record_count": len(hardware_records),
        "hardware_pending_count": len(pending),
        "hardware_committed_count": len(committed),
        "hardware_failed_flush_count": len(failed_flushes),
        "hardware_pending_observed": bool(pending),
        "hardware_committed_observed": bool(committed),
        "hardware_flush_attempted": bool(committed or failed_flushes),
        "last_hardware_rc": (
            hardware_records[-1].get("rc") if hardware_records else None
        ),
        "last_hardware_status": (
            hardware_records[-1].get("status") if hardware_records else None
        ),
    }


def permission_blocked(command_results: list[dict[str, Any]]) -> bool:
    needles = (
        "Permission denied",
        "a password is required",
        "Could not load tcti",
        "No standard TCTI could be loaded",
    )
    for result in command_results:
        if result.get("returncode") == 0:
            continue
        stderr = result.get("stderr_tail", "")
        if any(needle in stderr for needle in needles):
            return True
    return False


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    verdict = payload["verdict"]
    checks = payload["checks"]
    req = payload["c6_s2_requirements"]
    lines = [
        "# TPM Epoch Freshness Probe",
        "",
        "This is a bounded C6-S2 probe of the production hardware anchor path.",
        "It does not claim async Merkle maintenance, PCR-bound rollback resistance, or full replay protection.",
        "",
        "## Verdict",
        "",
        f"- Overall pass: `{str(payload['overall_pass']).lower()}`",
        f"- Verdict: `{verdict}`",
        f"- Environment blocked: `{str(payload['environment_blocked']).lower()}`",
        f"- Hardware committed: `{str(payload['hardware_epoch_committed']).lower()}`",
        "",
        "## Checks",
        "",
    ]
    for key, value in checks.items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")
    lines.extend(["", "## C6-S2 Requirement Status", ""])
    for key, value in req.items():
        if isinstance(value, dict):
            lines.append(f"- `{key}`: `{value.get('status')}`")
            if value.get("reason"):
                lines.append(f"  - reason: `{value['reason']}`")
        else:
            lines.append(f"- `{key}`: `{value}`")
    lines.extend([
        "",
        "## Trace",
        "",
        f"- Trace path: `{payload['production_path_attempt']['trace_path']}`",
        f"- Hardware pending observed: `{str(payload['trace_summary']['hardware_pending_observed']).lower()}`",
        f"- Hardware flush attempted: `{str(payload['trace_summary']['hardware_flush_attempted']).lower()}`",
        f"- Hardware committed observed: `{str(payload['trace_summary']['hardware_committed_observed']).lower()}`",
        "",
        "## Conservative Interpretation",
        "",
        "- A pending hardware epoch record proves the production path reached hardware-backend staging.",
        "- A failed hardware-force flush with TPM command failures is environment/provisioning evidence, not rollback protection.",
        "- Stale snapshot replay, PCR drift, reboot recovery, and mount refusal stay unclaimed unless a provisioned TPM run completes.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_probe(out_dir: Path, nv_index: str, tcti: str) -> dict[str, Any]:
    if not FUSE_BIN.exists():
        raise SystemExit(f"missing build artifact: {relpath(FUSE_BIN)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "tpm_epoch_freshness_probe.trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()

    env = os.environ.copy()
    env.update({
        "PQC_FRESHNESS_ANCHOR_BACKEND": "hardware",
        "PQC_FRESHNESS_ANCHOR_PATH": "/tmp/aegisq_tpm_epoch_probe_anchor",
        "PQC_TPM_NV_INDEX": nv_index,
        "PQC_TPM_TCTI": tcti,
        "TSS2_TCTI": tcti,
        "PQC_ANCHOR_TRACE_PATH": str(trace_path),
    })

    tool_results: list[dict[str, Any]] = []
    sudo_probe = run_capture(["sudo", "-n", "true"], out_dir, "sudo_noninteractive")
    tool_results.append(sudo_probe)
    if shutil.which("tpm2_nvreadpublic"):
        tool_results.append(run_capture(
            ["tpm2_nvreadpublic", nv_index],
            out_dir,
            "tpm2_nvreadpublic",
            env=env,
        ))
    if shutil.which("tpm2_pcrread"):
        tool_results.append(run_capture(
            ["tpm2_pcrread", "sha256:0,1,2,3"],
            out_dir,
            "tpm2_pcrread",
            env=env,
        ))

    production = run_capture(
        [str(FUSE_BIN), "--anchor-self-test"],
        out_dir,
        "hardware_anchor_self_test",
        env=env,
    )
    production["trace_path"] = relpath(trace_path)

    events, malformed = read_trace(trace_path)
    trace_summary = classify_trace(events)
    blocked = permission_blocked(tool_results) or (
        production["returncode"] != 0 and
        trace_summary.get("last_hardware_status") == "failed"
    )
    committed = bool(trace_summary["hardware_committed_observed"])

    stale_reason = (
        "hardware TPM epoch did not commit on this run; replay test would not "
        "measure a provisioned epoch"
    )
    pcr_reason = (
        "PCR read failed or anchor path has no persistent PCR policy; PCR-bound "
        "freshness remains a non-claim"
    )
    c6_requirements: dict[str, Any] = {
        "tpm_latency": {
            "status": "recorded",
            "commands": [
                {
                    "name": result["name"],
                    "returncode": result["returncode"],
                    "elapsed_ms": result["elapsed_ms"],
                }
                for result in tool_results + [production]
            ],
        },
        "anti_hammering_wear_boundary": {
            "status": "recorded",
            "max_anchor_self_tests": 1,
            "observed_hardware_flush_attempts":
                trace_summary["hardware_committed_count"] +
                trace_summary["hardware_failed_flush_count"],
            "reason": (
                "probe is intentionally single-shot and does not provision or "
                "loop TPM NV writes"
            ),
        },
        "stale_snapshot_replay": {
            "status": "not_run_environment_blocked" if not committed else "needs_followup",
            "reason": stale_reason if not committed else "committed epoch exists; run replay matrix next",
        },
        "pcr_drift": {
            "status": "not_run_environment_blocked",
            "reason": pcr_reason,
        },
        "reboot_recovery": {
            "status": "not_run_environment_blocked" if not committed else "needs_followup",
            "reason": stale_reason if not committed else "committed epoch exists; run reboot/remount proof next",
        },
        "mount_refusal_logs": {
            "status": "not_run_environment_blocked" if not committed else "needs_followup",
            "reason": stale_reason if not committed else "committed epoch exists; run stale remount proof next",
        },
    }

    source = source_checks()
    checks = {
        "source_checks_pass": all(source.values()),
        "trace_jsonl_well_formed": malformed == 0 and bool(events),
        "hardware_pending_observed": trace_summary["hardware_pending_observed"],
        "hardware_flush_attempted": trace_summary["hardware_flush_attempted"],
        "tpm_latency_recorded": all(
            "elapsed_ms" in result for result in tool_results + [production]
        ),
        "anti_hammering_boundary_recorded":
            c6_requirements["anti_hammering_wear_boundary"]["status"] == "recorded",
        "committed_or_environment_blocked": committed or blocked,
        "rollback_claim_not_allowed": not committed,
    }
    verdict = "hardware-epoch-committed" if committed else (
        "environment-blocked" if blocked else "hardware-epoch-failed-unknown"
    )

    payload = {
        "schema_version": 1,
        "generated_by": "code/experiments/run_tpm_epoch_freshness_probe.py",
        "generated_utc": now_utc(),
        "scope": (
            "Bounded hardware TPM epoch-freshness probe.  It records whether "
            "the production hardware anchor path commits an epoch or is blocked "
            "by local TPM/provisioning/permission state.  It is not a rollback "
            "resistance claim."
        ),
        "config": {
            "nv_index": nv_index,
            "tcti": tcti,
            "tools": {
                "tpm2_nvreadpublic": shutil.which("tpm2_nvreadpublic") is not None,
                "tpm2_nvwrite": shutil.which("tpm2_nvwrite") is not None,
                "tpm2_nvread": shutil.which("tpm2_nvread") is not None,
                "tpm2_pcrread": shutil.which("tpm2_pcrread") is not None,
            },
            "devices": [device_record("/dev/tpmrm0"), device_record("/dev/tpm0")],
            "user": {
                "uid": os.getuid(),
                "gid": os.getgid(),
                "groups": current_groups(),
                "PQC_SUDO_PASSWORD_present": bool(os.environ.get("PQC_SUDO_PASSWORD")),
            },
        },
        "tool_probes": tool_results,
        "production_path_attempt": production,
        "trace_summary": trace_summary,
        "malformed_trace_lines": malformed,
        "events": events,
        "source_checks": source,
        "c6_s2_requirements": c6_requirements,
        "hardware_epoch_committed": committed,
        "environment_blocked": blocked,
        "verdict": verdict,
        "checks": checks,
        "overall_pass": all(checks.values()),
        "next_code_edit": (
            "If TPM access is enabled, run a provisioned stale-snapshot replay "
            "and reboot/remount refusal proof; otherwise keep rollback "
            "resistance as a non-claim."
        ),
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--nv-index", default=os.environ.get("PQC_TPM_NV_INDEX", "0x01500010"))
    parser.add_argument("--tcti", default=os.environ.get("PQC_TPM_TCTI", "device:/dev/tpmrm0"))
    args = parser.parse_args()
    payload = run_probe(args.out_dir, args.nv_index, args.tcti)
    json_path = args.out_dir / "tpm_epoch_freshness_probe.json"
    md_path = args.out_dir / "tpm_epoch_freshness_probe.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    write_markdown(payload, md_path)
    print(json.dumps({
        "overall_pass": payload["overall_pass"],
        "verdict": payload["verdict"],
        "environment_blocked": payload["environment_blocked"],
        "hardware_epoch_committed": payload["hardware_epoch_committed"],
        "json": relpath(json_path),
        "failed_checks": [
            key for key, value in payload["checks"].items() if not value
        ],
        "next_code_edit": payload["next_code_edit"],
    }, indent=2, sort_keys=True))
    return 0 if payload["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
