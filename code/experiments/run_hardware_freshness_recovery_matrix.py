#!/usr/bin/env python3
"""Build a hardware-freshness recovery-case matrix.

The real TPM artifacts cover replay-after-advance, missing-index startup, PCR
probe drift, and owner-authorization failure.  Two cases are unsafe to exercise
by rewriting the live NV index, so this harness uses command-path fault
injection around the final pqc_fuse binary: one run feeds a valid-but-stale TPM
anchor back to the loader, and one run interrupts nvwrite before the anchor is
committed.  The matrix records that scope explicitly instead of treating the
fault-injected rows as physical power-loss evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "hardware_freshness_recovery_matrix"
FUSE_BIN = ROOT / "build" / "pqc_fuse"
NV_INDEX = "0x01500010"

TPM_MONOTONIC = ROOT / "artifacts" / "validation" / "tpm_monotonic_replay" / "tpm_monotonic_replay.json"
TPM_UNPROVISIONED = ROOT / "artifacts" / "validation" / "tpm_unprovisioned.json"
TPM_UNPROVISIONED_STDOUT = ROOT / "artifacts" / "validation" / "tpm_unprovisioned.stdout"
TPM_UNPROVISIONED_STDERR = ROOT / "artifacts" / "validation" / "tpm_unprovisioned.stderr"
TPM_PCR = ROOT / "artifacts" / "validation" / "tpm_pcr_policy_probe" / "tpm_pcr_policy_probe.json"
TPM_PCR_DRIFT_STDERR = ROOT / "artifacts" / "validation" / "tpm_pcr_policy_probe" / "unseal_drift.stderr.txt"
COMBINED = ROOT / "artifacts" / "validation" / "combined_durability_bundle" / "combined_durability_bundle.json"

REQUIRED_CASES = (
    "stale_disk_new_tpm",
    "new_disk_stale_tpm",
    "missing_index",
    "changed_pcrs",
    "authorization_failure",
    "interrupted_nv_update",
    "normal_replay_after_advance",
)


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def relpath_text(path_text: str) -> str:
    path = Path(path_text)
    if not path.is_absolute():
        return str(path)
    return relpath(path)


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_capture(
    cmd: list[str],
    *,
    out_dir: Path,
    name: str,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
) -> dict[str, Any]:
    stdout_path = out_dir / f"{name}.stdout.txt"
    stderr_path = out_dir / f"{name}.stderr.txt"
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        input=input_text,
        text=True,
        stdout=stdout_path.open("w", encoding="utf-8"),
        stderr=stderr_path.open("w", encoding="utf-8"),
        check=False,
    )
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": relpath(stdout_path),
        "stderr": relpath(stderr_path),
    }


def make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def write_script(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    make_executable(path)


def run_authorization_failure(out_dir: Path) -> dict[str, Any]:
    probes = out_dir / "probes"
    probes.mkdir(parents=True, exist_ok=True)
    password = os.environ.get("PQC_SUDO_PASSWORD")
    if not password:
        result = {
            "command": ["sudo", "-S", "tpm2_nvread", "-C", "o", "-P", "<intentional-wrong-auth>", "-s", "1", NV_INDEX],
            "returncode": None,
            "stdout": relpath(probes / "auth_failure.stdout.txt"),
            "stderr": relpath(probes / "auth_failure.stderr.txt"),
            "skipped": "PQC_SUDO_PASSWORD not set",
        }
        write_json(probes / "auth_failure.json", result)
        return result

    result = run_capture(
        ["sudo", "-S", "-p", "", "tpm2_nvread", "-C", "o", "-P", "aegisq-intentional-wrong-auth", "-s", "1", NV_INDEX],
        out_dir=probes,
        name="auth_failure",
        input_text=password + "\n",
    )
    result["command"] = ["sudo", "-S", "tpm2_nvread", "-C", "o", "-P", "<intentional-wrong-auth>", "-s", "1", NV_INDEX]
    write_json(probes / "auth_failure.json", result)
    return result


def prepare_common_fake_bin(bin_dir: Path) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    write_script(
        bin_dir / "tpm2_nvreadpublic",
        "#!/bin/sh\n"
        "printf 'fault-injection nvreadpublic success for %s\\n' \"$*\" >&2\n"
        "exit 0\n",
    )


def run_new_disk_stale_tpm(out_dir: Path) -> dict[str, Any]:
    case_dir = out_dir / "fault_injection" / "new_disk_stale_tpm"
    bin_dir = case_dir / "bin"
    capture_dir = case_dir / "capture"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    capture_dir.mkdir(parents=True, exist_ok=True)
    prepare_common_fake_bin(bin_dir)

    write_script(
        bin_dir / "tpm2_nvwrite",
        "#!/usr/bin/env python3\n"
        "import hashlib, hmac, os, pathlib, struct, sys\n"
        "capture = pathlib.Path(os.environ['AEGIS_FAULT_CAPTURE_DIR'])\n"
        "capture.mkdir(parents=True, exist_ok=True)\n"
        "input_path = None\n"
        "for i, arg in enumerate(sys.argv):\n"
        "    if arg == '-i' and i + 1 < len(sys.argv):\n"
        "        input_path = pathlib.Path(sys.argv[i + 1])\n"
        "if input_path is None:\n"
        "    print('missing -i input', file=sys.stderr)\n"
        "    sys.exit(2)\n"
        "data = bytearray(input_path.read_bytes())\n"
        "(capture / 'nvwrite_input.bin').write_bytes(data)\n"
        "if len(data) != 88:\n"
        "    print(f'unexpected anchor size {len(data)}', file=sys.stderr)\n"
        "    sys.exit(3)\n"
        "seq = struct.unpack_from('<Q', data, 8)[0]\n"
        "struct.pack_into('<Q', data, 8, max(0, seq - 1))\n"
        "data[56:88] = hmac.new(bytes(32), bytes(data[:56]), hashlib.sha256).digest()\n"
        "(capture / 'stale_tpm_anchor.bin').write_bytes(data)\n"
        "print(f'fault-injection captured seq={seq} and staged stale seq={max(0, seq - 1)}', file=sys.stderr)\n"
        "sys.exit(0)\n",
    )
    write_script(
        bin_dir / "tpm2_nvread",
        "#!/usr/bin/env python3\n"
        "import os, pathlib, sys\n"
        "path = pathlib.Path(os.environ['AEGIS_FAULT_CAPTURE_DIR']) / 'stale_tpm_anchor.bin'\n"
        "if not path.exists():\n"
        "    print('stale TPM anchor fixture missing', file=sys.stderr)\n"
        "    sys.exit(2)\n"
        "sys.stdout.buffer.write(path.read_bytes())\n",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env.get('PATH', '')}",
            "AEGIS_FAULT_CAPTURE_DIR": str(capture_dir),
            "PQC_FRESHNESS_ANCHOR_BACKEND": "hardware",
            "PQC_FRESHNESS_ANCHOR_PATH": "hardware-fault-injection",
            "PQC_FRESHNESS_WINDOW_N": "1",
        }
    )
    result = run_capture([str(FUSE_BIN), "--anchor-self-test"], out_dir=case_dir, name="anchor_self_test", env=env)
    result["fixture_artifacts"] = {
        "nvwrite_input": relpath(capture_dir / "nvwrite_input.bin"),
        "stale_tpm_anchor": relpath(capture_dir / "stale_tpm_anchor.bin"),
    }
    write_json(case_dir / "new_disk_stale_tpm.json", result)
    return result


def run_interrupted_nv_update(out_dir: Path) -> dict[str, Any]:
    case_dir = out_dir / "fault_injection" / "interrupted_nv_update"
    bin_dir = case_dir / "bin"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    prepare_common_fake_bin(bin_dir)

    write_script(
        bin_dir / "tpm2_nvwrite",
        "#!/bin/sh\n"
        "printf 'fault-injection interrupted tpm2_nvwrite before commit\\n' >&2\n"
        "exit 130\n",
    )
    write_script(
        bin_dir / "tpm2_nvread",
        "#!/bin/sh\n"
        "printf 'unexpected nvread after interrupted update\\n' >&2\n"
        "exit 2\n",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env.get('PATH', '')}",
            "PQC_FRESHNESS_ANCHOR_BACKEND": "hardware",
            "PQC_FRESHNESS_ANCHOR_PATH": "hardware-fault-injection",
            "PQC_FRESHNESS_WINDOW_N": "1",
        }
    )
    result = run_capture([str(FUSE_BIN), "--anchor-self-test"], out_dir=case_dir, name="anchor_self_test", env=env)
    write_json(case_dir / "interrupted_nv_update.json", result)
    return result


def monotonic_row() -> dict[str, Any]:
    report = load_json(TPM_MONOTONIC)
    replay = ((report.get("result") or {}).get("replay_result") or {})
    passed = replay.get("mode") == "fail_closed" and ((report.get("result") or {}).get("fail_closed") is True)
    logs = (report.get("result") or {}).get("replay_mount_logs") or {}
    return {
        "case": "stale_disk_new_tpm",
        "scenario": "restore stale backing-directory snapshot after the TPM-backed anchor has advanced",
        "oracle_verdict": "fail_closed" if passed else "missing_or_unexpected",
        "scope": "real_tpm_replay_after_advance",
        "raw_artifacts": [relpath(TPM_MONOTONIC), *(relpath_text(str(v)) for v in logs.values() if v)],
        "evidence": replay.get("detail", ""),
        "pass": passed,
    }


def normal_replay_row() -> dict[str, Any]:
    report = load_json(COMBINED)
    sqlite = ((report.get("unified_campaign") or {}).get("replay") or {})
    dbm = ((report.get("unified_dbm_campaign") or {}).get("replay") or {})
    passed = sqlite.get("verdict") == "fail_closed" and sqlite.get("acceptable") is True and dbm.get("verdict") == "fail_closed" and dbm.get("acceptable") is True
    raw = [relpath(COMBINED)]
    for replay in (sqlite, dbm):
        read = replay.get("read") or {}
        for key in ("stdout", "stderr"):
            if read.get(key):
                raw.append(str(read[key]))
    return {
        "case": "normal_replay_after_advance",
        "scenario": "same-backing-store SQLite and dbm.dumb replay-after-advance campaigns",
        "oracle_verdict": "fail_closed" if passed else "missing_or_unexpected",
        "scope": "real_tpm_application_replay_after_advance",
        "raw_artifacts": raw,
        "evidence": f"sqlite={sqlite.get('detail', '')}; dbm={dbm.get('detail', '')}",
        "pass": passed,
    }


def missing_index_row() -> dict[str, Any]:
    report = load_json(TPM_UNPROVISIONED)
    stderr = read_text(TPM_UNPROVISIONED_STDERR)
    exit_code = report.get("hardware_anchor_without_preprovisioning_exit")
    passed = exit_code not in (None, 0) and "Freshness anchor probe failed" in stderr and "No such device" in stderr
    return {
        "case": "missing_index",
        "scenario": "start hardware-anchor mode when the configured NV index is absent",
        "oracle_verdict": "fail_closed" if passed else "missing_or_unexpected",
        "scope": "retained_final_binary_startup_probe",
        "raw_artifacts": [relpath(TPM_UNPROVISIONED), relpath(TPM_UNPROVISIONED_STDOUT), relpath(TPM_UNPROVISIONED_STDERR)],
        "evidence": f"exit={exit_code}; stderr contains freshness-anchor ENODEV={passed}",
        "pass": passed,
    }


def changed_pcr_row() -> dict[str, Any]:
    report = load_json(TPM_PCR)
    results = report.get("results") or {}
    passed = results.get("good_unseal_matches_secret") is True and results.get("drift_rejected") is True
    return {
        "case": "changed_pcrs",
        "scenario": "transient PCR-policy object unseals on current PCR digest and rejects a drifted PCR digest",
        "oracle_verdict": "transient_probe_rejects_drift" if passed else "missing_or_unexpected",
        "scope": "transient_pcr_probe_not_persistent_filesystem_anchor",
        "raw_artifacts": [relpath(TPM_PCR), relpath(TPM_PCR_DRIFT_STDERR)],
        "evidence": f"good_unseal={results.get('good_unseal_matches_secret')}; drift_rejected={results.get('drift_rejected')}",
        "pass": passed,
    }


def authorization_row(out_dir: Path, probe: dict[str, Any]) -> dict[str, Any]:
    stderr_path = ROOT / probe.get("stderr", "")
    stderr = read_text(stderr_path)
    passed = probe.get("returncode") not in (None, 0) and "authorization failure" in stderr
    return {
        "case": "authorization_failure",
        "scenario": "read the provisioned NV index with deliberate wrong owner authorization",
        "oracle_verdict": "fail_closed" if passed else "missing_or_unexpected",
        "scope": "real_tpm_non_destructive_wrong_auth_probe",
        "raw_artifacts": [relpath(out_dir / "probes" / "auth_failure.json"), str(probe.get("stdout", "")), str(probe.get("stderr", ""))],
        "evidence": "TPM tool reports authorization failure without exposing NV data" if passed else probe.get("skipped", ""),
        "pass": passed,
    }


def new_disk_stale_row(out_dir: Path, probe: dict[str, Any]) -> dict[str, Any]:
    stderr = read_text(ROOT / probe.get("stderr", ""))
    passed = probe.get("returncode") == 0 and "PQC-FUSE anchor self-test: PASS" in stderr
    artifacts = [
        relpath(out_dir / "fault_injection" / "new_disk_stale_tpm" / "new_disk_stale_tpm.json"),
        str(probe.get("stdout", "")),
        str(probe.get("stderr", "")),
    ]
    artifacts.extend(str(v) for v in (probe.get("fixture_artifacts") or {}).values())
    return {
        "case": "new_disk_stale_tpm",
        "scenario": "local committed prefix is ahead of the TPM anchor fixture",
        "oracle_verdict": "latest_committed" if passed else "missing_or_unexpected",
        "scope": "final_binary_command_path_fault_injection_no_live_nv_mutation",
        "raw_artifacts": artifacts,
        "evidence": "anchor load accepts stored.global_sequence <= local sequence; stale TPM lag is not treated as disk rollback" if passed else stderr[-500:],
        "pass": passed,
    }


def interrupted_update_row(out_dir: Path, probe: dict[str, Any]) -> dict[str, Any]:
    stderr = read_text(ROOT / probe.get("stderr", ""))
    passed = probe.get("returncode") not in (None, 0) and "PQC-FUSE anchor self-test: FAIL" in stderr
    return {
        "case": "interrupted_nv_update",
        "scenario": "interrupt tpm2_nvwrite before the anchor update commits",
        "oracle_verdict": "fail_closed_update_not_committed" if passed else "missing_or_unexpected",
        "scope": "final_binary_command_path_fault_injection_not_physical_power_loss",
        "raw_artifacts": [
            relpath(out_dir / "fault_injection" / "interrupted_nv_update" / "interrupted_nv_update.json"),
            str(probe.get("stdout", "")),
            str(probe.get("stderr", "")),
        ],
        "evidence": "final binary returns nonzero when the hardware-anchor update command is interrupted" if passed else stderr[-500:],
        "pass": passed,
    }


def normalize_artifact(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def build_report(out_dir: Path) -> dict[str, Any]:
    auth_probe = run_authorization_failure(out_dir)
    stale_probe = run_new_disk_stale_tpm(out_dir)
    interrupted_probe = run_interrupted_nv_update(out_dir)

    rows = [
        monotonic_row(),
        new_disk_stale_row(out_dir, stale_probe),
        missing_index_row(),
        changed_pcr_row(),
        authorization_row(out_dir, auth_probe),
        interrupted_update_row(out_dir, interrupted_probe),
        normal_replay_row(),
    ]

    present_cases = {row["case"] for row in rows}
    for row in rows:
        row["raw_artifacts_present"] = all(
            artifact and normalize_artifact(artifact).exists()
            for artifact in row.get("raw_artifacts", [])
        )
    checks = {
        "all_required_cases_present": present_cases == set(REQUIRED_CASES),
        "all_rows_have_oracle_verdict": all(row.get("oracle_verdict") and row.get("oracle_verdict") != "missing_or_unexpected" for row in rows),
        "all_rows_have_raw_artifacts": all(row.get("raw_artifacts_present") for row in rows),
        "all_case_oracles_pass": all(row.get("pass") is True for row in rows),
    }

    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": all(checks.values()),
        "scope": (
            "Recovery-case matrix for the current hardware-freshness scope.  "
            "Real TPM artifacts cover provisioned replay-after-advance, missing "
            "index, changed-PCR transient probe behavior, and wrong authorization.  "
            "New-disk/stale-TPM and interrupted-NV-update rows use final-binary "
            "command-path fault injection and do not model physical power loss."
        ),
        "required_cases": list(REQUIRED_CASES),
        "checks": checks,
        "rows": rows,
    }


def write_markdown(out_dir: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Hardware Freshness Recovery Matrix",
        "",
        f"- Overall pass: `{str(report['overall_pass']).lower()}`",
        f"- Scope: {report['scope']}",
        "",
        "| Case | Oracle verdict | Scope | Pass |",
        "| --- | --- | --- | --- |",
    ]
    for row in report["rows"]:
        lines.append(
            f"| `{row['case']}` | `{row['oracle_verdict']}` | {row['scope']} | `{str(row['pass']).lower()}` |"
        )
    lines.extend(["", "## Raw Artifacts", ""])
    for row in report["rows"]:
        lines.append(f"### `{row['case']}`")
        lines.append("")
        lines.append(f"- Scenario: {row['scenario']}")
        lines.append(f"- Evidence: {row['evidence']}")
        for artifact in row.get("raw_artifacts", []):
            lines.append(f"- `{artifact}`")
        lines.append("")
    (out_dir / "hardware_freshness_recovery_matrix.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build hardware-freshness recovery matrix")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not FUSE_BIN.exists():
        raise FileNotFoundError(relpath(FUSE_BIN))

    report = build_report(args.out_dir)
    write_json(args.out_dir / "hardware_freshness_recovery_matrix.json", report)
    write_markdown(args.out_dir, report)
    print(json.dumps({"out_dir": relpath(args.out_dir), "overall_pass": report["overall_pass"]}, indent=2))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
