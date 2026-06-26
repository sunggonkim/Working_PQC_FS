#!/usr/bin/env python3
"""Run a non-destructive TPM PCR policy probe.

The probe creates a transient sealed object bound to the current PCR values,
then verifies two outcomes:

  1. unseal succeeds when a policy session is built from the current PCR file;
  2. policy construction or unseal fails when the PCR file is deliberately
     modified to represent drift.

This is PCR-policy evidence for a transient object.  It does not mutate PCRs,
does not provision the persistent AEGIS-Q NV index, and does not prove the full
hardware-backed freshness/recovery flow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "tpm_pcr_policy_probe"


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    sudo_password: str | None,
    name: str,
    out_dir: Path,
    allow_fail: bool = False,
) -> dict[str, Any]:
    actual_cmd = cmd
    stdin = None
    if sudo_password is not None:
        actual_cmd = ["sudo", "-S"] + cmd
        stdin = sudo_password + "\n"
    proc = subprocess.run(actual_cmd, cwd=cwd, env=env, input=stdin, text=True, capture_output=True)
    stdout_path = out_dir / f"{name}.stdout.txt"
    stderr_path = out_dir / f"{name}.stderr.txt"
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    rec = {
        "name": name,
        "command": actual_cmd,
        "returncode": proc.returncode,
        "stdout": str(stdout_path.relative_to(ROOT)),
        "stderr": str(stderr_path.relative_to(ROOT)),
    }
    if proc.returncode != 0 and not allow_fail:
        raise RuntimeError(f"{name} failed with {proc.returncode}; see {stderr_path}")
    return rec


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chown_to_user(path: Path, cwd: Path, env: dict[str, str], sudo_password: str | None, out_dir: Path, name: str) -> dict[str, Any]:
    return run_cmd(
        ["chown", f"{os.getuid()}:{os.getgid()}", str(path)],
        cwd=cwd,
        env=env,
        sudo_password=sudo_password,
        out_dir=out_dir,
        name=name,
    )


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# TPM PCR policy probe",
        "",
        "This artifact records a non-destructive PCR-policy seal/unseal probe.",
        "It does not provision AEGIS-Q's persistent NV index and does not prove the full hardware-backed freshness flow.",
        "",
        "## Results",
        "",
        f"- TCTI: `{report['config']['tcti']}`",
        f"- PCR list: `{report['config']['pcr_list']}`",
        f"- Current PCR digest file SHA-256: `{report['digests']['pcr_current_sha256']}`",
        f"- Drifted PCR digest file SHA-256: `{report['digests']['pcr_drift_sha256']}`",
        f"- Good unseal return code: `{report['results']['good_unseal_rc']}`",
        f"- Good unseal matches secret: `{report['results']['good_unseal_matches_secret']}`",
        f"- Drift policy return code: `{report['results']['drift_policy_rc']}`",
        f"- Drift unseal return code: `{report['results']['drift_unseal_rc']}`",
        f"- Drift rejected: `{report['results']['drift_rejected']}`",
        "",
        "## Conservative interpretation",
        "",
        "- This closes a transient PCR-policy probe: current PCR policy authorizes unseal, while a drifted PCR digest is rejected.",
        "- This is not a persistent freshness proof for the filesystem anchor.",
        "- Full closure still requires monotonic update semantics and mount/recovery behavior tied to the hardware-backed anchor state.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tcti", default=os.environ.get("PQC_TPM_TCTI", "device:/dev/tpmrm0"))
    ap.add_argument("--pcr-list", default="sha256:0,1,2,3")
    ap.add_argument("--sudo", action="store_true", default=True)
    ap.add_argument("--no-sudo", dest="sudo", action="store_false")
    ap.add_argument("--sudo-password-env", default="PQC_SUDO_PASSWORD")
    args = ap.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    required = [
        "tpm2_pcrread",
        "tpm2_createpolicy",
        "tpm2_createprimary",
        "tpm2_create",
        "tpm2_load",
        "tpm2_startauthsession",
        "tpm2_policypcr",
        "tpm2_unseal",
        "tpm2_flushcontext",
    ]
    missing = [tool for tool in required if shutil.which(tool) is None]
    if missing:
        raise SystemExit(f"missing tpm2-tools: {missing}")

    env = os.environ.copy()
    env["TSS2_TCTI"] = args.tcti
    sudo_password = os.environ.get(args.sudo_password_env, "") if args.sudo else None
    if args.sudo and not sudo_password:
        raise SystemExit(f"missing sudo password env {args.sudo_password_env}")

    commands: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="aegis_tpm_pcr_") as tmp_s:
        tmp = Path(tmp_s)
        secret = tmp / "secret.dat"
        secret.write_text("aegis-q-pcr-policy-secret", encoding="utf-8")

        commands.append(run_cmd(["tpm2_pcrread", "-o", "pcr.bin", args.pcr_list], cwd=tmp, env=env, sudo_password=sudo_password, name="pcrread", out_dir=out_dir))
        commands.append(chown_to_user(tmp / "pcr.bin", tmp, env, sudo_password, out_dir, "chown_pcr"))
        pcr_current = tmp / "pcr.bin"
        pcr_drift = tmp / "pcr_drift.bin"
        b = bytearray(pcr_current.read_bytes())
        b[-1] ^= 0xFF
        pcr_drift.write_bytes(b)

        commands.append(run_cmd(["tpm2_createpolicy", "--policy-pcr", "-l", args.pcr_list, "-f", "pcr.bin", "-L", "policy.dat"], cwd=tmp, env=env, sudo_password=sudo_password, name="createpolicy_current", out_dir=out_dir))
        commands.append(run_cmd(["tpm2_createprimary", "-C", "o", "-c", "primary.ctx"], cwd=tmp, env=env, sudo_password=sudo_password, name="createprimary", out_dir=out_dir))
        commands.append(run_cmd(["tpm2_create", "-C", "primary.ctx", "-i", "secret.dat", "-L", "policy.dat", "-u", "seal.pub", "-r", "seal.priv"], cwd=tmp, env=env, sudo_password=sudo_password, name="create_sealed", out_dir=out_dir))
        commands.append(run_cmd(["tpm2_load", "-C", "primary.ctx", "-u", "seal.pub", "-r", "seal.priv", "-c", "seal.ctx"], cwd=tmp, env=env, sudo_password=sudo_password, name="load_sealed", out_dir=out_dir))

        commands.append(run_cmd(["tpm2_startauthsession", "--policy-session", "-S", "good.sess"], cwd=tmp, env=env, sudo_password=sudo_password, name="start_good_session", out_dir=out_dir))
        commands.append(run_cmd(["tpm2_policypcr", "-S", "good.sess", "-l", args.pcr_list, "-f", "pcr.bin"], cwd=tmp, env=env, sudo_password=sudo_password, name="policypcr_current", out_dir=out_dir))
        good_unseal = run_cmd(["tpm2_unseal", "-c", "seal.ctx", "-p", "session:good.sess", "-o", "unsealed.dat"], cwd=tmp, env=env, sudo_password=sudo_password, name="unseal_current", out_dir=out_dir)
        commands.append(good_unseal)
        commands.append(chown_to_user(tmp / "unsealed.dat", tmp, env, sudo_password, out_dir, "chown_unsealed"))
        good_match = (tmp / "unsealed.dat").read_bytes() == secret.read_bytes()
        commands.append(run_cmd(["tpm2_flushcontext", "good.sess"], cwd=tmp, env=env, sudo_password=sudo_password, name="flush_good", out_dir=out_dir, allow_fail=True))

        commands.append(run_cmd(["tpm2_startauthsession", "--policy-session", "-S", "drift.sess"], cwd=tmp, env=env, sudo_password=sudo_password, name="start_drift_session", out_dir=out_dir))
        drift_policy = run_cmd(["tpm2_policypcr", "-S", "drift.sess", "-l", args.pcr_list, "-f", "pcr_drift.bin"], cwd=tmp, env=env, sudo_password=sudo_password, name="policypcr_drift", out_dir=out_dir, allow_fail=True)
        commands.append(drift_policy)
        drift_unseal = run_cmd(["tpm2_unseal", "-c", "seal.ctx", "-p", "session:drift.sess", "-o", "drift_unsealed.dat"], cwd=tmp, env=env, sudo_password=sudo_password, name="unseal_drift", out_dir=out_dir, allow_fail=True)
        commands.append(drift_unseal)
        commands.append(run_cmd(["tpm2_flushcontext", "drift.sess"], cwd=tmp, env=env, sudo_password=sudo_password, name="flush_drift", out_dir=out_dir, allow_fail=True))

        report = {
            "note": "Transient PCR policy probe only; not full filesystem freshness proof.",
            "config": {
                "tcti": args.tcti,
                "pcr_list": args.pcr_list,
                "sudo": args.sudo,
            },
            "digests": {
                "pcr_current_sha256": sha256(pcr_current),
                "pcr_drift_sha256": sha256(pcr_drift),
            },
            "results": {
                "good_unseal_rc": good_unseal["returncode"],
                "good_unseal_matches_secret": good_match,
                "drift_policy_rc": drift_policy["returncode"],
                "drift_unseal_rc": drift_unseal["returncode"],
                "drift_rejected": drift_policy["returncode"] != 0 or drift_unseal["returncode"] != 0,
            },
            "commands": commands,
        }

    json_path = out_dir / "tpm_pcr_policy_probe.json"
    md_path = out_dir / "tpm_pcr_policy_probe.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({"out_dir": str(out_dir), "drift_rejected": report["results"]["drift_rejected"]}, indent=2))
    return 0 if report["results"]["good_unseal_matches_secret"] and report["results"]["drift_rejected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
