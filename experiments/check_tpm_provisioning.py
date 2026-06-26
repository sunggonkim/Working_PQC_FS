#!/usr/bin/env python3
"""Record TPM provisioning state without claiming any freshness proof.

The purpose of this script is to provide a single conservative entry point
for later TPM freshness bundles:
  - check whether tpm2-tools are available,
  - record the configured TCTI,
  - optionally probe the current NV index if one is configured,
  - emit a JSON manifest that can be reused by later proof runs.

This does not create or provision any persistent TPM state.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "tpm_provisioning_probe"


def run(cmd: list[str], *, env: dict[str, str] | None = None, sudo_password: str | None = None) -> dict:
    actual_cmd = cmd
    stdin = None
    if sudo_password is not None:
        actual_cmd = ["sudo", "-S"] + cmd
        stdin = sudo_password + "\n"
    proc = subprocess.run(
        actual_cmd,
        cwd=ROOT,
        text=True,
        input=stdin,
        capture_output=True,
        env=env,
    )
    return {
        "command": actual_cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# TPM provisioning probe",
        "",
        "This artifact records TPM/TCTI/provisioning state only. It does not establish PCR sealing, monotonic freshness, or hardware-backed recovery.",
        "",
        "## Configuration",
        "",
        f"- NV index: `{report['configured']['nv_index']}`",
        f"- Requested TCTI: `{report['configured']['tcti'] or '<default>'}`",
        f"- Used sudo: `{report['configured'].get('sudo', False)}`",
        "",
        "## Tool availability",
        "",
    ]
    for tool, present in report["tpm2_tools"].items():
        lines.append(f"- {tool}: `{present}`")
    lines.extend(["", "## Probe results", ""])
    for check in report["checks"]:
        cmd = " ".join(check.get("command", []))
        tcti = check.get("tcti", "<unset>")
        stdout = check.get("stdout", "")
        stderr = check.get("stderr", "")
        lines.append(f"### `{cmd}`")
        lines.append("")
        lines.append(f"- TCTI: `{tcti}`")
        lines.append(f"- Return code: `{check.get('returncode')}`")
        if "tpm2_nvreadpublic" in cmd and stdout:
            for needle in ("0x", "name:", "hash algorithm:", "friendly:", "attributes:", "value:", "size:"):
                for line in stdout.splitlines():
                    if line.strip().startswith(needle):
                        lines.append(f"- `{line.strip()}`")
        elif "tpm2_getcap" in cmd and stdout:
            for key in ("TPM2_PT_FAMILY_INDICATOR", "TPM2_PT_MANUFACTURER", "TPM2_PT_VENDOR_STRING_2", "TPM2_PT_PCR_COUNT", "TPM2_PT_NV_INDEX_MAX", "TPM2_PT_MODES"):
                for i, line in enumerate(stdout.splitlines()):
                    if line.startswith(key):
                        snippet = [line.strip()]
                        snippet.extend(x.strip() for x in stdout.splitlines()[i + 1 : i + 3] if x.strip())
                        lines.append(f"- `{' / '.join(snippet)}`")
                        break
        elif "tpm2_pcrread" in cmd and stdout:
            pcr_lines = [line.strip() for line in stdout.splitlines() if line.strip().startswith(tuple(str(i) for i in range(8)))]
            lines.append(f"- PCR rows: `{len(pcr_lines)}`")
        if stderr:
            first = stderr.strip().splitlines()[0] if stderr.strip() else ""
            if first:
                lines.append(f"- First stderr line: `{first}`")
        lines.append("")
    lines.extend(
        [
            "## Conservative interpretation",
            "",
            "- A zero return code from `tpm2_nvreadpublic` records that the NV index exists and exposes owner read/write attributes.",
            "- A zero return code from `tpm2_pcrread` records current PCR values only; it is not PCR binding or PCR-drift rejection.",
            "- No monotonic freshness update or recovery verdict is claimed by this probe.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--nv-index", default=os.environ.get("PQC_TPM_NV_INDEX", "0x01500010"))
    ap.add_argument("--tcti", default=os.environ.get("PQC_TPM_TCTI", os.environ.get("TSS2_TCTI", "")))
    ap.add_argument(
        "--sudo",
        action="store_true",
        help="Run tpm2-tools through sudo -S. The password is read from --sudo-password-env.",
    )
    ap.add_argument("--sudo-password-env", default="PQC_SUDO_PASSWORD")
    ap.add_argument(
        "--probe-tcti",
        action="append",
        default=[],
        help="Additional explicit TCTI value to probe, e.g. device:/dev/tpmrm0. Can be repeated.",
    )
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "note": "Provisioning probe only; does not establish PCR sealing or hardware-backed freshness.",
        "tpm2_tools": {
            "tpm2_nvreadpublic": shutil.which("tpm2_nvreadpublic") is not None,
            "tpm2_nvread": shutil.which("tpm2_nvread") is not None,
            "tpm2_nvwrite": shutil.which("tpm2_nvwrite") is not None,
            "tpm2_getcap": shutil.which("tpm2_getcap") is not None,
        },
        "configured": {
            "nv_index": args.nv_index,
            "tcti": args.tcti,
            "sudo": args.sudo,
            "sudo_password_env": args.sudo_password_env if args.sudo else "",
        },
        "checks": [],
    }

    sudo_password = os.environ.get(args.sudo_password_env, "") if args.sudo else None
    if args.sudo and not sudo_password:
        report["checks"].append({
            "command": ["sudo", "-S"],
            "returncode": 1,
            "stdout": "",
            "stderr": f"missing sudo password env {args.sudo_password_env}",
        })

    tcti_values = []
    for value in [args.tcti, *args.probe_tcti]:
        if value and value not in tcti_values:
            tcti_values.append(value)
    if not tcti_values:
        tcti_values.append("")

    for tcti in tcti_values:
        env = os.environ.copy()
        if tcti:
            env["TSS2_TCTI"] = tcti
        label = tcti or "<default>"
        if shutil.which("tpm2_getcap"):
            item = run(["tpm2_getcap", "properties-fixed"], env=env, sudo_password=sudo_password)
            item["tcti"] = label
            report["checks"].append(item)
        if shutil.which("tpm2_nvreadpublic"):
            item = run(["tpm2_nvreadpublic", args.nv_index], env=env, sudo_password=sudo_password)
            item["tcti"] = label
            report["checks"].append(item)
        if shutil.which("tpm2_pcrread"):
            item = run(["tpm2_pcrread", "sha256:0,1,2,3,4,5,6,7"], env=env, sudo_password=sudo_password)
            item["tcti"] = label
            report["checks"].append(item)

    manifest = out_dir / "tpm_provisioning_probe.json"
    manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, out_dir / "tpm_provisioning_probe.md")
    print(json.dumps({"out_dir": str(out_dir), "checks": len(report["checks"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
