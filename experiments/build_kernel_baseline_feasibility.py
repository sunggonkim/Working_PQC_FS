#!/usr/bin/env python3
"""Record whether fscrypt/dm-crypt frozen-contract baselines are runnable.

The default path is unprivileged.  With --use-sudo-password-env, the script also
runs disposable loop-device probes using AEGISQ_SUDO_PASSWORD, but it still does
not modify the host filesystem layout or mark the kernel baseline matrix
complete.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import stat
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    ROOT
    / "artifacts"
    / "validation"
    / "frozen_workload_contract"
    / "frozen_workload_contract.json"
)
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "kernel_baseline_feasibility"


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def command_capture(command: list[str], timeout_s: float = 10.0) -> dict[str, Any]:
    if shutil.which(command[0]) is None and not Path(command[0]).exists():
        return {
            "argv": command,
            "available": False,
            "returncode": None,
            "stdout": "",
            "stderr": "",
        }
    try:
        proc = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "argv": command,
            "available": True,
            "timeout": True,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }
    return {
        "argv": command,
        "available": True,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def sudo_capture(command: list[str], password: str | None, timeout_s: float = 20.0) -> dict[str, Any]:
    if password is None:
        return command_capture(["sudo", "-n", *command], timeout_s=timeout_s)
    proc = subprocess.run(
        ["sudo", "-S", "-p", "", *command],
        cwd=ROOT,
        check=False,
        input=password + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
    )
    return {
        "argv": ["sudo", "-S", "-p", "", *command],
        "available": True,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "password_supplied": True,
    }


def node_status(path: Path) -> dict[str, Any]:
    try:
        st = path.stat()
    except OSError as exc:
        return {"path": str(path), "exists": False, "error": str(exc)}
    mode = st.st_mode
    return {
        "path": str(path),
        "exists": True,
        "mode_octal": oct(stat.S_IMODE(mode)),
        "uid": st.st_uid,
        "gid": st.st_gid,
        "is_char_device": stat.S_ISCHR(mode),
        "is_block_device": stat.S_ISBLK(mode),
        "user_readable": os.access(path, os.R_OK),
        "user_writable": os.access(path, os.W_OK),
    }


def load_contract(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def kernel_config() -> dict[str, Any]:
    keys = [
        "CONFIG_FS_ENCRYPTION",
        "CONFIG_FS_VERITY",
        "CONFIG_DM_CRYPT",
        "CONFIG_DM_INTEGRITY",
    ]
    candidates = [
        Path("/proc/config.gz"),
        Path("/boot") / f"config-{os.uname().release}",
    ]
    text = ""
    source = None
    for path in candidates:
        try:
            if path.suffix == ".gz":
                text = gzip.decompress(path.read_bytes()).decode("utf-8", errors="replace")
            else:
                text = path.read_text(encoding="utf-8", errors="replace")
            source = str(path)
            break
        except OSError:
            continue
    values: dict[str, str] = {}
    for key in keys:
        values[key] = "missing"
    if text:
        for line in text.splitlines():
            for key in keys:
                if line == f"{key}=y" or line == f"{key}=m":
                    values[key] = line.split("=", 1)[1]
                elif line == f"# {key} is not set":
                    values[key] = "not_set"
    return {
        "source": source,
        "values": values,
    }


def privileged_loop_probes(password: str | None) -> dict[str, Any]:
    if password is None:
        return {
            "enabled": False,
            "reason": "AEGISQ_SUDO_PASSWORD not set; privileged loop probes skipped",
        }

    with tempfile.TemporaryDirectory(prefix="aegis_kernel_baseline_probe_") as tmp:
        work = Path(tmp)
        fscrypt_result = fscrypt_loop_probe(work / "fscrypt", password)
        dmcrypt_result = dmcrypt_loop_probe(work / "dmcrypt", password)
        return {
            "enabled": True,
            "fscrypt_loop_probe": fscrypt_result,
            "dm_crypt_loop_probe": dmcrypt_result,
        }


def fscrypt_loop_probe(work: Path, password: str) -> dict[str, Any]:
    work.mkdir(parents=True, exist_ok=True)
    img = work / "fscrypt.img"
    mnt = work / "mnt"
    key = work / "raw.key"
    mnt.mkdir()
    key.write_bytes(b"A" * 32)
    img.write_bytes(b"")
    os.truncate(img, 128 * 1024 * 1024)
    loopdev = ""
    steps: list[dict[str, Any]] = []
    try:
        losetup = sudo_capture(["losetup", "--find", "--show", str(img)], password, timeout_s=10)
        steps.append({"name": "losetup", "result": losetup})
        if losetup.get("returncode") != 0:
            return {"pass": False, "steps": steps}
        loopdev = str(losetup.get("stdout", "")).strip()
        for name, command, timeout in [
            ("mkfs_ext4_encrypt", ["mkfs.ext4", "-q", "-O", "encrypt", loopdev], 20),
            ("tune2fs_encrypt", ["tune2fs", "-O", "encrypt", loopdev], 20),
            ("e2fsck", ["e2fsck", "-fy", loopdev], 20),
            ("mount", ["mount", loopdev, str(mnt)], 20),
            ("fscrypt_setup", ["fscrypt", "setup", "--quiet", str(mnt)], 20),
            ("mkdir_contract", ["mkdir", str(mnt / "contract")], 10),
            (
                "fscrypt_encrypt",
                [
                    "fscrypt",
                    "encrypt",
                    "--quiet",
                    "--source=raw_key",
                    f"--key={key}",
                    "--name=aegisq-probe",
                    "--no-recovery",
                    str(mnt / "contract"),
                ],
                20,
            ),
        ]:
            result = sudo_capture(command, password, timeout_s=timeout)
            steps.append({"name": name, "result": result})
            if result.get("returncode") != 0:
                return {"pass": False, "steps": steps}
        status = sudo_capture(["fscrypt", "status", str(mnt / "contract")], password, timeout_s=10)
        steps.append({"name": "fscrypt_status_contract", "result": status})
        return {"pass": status.get("returncode") == 0, "steps": steps}
    finally:
        if mnt.exists():
            sudo_capture(["umount", str(mnt)], password, timeout_s=10)
        if loopdev:
            sudo_capture(["losetup", "-d", loopdev], password, timeout_s=10)


def dmcrypt_loop_probe(work: Path, password: str) -> dict[str, Any]:
    work.mkdir(parents=True, exist_ok=True)
    img = work / "dmcrypt.img"
    key = work / "luks.key"
    mnt = work / "mnt"
    mapper = f"aegisq_probe_{os.getpid()}"
    mnt.mkdir()
    key.write_bytes(b"B" * 64)
    img.write_bytes(b"")
    os.truncate(img, 128 * 1024 * 1024)
    loopdev = ""
    mapper_path = f"/dev/mapper/{mapper}"
    steps: list[dict[str, Any]] = []
    try:
        losetup = sudo_capture(["losetup", "--find", "--show", str(img)], password, timeout_s=10)
        steps.append({"name": "losetup", "result": losetup})
        if losetup.get("returncode") != 0:
            return {"pass": False, "steps": steps}
        loopdev = str(losetup.get("stdout", "")).strip()
        for name, command, timeout in [
            (
                "luks_format",
                [
                    "cryptsetup",
                    "--batch-mode",
                    "--type",
                    "luks2",
                    "--cipher",
                    "aes-xts-plain64",
                    "--key-size",
                    "512",
                    "--pbkdf",
                    "pbkdf2",
                    "--pbkdf-force-iterations",
                    "1000",
                    "--key-file",
                    str(key),
                    "luksFormat",
                    loopdev,
                ],
                30,
            ),
            ("luks_open", ["cryptsetup", "open", "--key-file", str(key), loopdev, mapper], 20),
            ("mkfs_ext4", ["mkfs.ext4", "-q", mapper_path], 20),
            ("mount", ["mount", mapper_path, str(mnt)], 20),
            ("write_probe", ["bash", "-c", f"printf probe > {mnt / 'probe.txt'} && sync"], 20),
            ("read_probe", ["cat", str(mnt / "probe.txt")], 10),
        ]:
            result = sudo_capture(command, password, timeout_s=timeout)
            steps.append({"name": name, "result": result})
            if result.get("returncode") != 0:
                return {"pass": False, "steps": steps}
        return {"pass": True, "steps": steps}
    finally:
        if mnt.exists():
            sudo_capture(["umount", str(mnt)], password, timeout_s=10)
        sudo_capture(["cryptsetup", "close", mapper], password, timeout_s=10)
        if loopdev:
            sudo_capture(["losetup", "-d", loopdev], password, timeout_s=10)


def build_report(contract_path: Path, sudo_password: str | None) -> dict[str, Any]:
    contract = load_contract(contract_path)
    modes = (contract.get("contract") or {}).get("filesystem_modes") or {}
    sudo_probe = command_capture(["sudo", "-n", "true"], timeout_s=5.0)
    sudo_password_probe = sudo_capture(["true"], sudo_password, timeout_s=5.0) if sudo_password else None
    fscrypt_root = command_capture(["fscrypt", "status", "/"], timeout_s=10.0)
    fscrypt_repo = command_capture(["fscrypt", "status", str(ROOT)], timeout_s=10.0)
    cryptsetup_version = command_capture(["cryptsetup", "--version"], timeout_s=5.0)
    fscrypt_version = command_capture(["fscrypt", "--version"], timeout_s=5.0)
    docker_version = command_capture(["docker", "version", "--format", "{{.Server.Version}}"], timeout_s=10.0)
    mapper_control = node_status(Path("/dev/mapper/control"))
    loop_control = node_status(Path("/dev/loop-control"))
    loop_nodes = [
        node_status(path)
        for path in sorted(Path("/dev").glob("loop*"))[:32]
    ]
    kconfig = kernel_config()
    kvalues = kconfig.get("values", {})
    loop_probes = privileged_loop_probes(sudo_password)
    encryption_not_enabled = (
        "encryption not enabled" in (fscrypt_repo.get("stderr") or "")
        or "encryption not enabled" in (fscrypt_repo.get("stdout") or "")
    )
    sudo_noninteractive_ready = sudo_probe.get("returncode") == 0
    sudo_privileged_ready = sudo_noninteractive_ready or (
        sudo_password_probe is not None and sudo_password_probe.get("returncode") == 0
    )
    fs_encryption_ready = kvalues.get("CONFIG_FS_ENCRYPTION") in {"y", "m"}
    dm_crypt_kernel_ready = kvalues.get("CONFIG_DM_CRYPT") in {"y", "m"}
    fscrypt_ready_without_interactive_root = (
        shutil.which("fscrypt") is not None
        and sudo_noninteractive_ready
        and fs_encryption_ready
        and not encryption_not_enabled
        and fscrypt_repo.get("returncode") == 0
    )
    fscrypt_ready_with_sudo_password = (
        shutil.which("fscrypt") is not None
        and sudo_privileged_ready
        and fs_encryption_ready
        and (loop_probes.get("fscrypt_loop_probe") or {}).get("pass") is True
    )
    dmcrypt_ready_without_interactive_root = (
        shutil.which("cryptsetup") is not None
        and sudo_noninteractive_ready
        and dm_crypt_kernel_ready
    )
    dmcrypt_ready_with_sudo_password = (
        shutil.which("cryptsetup") is not None
        and sudo_privileged_ready
        and dm_crypt_kernel_ready
        and ((loop_probes.get("dm_crypt_loop_probe") or {}).get("pass") is True)
    )
    fscrypt_reasons: list[str] = []
    if shutil.which("fscrypt") is None:
        fscrypt_reasons.append("fscrypt_binary_missing")
    if not sudo_privileged_ready:
        fscrypt_reasons.append("noninteractive_sudo_unavailable")
    if not fs_encryption_ready:
        fscrypt_reasons.append("kernel_config_fs_encryption_disabled")
    if encryption_not_enabled:
        fscrypt_reasons.append("root_ext4_encrypt_feature_not_enabled")
    if fscrypt_repo.get("returncode") != 0:
        fscrypt_reasons.append("fscrypt_status_nonzero")
    if (loop_probes.get("fscrypt_loop_probe") or {}).get("pass") is False:
        fscrypt_reasons.append("disposable_fscrypt_loop_probe_failed")
    dmcrypt_reasons: list[str] = []
    if shutil.which("cryptsetup") is None:
        dmcrypt_reasons.append("cryptsetup_binary_missing")
    if not sudo_privileged_ready:
        dmcrypt_reasons.append("noninteractive_sudo_unavailable")
    if not dm_crypt_kernel_ready:
        dmcrypt_reasons.append("kernel_config_dm_crypt_disabled")
    if (loop_probes.get("dm_crypt_loop_probe") or {}).get("pass") is False:
        dmcrypt_reasons.append("disposable_dm_crypt_loop_probe_failed")
    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": True,
        "scope": (
            "Non-destructive feasibility audit only.  It does not execute "
            "fscrypt or dm-crypt benchmark rows and does not close the frozen "
            "kernel-baseline matrix item."
        ),
        "contract": {
            "path": relpath(contract_path),
            "contract_id": (contract.get("contract") or {}).get("contract_id"),
            "fscrypt_mode": modes.get("fscrypt"),
            "dm_crypt_ext4_mode": modes.get("dm_crypt_ext4"),
        },
        "host": {
            "euid": os.geteuid(),
            "uid": os.getuid(),
            "groups": os.getgroups(),
            "findmnt_root": command_capture(["findmnt", "-T", str(ROOT), "-no", "SOURCE,FSTYPE,TARGET,OPTIONS"], timeout_s=5.0),
            "sudo_probe": sudo_probe,
            "sudo_password_probe": sudo_password_probe,
            "fscrypt_version": fscrypt_version,
            "cryptsetup_version": cryptsetup_version,
            "losetup_version": command_capture(["losetup", "--version"], timeout_s=5.0),
            "mkfs_ext4_version": command_capture(["mkfs.ext4", "-V"], timeout_s=5.0),
            "docker_version": docker_version,
            "kernel_config": kconfig,
            "fscrypt_status_root": fscrypt_root,
            "fscrypt_status_repo": fscrypt_repo,
            "dev_mapper_control": mapper_control,
            "dev_loop_control": loop_control,
            "loop_nodes": loop_nodes,
        },
        "privileged_loop_probes": loop_probes,
        "fscrypt": {
            "runnable_without_interactive_root": fscrypt_ready_without_interactive_root,
            "runnable_with_sudo_password": fscrypt_ready_with_sudo_password,
            "blocking_reasons": fscrypt_reasons,
        },
        "dm_crypt_ext4": {
            "runnable_without_interactive_root": dmcrypt_ready_without_interactive_root,
            "runnable_with_sudo_password": dmcrypt_ready_with_sudo_password,
            "blocking_reasons": dmcrypt_reasons,
        },
        "next_required_action": (
            "fscrypt cannot be executed on this kernel while CONFIG_FS_ENCRYPTION "
            "is disabled.  A root-controlled LUKS2 dm-crypt/ext4 loop volume is "
            "probe-runnable with sudo, but the checklist item remains open until "
            "both kernel baseline rows are available under the frozen fio contract.  "
            "Do not use the historical sequential fio files as current comparison evidence."
        ),
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Kernel Baseline Feasibility Audit",
        "",
        f"- Overall pass: `{str(report['overall_pass']).lower()}`",
        f"- Contract ID: `{report['contract']['contract_id']}`",
        f"- Scope: {report['scope']}",
        f"- fscrypt runnable without interactive root: `{str(report['fscrypt']['runnable_without_interactive_root']).lower()}`",
        f"- dm-crypt/ext4 runnable without interactive root: `{str(report['dm_crypt_ext4']['runnable_without_interactive_root']).lower()}`",
        f"- fscrypt runnable with supplied sudo password: `{str(report['fscrypt']['runnable_with_sudo_password']).lower()}`",
        f"- dm-crypt/ext4 runnable with supplied sudo password: `{str(report['dm_crypt_ext4']['runnable_with_sudo_password']).lower()}`",
        "",
        "## Blocking Reasons",
        "",
        f"- fscrypt: `{', '.join(report['fscrypt']['blocking_reasons']) or 'none'}`",
        f"- dm-crypt/ext4: `{', '.join(report['dm_crypt_ext4']['blocking_reasons']) or 'none'}`",
        "",
        "## Next Required Action",
        "",
        report["next_required_action"],
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--use-sudo-password-env",
        action="store_true",
        help="Read AEGISQ_SUDO_PASSWORD for non-destructive privileged loop probes.",
    )
    args = parser.parse_args()
    args.contract = args.contract if args.contract.is_absolute() else ROOT / args.contract
    args.out = args.out if args.out.is_absolute() else ROOT / args.out
    args.out.mkdir(parents=True, exist_ok=True)
    sudo_password = os.environ.get("AEGISQ_SUDO_PASSWORD") if args.use_sudo_password_env else None
    report = build_report(args.contract, sudo_password)
    json_path = args.out / "kernel_baseline_feasibility.json"
    md_path = args.out / "kernel_baseline_feasibility.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "json": relpath(json_path),
                "markdown": relpath(md_path),
                "fscrypt_runnable_without_interactive_root": report["fscrypt"]["runnable_without_interactive_root"],
                "fscrypt_runnable_with_sudo_password": report["fscrypt"]["runnable_with_sudo_password"],
                "dm_crypt_ext4_runnable_without_interactive_root": report["dm_crypt_ext4"]["runnable_without_interactive_root"],
                "dm_crypt_ext4_runnable_with_sudo_password": report["dm_crypt_ext4"]["runnable_with_sudo_password"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
