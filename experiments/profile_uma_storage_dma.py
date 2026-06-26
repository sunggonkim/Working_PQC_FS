#!/usr/bin/env python3
"""Optional profiler wrapper for the UMA / storage-DMA probe.

This is deliberately conservative:
  - If `nsys` is available, it profiles the probe command.
  - If requested and `ncu` is available, it profiles the checksum kernel in the
    existing io_uring_uvm path and exports the report to CSV.
  - Otherwise it records the exact command that should be run manually.
  - It does not infer DMA semantics or counter-backed evidence on its own.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "uma_storage_dma_profile"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--nvme-device", default="/dev/nvme0n1")
    ap.add_argument(
        "--include-um-smoke",
        action="store_true",
        help="Also profile the existing pqc_fuse --um-smoke path in the same bundle.",
    )
    ap.add_argument(
        "--include-managed-storage",
        action="store_true",
        help="Also profile the managed-buffer storage probe in the same bundle.",
    )
    ap.add_argument(
        "--trace",
        default="cuda,osrt,tegra-accelerators",
        help="Nsight Systems trace set to use when profiling is available.",
    )
    ap.add_argument(
        "--run-as-root",
        action="store_true",
        help="Run the profiler command under sudo -S when a password is available.",
    )
    ap.add_argument(
        "--sudo-password-env",
        default="PQC_SUDO_PASSWORD",
        help="Environment variable name containing the sudo password, if needed.",
    )
    ap.add_argument(
        "--nsys-output",
        default="uma_storage_dma_nsys",
        help="Output basename for Nsight Systems if available.",
    )
    ap.add_argument(
        "--include-ncu",
        action="store_true",
        help="Also run Nsight Compute on the existing io_uring_uvm checksum kernel.",
    )
    ap.add_argument(
        "--ncu-set",
        default="detailed",
        help="Nsight Compute set for the checksum-kernel profile.",
    )
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_cmd = [
        "python3",
        "experiments/run_uma_storage_dma_probe.py",
        "--nvme-device",
        args.nvme_device,
        "--out-dir",
        str(out_dir / "probe"),
    ]

    um_smoke_cmd = [str(ROOT / "build" / "pqc_fuse"), "--um-smoke"]
    managed_storage_cmd = [str(ROOT / "build" / "io_uring_uvm"), "--managed-buffer", args.nvme_device]

    nsys = shutil.which("nsys")
    ncu = shutil.which("ncu")
    runs = []

    def run_profiled(name: str, command: list[str], output_base: Path) -> dict:
        output_base.parent.mkdir(parents=True, exist_ok=True)
        if nsys:
            cmd = [nsys, "profile", f"--trace={args.trace}", "--output", str(output_base)] + command
            sudo_pw = os.environ.get(args.sudo_password_env, "")
            if args.run_as_root and sudo_pw:
                cmd = ["sudo", "-S"] + cmd
                proc = subprocess.run(
                    cmd,
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    input=sudo_pw + "\n",
                )
            else:
                proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
            mode = "nsys"
        else:
            cmd = command
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
            mode = "probe-only"

        if nsys and proc.returncode == 0:
            stats_cmd = [
                nsys,
                "stats",
                "--force-overwrite=true",
                "--format",
                "csv",
                "--report",
                "um_sum,um_cpu_page_faults_sum,cuda_api_sum,osrt_sum",
                "--output",
                str(output_base.parent / "stats"),
                str(output_base.with_suffix(".nsys-rep")),
            ]
            if args.run_as_root and os.environ.get(args.sudo_password_env, ""):
                stats_cmd = ["sudo", "-S"] + stats_cmd
                subprocess.run(
                    stats_cmd,
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    input=os.environ.get(args.sudo_password_env, "") + "\n",
                )
            else:
                subprocess.run(stats_cmd, cwd=ROOT, text=True, capture_output=True)
        return {
            "name": name,
            "mode": mode,
            "command": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "output_base": str(output_base),
        }

    def run_ncu(name: str, command: list[str], output_base: Path) -> dict:
        output_base.parent.mkdir(parents=True, exist_ok=True)
        sudo_pw = os.environ.get(args.sudo_password_env, "")
        if ncu:
            cmd = [
                ncu,
                "--set",
                args.ncu_set,
                "--kernel-name",
                "regex:checksum_kernel",
                "--target-processes",
                "all",
                "--export",
                str(output_base),
                "--force-overwrite",
            ] + command
            if args.run_as_root and sudo_pw:
                cmd = ["sudo", "-S"] + cmd
                proc = subprocess.run(
                    cmd,
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    input=sudo_pw + "\n",
                )
            else:
                proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)

            csv_path = output_base.with_suffix(".csv")
            import_cmd = [
                ncu,
                "--import",
                str(output_base.with_suffix(".ncu-rep")),
                "--csv",
            ]
            import_proc = None
            if proc.returncode == 0:
                with csv_path.open("w", encoding="utf-8") as f:
                    import_proc = subprocess.run(
                        import_cmd,
                        cwd=ROOT,
                        text=True,
                        stdout=f,
                        stderr=subprocess.PIPE,
                    )
            mode = "ncu"
        else:
            cmd = command
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
            import_cmd = []
            import_proc = None
            csv_path = output_base.with_suffix(".csv")
            mode = "probe-only"

        return {
            "name": name,
            "mode": mode,
            "command": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "output_base": str(output_base),
            "csv_path": str(csv_path),
            "import_command": import_cmd,
            "import_returncode": None if import_proc is None else import_proc.returncode,
            "import_stderr": None if import_proc is None else import_proc.stderr,
        }

    runs.append(run_profiled("raw_probe", probe_cmd, out_dir / "probe" / "raw_probe"))
    if args.include_um_smoke:
        runs.append(run_profiled("um_smoke", um_smoke_cmd, out_dir / "um_smoke" / "um_smoke"))
    if args.include_managed_storage:
        runs.append(
            run_profiled(
                "managed_storage_probe",
                managed_storage_cmd,
                out_dir / "managed_storage" / "managed_storage",
            )
        )
    if args.include_ncu:
        runs.append(
            run_ncu(
                "raw_probe_ncu",
                [str(ROOT / "build" / "io_uring_uvm"), args.nvme_device],
                out_dir / "ncu" / "io_uring_uvm_checksum",
            )
        )

    record = {
        "trace": args.trace,
        "run_as_root": args.run_as_root,
        "sudo_password_env": args.sudo_password_env,
        "include_um_smoke": args.include_um_smoke,
        "include_managed_storage": args.include_managed_storage,
        "include_ncu": args.include_ncu,
        "ncu_set": args.ncu_set,
        "runs": runs,
        "note": "Profiler wrapper only; not a proof of verified NVMe-to-UVM DMA semantics.",
    }
    (out_dir / "profile_uma_storage_dma.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "runs": len(runs), "returncodes": [r["returncode"] for r in runs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
