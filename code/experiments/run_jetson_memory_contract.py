#!/usr/bin/env python3
"""Gate 0.10-S0 CUDA/Jetson memory capability and claim contract."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build"
PROBE = BUILD / "jetson_memory_contract_probe"
OUT = ROOT / "artifacts" / "validation" / "jetson_memory_contract"

OFFICIAL_BASIS = [
    {
        "name": "NVIDIA CUDA for Tegra memory/coherency guidance",
        "url": "https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html",
        "used_for": [
            "Tegra memory model scope",
            "host registration and pageable memory caution",
            "UVM and coherency claim boundary",
        ],
    },
    {
        "name": "NVIDIA GPUDirect RDMA caveats",
        "url": "https://docs.nvidia.com/cuda/gpudirect-rdma/index.html",
        "used_for": [
            "GPUDirect/RDMA applicability boundary",
            "non-claim guard for NVMe-to-UVM/direct DMA",
        ],
    },
]


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def run_command(argv: list[str], timeout: float = 10.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            argv,
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return {
            "argv": argv,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "argv": argv,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timed_out": True,
        }


def list_paths(path: Path) -> list[str]:
    if not path.exists():
        return []
    if path.is_file():
        return [str(path)]
    return sorted(str(candidate) for candidate in path.iterdir())


def proc_modules() -> list[str]:
    text = read_file(Path("/proc/modules")) or ""
    modules = []
    for line in text.splitlines():
        parts = line.split()
        if parts:
            modules.append(parts[0])
    return sorted(modules)


def run_probe(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not PROBE.exists():
        return (
            {
                "available": False,
                "reason": f"missing build artifact: {relpath(PROBE)}",
                "path": relpath(PROBE),
            },
            {
                "schema_version": 1,
                "device_count": 0,
                "devices": [],
                "probes": {},
            },
        )

    result = run_command([str(PROBE)], timeout=30.0)
    (out_dir / "jetson_memory_probe.stdout.json").write_text(
        result["stdout"], encoding="utf-8")
    (out_dir / "jetson_memory_probe.stderr.txt").write_text(
        result["stderr"], encoding="utf-8")
    try:
        payload = json.loads(result["stdout"])
        parse_error = None
    except json.JSONDecodeError as exc:
        payload = {"schema_version": 1, "device_count": 0, "devices": [],
                   "probes": {}}
        parse_error = repr(exc)
    return (
        {
            "available": True,
            "path": relpath(PROBE),
            "returncode": result["returncode"],
            "timed_out": result["timed_out"],
            "stdout_path": relpath(out_dir / "jetson_memory_probe.stdout.json"),
            "stderr_path": relpath(out_dir / "jetson_memory_probe.stderr.txt"),
            "json_parse_error": parse_error,
        },
        payload,
    )


def platform_manifest() -> dict[str, Any]:
    modules = proc_modules()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "uname": platform.uname()._asdict(),
        "kernel_release": platform.release(),
        "nv_tegra_release": read_file(Path("/etc/nv_tegra_release")),
        "l4t_version_files": {
            "/etc/nv_tegra_release": read_file(Path("/etc/nv_tegra_release")),
            "/etc/nv_boot_control.conf": read_file(Path("/etc/nv_boot_control.conf")),
        },
        "nvcc_version": run_command(["nvcc", "--version"], timeout=10.0),
        "nvidia_smi": run_command(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            timeout=10.0,
        ) if shutil.which("nvidia-smi") else {
            "argv": ["nvidia-smi"],
            "returncode": None,
            "stdout": "",
            "stderr": "nvidia-smi not found",
            "timed_out": False,
        },
        "device_nodes": {
            "/dev/nvidia*": sorted(
                str(path) for path in Path("/dev").glob("nvidia*")),
            "/dev/nvidia-caps": list_paths(Path("/dev/nvidia-caps")),
            "/dev/dma_heap": list_paths(Path("/dev/dma_heap")),
            "/dev/cpu_dma_latency": Path("/dev/cpu_dma_latency").exists(),
        },
        "sysfs": {
            "/sys/class/dma_heap": list_paths(Path("/sys/class/dma_heap")),
            "/sys/class/infiniband": list_paths(Path("/sys/class/infiniband")),
            "/sys/kernel/dmabuf": list_paths(Path("/sys/kernel/dmabuf")),
        },
        "kernel_modules_relevant": [
            module for module in modules
            if module in {
                "nvidia",
                "nvidia_uvm",
                "nvidia_modeset",
                "nvidia_peermem",
                "nv_peer_mem",
                "ib_core",
                "mlx5_core",
                "mlx5_ib",
            }
        ],
    }


def first_device(probe_payload: dict[str, Any]) -> dict[str, Any]:
    devices = probe_payload.get("devices", [])
    if isinstance(devices, list) and devices:
        first = devices[0]
        return first if isinstance(first, dict) else {}
    return {}


def probe_status(probe_payload: dict[str, Any], name: str) -> str:
    probe = probe_payload.get("probes", {}).get(name, {})
    return str(probe.get("status", "missing")) if isinstance(probe, dict) else "missing"


def claim(name: str, level: str, eligible: bool, evidence: list[str],
          guard: str) -> dict[str, Any]:
    return {
        "name": name,
        "evidence_level": level,
        "paper_mechanism_eligible": eligible,
        "evidence": evidence,
        "negative_guard": guard,
    }


def claim_contract(probe_payload: dict[str, Any],
                   manifest: dict[str, Any]) -> list[dict[str, Any]]:
    device = first_device(probe_payload)
    device_count = int(probe_payload.get("device_count", 0) or 0)
    host_alloc = probe_status(probe_payload, "cudaHostAlloc")
    host_register = probe_status(probe_payload, "cudaHostRegister")
    managed = probe_status(probe_payload, "cudaMallocManaged")
    managed_prefetch = probe_status(probe_payload, "cudaMemPrefetchAsyncManaged")
    pageable_prefetch = probe_status(probe_payload, "pageableMemoryPrefetch")
    mapped = probe_status(probe_payload, "cudaHostAllocMappedKernel")

    has_dma_heap = bool(manifest.get("device_nodes", {}).get("/dev/dma_heap"))
    rdma_devices = manifest.get("sysfs", {}).get("/sys/class/infiniband", [])
    modules = set(manifest.get("kernel_modules_relevant", []))
    gpudirect_attr = int(device.get("gpudirect_rdma_supported", 0) or 0)
    peermem = "nvidia_peermem" in modules or "nv_peer_mem" in modules

    return [
        claim(
            "cudaDeviceProp/cudaDeviceGetAttribute inventory",
            "local_probe" if device_count > 0 else "unavailable",
            device_count > 0,
            [
                f"device_count={device_count}",
                f"selected_device_name={device.get('name')}",
                f"compute_capability={device.get('compute_capability')}",
            ],
            "No CUDA/Jetson hardware mechanism claim may be made if device_count is zero or the property is absent.",
        ),
        claim(
            "pinned host allocation via cudaHostAlloc",
            "local_probe_pass" if host_alloc == "pass" else "local_probe_fail",
            host_alloc == "pass",
            [f"cudaHostAlloc={host_alloc}", f"mapped_kernel={mapped}"],
            "Do not claim pinned-memory benefit or zero-copy unless a production mounted-path benchmark uses this path.",
        ),
        claim(
            "registered host memory via cudaHostRegister",
            "local_probe_pass" if host_register == "pass" else "local_probe_fail",
            host_register == "pass",
            [
                f"hostRegisterSupported={device.get('host_register_supported')}",
                f"cudaHostRegister={host_register}",
            ],
            "Do not claim registered host memory for production I/O unless the mounted path records this allocation mode.",
        ),
        claim(
            "managed memory / UVM allocation",
            "local_probe_pass" if managed == "pass" else "local_probe_fail",
            managed == "pass",
            [
                f"managedMemory={device.get('managed_memory')}",
                f"concurrentManagedAccess={device.get('concurrent_managed_access')}",
                f"cudaMallocManaged={managed}",
                f"cudaMemPrefetchAsyncManaged={managed_prefetch}",
            ],
            "Do not claim direct storage-to-UVM DMA or production UVM benefit from allocation-only evidence.",
        ),
        claim(
            "pageable memory access",
            "local_probe_pass" if pageable_prefetch == "pass" else "non_claim",
            pageable_prefetch == "pass",
            [
                f"pageableMemoryAccess={device.get('pageable_memory_access')}",
                f"pageableMemoryAccessUsesHostPageTables={device.get('pageable_memory_access_uses_host_page_tables')}",
                f"pageableMemoryPrefetch={pageable_prefetch}",
            ],
            "Do not claim coherent pageable GPU access unless the probe passes and the mounted path uses pageable memory intentionally.",
        ),
        claim(
            "dma-buf export/import",
            "kernel_interface_observed_not_cuda_proven" if has_dma_heap else "non_claim",
            False,
            [
                f"/dev/dma_heap_present={has_dma_heap}",
                "no CUDA dma-buf export/import production probe is implemented",
            ],
            "dma-buf zero-copy remains a non-claim until CUDA allocation export/import and mounted-path use are proven.",
        ),
        claim(
            "GPUDirect/RDMA applicability",
            "environment_possible_not_production" if (
                gpudirect_attr and peermem and rdma_devices
            ) else "non_claim",
            False,
            [
                f"cudaDevAttrGPUDirectRDMASupported={gpudirect_attr}",
                f"nvidia_peermem_or_nv_peer_mem={peermem}",
                f"infiniband_devices={rdma_devices}",
            ],
            "GPUDirect/RDMA and direct NVMe-to-UVM DMA remain non-claims without driver, peer-memory, RDMA/NVMe, and production mounted-path evidence.",
        ),
        claim(
            "NVMe-to-UVM direct DMA",
            "non_claim",
            False,
            [
                "no NVMe peer-DMA proof",
                "no UVM direct-DMA proof",
                "no production mounted-path trace",
            ],
            "The paper must not claim direct NVMe-to-UVM DMA.",
        ),
    ]


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "jetson_memory_contract.json"
    md_path = out_dir / "jetson_memory_contract.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")

    lines = [
        "# Jetson Memory Contract",
        "",
        f"- Generated: `{payload['generated_utc']}`",
        f"- Overall pass: `{str(payload['overall_pass']).lower()}`",
        f"- Probe executable: `{payload['probe_run'].get('path')}`",
        f"- Raw stdout: `{payload['probe_run'].get('stdout_path')}`",
        f"- Raw stderr: `{payload['probe_run'].get('stderr_path')}`",
        "",
        "## Official Basis",
        "",
    ]
    for basis in payload["official_basis"]:
        lines.append(f"- [{basis['name']}]({basis['url']})")
    lines.extend([
        "",
        "## Device Summary",
        "",
        f"- Device count: `{payload['probe_payload'].get('device_count')}`",
        f"- CUDA runtime version: `{payload['probe_payload'].get('cuda_runtime_version')}`",
        f"- CUDA driver version: `{payload['probe_payload'].get('cuda_driver_version')}`",
        "",
        "## Claim Contract",
        "",
        "| Claim | Evidence level | Paper mechanism eligible | Evidence |",
        "| --- | --- | --- | --- |",
    ])
    for item in payload["claim_contract"]:
        lines.append(
            f"| {item['name']} | `{item['evidence_level']}` | "
            f"`{str(item['paper_mechanism_eligible']).lower()}` | "
            f"{'; '.join(item['evidence'])} |"
        )
    lines.extend([
        "",
        "## Negative Claim Guard",
        "",
        payload["negative_claim_guard"],
        "",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    probe_run, probe_payload = run_probe(OUT)
    manifest = platform_manifest()
    claims = claim_contract(probe_payload, manifest)
    required_names = {
        "cudaDeviceProp/cudaDeviceGetAttribute inventory",
        "pinned host allocation via cudaHostAlloc",
        "registered host memory via cudaHostRegister",
        "managed memory / UVM allocation",
        "pageable memory access",
        "dma-buf export/import",
        "GPUDirect/RDMA applicability",
        "NVMe-to-UVM direct DMA",
    }
    present_names = {item["name"] for item in claims}
    payload = {
        "schema_version": 1,
        "generated_by": "code/experiments/run_jetson_memory_contract.py",
        "generated_utc": now_utc(),
        "scope": "Gate 0.10-S0 CUDA/Jetson memory capability and claim contract.",
        "official_basis": OFFICIAL_BASIS,
        "probe_run": probe_run,
        "probe_payload": probe_payload,
        "platform_manifest": manifest,
        "claim_contract": claims,
        "negative_claim_guard": (
            "No paper or README text may claim direct NVMe-to-UVM DMA, "
            "GPUDirect/RDMA, dma-buf zero-copy, pageable-memory coherency, "
            "UVM production benefit, or pinned/registered host-memory benefit "
            "unless this contract reports paper_mechanism_eligible=true for "
            "that mechanism and a later production mounted-path artifact uses it."
        ),
        "overall_pass": (
            probe_run.get("available") is True and
            probe_run.get("returncode") == 0 and
            probe_run.get("json_parse_error") is None and
            required_names == present_names
        ),
        "paper_text_status": "not_updated",
        "parent_checklist_closed": False,
    }
    write_outputs(OUT, payload)
    print(json.dumps({
        "overall_pass": payload["overall_pass"],
        "json": relpath(OUT / "jetson_memory_contract.json"),
        "markdown": relpath(OUT / "jetson_memory_contract.md"),
        "device_count": probe_payload.get("device_count"),
        "non_claims": [
            item["name"] for item in claims
            if item["paper_mechanism_eligible"] is False
        ],
    }, indent=2, sort_keys=True))
    return 0 if payload["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
