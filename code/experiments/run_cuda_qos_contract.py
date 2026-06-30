#!/usr/bin/env python3
"""Gate 0.12-S0 CUDA scheduling/QoS trace and claim contract."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build"
PROBE = BUILD / "cuda_qos_contract_probe"
OUT = ROOT / "artifacts" / "validation" / "cuda_qos_contract"
CODE = ROOT / "code"

OFFICIAL_BASIS = [
    {
        "name": "NVIDIA CUDA Runtime API stream management",
        "url": "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html",
        "used_for": [
            "cudaDeviceGetStreamPriorityRange",
            "cudaStreamCreateWithPriority",
            "cudaStreamSynchronize and stream-scoped completion boundaries",
        ],
    },
    {
        "name": "NVIDIA CUDA Runtime API memory management",
        "url": "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html",
        "used_for": [
            "cudaMemPrefetchAsync",
            "cudaStreamAttachMemAsync",
            "managed-memory movement and attach claim boundary",
        ],
    },
    {
        "name": "NVIDIA CUPTI documentation",
        "url": "https://docs.nvidia.com/cupti/",
        "used_for": [
            "CUPTI availability versus retained trace evidence",
            "non-claim boundary for hardware-counter-backed QoS",
        ],
    },
]

CUDA_CALLS = [
    "cudaDeviceGetStreamPriorityRange",
    "cudaStreamCreateWithPriority",
    "cudaStreamCreateWithFlags",
    "cudaStreamCreate",
    "cudaStreamAttachMemAsync",
    "cudaMemPrefetchAsync",
    "cudaMemcpyAsync",
    "cudaMemcpy",
    "cudaStreamSynchronize",
    "cudaDeviceSynchronize",
    "cudaEventCreate",
    "cudaEventRecord",
    "cudaEventSynchronize",
    "cudaEventElapsedTime",
    "cudaLaunchKernel",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def run_command(argv: list[str], timeout: float = 20.0) -> dict[str, Any]:
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
    except FileNotFoundError:
        return {
            "argv": argv,
            "returncode": None,
            "stdout": "",
            "stderr": "command not found",
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


def write_raw(out_dir: Path, stem: str, result: dict[str, Any]) -> dict[str, str]:
    stdout_path = out_dir / f"{stem}.stdout.txt"
    stderr_path = out_dir / f"{stem}.stderr.txt"
    stdout_path.write_text(str(result.get("stdout", "")), encoding="utf-8")
    stderr_path.write_text(str(result.get("stderr", "")), encoding="utf-8")
    return {
        "stdout_path": relpath(stdout_path),
        "stderr_path": relpath(stderr_path),
    }


def production_sources() -> list[Path]:
    return sorted(
        path for path in CODE.iterdir()
        if path.is_file() and path.suffix in {".c", ".h", ".cu", ".cuh", ".cpp", ".hpp"}
    )


def source_scan() -> dict[str, Any]:
    per_file: list[dict[str, Any]] = []
    aggregate_calls = {name: 0 for name in CUDA_CALLS}
    aggregate_kernel_launches: list[dict[str, Any]] = []

    launch_re = re.compile(r"([A-Za-z_][A-Za-z0-9_:]*)\s*<<<([^>]*)>>>")
    for path in production_sources():
        text = read_text(path)
        lines = text.splitlines()
        calls: dict[str, list[int]] = {name: [] for name in CUDA_CALLS}
        launches: list[dict[str, Any]] = []
        for line_no, line in enumerate(lines, start=1):
            for name in CUDA_CALLS:
                if name in line:
                    calls[name].append(line_no)
                    aggregate_calls[name] += line.count(name)
            for match in launch_re.finditer(line):
                launch = {
                    "file": relpath(path),
                    "line": line_no,
                    "kernel": match.group(1),
                    "configuration_text": match.group(2).strip(),
                }
                launches.append(launch)
                aggregate_kernel_launches.append(launch)
        if any(calls.values()) or launches:
            per_file.append({
                "file": relpath(path),
                "cuda_calls": {name: locs for name, locs in calls.items() if locs},
                "kernel_launches": launches,
            })

    return {
        "source_root": relpath(CODE),
        "files_scanned": [relpath(path) for path in production_sources()],
        "per_file": per_file,
        "aggregate_call_counts": aggregate_calls,
        "kernel_launches": aggregate_kernel_launches,
        "production_uses_stream_priority": aggregate_calls["cudaStreamCreateWithPriority"] > 0,
        "production_uses_nonblocking_stream": aggregate_calls["cudaStreamCreateWithFlags"] > 0,
        "production_uses_prefetch": aggregate_calls["cudaMemPrefetchAsync"] > 0,
        "production_uses_attach": aggregate_calls["cudaStreamAttachMemAsync"] > 0,
        "production_uses_stream_sync": aggregate_calls["cudaStreamSynchronize"] > 0,
        "production_uses_device_sync": aggregate_calls["cudaDeviceSynchronize"] > 0,
    }


def find_existing(paths: list[str]) -> list[str]:
    return [path for path in paths if Path(path).exists()]


def profiler_availability(out_dir: Path) -> dict[str, Any]:
    nsys = shutil.which("nsys")
    ncu = shutil.which("ncu")
    nvprof = shutil.which("nvprof")
    cupti_roots = sorted(str(path) for path in Path("/usr/local").glob("cuda*/extras/CUPTI"))
    cupti_headers = sorted(str(path) for path in Path("/usr/local").glob("cuda*/extras/CUPTI/include/cupti.h"))
    cupti_libs = sorted(str(path) for path in Path("/usr/local").glob("cuda*/extras/CUPTI/lib64/libcupti.so*"))
    sample_dirs = find_existing([
        "/usr/local/cuda/extras/CUPTI/samples/pm_sampling",
        "/usr/local/cuda-13.0/extras/CUPTI/samples/pm_sampling",
        "/usr/local/cuda-12.6/extras/CUPTI/samples/pm_sampling",
    ])
    versions: dict[str, Any] = {}
    if nsys:
        result = run_command([nsys, "--version"], timeout=10.0)
        versions["nsys"] = result | write_raw(out_dir, "nsys_version", result)
    if ncu:
        result = run_command([ncu, "--version"], timeout=10.0)
        versions["ncu"] = result | write_raw(out_dir, "ncu_version", result)
    if nvprof:
        result = run_command([nvprof, "--version"], timeout=10.0)
        versions["nvprof"] = result | write_raw(out_dir, "nvprof_version", result)
    return {
        "nsys_available": bool(nsys),
        "nsys_path": nsys,
        "ncu_available": bool(ncu),
        "ncu_path": ncu,
        "nvprof_available": bool(nvprof),
        "nvprof_path": nvprof,
        "cupti_roots": cupti_roots,
        "cupti_headers": cupti_headers,
        "cupti_libs": cupti_libs,
        "cupti_pm_sampling_samples": sample_dirs,
        "version_commands": versions,
        "retained_profiler_trace": False,
        "retained_profiler_trace_reason": (
            "This contract records profiler availability only. It does not run "
            "Nsight/CUPTI profiling, so it cannot support a hardware-counter or "
            "foreground-inference QoS claim."
        ),
    }


def run_probe(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not PROBE.exists():
        return (
            {
                "available": False,
                "path": relpath(PROBE),
                "reason": "probe binary is missing; run cmake and build first",
            },
            {"overall_pass": False},
        )
    result = run_command([str(PROBE)], timeout=30.0)
    raw_paths = write_raw(out_dir, "cuda_qos_contract_probe", result)
    try:
        payload = json.loads(result["stdout"])
        parse_error = None
    except json.JSONDecodeError as exc:
        payload = {"overall_pass": False}
        parse_error = repr(exc)
    return (
        {
            "available": True,
            "path": relpath(PROBE),
            "returncode": result["returncode"],
            "timed_out": result["timed_out"],
            "json_parse_error": parse_error,
            **raw_paths,
        },
        payload,
    )


def status_of(payload: dict[str, Any], key: str) -> str:
    node = payload.get(key)
    if isinstance(node, dict):
        return str(node.get("status", "missing"))
    return "missing"


def claim_verdicts(
    probe_payload: dict[str, Any],
    scan: dict[str, Any],
    profiler: dict[str, Any],
) -> list[dict[str, Any]]:
    profiler_trace = bool(profiler.get("retained_profiler_trace"))
    return [
        {
            "claim": "CUDA stream priority availability",
            "evidence_level": "runtime_probe",
            "probe_pass": status_of(probe_payload, "stream_priority_range") == "pass",
            "production_path_uses_feature": bool(scan["production_uses_stream_priority"]),
            "paper_mechanism_eligible": False,
            "guard": (
                "API availability alone is not a scheduling mechanism. Do not claim "
                "stream-priority QoS until a mounted-path executor uses priority streams "
                "and a same-run trace proves the path executes."
            ),
        },
        {
            "claim": "Managed-memory prefetch scheduling",
            "evidence_level": "production_source_plus_runtime_probe",
            "probe_pass": status_of(probe_payload, "prefetch_device") == "pass",
            "production_path_uses_feature": bool(scan["production_uses_prefetch"]),
            "paper_mechanism_eligible": False,
            "guard": (
                "Production source contains prefetch calls and the probe exercises them, "
                "but this is not a mounted-path Nsight/CUPTI trace or a QoS result."
            ),
        },
        {
            "claim": "Stream attach scheduling",
            "evidence_level": "runtime_probe",
            "probe_pass": status_of(probe_payload, "attach_managed") == "pass",
            "production_path_uses_feature": bool(scan["production_uses_attach"]),
            "paper_mechanism_eligible": False,
            "guard": (
                "Stream attach remains diagnostic unless production mounted-path code "
                "uses it and a retained trace proves execution."
            ),
        },
        {
            "claim": "Kernel launch shape accounting",
            "evidence_level": "source_scan_plus_runtime_probe",
            "probe_pass": probe_payload.get("kernel_launch_shape", {}).get("launch_status") == "pass",
            "production_kernel_launches": len(scan.get("kernel_launches", [])),
            "paper_mechanism_eligible": False,
            "guard": (
                "Kernel launch shapes are implementation evidence only. They are not a "
                "CUDA scheduling mechanism until a mounted-path trace proves execution."
            ),
        },
        {
            "claim": "Nsight/CUPTI-backed GPU QoS",
            "evidence_level": "availability_or_unavailable",
            "probe_pass": bool(profiler.get("nsys_available") or profiler.get("cupti_headers")),
            "retained_profiler_trace": profiler_trace,
            "paper_mechanism_eligible": False,
            "guard": (
                "No hardware-counter, SM-occupancy, or foreground AI QoS claim is allowed "
                "without retained same-run Nsight/CUPTI traces tied to mounted storage pressure."
            ),
        },
        {
            "claim": "Foreground AI QoS recovery or GPU isolation",
            "evidence_level": "negative_claim_guard",
            "probe_pass": False,
            "production_path_uses_feature": False,
            "paper_mechanism_eligible": False,
            "guard": (
                "This contract explicitly does not prove foreground AI QoS recovery, "
                "GPU isolation, side-channel defense, or multi-tenant scheduling."
            ),
        },
    ]


def build_payload(out_dir: Path) -> dict[str, Any]:
    probe_meta, probe_payload = run_probe(out_dir)
    scan = source_scan()
    profiler = profiler_availability(out_dir)
    verdicts = claim_verdicts(probe_payload, scan, profiler)
    required_sections_present = {
        "probe_metadata": probe_meta.get("available") is True,
        "probe_json_parse": probe_meta.get("json_parse_error") is None,
        "stream_priority_probe": "stream_priority_range" in probe_payload,
        "kernel_launch_shape": "kernel_launch_shape" in probe_payload,
        "prefetch_probe": "prefetch_device" in probe_payload and "prefetch_host" in probe_payload,
        "stream_attach_probe": "attach_managed" in probe_payload,
        "synchronization_probe": "synchronization" in probe_payload,
        "production_source_scan": bool(scan.get("files_scanned")),
        "profiler_availability": "nsys_available" in profiler,
        "non_claim_guards": len(verdicts) >= 5,
    }
    return {
        "schema_version": 1,
        "generated_by": "code/experiments/run_cuda_qos_contract.py",
        "generated_utc": now_utc(),
        "scope": "Gate 0.12-S0 CUDA scheduling/QoS trace contract without promoting GPU isolation claims.",
        "official_basis": OFFICIAL_BASIS,
        "environment": {
            "PATH": os.environ.get("PATH", ""),
        },
        "probe": {
            "metadata": probe_meta,
            "payload": probe_payload,
        },
        "production_source_scan": scan,
        "profiler_availability": profiler,
        "claim_verdicts": verdicts,
        "required_sections_present": required_sections_present,
        "artifact_verdict": {
            "runner_completed": True,
            "overall_pass": all(required_sections_present.values()) and probe_payload.get("overall_pass") is True,
            "paper_claim_guard": (
                "CUDA stream priority, prefetch, attach, Nsight, CUPTI, foreground AI QoS, "
                "and GPU isolation language is not paper-mechanism eligible unless this "
                "contract is later replaced by same-run mounted-path trace evidence."
            ),
        },
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    probe = payload["probe"]["payload"]
    scan = payload["production_source_scan"]
    profiler = payload["profiler_availability"]
    verdict = payload["artifact_verdict"]
    lines = [
        "# CUDA QoS Contract",
        "",
        f"- Generated: `{payload['generated_utc']}`",
        f"- Generated by: `{payload['generated_by']}`",
        f"- Overall artifact pass: `{verdict['overall_pass']}`",
        f"- Probe overall pass: `{probe.get('overall_pass')}`",
        "",
        "## Runtime Probe",
        "",
        f"- Device count: `{probe.get('device_count', {}).get('value')}`",
        f"- Stream priority range status: `{probe.get('stream_priority_range', {}).get('status')}`",
        f"- Distinct priority range: `{probe.get('stream_priority_range', {}).get('distinct_range')}`",
        f"- Prefetch device status: `{probe.get('prefetch_device', {}).get('status')}`",
        f"- Stream attach status: `{probe.get('attach_managed', {}).get('status')}`",
        f"- Kernel launch status: `{probe.get('kernel_launch_shape', {}).get('launch_status')}`",
        f"- Stream synchronize status: `{probe.get('synchronization', {}).get('stream_synchronize', {}).get('status')}`",
        "",
        "## Production Source Scan",
        "",
        f"- Production CUDA kernel launches: `{len(scan.get('kernel_launches', []))}`",
        f"- Uses stream priority in production source: `{scan.get('production_uses_stream_priority')}`",
        f"- Uses nonblocking stream in production source: `{scan.get('production_uses_nonblocking_stream')}`",
        f"- Uses prefetch in production source: `{scan.get('production_uses_prefetch')}`",
        f"- Uses stream attach in production source: `{scan.get('production_uses_attach')}`",
        f"- Uses stream sync in production source: `{scan.get('production_uses_stream_sync')}`",
        f"- Uses device sync in production source: `{scan.get('production_uses_device_sync')}`",
        "",
        "## Profiler Availability",
        "",
        f"- nsys available: `{profiler.get('nsys_available')}`",
        f"- ncu available: `{profiler.get('ncu_available')}`",
        f"- CUPTI headers: `{len(profiler.get('cupti_headers', []))}`",
        f"- CUPTI libraries: `{len(profiler.get('cupti_libs', []))}`",
        f"- Retained profiler trace: `{profiler.get('retained_profiler_trace')}`",
        "",
        "## Claim Verdicts",
        "",
    ]
    for item in payload["claim_verdicts"]:
        lines.append(
            f"- `{item['claim']}`: paper mechanism eligible = "
            f"`{item['paper_mechanism_eligible']}`; guard: {item['guard']}"
        )
    lines.extend([
        "",
        "## Claim Guard",
        "",
        f"- {verdict['paper_claim_guard']}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = build_payload(OUT)
    json_path = OUT / "cuda_qos_contract.json"
    md_path = OUT / "cuda_qos_contract.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(payload, md_path)
    print(json.dumps({
        "overall_pass": payload["artifact_verdict"]["overall_pass"],
        "probe_overall_pass": payload["probe"]["payload"].get("overall_pass"),
        "production_kernel_launches": len(payload["production_source_scan"].get("kernel_launches", [])),
        "production_uses_prefetch": payload["production_source_scan"].get("production_uses_prefetch"),
        "production_uses_stream_priority": payload["production_source_scan"].get("production_uses_stream_priority"),
        "nsys_available": payload["profiler_availability"].get("nsys_available"),
        "cupti_headers": len(payload["profiler_availability"].get("cupti_headers", [])),
        "json": relpath(json_path),
        "markdown": relpath(md_path),
    }, indent=2, sort_keys=True))
    return 0 if payload["artifact_verdict"]["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
