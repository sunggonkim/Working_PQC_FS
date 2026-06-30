#!/usr/bin/env python3
"""Gate 0.18-S0 NVIDIA/Jetson optimization claim ladder builder."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
OUT = ROOT / "artifacts" / "validation" / "jetson_optimization_ladder"
MEMORY_CONTRACT = ROOT / "artifacts" / "validation" / "jetson_memory_contract" / "jetson_memory_contract.json"
CUDA_QOS_CONTRACT = ROOT / "artifacts" / "validation" / "cuda_qos_contract" / "cuda_qos_contract.json"
POWER_THERMAL_CONTRACT = (
    ROOT / "artifacts" / "validation" / "jetson_power_thermal_contract" /
    "jetson_power_thermal_contract.json"
)

CLASSIFICATIONS = [
    "available-api",
    "probe-passes",
    "microbenchmark-benefit",
    "production-mounted-path-benefit",
    "paper-mechanism",
    "diagnostic-only",
    "unavailable",
    "non-claim",
]

OFFICIAL_BASIS = [
    {
        "name": "NVIDIA CUDA for Tegra memory/coherency guidance",
        "url": "https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html",
    },
    {
        "name": "NVIDIA CUDA Runtime API stream management",
        "url": "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html",
    },
    {
        "name": "NVIDIA CUDA Runtime API memory management",
        "url": "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html",
    },
    {
        "name": "NVIDIA GPUDirect RDMA caveats",
        "url": "https://docs.nvidia.com/cuda/gpudirect-rdma/index.html",
    },
    {
        "name": "NVIDIA CUPTI documentation",
        "url": "https://docs.nvidia.com/cupti/",
    },
]


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {relpath(path)}")
    return payload


def production_source_files() -> list[Path]:
    return sorted(
        path for path in CODE.iterdir()
        if path.is_file() and path.suffix in {".c", ".h", ".cu", ".cuh", ".cpp", ".hpp"}
    )


def scan_terms() -> dict[str, Any]:
    terms = [
        "cudaHostAlloc",
        "cudaHostRegister",
        "cudaMallocManaged",
        "cudaMemPrefetchAsync",
        "cudaStreamAttachMemAsync",
        "cudaStreamCreateWithPriority",
        "cudaStreamCreateWithFlags",
        "cudaStreamSynchronize",
        "cudaDeviceSynchronize",
        "dma_buf",
        "dmabuf",
        "GPUDirect",
        "nvidia_peermem",
        "nv_peer_mem",
        "CUPTI",
        "Nsight",
        "nsys",
    ]
    per_term: dict[str, list[dict[str, Any]]] = {term: [] for term in terms}
    kernel_launches: list[dict[str, Any]] = []
    launch_re = re.compile(r"([A-Za-z_][A-Za-z0-9_:]*)\s*<<<([^>]*)>>>")
    for path in production_source_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), start=1):
            for term in terms:
                if term in line:
                    per_term[term].append({
                        "file": relpath(path),
                        "line": line_no,
                        "text": line.strip()[:220],
                    })
            for match in launch_re.finditer(line):
                kernel_launches.append({
                    "file": relpath(path),
                    "line": line_no,
                    "kernel": match.group(1),
                    "configuration_text": match.group(2).strip(),
                })
    return {
        "files_scanned": [relpath(path) for path in production_source_files()],
        "term_hits": per_term,
        "kernel_launches": kernel_launches,
    }


def probe_status(memory: dict[str, Any], name: str) -> str:
    probes = memory.get("probe_payload", {}).get("probes", {})
    probe = probes.get(name, {}) if isinstance(probes, dict) else {}
    return str(probe.get("status", "missing")) if isinstance(probe, dict) else "missing"


def nested_status(payload: dict[str, Any], *path: str) -> str:
    node: Any = payload
    for key in path:
        if not isinstance(node, dict):
            return "missing"
        node = node.get(key)
    if isinstance(node, dict):
        return str(node.get("status", "missing"))
    return "missing"


def has_hit(scan: dict[str, Any], *terms: str) -> bool:
    hits = scan.get("term_hits", {})
    return any(bool(hits.get(term)) for term in terms)


def classify(
    name: str,
    available_api: bool,
    probe_pass: bool,
    production_source_present: bool,
    microbenchmark_benefit: bool,
    mounted_path_benefit: bool,
    paper_mechanism: bool,
    forced_non_claim: bool,
    evidence: list[str],
    missing: list[str],
    guard: str,
) -> dict[str, Any]:
    if forced_non_claim:
        highest = "non-claim"
    elif paper_mechanism:
        highest = "paper-mechanism"
    elif mounted_path_benefit:
        highest = "production-mounted-path-benefit"
    elif microbenchmark_benefit:
        highest = "microbenchmark-benefit"
    elif probe_pass:
        highest = "probe-passes"
    elif available_api:
        highest = "available-api"
    else:
        highest = "unavailable"

    paper_classification = "paper-mechanism" if paper_mechanism else (
        "non-claim" if forced_non_claim else (
            "unavailable" if highest == "unavailable" else "diagnostic-only"
        )
    )

    return {
        "name": name,
        "highest_evidence_level": highest,
        "paper_classification": paper_classification,
        "allowed_classifications": CLASSIFICATIONS,
        "available_api": available_api,
        "probe_pass": probe_pass,
        "production_source_present": production_source_present,
        "microbenchmark_benefit_retained": microbenchmark_benefit,
        "production_mounted_path_benefit_retained": mounted_path_benefit,
        "paper_mechanism_eligible": paper_mechanism,
        "evidence": evidence,
        "missing_for_paper_mechanism": missing,
        "negative_claim_guard": guard,
    }


def build_ladder(memory: dict[str, Any], qos: dict[str, Any],
                 thermal: dict[str, Any], scan: dict[str, Any]) -> list[dict[str, Any]]:
    qos_probe = qos.get("probe", {}).get("payload", {})
    profiler = qos.get("profiler_availability", {})
    production_scan = qos.get("production_source_scan", {})
    memory_probe = memory.get("probe_payload", {})
    devices = memory_probe.get("devices", [])
    device = devices[0] if isinstance(devices, list) and devices else {}
    claim_contract = {
        str(item.get("name")): item
        for item in memory.get("claim_contract", [])
        if isinstance(item, dict)
    }
    thermal_ok = thermal.get("artifact_verdict", {}).get("overall_pass") is True

    def memory_claim(name: str) -> dict[str, Any]:
        return claim_contract.get(name, {})

    return [
        classify(
            "pinned memory via cudaHostAlloc",
            available_api=probe_status(memory, "cudaHostAlloc") != "missing",
            probe_pass=probe_status(memory, "cudaHostAlloc") == "pass",
            production_source_present=has_hit(scan, "cudaHostAlloc"),
            microbenchmark_benefit=False,
            mounted_path_benefit=False,
            paper_mechanism=False,
            forced_non_claim=False,
            evidence=[
                f"cudaHostAlloc={probe_status(memory, 'cudaHostAlloc')}",
                f"cudaHostAllocMappedKernel={probe_status(memory, 'cudaHostAllocMappedKernel')}",
                f"production_source_present={has_hit(scan, 'cudaHostAlloc')}",
                f"source_contract={relpath(MEMORY_CONTRACT)}",
            ],
            missing=[
                "production mounted-path allocation uses cudaHostAlloc",
                "matched benchmark shows benefit versus pageable/managed staging",
                "thermal-qualified headline run carries this allocation mode",
            ],
            guard=memory_claim("pinned host allocation via cudaHostAlloc").get(
                "negative_guard",
                "Do not claim pinned-memory benefit without mounted-path evidence.",
            ),
        ),
        classify(
            "registered host memory via cudaHostRegister",
            available_api=probe_status(memory, "cudaHostRegister") != "missing",
            probe_pass=probe_status(memory, "cudaHostRegister") == "pass",
            production_source_present=has_hit(scan, "cudaHostRegister"),
            microbenchmark_benefit=False,
            mounted_path_benefit=False,
            paper_mechanism=False,
            forced_non_claim=False,
            evidence=[
                f"cudaHostRegister={probe_status(memory, 'cudaHostRegister')}",
                f"production_source_present={has_hit(scan, 'cudaHostRegister')}",
                f"source_contract={relpath(MEMORY_CONTRACT)}",
            ],
            missing=[
                "production mounted-path buffer registration path",
                "registration lifetime and unregister cleanup proof",
                "matched benchmark shows benefit and no unsafe lifetime exposure",
            ],
            guard=memory_claim("registered host memory via cudaHostRegister").get(
                "negative_guard",
                "Do not claim registered-memory production benefit without mounted-path evidence.",
            ),
        ),
        classify(
            "UVM / cudaMallocManaged",
            available_api=probe_status(memory, "cudaMallocManaged") != "missing",
            probe_pass=probe_status(memory, "cudaMallocManaged") == "pass",
            production_source_present=has_hit(scan, "cudaMallocManaged"),
            microbenchmark_benefit=False,
            mounted_path_benefit=False,
            paper_mechanism=False,
            forced_non_claim=False,
            evidence=[
                f"managedMemory={device.get('managedMemory')}",
                f"concurrentManagedAccess={device.get('concurrentManagedAccess')}",
                f"cudaMallocManaged={probe_status(memory, 'cudaMallocManaged')}",
                f"production_source_present={has_hit(scan, 'cudaMallocManaged')}",
                f"source_contract={relpath(MEMORY_CONTRACT)}",
            ],
            missing=[
                "production mounted-path benefit for managed buffers",
                "Nsight/CUPTI migration or stall trace for the same mounted workload",
                "no direct-storage-DMA implication in paper text",
            ],
            guard=memory_claim("managed memory / UVM allocation").get(
                "negative_guard",
                "Do not claim UVM production benefit from allocation-only evidence.",
            ),
        ),
        classify(
            "CUDA stream priority",
            available_api=nested_status(qos_probe, "stream_priority_range") != "missing",
            probe_pass=nested_status(qos_probe, "stream_priority_range") == "pass",
            production_source_present=bool(production_scan.get("production_uses_stream_priority")),
            microbenchmark_benefit=False,
            mounted_path_benefit=False,
            paper_mechanism=False,
            forced_non_claim=False,
            evidence=[
                f"stream_priority_range={nested_status(qos_probe, 'stream_priority_range')}",
                f"distinct_priority_range={qos_probe.get('stream_priority_range', {}).get('distinct_range')}",
                f"production_uses_stream_priority={production_scan.get('production_uses_stream_priority')}",
                f"source_contract={relpath(CUDA_QOS_CONTRACT)}",
            ],
            missing=[
                "mounted-path executor creates priority streams",
                "same-run trace shows storage work submitted on priority stream",
                "foreground latency evidence under competing GPU pressure",
            ],
            guard="Do not claim stream-priority QoS until mounted-path priority streams and same-run trace evidence exist.",
        ),
        classify(
            "CUDA prefetch",
            available_api=(
                probe_status(memory, "cudaMemPrefetchAsyncManaged") != "missing" or
                nested_status(qos_probe, "prefetch_device") != "missing"
            ),
            probe_pass=(
                probe_status(memory, "cudaMemPrefetchAsyncManaged") == "pass" and
                nested_status(qos_probe, "prefetch_device") == "pass" and
                nested_status(qos_probe, "prefetch_host") == "pass"
            ),
            production_source_present=bool(production_scan.get("production_uses_prefetch")),
            microbenchmark_benefit=False,
            mounted_path_benefit=False,
            paper_mechanism=False,
            forced_non_claim=False,
            evidence=[
                f"memory_contract_prefetch={probe_status(memory, 'cudaMemPrefetchAsyncManaged')}",
                f"qos_prefetch_device={nested_status(qos_probe, 'prefetch_device')}",
                f"qos_prefetch_host={nested_status(qos_probe, 'prefetch_host')}",
                f"production_uses_prefetch={production_scan.get('production_uses_prefetch')}",
                f"source_contract={relpath(CUDA_QOS_CONTRACT)}",
            ],
            missing=[
                "per-run mounted-path trace proving the prefetch calls execute",
                "microbenchmark or macrobenchmark benefit attributed to prefetch",
                "thermal-qualified same-run metadata for the benchmark",
            ],
            guard="Prefetch is implementation/diagnostic evidence only until a retained mounted-path benefit exists.",
        ),
        classify(
            "CUDA stream attach",
            available_api=(
                probe_status(memory, "cudaStreamAttachMemAsyncManaged") != "missing" or
                nested_status(qos_probe, "attach_managed") != "missing"
            ),
            probe_pass=(
                probe_status(memory, "cudaStreamAttachMemAsyncManaged") == "pass" and
                nested_status(qos_probe, "attach_managed") == "pass"
            ),
            production_source_present=bool(production_scan.get("production_uses_attach")),
            microbenchmark_benefit=False,
            mounted_path_benefit=False,
            paper_mechanism=False,
            forced_non_claim=False,
            evidence=[
                f"memory_contract_attach={probe_status(memory, 'cudaStreamAttachMemAsyncManaged')}",
                f"qos_attach_managed={nested_status(qos_probe, 'attach_managed')}",
                f"production_uses_attach={production_scan.get('production_uses_attach')}",
                f"source_contract={relpath(CUDA_QOS_CONTRACT)}",
            ],
            missing=[
                "production mounted-path code uses cudaStreamAttachMemAsync",
                "retained trace shows stream attach on mounted workload buffers",
                "benchmark isolates attach effect from prefetch and synchronization",
            ],
            guard="Do not claim stream-attach scheduling or coherency benefit without production mounted-path evidence.",
        ),
        classify(
            "Nsight Systems / Nsight Compute",
            available_api=bool(profiler.get("nsys_available") or profiler.get("ncu_available")),
            probe_pass=bool(profiler.get("nsys_available") or profiler.get("ncu_available")),
            production_source_present=False,
            microbenchmark_benefit=False,
            mounted_path_benefit=False,
            paper_mechanism=False,
            forced_non_claim=False,
            evidence=[
                f"nsys_available={profiler.get('nsys_available')}",
                f"ncu_available={profiler.get('ncu_available')}",
                f"retained_profiler_trace={profiler.get('retained_profiler_trace')}",
                f"thermal_contract_pass={thermal_ok}",
                f"source_contract={relpath(CUDA_QOS_CONTRACT)}",
            ],
            missing=[
                "retained Nsight report for the mounted workload",
                "exported trace summary tied to exact run command",
                "same-run thermal and power metadata",
            ],
            guard="Profiler availability is not trace evidence; do not claim Nsight-backed QoS without retained reports.",
        ),
        classify(
            "CUPTI",
            available_api=bool(profiler.get("cupti_headers") or profiler.get("cupti_libs")),
            probe_pass=bool(profiler.get("cupti_headers") and profiler.get("cupti_libs")),
            production_source_present=has_hit(scan, "CUPTI"),
            microbenchmark_benefit=False,
            mounted_path_benefit=False,
            paper_mechanism=False,
            forced_non_claim=not bool(profiler.get("cupti_headers") or profiler.get("cupti_libs")),
            evidence=[
                f"cupti_headers={len(profiler.get('cupti_headers', []))}",
                f"cupti_libs={len(profiler.get('cupti_libs', []))}",
                f"retained_profiler_trace={profiler.get('retained_profiler_trace')}",
                f"production_source_mentions_cupti={has_hit(scan, 'CUPTI')}",
                f"source_contract={relpath(CUDA_QOS_CONTRACT)}",
            ],
            missing=[
                "CUPTI headers and libraries available in build environment",
                "live CUPTI sampler linked or invoked for mounted-path run",
                "retained PM/occupancy trace tied to storage pressure",
            ],
            guard="CUPTI-backed hardware-counter claims are prohibited until retained CUPTI traces exist.",
        ),
        classify(
            "dma-buf zero-copy",
            available_api=bool(memory.get("platform_manifest", {}).get("sysfs", {}).get("/sys/class/dma_heap")),
            probe_pass=False,
            production_source_present=has_hit(scan, "dma_buf", "dmabuf"),
            microbenchmark_benefit=False,
            mounted_path_benefit=False,
            paper_mechanism=False,
            forced_non_claim=True,
            evidence=[
                f"/sys/class/dma_heap={memory.get('platform_manifest', {}).get('sysfs', {}).get('/sys/class/dma_heap')}",
                f"/dev/dma_heap={memory.get('platform_manifest', {}).get('device_nodes', {}).get('/dev/dma_heap')}",
                "no CUDA dma-buf export/import probe is retained",
                f"production_source_mentions_dmabuf={has_hit(scan, 'dma_buf', 'dmabuf')}",
                f"source_contract={relpath(MEMORY_CONTRACT)}",
            ],
            missing=[
                "CUDA allocation export/import proof",
                "mounted-path use of dma-buf buffers",
                "copy-path comparison showing zero-copy effect",
            ],
            guard=memory_claim("dma-buf export/import").get(
                "negative_guard",
                "dma-buf zero-copy remains a non-claim without export/import and mounted-path proof.",
            ),
        ),
        classify(
            "GPUDirect/RDMA",
            available_api=False,
            probe_pass=False,
            production_source_present=has_hit(scan, "GPUDirect", "nvidia_peermem", "nv_peer_mem"),
            microbenchmark_benefit=False,
            mounted_path_benefit=False,
            paper_mechanism=False,
            forced_non_claim=True,
            evidence=[
                f"cudaDevAttrGPUDirectRDMASupported={device.get('gpuDirectRDMASupported')}",
                f"kernel_modules_relevant={memory.get('platform_manifest', {}).get('kernel_modules_relevant')}",
                f"infiniband={memory.get('platform_manifest', {}).get('sysfs', {}).get('/sys/class/infiniband')}",
                f"production_source_mentions_gd={has_hit(scan, 'GPUDirect', 'nvidia_peermem', 'nv_peer_mem')}",
                f"source_contract={relpath(MEMORY_CONTRACT)}",
            ],
            missing=[
                "GPUDirect/RDMA-supported device attribute and peer-memory driver",
                "RDMA/NVMe peer device evidence",
                "production mounted-path trace and benchmark",
            ],
            guard=memory_claim("GPUDirect/RDMA applicability").get(
                "negative_guard",
                "GPUDirect/RDMA and direct NVMe-to-UVM DMA remain non-claims without driver and production proof.",
            ),
        ),
        classify(
            "direct NVMe-to-UVM DMA",
            available_api=False,
            probe_pass=False,
            production_source_present=False,
            microbenchmark_benefit=False,
            mounted_path_benefit=False,
            paper_mechanism=False,
            forced_non_claim=True,
            evidence=[
                "no NVMe peer-DMA proof",
                "no UVM direct-DMA proof",
                "no mounted-path direct-DMA trace",
                f"source_contract={relpath(MEMORY_CONTRACT)}",
            ],
            missing=[
                "kernel/driver proof of direct NVMe peer DMA into UVM",
                "mounted-path implementation and trace",
                "correctness and fallback evidence",
            ],
            guard=memory_claim("NVMe-to-UVM direct DMA").get(
                "negative_guard",
                "The paper must not claim direct NVMe-to-UVM DMA.",
            ),
        ),
    ]


def build_payload() -> dict[str, Any]:
    memory = load_json(MEMORY_CONTRACT)
    qos = load_json(CUDA_QOS_CONTRACT)
    thermal = load_json(POWER_THERMAL_CONTRACT)
    scan = scan_terms()
    ladder = build_ladder(memory, qos, thermal, scan)
    paper_eligible = [item for item in ladder if item["paper_mechanism_eligible"]]
    missing_inputs = [
        relpath(path)
        for path in [MEMORY_CONTRACT, CUDA_QOS_CONTRACT, POWER_THERMAL_CONTRACT]
        if not path.exists()
    ]
    return {
        "schema_version": 1,
        "generated_by": "code/experiments/build_jetson_optimization_ladder.py",
        "generated_utc": now_utc(),
        "scope": "Gate 0.18-S0 NVIDIA/Jetson optimization evidence ladder.",
        "official_basis": OFFICIAL_BASIS,
        "input_artifacts": [
            relpath(MEMORY_CONTRACT),
            relpath(CUDA_QOS_CONTRACT),
            relpath(POWER_THERMAL_CONTRACT),
        ],
        "missing_inputs": missing_inputs,
        "source_scan": scan,
        "classifications": CLASSIFICATIONS,
        "ladder": ladder,
        "summary": {
            "terms_total": len(ladder),
            "paper_mechanism_eligible_count": len(paper_eligible),
            "paper_mechanism_eligible_terms": [item["name"] for item in paper_eligible],
            "non_claim_terms": [
                item["name"] for item in ladder
                if item["paper_classification"] == "non-claim"
            ],
            "diagnostic_only_terms": [
                item["name"] for item in ladder
                if item["paper_classification"] == "diagnostic-only"
            ],
            "unavailable_terms": [
                item["name"] for item in ladder
                if item["paper_classification"] == "unavailable"
            ],
        },
        "artifact_verdict": {
            "overall_pass": not missing_inputs and len(ladder) >= 10 and not paper_eligible,
            "paper_claim_guard": (
                "No NVIDIA/Jetson optimization term may be described as an AEGIS-Q paper "
                "mechanism unless this ladder marks it paper-mechanism after retained "
                "production mounted-path traces and benchmark evidence."
            ),
            "parent_checklist_closed": False,
            "paper_text_status": "not_updated",
        },
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    summary = payload["summary"]
    verdict = payload["artifact_verdict"]
    lines = [
        "# Jetson Optimization Claim Ladder",
        "",
        f"- Generated: `{payload['generated_utc']}`",
        f"- Generated by: `{payload['generated_by']}`",
        f"- Overall artifact pass: `{verdict['overall_pass']}`",
        f"- Parent checklist closed: `{verdict['parent_checklist_closed']}`",
        f"- Paper text status: `{verdict['paper_text_status']}`",
        f"- Paper-mechanism eligible terms: `{summary['paper_mechanism_eligible_count']}`",
        "",
        "## Inputs",
        "",
    ]
    lines.extend(f"- `{artifact}`" for artifact in payload["input_artifacts"])
    lines.extend([
        "",
        "## Ladder",
        "",
        "| Term | Highest Evidence | Paper Classification | Production Source | Paper Mechanism |",
        "| --- | --- | --- | --- | --- |",
    ])
    for item in payload["ladder"]:
        lines.append(
            f"| {item['name']} | `{item['highest_evidence_level']}` | "
            f"`{item['paper_classification']}` | "
            f"`{item['production_source_present']}` | "
            f"`{item['paper_mechanism_eligible']}` |"
        )
    lines.extend([
        "",
        "## Non-Claim Terms",
        "",
    ])
    lines.extend(f"- {term}" for term in summary["non_claim_terms"])
    lines.extend([
        "",
        "## Diagnostic-Only Terms",
        "",
    ])
    lines.extend(f"- {term}" for term in summary["diagnostic_only_terms"])
    lines.extend([
        "",
        "## Claim Guard",
        "",
        f"- {verdict['paper_claim_guard']}",
        "",
        "## Missing Evidence Pattern",
        "",
        "- Probe-passing APIs are not treated as production benefits.",
        "- Production source mentions are not treated as mounted-path benchmark evidence.",
        "- Nsight/CUPTI availability is not treated as a retained trace.",
        "- No term is paper-mechanism eligible in this artifact.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    json_path = OUT / "jetson_optimization_ladder.json"
    md_path = OUT / "jetson_optimization_ladder.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(payload, md_path)
    print(json.dumps({
        "overall_pass": payload["artifact_verdict"]["overall_pass"],
        "terms_total": payload["summary"]["terms_total"],
        "paper_mechanism_eligible_count": payload["summary"]["paper_mechanism_eligible_count"],
        "non_claim_terms": payload["summary"]["non_claim_terms"],
        "diagnostic_only_terms": payload["summary"]["diagnostic_only_terms"],
        "json": relpath(json_path),
        "markdown": relpath(md_path),
    }, indent=2, sort_keys=True))
    return 0 if payload["artifact_verdict"]["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
