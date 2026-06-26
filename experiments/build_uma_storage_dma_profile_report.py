#!/usr/bin/env python3
"""Package the profiler-backed UMA / storage-DMA probe into a retained report.

This report is intentionally conservative:
  - It records the exact profiler commands and the retained Nsight outputs.
  - It summarizes Unified Memory and CPU page-fault reports when present.
  - It does not upgrade the bundle into a verified NVMe-to-UVM DMA claim.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN = ROOT / "artifacts" / "validation" / "uma_storage_dma_profile"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "uma_storage_dma_profile_report"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def summarize_nsys_bundle(base_dir: Path) -> dict:
    return {
        "nsys_rep": str(base_dir.with_suffix(".nsys-rep")),
        "sqlite": str(base_dir.with_suffix(".sqlite")),
        "um_sum": read_csv_rows(base_dir.parent / "stats_um_sum.csv"),
        "um_cpu_page_faults_sum": read_csv_rows(base_dir.parent / "stats_um_cpu_page_faults_sum.csv"),
        "cuda_api_sum": read_csv_rows(base_dir.parent / "stats_cuda_api_sum.csv"),
        "osrt_sum": read_csv_rows(base_dir.parent / "stats_osrt_sum.csv"),
    }


def summarize_ncu_csv(path: Path) -> dict:
    rows = read_csv_rows(path)
    selected = []
    for row in rows:
        section = row.get("Section Name", "")
        name = row.get("Metric Name", "")
        if section not in {
            "GPU Speed Of Light Throughput",
            "Memory Workload Analysis",
            "GPU and Memory Workload Distribution",
            "Launch Statistics",
            "Occupancy",
        }:
            continue
        lowered = name.lower()
        if any(
            key in lowered
            for key in (
                "memory",
                "l1",
                "l2",
                "dram",
                "throughput",
                "duration",
                "sector",
                "byte",
                "cache",
                "active cycles",
                "elapsed cycles",
                "occupancy",
                "spilling",
            )
        ):
            selected.append(row)
    return {
        "csv": str(path),
        "selected_metric_count": len(selected),
        "selected_metrics": selected[:120],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    profile_record_path = args.in_dir / "profile_uma_storage_dma.json"
    profile_record = json.loads(profile_record_path.read_text(encoding="utf-8")) if profile_record_path.exists() else {}
    probe_json = args.in_dir / "probe" / "uma_storage_dma_probe.json"

    report = {
        "note": "Profile report only; not a proof of NVMe-to-UVM DMA semantics.",
        "input_dir": str(args.in_dir),
        "profile_record": profile_record,
        "probe_json": str(probe_json),
        "raw_probe": None,
        "um_smoke": None,
        "managed_storage_probe": None,
        "raw_probe_ncu": None,
    }

    runs = {r.get("name"): r for r in profile_record.get("runs", []) if isinstance(r, dict)}

    raw = runs.get("raw_probe")
    if raw:
        output_base = Path(raw.get("output_base", args.in_dir / "probe" / "raw_probe"))
        raw_stdout = read_text(args.in_dir / "probe" / "io_uring_uvm_nvme.stdout.txt")
        report["raw_probe"] = {
            "command": raw.get("command"),
            "returncode": raw.get("returncode"),
            "stdout": raw.get("stdout"),
            "stderr": raw.get("stderr"),
            "output_base": raw.get("output_base"),
            "nsys": summarize_nsys_bundle(output_base),
            "io_uring_uvm_stdout": raw_stdout,
            "same_buffer_checksum_match": "CHECKSUM_MATCH" in raw_stdout,
        }

    um = runs.get("um_smoke")
    if um:
        output_base = Path(um.get("output_base", args.in_dir / "um_smoke" / "um_smoke"))
        report["um_smoke"] = {
            "command": um.get("command"),
            "returncode": um.get("returncode"),
            "stdout": um.get("stdout"),
            "stderr": um.get("stderr"),
            "output_base": um.get("output_base"),
            "nsys": summarize_nsys_bundle(output_base),
        }

    managed = runs.get("managed_storage_probe")
    if managed:
        output_base = Path(managed.get("output_base", args.in_dir / "managed_storage" / "managed_storage"))
        managed_stdout = managed.get("stdout") or read_text(args.in_dir / "managed_storage" / "managed_storage.stdout.txt")
        report["managed_storage_probe"] = {
            "command": managed.get("command"),
            "returncode": managed.get("returncode"),
            "stdout": managed_stdout,
            "stderr": managed.get("stderr"),
            "output_base": managed.get("output_base"),
            "nsys": summarize_nsys_bundle(output_base),
            "same_buffer_checksum_match": "CHECKSUM_MATCH" in managed_stdout,
            "managed_pointer_attr": "type=managed" in managed_stdout,
            "device_prefetch_seen": "last_prefetch_location_type=1(device)" in managed_stdout,
            "host_prefetch_seen": "last_prefetch_location_type=2(host)" in managed_stdout,
        }

    ncu = runs.get("raw_probe_ncu")
    if ncu:
        output_base = Path(ncu.get("output_base", args.in_dir / "ncu" / "io_uring_uvm_checksum"))
        stdout = ncu.get("stdout") or ""
        csv_path = Path(ncu.get("csv_path") or output_base.with_suffix(".csv"))
        report["raw_probe_ncu"] = {
            "command": ncu.get("command"),
            "returncode": ncu.get("returncode"),
            "stdout": stdout,
            "stderr": ncu.get("stderr"),
            "output_base": ncu.get("output_base"),
            "ncu_rep": str(output_base.with_suffix(".ncu-rep")),
            "csv_path": str(csv_path),
            "import_returncode": ncu.get("import_returncode"),
            "same_buffer_checksum_match": "CHECKSUM_MATCH" in stdout,
            "ncu": summarize_ncu_csv(csv_path),
        }

    (args.out_dir / "uma_storage_dma_profile_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    md = [
        "# UMA / storage-DMA profile report",
        "",
        "This bundle packages the profiler-backed raw-read run and the retained Nsight summaries.",
        "",
        f"- Input directory: `{args.in_dir}`",
        f"- Probe JSON: `{probe_json}`",
        "",
        "## Raw probe",
        "",
    ]
    if report["raw_probe"]:
        md.append(f"- Command: `{report['raw_probe']['command']}`")
        md.append(f"- Return code: `{report['raw_probe']['returncode']}`")
        raw_nsys = report["raw_probe"]["nsys"]
        md.append(f"- Nsight report: `{raw_nsys['nsys_rep']}`")
        md.append(f"- SQLite export: `{raw_nsys['sqlite']}`")
        md.append(
            f"- Same storage-filled buffer observed by GPU checksum: "
            f"`{report['raw_probe']['same_buffer_checksum_match']}`"
        )
        md.append("")
        md.append("### Raw Probe same-buffer evidence")
        md.append("")
        stdout = report["raw_probe"].get("io_uring_uvm_stdout") or ""
        if stdout:
            for line in stdout.splitlines():
                if (
                    "Async read completed" in line
                    or "CPU checksum" in line
                    or "GPU mapped-buffer checksum" in line
                    or "CHECKSUM_MATCH" in line
                    or "device pointer" in line
                    or "POINTER_ATTR" in line
                    or "HOST_FLAGS" in line
                    or "RANGE_ATTR" in line
                ):
                    md.append(f"- `{line}`")
        else:
            md.append("- No `io_uring_uvm` stdout was retained.")
        md.append("")
        md.append("### Raw Probe CUDA API summary")
        md.append("")
        if raw_nsys["cuda_api_sum"]:
            for row in raw_nsys["cuda_api_sum"][:5]:
                md.append(
                    f"- {row.get('Name','')}: calls={row.get('Num Calls','')}, total_ns={row.get('Total Time (ns)','')}"
                )
        else:
            md.append("- No CUDA API rows were recorded.")

        md.append("")
        md.append("### Raw Probe Unified Memory summary")
        md.append("")
        if raw_nsys["um_sum"]:
            for row in raw_nsys["um_sum"]:
                md.append(
                    f"- VA={row.get('Virtual Address','')}, HtoD={row.get('HtoD Migration Size','')}, "
                    f"DtoH={row.get('DtoH Migration Size','')}, CPU faults={row.get('CPU Page Faults','')}, "
                    f"GPU faults={row.get('GPU Page Faults','')}, throughput={row.get('Migration Throughput','')}"
                )
        else:
            md.append("- No CUDA Unified Memory transfer rows were recorded for the raw probe.")
    else:
        md.append("- Raw probe record not found.")

    md.extend(["", "## Managed-memory smoke", ""])
    if report["um_smoke"]:
        md.append(f"- Command: `{report['um_smoke']['command']}`")
        md.append(f"- Return code: `{report['um_smoke']['returncode']}`")
        nsys = report["um_smoke"]["nsys"]
        md.append(f"- Nsight report: `{nsys['nsys_rep']}`")
        md.append(f"- SQLite export: `{nsys['sqlite']}`")
        md.append("")
        md.append("### Unified Memory summary")
        md.append("")
        if nsys["um_sum"]:
            for row in nsys["um_sum"]:
                md.append(
                    f"- VA={row.get('Virtual Address','')}, HtoD={row.get('HtoD Migration Size','')}, "
                    f"DtoH={row.get('DtoH Migration Size','')}, CPU faults={row.get('CPU Page Faults','')}, "
                    f"GPU faults={row.get('GPU Page Faults','')}, throughput={row.get('Migration Throughput','')}"
                )
        else:
            md.append("- No CUDA Unified Memory transfer rows were recorded in this profiling run.")

        md.extend(["", "### CUDA API summary", ""])
        if nsys["cuda_api_sum"]:
            for row in nsys["cuda_api_sum"][:5]:
                md.append(
                    f"- {row.get('Name','')}: calls={row.get('Num Calls','')}, total_ns={row.get('Total Time (ns)','')}"
                )
        else:
            md.append("- No CUDA API rows were recorded.")

        md.extend(["", "### OS runtime summary", ""])
        if nsys["osrt_sum"]:
            for row in nsys["osrt_sum"][:10]:
                md.append(
                    f"- {row.get('Name','')}: calls={row.get('Num Calls','')}, total_ns={row.get('Total Time (ns)','')}"
                )
        else:
            md.append("- No OS runtime rows were recorded.")
    else:
        md.append("- Managed-memory smoke record not found.")

    md.extend(["", "## Managed storage buffer probe", ""])
    if report["managed_storage_probe"]:
        managed = report["managed_storage_probe"]
        md.append(f"- Command: `{managed['command']}`")
        md.append(f"- Return code: `{managed['returncode']}`")
        md.append(f"- Same storage-filled managed buffer observed by GPU checksum: `{managed['same_buffer_checksum_match']}`")
        md.append(f"- Managed pointer attribute observed: `{managed['managed_pointer_attr']}`")
        md.append(f"- Device-prefetch location observed: `{managed['device_prefetch_seen']}`")
        md.append(f"- Host-prefetch location observed: `{managed['host_prefetch_seen']}`")
        md.append("")
        md.append("### Managed storage diagnostics")
        md.append("")
        stdout = managed.get("stdout") or ""
        if stdout:
            for line in stdout.splitlines():
                if (
                    "Managed buffer allocated" in line
                    or "Buffered pread completed" in line
                    or "CPU checksum" in line
                    or "GPU managed-buffer checksum" in line
                    or "CHECKSUM_MATCH" in line
                    or "POINTER_ATTR" in line
                    or "HOST_FLAGS" in line
                    or "RANGE_ATTR" in line
                ):
                    md.append(f"- `{line}`")
        else:
            md.append("- No managed-storage stdout was retained.")
    else:
        md.append("- Managed storage probe record not found.")

    md.extend(["", "## Nsight Compute storage-visible buffer counters", ""])
    if report["raw_probe_ncu"]:
        ncu_report = report["raw_probe_ncu"]
        md.append(f"- Command: `{ncu_report['command']}`")
        md.append(f"- Return code: `{ncu_report['returncode']}`")
        md.append(f"- NCU report: `{ncu_report['ncu_rep']}`")
        md.append(f"- CSV export: `{ncu_report['csv_path']}`")
        md.append(f"- Same-buffer checksum match in profiled run: `{ncu_report['same_buffer_checksum_match']}`")
        md.append(f"- Selected metric rows: `{ncu_report['ncu']['selected_metric_count']}`")
        md.extend(["", "### Selected NCU metrics", ""])
        if ncu_report["ncu"]["selected_metrics"]:
            for row in ncu_report["ncu"]["selected_metrics"][:40]:
                md.append(
                    f"- {row.get('Section Name','')} / {row.get('Metric Name','')}: "
                    f"{row.get('Metric Value','')} {row.get('Metric Unit','')}"
                )
        else:
            md.append("- No selected NCU memory metrics were exported.")
    else:
        md.append("- Nsight Compute run not found in this bundle.")

    md.extend(
        [
            "",
            "This report does not claim verified NVMe-to-UVM DMA semantics or migration suppression.",
            "",
        ]
    )
    (args.out_dir / "uma_storage_dma_profile_report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
