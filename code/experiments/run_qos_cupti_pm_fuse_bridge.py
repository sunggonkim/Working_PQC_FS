#!/usr/bin/env python3
"""Run mounted FUSE writes driven by live CUPTI PM sampling telemetry.

This is the PM/CUPTI counterpart to `run_qos_fuse_live_bridge.py`.
It deliberately reuses the existing mounted-FUSE telemetry-file path:

  * copy NVIDIA's installed CUPTI `pm_sampling` sample into the artifact dir;
  * patch the copied sample so each decoded PM sample writes
    `PQC_TELEMETRY_FILE` and a JSONL PM trace;
  * mount the real FUSE filesystem with `PQC_TELEMETRY_FILE`;
  * run a mounted writer while the CUPTI sample launches its CUDA workload;
  * retain FUSE runtime throttle trace and CUPTI PM sample trace.

The supported claim is narrow: live CUPTI PM samples can drive the mounted
FUSE daemon's existing telemetry-file throttle path in the same run.  This
does not prove foreground AI p99 recovery.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from run_qos_fuse_live_bridge import (
    DEFAULT_OUT as TEGRATESTATS_OUT,
    FUSE_BIN,
    ROOT,
    load_runtime_trace,
    start_fuse,
    stop_fuse,
    writer_worker,
    write_runtime_telemetry,
)


CUPTI_SAMPLE_SRC = Path("/usr/local/cuda-13.0/extras/CUPTI/samples/pm_sampling")
CUDA_ROOT = Path("/usr/local/cuda-13.0")
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "qos_cupti_pm_fuse_bridge"


PATCH_NEEDLE = """        for (size_t i = 0; i < metricsList.size(); ++i) {
            samplerRange.metricValues[metricsList[i]] = metricValues[i];
        }
"""

PATCH_REPLACEMENT = PATCH_NEEDLE + r"""
        const char* telemetryPath = std::getenv("PQC_CUPTI_TELEMETRY_FILE");
        const char* tracePath = std::getenv("PQC_CUPTI_TRACE_PATH");
        if (telemetryPath || tracePath)
        {
            double grActive = samplerRange.metricValues["gr__cycles_active.avg"];
            double grElapsed = samplerRange.metricValues["gr__cycles_elapsed.max"];
            double smInst = samplerRange.metricValues["sm__inst_executed_realtime.avg.per_cycle_active"];
            double memUtil = 0.0;
            if (grElapsed > 0.0) {
                memUtil = grActive / grElapsed;
            }
            if (memUtil < 0.0) memUtil = 0.0;
            if (memUtil > 1.0) memUtil = 1.0;
            double tensorUtil = smInst;
            if (tensorUtil < 0.0) tensorUtil = 0.0;
            if (tensorUtil > 1.0) tensorUtil = 1.0;
            if (telemetryPath)
            {
                std::ofstream telemetry(telemetryPath, std::ios::trunc);
                telemetry << std::fixed << std::setprecision(6)
                          << memUtil << " " << tensorUtil << " 2000000 0\n";
            }
            if (tracePath)
            {
                std::ofstream trace(tracePath, std::ios::app);
                trace << "{\"sample_index\":" << rangeIndex
                      << ",\"start_timestamp\":" << samplerRange.startTimestamp
                      << ",\"end_timestamp\":" << samplerRange.endTimestamp
                      << ",\"gr_cycles_active_avg\":" << grActive
                      << ",\"gr_cycles_elapsed_max\":" << grElapsed
                      << ",\"sm_inst_per_active_cycle\":" << smInst
                      << ",\"mem_bandwidth_util\":" << memUtil
                      << ",\"tensor_core_util\":" << tensorUtil
                      << "}\n";
            }
        }
"""


def sudo_password() -> str:
    value = os.environ.get("PQC_SUDO_PASSWORD")
    if not value:
        raise RuntimeError("PQC_SUDO_PASSWORD must be set for sudo CUPTI/tegrastats collection")
    return value


def prepare_cupti_sampler(out_dir: Path, iterations: int) -> Path:
    if not CUPTI_SAMPLE_SRC.exists():
        raise FileNotFoundError(f"missing CUPTI sample: {CUPTI_SAMPLE_SRC}")

    build_dir = out_dir / "cupti_pm_sampling_src"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    shutil.copytree(CUPTI_SAMPLE_SRC, build_dir)

    header = build_dir / "pm_sampling.h"
    text = header.read_text(encoding="utf-8")
    if PATCH_NEEDLE not in text:
        raise RuntimeError("CUPTI sample patch point not found")
    text = text.replace(PATCH_NEEDLE, PATCH_REPLACEMENT)
    text = text.replace("#include <unordered_map>\n", "#include <unordered_map>\n#include <cstdlib>\n")
    header.write_text(text, encoding="utf-8")

    cu = build_dir / "pm_sampling.cu"
    text = cu.read_text(encoding="utf-8")
    text = text.replace(
        "    const size_t NUM_OF_ITERATIONS = 100;\n",
        f"    const size_t NUM_OF_ITERATIONS = {int(iterations)};\n",
    )
    cu.write_text(text, encoding="utf-8")

    proc = subprocess.run(
        [
            "make",
            "-C",
            str(build_dir),
            f"CUDA_INSTALL_PATH={CUDA_ROOT}",
            f"CUPTI_INSTALL_PATH={CUDA_ROOT / 'extras' / 'CUPTI'}",
            "SMS=110",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    (out_dir / "cupti_build.stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (out_dir / "cupti_build.stderr.txt").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"CUPTI PM sampler build failed; see {out_dir / 'cupti_build.stderr.txt'}")
    return build_dir / "pm_sampling"


def run_cupti_sampler(binary: Path,
                      telemetry_file: Path,
                      pm_trace: Path,
                      out_dir: Path,
                      sampling_interval: int,
                      max_samples: int) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    ld_paths = [
        str(CUDA_ROOT / "extras" / "CUPTI" / "lib64"),
        str(CUDA_ROOT / "lib64"),
        str(CUDA_ROOT / "targets" / "sbsa-linux" / "lib"),
        env.get("LD_LIBRARY_PATH", ""),
    ]
    env["LD_LIBRARY_PATH"] = ":".join(path for path in ld_paths if path)
    env["PQC_CUPTI_TELEMETRY_FILE"] = str(telemetry_file)
    env["PQC_CUPTI_TRACE_PATH"] = str(pm_trace)
    cmd = [
        "sudo",
        "-S",
        "env",
        f"LD_LIBRARY_PATH={env['LD_LIBRARY_PATH']}",
        f"PQC_CUPTI_TELEMETRY_FILE={telemetry_file}",
        f"PQC_CUPTI_TRACE_PATH={pm_trace}",
        str(binary),
        "-d",
        "0",
        "-i",
        str(sampling_interval),
        "-s",
        str(max_samples),
    ]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        input=sudo_password() + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
    (out_dir / "cupti_pm_sampling.stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (out_dir / "cupti_pm_sampling.stderr.txt").write_text(proc.stderr, encoding="utf-8")
    (out_dir / "cupti_pm_sampling.command.json").write_text(json.dumps(cmd, indent=2), encoding="utf-8")
    return proc


def count_targets(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field, "missing"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--chunk-bytes", type=int, default=131072)
    parser.add_argument("--fsync-every", type=int, default=8)
    parser.add_argument("--cupti-iterations", type=int, default=600)
    parser.add_argument("--sampling-interval", type=int, default=1000000)
    parser.add_argument("--max-samples", type=int, default=512)
    args = parser.parse_args()

    if not FUSE_BIN.exists():
        raise SystemExit("missing build/pqc_fuse; run cmake --build build first")

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sampler = prepare_cupti_sampler(out_dir, args.cupti_iterations)

    storage_dir = Path(tempfile.mkdtemp(prefix="aegis_cupti_qos_store_"))
    mount_dir = Path(tempfile.mkdtemp(prefix="aegis_cupti_qos_mnt_"))
    mount_log_dir = out_dir / "mount_logs"
    mount_log_dir.mkdir(parents=True, exist_ok=True)
    telemetry_file = out_dir / "runtime_telemetry.txt"
    runtime_trace = out_dir / "runtime_fuse_admission_trace.jsonl"
    throttle_trace = out_dir / "runtime_fuse_throttle_trace.jsonl"
    pm_trace = out_dir / "cupti_pm_samples.jsonl"
    write_runtime_telemetry(telemetry_file, {"mem_bandwidth_util": 0.0, "tensor_core_util": 0.0})

    ctx = mp.get_context("spawn")
    stop_flag = ctx.Event()
    throttle_flag = ctx.Value("i", 0)
    result_queue: mp.Queue = ctx.Queue()
    fuse_proc: subprocess.Popen[str] | None = None
    writer_proc: mp.Process | None = None

    try:
        fuse_proc = start_fuse(storage_dir, mount_dir, mount_log_dir,
                               telemetry_file, runtime_trace, throttle_trace)
        writer_proc = ctx.Process(
            target=writer_worker,
            args=(str(mount_dir), stop_flag, throttle_flag, result_queue,
                  args.chunk_bytes, args.fsync_every),
            daemon=True,
        )
        writer_proc.start()
        proc = run_cupti_sampler(
            sampler,
            telemetry_file,
            pm_trace,
            out_dir,
            args.sampling_interval,
            args.max_samples,
        )
    finally:
        stop_flag.set()
        if writer_proc is not None:
            writer_proc.join(timeout=10)
            if writer_proc.is_alive():
                writer_proc.terminate()
                writer_proc.join(timeout=5)
        stop_fuse(fuse_proc, mount_dir)

    writer_stats: dict[str, Any] = {}
    if not result_queue.empty():
        writer_stats = result_queue.get()
    pm_rows = load_runtime_trace(pm_trace)
    throttle_rows = load_runtime_trace(throttle_trace)
    runtime_rows = load_runtime_trace(runtime_trace)
    throttle_counts = {
        "open": sum(1 for row in throttle_rows if int(row.get("throttled", 0)) == 0),
        "throttled": sum(1 for row in throttle_rows if int(row.get("throttled", 0)) != 0),
    }
    report = {
        "note": "Same-run mounted-FUSE QoS bridge driven by live CUPTI PM sampling telemetry.",
        "cupti_returncode": proc.returncode,
        "cupti_stdout": str((out_dir / "cupti_pm_sampling.stdout.txt").relative_to(ROOT)),
        "cupti_stderr": str((out_dir / "cupti_pm_sampling.stderr.txt").relative_to(ROOT)),
        "cupti_pm_trace": str(pm_trace.relative_to(ROOT)),
        "cupti_pm_samples": len(pm_rows),
        "runtime_throttle_trace": str(throttle_trace.relative_to(ROOT)),
        "runtime_throttle_trace_rows": len(throttle_rows),
        "runtime_throttle_counts": throttle_counts,
        "runtime_throttle_sleep_us_total": sum(int(row.get("sleep_us", 0)) for row in throttle_rows),
        "runtime_admission_trace_rows": len(runtime_rows),
        "writer_stats": writer_stats,
        "success_criteria": {
            "cupti_samples_present": len(pm_rows) > 0,
            "mounted_fuse_throttle_present": throttle_counts["throttled"] > 0,
            "writer_harness_throttle_disabled": writer_stats.get("throttle_sleeps", 0) == 0,
            "cupti_returncode_zero": proc.returncode == 0,
        },
        "sample_preview": pm_rows[:5],
    }
    report["verified"] = all(report["success_criteria"].values())
    (out_dir / "qos_cupti_pm_fuse_bridge.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# QoS CUPTI PM FUSE bridge",
        "",
        "This bundle writes through a real FUSE mount while a sudo CUPTI PM-sampling workload writes live PM-derived telemetry into the mounted daemon's telemetry file.",
        "",
        f"- Verified: `{report['verified']}`",
        f"- CUPTI return code: `{proc.returncode}`",
        f"- CUPTI PM samples: `{len(pm_rows)}`",
        f"- Runtime throttle counts: `{throttle_counts}`",
        f"- Runtime throttle sleep total: `{report['runtime_throttle_sleep_us_total']}` us",
        f"- Writer stats: `{writer_stats}`",
        f"- CUPTI trace: `{report['cupti_pm_trace']}`",
        f"- FUSE throttle trace: `{report['runtime_throttle_trace']}`",
        "",
        "Interpretation: this closes a same-run PM/CUPTI-to-mounted-FUSE throttle wiring check. It does not prove foreground AI p99 recovery.",
    ]
    (out_dir / "qos_cupti_pm_fuse_bridge.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    shutil.rmtree(storage_dir, ignore_errors=True)
    shutil.rmtree(mount_dir, ignore_errors=True)
    print(json.dumps({
        "out_dir": str(out_dir),
        "verified": report["verified"],
        "cupti_samples": len(pm_rows),
        "throttle_counts": throttle_counts,
    }, indent=2))
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
