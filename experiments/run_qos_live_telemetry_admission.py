#!/usr/bin/env python3
"""Drive the admission controller with live tegrastats samples.

This is a bridge test, not a completed closed-loop QoS controller.  It starts
an optional GPU pressure process, samples `tegrastats`, maps live GPU power /
GR3D activity into the existing telemetry inputs, and invokes
`pqc_fuse --admission-telemetry-smoke` for each sample.

The resulting bundle proves that live workload-time telemetry can reach the
same admission code path as the offline Nsight adapter.  It does not prove that
the controller protects foreground AI p99 in a production FUSE workload.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
PQC_FUSE = ROOT / "build" / "pqc_fuse"
GPU_BURNER = ROOT / "experiments" / "gpu_burner"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "qos_live_telemetry_admission"

GPU_POWER_RE = re.compile(r"VDD_GPU\s+(\d+)mW")
GR3D_RE = re.compile(r"GR3D_FREQ\s+(?:(\d+)%|@)")


def parse_tegrastats(line: str) -> dict[str, Any]:
    gpu_power = None
    gr3d = None
    m = GPU_POWER_RE.search(line)
    if m:
        gpu_power = int(m.group(1))
    m = GR3D_RE.search(line)
    if m and m.group(1) is not None:
        gr3d = int(m.group(1))
    return {
        "raw": line.rstrip("\n"),
        "gpu_power_mw": gpu_power,
        "gr3d_percent": gr3d,
        "mem_bandwidth_util": min(1.0, max(0.0, (gpu_power or 0) / 35000.0)),
        "tensor_core_util": min(1.0, max(0.0, (gr3d or 0) / 100.0)),
    }


def run_admission(out_dir: Path, idx: int, telemetry: dict[str, Any]) -> dict[str, Any]:
    trace = out_dir / f"sample_{idx:02d}.jsonl"
    stdout = out_dir / f"sample_{idx:02d}.stdout"
    stderr = out_dir / f"sample_{idx:02d}.stderr"
    env = os.environ.copy()
    env.update({
        "PQC_ADMISSION_TRACE_PATH": str(trace),
        "PQC_TELEMETRY_MEM_BANDWIDTH": f"{telemetry['mem_bandwidth_util']:.6f}",
        "PQC_TELEMETRY_TENSOR_CORE": f"{telemetry['tensor_core_util']:.6f}",
        "PQC_ADMISSION_SMOKE_AI_BUDGET_NS": "2000000",
        "PQC_ADMISSION_SMOKE_CPU_QUEUE_DEPTH": "1",
        "PQC_ADMISSION_SMOKE_GPU_QUEUE_DEPTH": "1",
        "PQC_ADMISSION_SMOKE_BYTES": "131072",
        "PQC_ADMISSION_SMOKE_GPU_KERNEL_NS": "100000",
        "PQC_ADMISSION_SMOKE_H2D_NS": "100000",
        "PQC_ADMISSION_SMOKE_D2H_NS": "100000",
    })
    proc = subprocess.run([str(PQC_FUSE), "--admission-telemetry-smoke"], cwd=ROOT, env=env, text=True, capture_output=True)
    stdout.write_text(proc.stdout, encoding="utf-8")
    stderr.write_text(proc.stderr, encoding="utf-8")
    summary = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            summary = json.loads(line)
            break
    trace_rows = []
    if trace.exists():
        for line in trace.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                trace_rows.append(json.loads(line))
    return {
        "sample": idx,
        "returncode": proc.returncode,
        "telemetry": telemetry,
        "summary": summary,
        "trace": str(trace.relative_to(ROOT)),
        "stdout": str(stdout.relative_to(ROOT)),
        "stderr": str(stderr.relative_to(ROOT)),
        "trace_rows": trace_rows,
    }


def terminate(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--interval-ms", type=int, default=250)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--start-gpu-burner", action="store_true", default=True)
    ap.add_argument("--no-gpu-burner", dest="start_gpu_burner", action="store_false")
    args = ap.parse_args()

    if not PQC_FUSE.exists():
        raise SystemExit("missing build/pqc_fuse; run cmake --build build first")
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    burner = None
    tegra = None
    samples = []
    try:
        if args.start_gpu_burner and GPU_BURNER.exists():
            burner = subprocess.Popen([str(GPU_BURNER), str(max(3, args.samples))], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.5)
        tegra_cmd = ["sudo", "-S", "tegrastats", "--interval", str(args.interval_ms)]
        tegra = subprocess.Popen(tegra_cmd, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert tegra.stdin is not None
        tegra.stdin.write(os.environ.get("PQC_SUDO_PASSWORD", "1234qwer") + "\n")
        tegra.stdin.flush()
        assert tegra.stdout is not None
        for idx in range(args.samples):
            line = tegra.stdout.readline()
            if not line:
                break
            telemetry = parse_tegrastats(line)
            samples.append(run_admission(out_dir, idx + 1, telemetry))
    finally:
        terminate(tegra)
        terminate(burner)

    decisions: dict[str, int] = {}
    for sample in samples:
        target = (sample.get("summary") or {}).get("chosen_target", "missing")
        decisions[target] = decisions.get(target, 0) + 1
    report = {
        "note": "Live tegrastats-to-admission bridge only; not full closed-loop QoS certification.",
        "samples_requested": args.samples,
        "samples_recorded": len(samples),
        "gpu_burner_started": bool(args.start_gpu_burner and GPU_BURNER.exists()),
        "decision_counts": decisions,
        "samples": samples,
    }
    json_path = out_dir / "qos_live_telemetry_admission.json"
    md_path = out_dir / "qos_live_telemetry_admission.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# QoS live telemetry admission bridge",
        "",
        "This bundle samples live `tegrastats` output while optional GPU pressure is active and feeds each sample into `pqc_fuse --admission-telemetry-smoke`.",
        "It is path evidence only, not full closed-loop QoS certification.",
        "",
        f"- Samples requested: `{args.samples}`",
        f"- Samples recorded: `{len(samples)}`",
        f"- GPU burner started: `{report['gpu_burner_started']}`",
        f"- Decision counts: `{decisions}`",
        "",
        "| sample | gpu_power_mw | gr3d_percent | mem_util | tensor_util | target | trace |",
        "|---:|---:|---:|---:|---:|---|---|",
    ]
    for sample in samples:
        t = sample["telemetry"]
        s = sample.get("summary") or {}
        lines.append(
            f"| {sample['sample']} | {t.get('gpu_power_mw')} | {t.get('gr3d_percent')} | "
            f"{t['mem_bandwidth_util']:.4f} | {t['tensor_core_util']:.4f} | "
            f"{s.get('chosen_target')} | `{sample['trace']}` |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "samples": len(samples), "decision_counts": decisions}, indent=2))
    return 0 if samples else 1


if __name__ == "__main__":
    raise SystemExit(main())
