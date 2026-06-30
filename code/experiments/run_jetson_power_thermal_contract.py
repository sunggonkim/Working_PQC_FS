#!/usr/bin/env python3
"""Gate 0.11-S0 Jetson power, thermal, and platform-state contract."""

from __future__ import annotations

import glob
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "validation" / "jetson_power_thermal_contract"

OFFICIAL_BASIS = [
    {
        "name": "NVIDIA Jetson tegrastats utility",
        "url": (
            "https://docs.nvidia.com/jetson/archives/r38.2/DeveloperGuide/"
            "AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html"
        ),
        "used_for": [
            "tegrastats reports memory, CPU/GPU/EMC frequency, temperature, and power rails",
            "foreground tegrastats sampling with --interval",
            "invalid-run metadata requirements for headline performance runs",
        ],
    },
    {
        "name": "NVIDIA Jetson Thor power and thermal guidance",
        "url": (
            "https://docs.nvidia.com/jetson/archives/r38.2/DeveloperGuide/"
            "SD/PlatformPowerAndPerformance/JetsonThor.html"
        ),
        "used_for": [
            "nvpmodel -q as the current power-mode query",
            "power mode persistence and reboot caveats",
            "thermal throttling and shutdown boundary context",
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


def path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def read_first_existing(paths: list[Path]) -> str | None:
    for path in paths:
        value = read_file(path)
        if value is not None:
            return value
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


def run_streaming_command(argv: list[str], seconds: float) -> dict[str, Any]:
    started = time.monotonic()
    try:
        proc = subprocess.Popen(
            argv,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except FileNotFoundError:
        return {
            "argv": argv,
            "returncode": None,
            "stdout": "",
            "stderr": "command not found",
            "timed_out": False,
            "duration_seconds": 0.0,
        }

    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            stdout, stderr = proc.communicate(timeout=2.0)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
            stdout, stderr = proc.communicate()

    return {
        "argv": argv,
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": timed_out,
        "duration_seconds": time.monotonic() - started,
    }


def write_raw(out_dir: Path, stem: str, result: dict[str, Any]) -> dict[str, Any]:
    stdout_path = out_dir / f"{stem}.stdout.txt"
    stderr_path = out_dir / f"{stem}.stderr.txt"
    stdout_path.write_text(str(result.get("stdout", "")), encoding="utf-8")
    stderr_path.write_text(str(result.get("stderr", "")), encoding="utf-8")
    return {
        "stdout_path": relpath(stdout_path),
        "stderr_path": relpath(stderr_path),
    }


def parse_int(text: str | None) -> int | None:
    if text is None:
        return None
    try:
        return int(text.strip())
    except ValueError:
        return None


def temp_millic_to_celsius(value: str | None) -> float | None:
    number = parse_int(value)
    if number is None:
        return None
    if abs(number) > 1000:
        return number / 1000.0
    return float(number)


def collect_cpu_frequency() -> list[dict[str, Any]]:
    cpus: list[dict[str, Any]] = []
    for cpu_dir in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*")):
        cpufreq = cpu_dir / "cpufreq"
        cpu = {
            "cpu": cpu_dir.name,
            "online": read_file(cpu_dir / "online"),
            "cpufreq_available": cpufreq.exists(),
        }
        if cpufreq.exists():
            fields = [
                "scaling_cur_freq",
                "cpuinfo_cur_freq",
                "scaling_min_freq",
                "scaling_max_freq",
                "cpuinfo_min_freq",
                "cpuinfo_max_freq",
                "scaling_governor",
                "scaling_driver",
                "affected_cpus",
            ]
            for field in fields:
                cpu[field] = read_file(cpufreq / field)
        cpus.append(cpu)
    return cpus


def collect_devfreq() -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    for path_text in sorted(glob.glob("/sys/class/devfreq/*")):
        path = Path(path_text)
        fields = [
            "name",
            "cur_freq",
            "min_freq",
            "max_freq",
            "available_frequencies",
            "governor",
            "available_governors",
            "load",
            "target_freq",
        ]
        record = {"path": str(path)}
        for field in fields:
            record[field] = read_file(path / field)
        devices.append(record)
    return devices


def collect_thermal_zones() -> list[dict[str, Any]]:
    zones: list[dict[str, Any]] = []
    for path in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        record: dict[str, Any] = {
            "path": str(path),
            "type": read_file(path / "type"),
            "temp_raw": read_file(path / "temp"),
            "temp_celsius": temp_millic_to_celsius(read_file(path / "temp")),
            "mode": read_file(path / "mode"),
            "policy": read_file(path / "policy"),
        }
        trips: list[dict[str, Any]] = []
        for trip_temp in sorted(path.glob("trip_point_*_temp")):
            prefix = trip_temp.name.removesuffix("_temp")
            trips.append({
                "name": prefix,
                "temp_raw": read_file(trip_temp),
                "temp_celsius": temp_millic_to_celsius(read_file(trip_temp)),
                "type": read_file(path / f"{prefix}_type"),
            })
        record["trip_points"] = trips
        zones.append(record)
    return zones


def collect_hwmon() -> list[dict[str, Any]]:
    monitors: list[dict[str, Any]] = []
    for path in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        record: dict[str, Any] = {
            "path": str(path),
            "name": read_file(path / "name"),
            "temps": [],
            "power": [],
            "freq": [],
            "pwm": [],
        }
        for temp in sorted(path.glob("temp*_input")):
            prefix = temp.name.removesuffix("_input")
            record["temps"].append({
                "sensor": prefix,
                "label": read_file(path / f"{prefix}_label"),
                "temp_raw": read_file(temp),
                "temp_celsius": temp_millic_to_celsius(read_file(temp)),
                "crit_raw": read_file(path / f"{prefix}_crit"),
                "max_raw": read_file(path / f"{prefix}_max"),
            })
        for power in sorted(path.glob("power*_input")):
            prefix = power.name.removesuffix("_input")
            record["power"].append({
                "sensor": prefix,
                "label": read_file(path / f"{prefix}_label"),
                "microwatts": parse_int(read_file(power)),
                "average_raw": read_file(path / f"{prefix}_average"),
            })
        for freq in sorted(path.glob("freq*_input")):
            prefix = freq.name.removesuffix("_input")
            record["freq"].append({
                "sensor": prefix,
                "label": read_file(path / f"{prefix}_label"),
                "hz": parse_int(read_file(freq)),
            })
        for pwm in sorted(path.glob("pwm*")):
            if re.fullmatch(r"pwm[0-9]+", pwm.name):
                record["pwm"].append({
                    "sensor": pwm.name,
                    "value": read_file(pwm),
                    "enable": read_file(path / f"{pwm.name}_enable"),
                })
        monitors.append(record)
    return monitors


def collect_throttle_sysfs() -> dict[str, Any]:
    records: dict[str, Any] = {
        "cpu_thermal_throttle": [],
        "thermal_cooling_devices": [],
        "bpmp_debug_available": path_exists(Path("/sys/kernel/debug/bpmp/debug")),
    }
    for path in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/thermal_throttle")):
        entry = {"path": str(path)}
        for child in sorted(path.iterdir()):
            if child.is_file():
                entry[child.name] = read_file(child)
        records["cpu_thermal_throttle"].append(entry)
    for path in sorted(Path("/sys/class/thermal").glob("cooling_device*")):
        records["thermal_cooling_devices"].append({
            "path": str(path),
            "type": read_file(path / "type"),
            "cur_state": read_file(path / "cur_state"),
            "max_state": read_file(path / "max_state"),
        })
    return records


def platform_manifest() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "uname": platform.uname()._asdict(),
        "kernel_release": platform.release(),
        "nv_tegra_release": read_file(Path("/etc/nv_tegra_release")),
        "nv_boot_control": read_file(Path("/etc/nv_boot_control.conf")),
        "nvpmodel_conf_exists": Path("/etc/nvpmodel.conf").exists(),
        "python": platform.python_version(),
    }


def find_nvpmodel() -> str | None:
    candidates = [
        shutil.which("nvpmodel"),
        "/usr/sbin/nvpmodel",
        "/sbin/nvpmodel",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def sample_tegrastats(out_dir: Path) -> dict[str, Any]:
    path = shutil.which("tegrastats")
    if not path:
        return {
            "available": False,
            "path": None,
            "command": ["tegrastats", "--interval", "200"],
            "sample_count": 0,
            "raw_paths": {},
            "parsed": parse_tegrastats(""),
        }

    result = run_streaming_command([path, "--interval", "200"], seconds=1.25)
    raw_paths = write_raw(out_dir, "tegrastats", result)
    parsed = parse_tegrastats(str(result.get("stdout", "")))
    return {
        "available": True,
        "path": path,
        "command": result["argv"],
        "returncode": result["returncode"],
        "timed_out_for_sampling": result["timed_out"],
        "duration_seconds": result["duration_seconds"],
        "sample_count": parsed["sample_count"],
        "raw_paths": raw_paths,
        "parsed": parsed,
    }


def parse_tegrastats(stdout: str) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    temperatures: list[dict[str, Any]] = []
    power_rails: list[dict[str, Any]] = []
    cpu_freq_mhz: list[int] = []
    gr3d_freq_mhz: list[int] = []
    emc_freq_mhz: list[int] = []
    throttle_markers: list[str] = []

    for line in lines:
        lower = line.lower()
        if any(marker in lower for marker in (
            "throt",
            "over-current",
            "undervoltage",
            "under-voltage",
            "thermal shutdown",
            "shutdown",
        )):
            throttle_markers.append(line)
        for name, value in re.findall(r"([A-Za-z0-9_+-]+)@([0-9]+(?:\.[0-9]+)?)C", line):
            temperatures.append({"name": name, "celsius": float(value), "line": line})
        for name, current, average in re.findall(
            r"(VDD_[A-Za-z0-9_]+)\s+([0-9]+)mW/([0-9]+)mW", line
        ):
            power_rails.append({
                "name": name,
                "current_mw": int(current),
                "average_mw": int(average),
                "line": line,
            })
        cpu_match = re.search(r"CPU\s+\[([^\]]+)\]", line)
        if cpu_match:
            cpu_freq_mhz.extend(int(value) for value in re.findall(r"@([0-9]+)", cpu_match.group(1)))
        gr3d_match = re.search(r"GR3D_FREQ\s+(?:[0-9]+%?)?@\[?([0-9,]+)\]?", line)
        if gr3d_match:
            gr3d_freq_mhz.extend(int(value) for value in gr3d_match.group(1).split(",") if value)
        emc_match = re.search(r"EMC_FREQ\s+(?:[0-9]+%?)?@?([0-9]+)", line)
        if emc_match:
            emc_freq_mhz.append(int(emc_match.group(1)))

    return {
        "sample_count": len(lines),
        "temperatures": temperatures,
        "power_rails": power_rails,
        "cpu_freq_mhz": cpu_freq_mhz,
        "gr3d_freq_mhz": gr3d_freq_mhz,
        "emc_freq_mhz": emc_freq_mhz,
        "throttle_markers": throttle_markers,
        "raw_line_preview": lines[:5],
    }


def query_nvpmodel(out_dir: Path) -> dict[str, Any]:
    path = find_nvpmodel()
    if not path:
        return {
            "available": False,
            "path": None,
            "command": ["nvpmodel", "-q"],
            "returncode": None,
            "raw_paths": {},
        }
    result = run_command([path, "-q"], timeout=10.0)
    raw_paths = write_raw(out_dir, "nvpmodel_q", result)
    return {
        "available": True,
        "path": path,
        "command": result["argv"],
        "returncode": result["returncode"],
        "timed_out": result["timed_out"],
        "raw_paths": raw_paths,
        "mode_lines": [
            line.strip()
            for line in str(result.get("stdout", "")).splitlines()
            if line.strip()
        ],
    }


def collect_platform_state() -> dict[str, Any]:
    return {
        "cpu_frequency": collect_cpu_frequency(),
        "devfreq": collect_devfreq(),
        "thermal_zones": collect_thermal_zones(),
        "hwmon": collect_hwmon(),
        "throttling_sysfs": collect_throttle_sysfs(),
        "jetson_clocks": {
            "path": shutil.which("jetson_clocks"),
            "show": run_command([shutil.which("jetson_clocks") or "jetson_clocks", "--show"], timeout=10.0)
            if shutil.which("jetson_clocks") else {
                "argv": ["jetson_clocks", "--show"],
                "returncode": None,
                "stdout": "",
                "stderr": "jetson_clocks not found",
                "timed_out": False,
            },
        },
    }


def sysfs_throttle_observed(throttle: dict[str, Any]) -> bool:
    for entry in throttle.get("cpu_thermal_throttle", []):
        for key, value in entry.items():
            if key == "path":
                continue
            parsed = parse_int(str(value) if value is not None else None)
            if parsed and parsed > 0:
                return True
    for device in throttle.get("thermal_cooling_devices", []):
        cur_state = parse_int(device.get("cur_state"))
        if cur_state and cur_state > 0:
            return True
    return False


def build_eligibility(
    tegrastats: dict[str, Any],
    nvpmodel: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    warnings: list[str] = []
    parsed = tegrastats.get("parsed", {})
    if not tegrastats.get("available"):
        reasons.append("tegrastats command unavailable")
    elif int(tegrastats.get("sample_count", 0) or 0) <= 0:
        reasons.append("tegrastats produced no retained samples")
    if not nvpmodel.get("available"):
        reasons.append("nvpmodel command unavailable")
    elif nvpmodel.get("returncode") not in (0, None):
        reasons.append("nvpmodel -q did not complete cleanly")
    if not state.get("thermal_zones"):
        reasons.append("no thermal zones captured from sysfs")
    if not state.get("cpu_frequency"):
        warnings.append("no CPU frequency state captured")
    if not state.get("devfreq"):
        warnings.append("no devfreq clock/frequency state captured")
    if parsed.get("throttle_markers"):
        reasons.append("tegrastats output contains throttling/thermal shutdown markers")
    if sysfs_throttle_observed(state.get("throttling_sysfs", {})):
        reasons.append("sysfs cooling/throttle state is active or nonzero")

    return {
        "headline_run_eligible": not reasons,
        "invalid_reasons": reasons,
        "warnings": warnings,
        "rules": [
            "A headline run is invalid if tegrastats is unavailable or produces no retained samples.",
            "A headline run is invalid if current nvpmodel power mode cannot be retained.",
            "A headline run is invalid if thermal-zone metadata cannot be retained.",
            "A headline run is invalid if tegrastats or sysfs reports throttling, over-current, undervoltage, or thermal shutdown markers.",
            "A headline run must retain raw tegrastats/nvpmodel logs, CPU/devfreq clock state, thermal zones, and platform manifest next to the benchmark artifact.",
        ],
    }


def build_payload(out_dir: Path) -> dict[str, Any]:
    tegrastats = sample_tegrastats(out_dir)
    nvpmodel = query_nvpmodel(out_dir)
    state = collect_platform_state()
    eligibility = build_eligibility(tegrastats, nvpmodel, state)
    required_sections_present = {
        "tegrastats_section": "available" in tegrastats and "parsed" in tegrastats,
        "nvpmodel_section": "available" in nvpmodel,
        "clock_frequency_state": bool(state.get("cpu_frequency")) or bool(state.get("devfreq")),
        "thermal_state": bool(state.get("thermal_zones")) or bool(state.get("hwmon")),
        "throttling_markers": "throttling_sysfs" in state,
        "invalid_run_rules": bool(eligibility.get("rules")),
    }
    return {
        "schema_version": 1,
        "generated_by": "code/experiments/run_jetson_power_thermal_contract.py",
        "generated_utc": now_utc(),
        "scope": "Gate 0.11-S0 Jetson power/thermal/platform-state capture for headline-run eligibility.",
        "official_basis": OFFICIAL_BASIS,
        "platform_manifest": platform_manifest(),
        "tegrastats": tegrastats,
        "nvpmodel": nvpmodel,
        "platform_state": state,
        "headline_run_eligibility": eligibility,
        "required_sections_present": required_sections_present,
        "artifact_verdict": {
            "runner_completed": True,
            "overall_pass": all(required_sections_present.values()),
            "paper_claim_guard": (
                "No headline performance result is paper-eligible unless it carries "
                "this contract or an equivalent per-run contract with no invalid reasons."
            ),
        },
        "non_claims": [
            "This artifact does not prove stable clocks during any future benchmark by itself.",
            "This artifact does not prove absence of throttling for a benchmark unless captured during that same run.",
            "This artifact does not upgrade any performance number into a paper claim.",
        ],
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    eligibility = payload["headline_run_eligibility"]
    tegrastats = payload["tegrastats"]
    nvpmodel = payload["nvpmodel"]
    parsed = tegrastats.get("parsed", {})
    verdict = payload["artifact_verdict"]

    lines = [
        "# Jetson Power/Thermal Contract",
        "",
        f"- Generated: `{payload['generated_utc']}`",
        f"- Generated by: `{payload['generated_by']}`",
        f"- Overall artifact pass: `{verdict['overall_pass']}`",
        f"- Headline-run eligible for this sample: `{eligibility['headline_run_eligible']}`",
        "",
        "## Raw Evidence",
        "",
        f"- tegrastats available: `{tegrastats.get('available')}`",
        f"- tegrastats samples: `{tegrastats.get('sample_count', 0)}`",
        f"- tegrastats stdout: `{tegrastats.get('raw_paths', {}).get('stdout_path')}`",
        f"- nvpmodel available: `{nvpmodel.get('available')}`",
        f"- nvpmodel return code: `{nvpmodel.get('returncode')}`",
        f"- nvpmodel stdout: `{nvpmodel.get('raw_paths', {}).get('stdout_path')}`",
        "",
        "## Parsed Snapshot",
        "",
        f"- tegrastats temperature readings: `{len(parsed.get('temperatures', []))}`",
        f"- tegrastats power rails: `{len(parsed.get('power_rails', []))}`",
        f"- tegrastats throttle markers: `{len(parsed.get('throttle_markers', []))}`",
        f"- CPU frequency records: `{len(payload['platform_state'].get('cpu_frequency', []))}`",
        f"- devfreq records: `{len(payload['platform_state'].get('devfreq', []))}`",
        f"- thermal zones: `{len(payload['platform_state'].get('thermal_zones', []))}`",
        f"- hwmon devices: `{len(payload['platform_state'].get('hwmon', []))}`",
        "",
        "## Invalid-Run Rules",
        "",
    ]
    lines.extend(f"- {rule}" for rule in eligibility["rules"])
    if eligibility["invalid_reasons"]:
        lines.extend(["", "## Current Invalid Reasons", ""])
        lines.extend(f"- {reason}" for reason in eligibility["invalid_reasons"])
    if eligibility["warnings"]:
        lines.extend(["", "## Current Warnings", ""])
        lines.extend(f"- {warning}" for warning in eligibility["warnings"])
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
    json_path = OUT / "jetson_power_thermal_contract.json"
    md_path = OUT / "jetson_power_thermal_contract.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(payload, md_path)
    print(json.dumps({
        "overall_pass": payload["artifact_verdict"]["overall_pass"],
        "headline_run_eligible": payload["headline_run_eligibility"]["headline_run_eligible"],
        "invalid_reasons": payload["headline_run_eligibility"]["invalid_reasons"],
        "json": relpath(json_path),
        "markdown": relpath(md_path),
    }, indent=2, sort_keys=True))
    return 0 if payload["artifact_verdict"]["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
