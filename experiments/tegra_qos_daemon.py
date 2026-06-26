#!/usr/bin/env python3
import os
import time
import multiprocessing
import math
import subprocess
import json
from pathlib import Path

TRACE_OUT = Path("artifacts/validation/tegra_qos_daemon_trace.jsonl")

class HysteresisController:
    def __init__(self, enter_threshold, exit_threshold, hold_samples=2):
        if exit_threshold > enter_threshold:
            raise ValueError("exit_threshold must be <= enter_threshold")
        self.enter_threshold = float(enter_threshold)
        self.exit_threshold = float(exit_threshold)
        self.hold_samples = max(1, int(hold_samples))
        self.state = "open"
        self._below_exit_count = 0

    def update(self, value):
        value = float(value)
        event = "hold"
        if self.state == "open":
            self._below_exit_count = 0
            if value >= self.enter_threshold:
                self.state = "throttled"
                event = "enter"
        else:
            if value <= self.exit_threshold:
                self._below_exit_count += 1
                if self._below_exit_count >= self.hold_samples:
                    self.state = "open"
                    self._below_exit_count = 0
                    event = "exit"
            else:
                self._below_exit_count = 0
        return {
            "state": self.state,
            "event": event,
            "throttle": 1 if self.state == "throttled" else 0,
            "pressure_value": value,
            "enter_threshold": self.enter_threshold,
            "exit_threshold": self.exit_threshold,
            "hold_samples": self.hold_samples,
            "below_exit_count": self._below_exit_count,
        }

def background_io_worker(shared_throttle_flag, total_bytes):
    print("[I/O Worker] Started background I/O processing...")
    processed = 0
    chunk_size = 1024 * 1024
    start_time = time.time()
    
    while processed < total_bytes:
        if shared_throttle_flag.value == 1:
            time.sleep(0.05)
            continue
            
        end_busy = time.time() + 0.005
        while time.time() < end_busy:
            math.sqrt(12345.6789)
        processed += chunk_size
        
    duration = time.time() - start_time
    throughput = (total_bytes / (1024*1024)) / duration
    print(f"\n[I/O Worker] Finished {total_bytes / (1024*1024):.0f} MB in {duration:.2f} s ({throughput:.2f} MB/s)")

def ai_inference_worker(duration_sec, latencies):
    print("[AI Worker] Started foreground AI inference...")
    end_time = time.time() + duration_sec
    while time.time() < end_time:
        t_start = time.time()
        end_busy = time.time() + 0.02
        while time.time() < end_busy:
            math.sqrt(9876.54321)
        latencies.append((time.time() - t_start) * 1000)
        time.sleep(0.01)

def telemetry_daemon(shared_throttle_flag, duration_sec):
    print("[Telemetry] Daemon started, parsing tegrastats output for a prototype throttle loop...")
    # Read tegrastats output
    cmd = 'echo "1234qwer" | sudo -S tegrastats --interval 100'
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    TRACE_OUT.parent.mkdir(parents=True, exist_ok=True)
    trace_fp = TRACE_OUT.open("w", encoding="utf-8")
    controller = HysteresisController(
        enter_threshold=float(os.environ.get("PQC_QOS_CPU_ENTER_THRESHOLD", "35.0")),
        exit_threshold=float(os.environ.get("PQC_QOS_CPU_EXIT_THRESHOLD", "15.0")),
        hold_samples=int(os.environ.get("PQC_QOS_HOLD_SAMPLES", "2")),
    )
    
    end_time = time.time() + duration_sec
    try:
        while time.time() < end_time:
            line = process.stdout.readline()
            if not line:
                continue
            
            # Parse CPU load from tegrastats: CPU [1%@2601,...]
            record = {"raw": line.rstrip("\n"), "throttle": int(shared_throttle_flag.value)}
            if 'CPU [' in line:
                cpu_str = line.split('CPU [')[1].split(']')[0]
                cores = cpu_str.split(',')
                loads = [int(c.split('%')[0]) for c in cores if '%' in c]
                avg_load = sum(loads) / len(loads) if loads else 0
                record["avg_cpu_load"] = avg_load
                decision = controller.update(avg_load)
                shared_throttle_flag.value = decision["throttle"]
                record.update({
                    "throttle": int(shared_throttle_flag.value),
                    "hysteresis_state": decision["state"],
                    "hysteresis_event": decision["event"],
                    "pressure_value": decision["pressure_value"],
                    "enter_threshold": decision["enter_threshold"],
                    "exit_threshold": decision["exit_threshold"],
                    "hold_samples": decision["hold_samples"],
                    "below_exit_count": decision["below_exit_count"],
                })
            trace_fp.write(json.dumps(record) + "\n")
            trace_fp.flush()
    finally:
        process.kill()
        trace_fp.close()

if __name__ == "__main__":
    print("=== Real Telemetry Admission Control Experiment ===")
    
    manager = multiprocessing.Manager()
    shared_throttle = manager.Value('i', 0)
    latencies = manager.list()
    
    print("\n--- RUN: Telemetry ENABLED (AEGIS-Q QoS) ---")
    p_io = multiprocessing.Process(target=background_io_worker, args=(shared_throttle, 50 * 1024 * 1024))
    p_ai = multiprocessing.Process(target=ai_inference_worker, args=(3, latencies))
    p_tel = multiprocessing.Process(target=telemetry_daemon, args=(shared_throttle, 4))
    
    p_io.start()
    p_ai.start()
    p_tel.start()
    
    p_io.join()
    p_ai.join()
    p_tel.join()
    
    latencies_list = list(latencies)
    if latencies_list:
        p99_enabled = sorted(latencies_list)[int(len(latencies_list)*0.99)]
        print(f"[Result] AI p99 Latency (Enabled): {p99_enabled:.2f} ms")
        summary = {
            "enabled": True,
            "ai_p99_ms": p99_enabled,
            "trace_file": str(TRACE_OUT),
        }
        (TRACE_OUT.parent / "tegra_qos_daemon_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        
    print("\nExperiment Complete. tegrastats parsing prototype finished; this is not a validated closed-loop controller.")
