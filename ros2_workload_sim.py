import time
import os
import csv

def set_state(c, g, t):
    with open("/tmp/telemetry.csv.tmp", "w") as f:
        f.write(f"{c},{g},{t}\n")
    os.rename("/tmp/telemetry.csv.tmp", "/tmp/telemetry.csv")

phases = [
    {"duration": 10, "c":0, "g":0, "t":0, "name": "Idle"},
    {"duration": 10, "c":1, "g":1, "t":1, "name": "Normal Autonomous Driving"},
    {"duration": 15, "c":1, "g":3, "t":1, "name": "Disruption: YOLO Burst (GPU=Max)"},
    {"duration": 15, "c":3, "g":1, "t":1, "name": "Disruption: SLAM Burst (CPU=Max)"},
    {"duration": 15, "c":1, "g":1, "t":3, "name": "Disruption: Thermal Warning"},
    {"duration": 10, "c":0, "g":0, "t":0, "name": "Cooldown"}
]

print("Starting ROS2 Telemetry Simulator (CITADEL Q-Learning Context)...")
start_time = time.time()

with open("citadel_telemetry_log.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["time_s", "phase", "c_cpu", "c_gpu", "thermal"])
    
    for p in phases:
        print(f"[{time.time()-start_time:05.1f}s] {p['name']} (CPU:{p['c']} GPU:{p['g']} TMP:{p['t']})")
        end_time = time.time() + p["duration"]
        while time.time() < end_time:
            set_state(p["c"], p["g"], p["t"])
            writer.writerow([time.time() - start_time, p['name'], p['c'], p['g'], p['t']])
            f.flush()
            time.sleep(0.1)

print("Simulation Complete.")
if os.path.exists("/tmp/telemetry.csv"):
    os.remove("/tmp/telemetry.csv")
