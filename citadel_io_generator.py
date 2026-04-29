import time
import os
import csv
import sys

mount_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pqc_mnt"
log_file = "citadel_io_log.csv"

# Write a 1MB chunk continuously
chunk_size = 1024 * 1024
data = os.urandom(chunk_size)

with open(log_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["time_s", "throughput_mbps"])
    
    start_time = time.time()
    file_idx = 0
    
    while True:
        t0 = time.time()
        elapsed = t0 - start_time
        if elapsed > 75: # Stop with the telemetry sim
            break
            
        filepath = os.path.join(mount_dir, f"sensor_dump_{file_idx}.bin")
        try:
            with open(filepath, "wb") as out:
                out.write(data)
                out.flush()
                os.fsync(out.fileno())
            t1 = time.time()
            mbps = (chunk_size / (1024*1024)) / (t1 - t0)
            writer.writerow([t1 - start_time, mbps])
            f.flush()
            os.remove(filepath)
        except Exception as e:
            pass
        file_idx += 1
        time.sleep(0.01) # Small pacing to prevent artificial 100% lock up
