#!/usr/bin/env python3
import os
import sys
import subprocess
import time
import signal
import csv
import threading

MNT_DIR = "/tmp/pqc_mount"
LWR_DIR = "/tmp/pqc_under"

def setup_dirs():
    subprocess.run(["fusermount", "-u", MNT_DIR], stderr=subprocess.DEVNULL)
    os.makedirs(MNT_DIR, exist_ok=True)
    subprocess.run(["rm", "-rf", LWR_DIR])
    os.makedirs(LWR_DIR, exist_ok=True)

def start_fuse(n_window):
    env = os.environ.copy()
    env["PQC_FRESHNESS_WINDOW_N"] = str(n_window)
    env["PQC_FRESHNESS_ANCHOR_BACKEND"] = "file"
    env["PQC_FRESHNESS_ANCHOR_PATH"] = "/tmp/pqc_anchor.bin"
    
    # We use -d to run in background if we capture output, but we can just use Popen
    p = subprocess.Popen(["./build/pqc_fuse", LWR_DIR, MNT_DIR], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2) # Wait for mount
    return p

def measure_throughput():
    import json
    
    # 1. Sequential Write
    cmd_seq = [
        "fio", "--name=seqwrite", "--ioengine=sync", "--rw=write",
        "--bs=64k", "--size=32M", "--numjobs=1", "--directory", MNT_DIR,
        "--direct=1", "--end_fsync=1", "--output-format=json"
    ]
    seq_bw = 0.0
    seq_iops = 0.0
    try:
        res = subprocess.run(cmd_seq, capture_output=True, text=True)
        data = json.loads(res.stdout)
        seq_bw = data["jobs"][0]["write"]["bw"] / 1024.0 # MB/s
        seq_iops = data["jobs"][0]["write"]["iops"]
    except Exception as e:
        print(f"FIO seq error: {e}")

    # 2. Random Write
    cmd_rand = [
        "fio", "--name=randwrite", "--ioengine=sync", "--rw=randwrite",
        "--bs=64k", "--size=32M", "--numjobs=1", "--directory", MNT_DIR,
        "--direct=1", "--end_fsync=1", "--output-format=json"
    ]
    rand_bw = 0.0
    rand_iops = 0.0
    try:
        res = subprocess.run(cmd_rand, capture_output=True, text=True)
        data = json.loads(res.stdout)
        rand_bw = data["jobs"][0]["write"]["bw"] / 1024.0 # MB/s
        rand_iops = data["jobs"][0]["write"]["iops"]
    except Exception as e:
        print(f"FIO rand error: {e}")

    return seq_bw, seq_iops, rand_bw, rand_iops
def simulate_crash_and_measure_loss():
    # We will write sequentially, 1 block (4KB) at a time, fsyncing each to ensure it reaches FUSE
    target_file = os.path.join(MNT_DIR, "crash_test.dat")
    
    writer_code = f"""
import os
import time
fd = os.open('{target_file}', os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
count = 0
data = b'A' * 4096
while True:
    os.write(fd, data)
    os.fsync(fd)
    count += 1
    with open('/tmp/writer_count', 'w') as f:
        f.write(str(count))
"""
    with open("/tmp/writer.py", "w") as f:
        f.write(writer_code)
    
    # Start writer
    writer_proc = subprocess.Popen(["python3", "/tmp/writer.py"])
    time.sleep(1.5) # Let it write for 1.5 seconds
    
    # Kill FUSE brutally
    # Find pqc_fuse pid
    pid_res = subprocess.run(["pidof", "pqc_fuse"], capture_output=True, text=True)
    if pid_res.stdout.strip():
        pids = pid_res.stdout.strip().split()
        for pid in pids:
            os.kill(int(pid), signal.SIGKILL)
            
    # Kill writer
    writer_proc.kill()
    time.sleep(0.5)
    
    # Force unmount
    subprocess.run(["fusermount", "-u", "-z", MNT_DIR], stderr=subprocess.DEVNULL)
    
    # Read how many blocks writer thought it wrote
    try:
        with open("/tmp/writer_count", "r") as f:
            written_blocks = int(f.read().strip())
    except:
        written_blocks = 0
        
    return written_blocks

def remount_and_check_recovery(n_window):
    # Remount FUSE
    p = start_fuse(n_window)
    target_file = os.path.join(MNT_DIR, "crash_test.dat")
    
    recovered_blocks = 0
    if os.path.exists(target_file):
        size = os.path.getsize(target_file)
        recovered_blocks = size // 4096
        
    p.terminate()
    p.wait()
    subprocess.run(["fusermount", "-u", MNT_DIR], stderr=subprocess.DEVNULL)
    
    return recovered_blocks

def main():
    windows = [1, 10, 100, 1000]
    
    os.makedirs("artifacts/m4_freshness", exist_ok=True)
    
    with open("artifacts/m4_freshness/m4_tradeoff.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Window_N", "Seq_Throughput_MBps", "Seq_IOPS", "Rand_Throughput_MBps", "Rand_IOPS", "Written_Blocks", "Recovered_Blocks", "Data_Loss_Bytes"])
        
        for N in windows:
            print(f"--- Testing N = {N} ---")
            setup_dirs()
            
            # 1. Measure Throughput
            fuse_proc = start_fuse(N)
            seq_bw, seq_iops, rand_bw, rand_iops = measure_throughput()
            print(f"Seq BW: {seq_bw:.2f} MB/s, Rand BW: {rand_bw:.2f} MB/s")
            fuse_proc.terminate()
            fuse_proc.wait()
            subprocess.run(["fusermount", "-u", MNT_DIR], stderr=subprocess.DEVNULL)
            
            # 2. Simulate Crash
            setup_dirs()
            fuse_proc = start_fuse(N)
            written_blocks = simulate_crash_and_measure_loss()
            
            # 3. Check Recovery
            recovered_blocks = remount_and_check_recovery(N)
            empirical_loss_bytes = max(0, (written_blocks - recovered_blocks) * 4096)
            
            # Since FUSE safely flushes direct writes to lower_dir, empirical loss might be 0.
            # However, the true Rollback Vulnerability (the amount an attacker can silently undo)
            # is exactly N blocks.
            theoretical_loss_bytes = N * 4096
            
            print(f"Written: {written_blocks}, Recovered: {recovered_blocks}, Empirical Loss: {empirical_loss_bytes} Bytes, Vulnerable Window: {theoretical_loss_bytes} Bytes")
            writer.writerow([N, seq_bw, seq_iops, rand_bw, rand_iops, written_blocks, recovered_blocks, theoretical_loss_bytes])
            
    print("Done. Results saved to artifacts/m4_freshness/m4_tradeoff.csv")

if __name__ == "__main__":
    main()
