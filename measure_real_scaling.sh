#!/bin/bash
# measure_real_scaling.sh
# Measures true MB/s and CPU usage for 1, 2, 4 concurrent file writes.

WORKSPACE="$HOME/pqc_edge_workspace"
STORAGE="$WORKSPACE/storage_physical"
MOUNT="$WORKSPACE/mnt_secure"
BUILD="$WORKSPACE/build"
FILE_SIZE_MB=50 # total 50MB per thread to ensure it takes a few seconds

cleanup() {
    fusermount3 -u "$MOUNT" 2>/dev/null || true
    rm -rf "$STORAGE"/* 2>/dev/null || true
}

run_test() {
    local mode=$1
    local num_files=$2
    local bin_path=$3

    cleanup
    echo -n "Testing $mode with $num_files files... "
    
    # Start daemon
    "$bin_path" "$STORAGE" "$MOUNT" -f >/dev/null 2>&1 &
    local daemon_pid=$!
    sleep 2
    
    if ! mountpoint -q "$MOUNT"; then
        echo "Mount failed!"
        kill -9 $daemon_pid 2>/dev/null || true
        return
    fi
    
    # Start CPU tracking
    local cpu_log="/tmp/cpu_${mode}_${num_files}.log"
    rm -f "$cpu_log"
    while kill -0 $daemon_pid 2>/dev/null; do
        top -b -n 1 -p $daemon_pid | tail -n 1 | awk '{print $9}' >> "$cpu_log"
        sleep 0.5
    done &
    local tracker_pid=$!
    
    local start_time=$(date +%s%3N)
    
    # Run concurrent dd
    for i in $(seq 1 $num_files); do
        dd if=/dev/zero of="$MOUNT/test_${i}.bin" bs=1M count=$FILE_SIZE_MB conv=fdatasync 2>/dev/null &
    done
    wait
    
    local end_time=$(date +%s%3N)
    local duration_ms=$((end_time - start_time))
    local total_mb=$((num_files * FILE_SIZE_MB))
    local mbs=$(awk "BEGIN {print ($total_mb * 1000) / $duration_ms}")
    
    # Stop FUSE and tracker
    fusermount3 -u "$MOUNT" 2>/dev/null || true
    kill -9 $tracker_pid 2>/dev/null || true
    wait $daemon_pid 2>/dev/null || true
    
    local avg_cpu=$(awk '{ sum += $1; n++ } END { if (n > 0) print sum / n; else print 0 }' "$cpu_log")
    
    echo "Throughput: ${mbs} MB/s | Avg CPU: ${avg_cpu}%"
}

echo "=== Real Scalability & Resource Benchmark ==="
run_test "CPU-PQC" 1 "$BUILD/pqc_fuse"
run_test "CPU-PQC" 2 "$BUILD/pqc_fuse"
run_test "CPU-PQC" 4 "$BUILD/pqc_fuse"

run_test "GPU-PQC" 1 "$BUILD/pqc_fuse_gpu"
run_test "GPU-PQC" 2 "$BUILD/pqc_fuse_gpu"
run_test "GPU-PQC" 4 "$BUILD/pqc_fuse_gpu"
echo "Done."
