#!/bin/bash
# measure_realistic_scaling.sh
# Uses camera_capture_test.py (JPEG frames) instead of dd for realistic scaling metrics.

WORKSPACE="/home/thor/skim/pqc_encrpyted_fs"
STORAGE="$WORKSPACE/storage_physical"
MOUNT="$WORKSPACE/mnt_secure"
BUILD="$WORKSPACE/build"
CAMERA_TEST="/home/thor/skim/pqc_encrpyted_fs/camera_capture_test.py"
DURATION=5
FPS=30

cleanup() {
    fusermount3 -u "$MOUNT" 2>/dev/null || true
    rm -rf "$STORAGE"/* 2>/dev/null || true
}

run_test() {
    local mode=$1
    local num_cams=$2
    local bin_path=$3

    cleanup
    echo -n "Testing $mode with $num_cams Camera(s)... "
    
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
    local cpu_log="/tmp/cpu_${mode}_${num_cams}.log"
    rm -f "$cpu_log"
    while kill -0 $daemon_pid 2>/dev/null; do
        ps -p $daemon_pid -o %cpu= | tr -d ' ' >> "$cpu_log"
        sleep 0.5
    done &
    local tracker_pid=$!
    
    local py_pids=""
    for i in $(seq 1 $num_cams); do
        python3 "$CAMERA_TEST" --mount "$MOUNT" --fps $FPS --duration $DURATION --label "cam_${mode}_${i}" >/dev/null 2>&1 &
        py_pids="$py_pids $!"
    done
    wait $py_pids
    
    # Stop FUSE and tracker
    fusermount3 -u "$MOUNT" 2>/dev/null || true
    kill -9 $tracker_pid 2>/dev/null || true
    wait $daemon_pid 2>/dev/null || true
    
    # Aggregate results from JSONs
    local total_mbs=0
    for i in $(seq 1 $num_cams); do
        local json_file="/tmp/camera_bench_cam_${mode}_${i}.json"
        if [ -f "$json_file" ]; then
            local mbs=$(python3 -c "import json; print(json.load(open('$json_file'))['throughput_mbs'])")
            total_mbs=$(echo "$total_mbs + $mbs" | bc)
        fi
    done
    
    local avg_cpu=$(awk '{ sum += $1; n++ } END { if (n > 0) print sum / n; else print 0 }' "$cpu_log")
    
    echo "Aggregate Throughput: ${total_mbs} MB/s | Avg CPU: ${avg_cpu}%"
}

echo "=== Realistic Camera Scalability Benchmark (Duration: ${DURATION}s) ==="
run_test "CPU-PQC" 1 "$BUILD/pqc_fuse"
run_test "CPU-PQC" 2 "$BUILD/pqc_fuse"
run_test "CPU-PQC" 4 "$BUILD/pqc_fuse"

run_test "GPU-PQC" 1 "$BUILD/pqc_fuse_gpu"
run_test "GPU-PQC" 2 "$BUILD/pqc_fuse_gpu"
run_test "GPU-PQC" 4 "$BUILD/pqc_fuse_gpu"
echo "Done."
