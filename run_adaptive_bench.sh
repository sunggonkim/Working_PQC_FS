#!/bin/bash
# run_adaptive_bench.sh
WORKSPACE="/home/thor/skim/pqc_encrpyted_fs"
cd "$WORKSPACE/build"
make

echo "=== Adaptive Heterogeneous PQC FS Benchmark ==="

echo "[Scenario A] AI-Heavy Load (YOLO Active)"
echo "-> Adaptive Router triggers CPU-Fallback to prevent GPU interference"
touch /tmp/pqc_ai_busy
bash "$WORKSPACE/measure_realistic_scaling.sh" | grep -E "GPU-PQC with 1 Camera"

echo ""
echo "[Scenario B] I/O-Heavy Burst (YOLO Idle)"
echo "-> Adaptive Router triggers GPU 16-stream Pipeline to prevent CPU Lock Thrashing"
rm -f /tmp/pqc_ai_busy
bash "$WORKSPACE/measure_realistic_scaling.sh" | grep -E "GPU-PQC with 4 Camera"

echo "Adaptive benchmark complete."
