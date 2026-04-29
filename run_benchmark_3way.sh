#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${HOME}/pqc_edge_workspace"
BUILD="${WORKSPACE}/build"
MOUNT="${WORKSPACE}/mnt_secure"
STORE="${WORKSPACE}/storage_physical"
COUNT=2560  # 2560 * 4K = 10MB

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     3-Way PQC-FUSE Benchmark (10 MB, bs=4K)                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# === 1. Raw I/O ===
echo "▶ [1/3] Raw I/O (no FUSE, no encryption)..."
rm -f "${STORE}/"*
sync

T0=$(date +%s%N)
dd if=/dev/urandom of="${STORE}/raw_bench.bin" bs=4K count=${COUNT} conv=fdatasync 2>&1
T1=$(date +%s%N)
RAW_MS=$(( (T1 - T0) / 1000000 ))
RAW_MBS=$(echo "scale=2; 10.0 * 1000 / ${RAW_MS}" | bc)
echo "  → Raw I/O: ${RAW_MS} ms (${RAW_MBS} MB/s)"
rm -f "${STORE}/"*

# === 2. CPU PQC ===
echo ""
echo "▶ [2/3] CPU PQC-FUSE (ML-KEM-512 via liboqs)..."
fusermount3 -u "${MOUNT}" 2>/dev/null || true
sleep 1

"${BUILD}/pqc_fuse" "${STORE}" "${MOUNT}" -f 2>/dev/null &
PID=$!
sleep 2

if ! mountpoint -q "${MOUNT}"; then
    echo "  ✗ CPU FUSE mount failed"; exit 1
fi

T0=$(date +%s%N)
dd if=/dev/urandom of="${MOUNT}/cpu_bench.bin" bs=4K count=${COUNT} conv=fdatasync 2>&1
T1=$(date +%s%N)
CPU_MS=$(( (T1 - T0) / 1000000 ))
CPU_MBS=$(echo "scale=2; 10.0 * 1000 / ${CPU_MS}" | bc)
echo "  → CPU PQC: ${CPU_MS} ms (${CPU_MBS} MB/s)"

fusermount3 -u "${MOUNT}" 2>/dev/null || true
wait ${PID} 2>/dev/null || true
sleep 2
rm -f "${STORE}/"*

# === 3. GPU PQC ===
echo ""
echo "▶ [3/3] GPU PQC-FUSE (CUDA Zero-copy)..."

"${BUILD}/pqc_fuse_gpu" "${STORE}" "${MOUNT}" -f 2>/dev/null &
PID=$!
sleep 3

if ! mountpoint -q "${MOUNT}"; then
    echo "  ✗ GPU FUSE mount failed"; exit 1
fi

T0=$(date +%s%N)
dd if=/dev/urandom of="${MOUNT}/gpu_bench.bin" bs=4K count=${COUNT} conv=fdatasync 2>&1
T1=$(date +%s%N)
GPU_MS=$(( (T1 - T0) / 1000000 ))
GPU_MBS=$(echo "scale=2; 10.0 * 1000 / ${GPU_MS}" | bc)
echo "  → GPU PQC: ${GPU_MS} ms (${GPU_MBS} MB/s)"

fusermount3 -u "${MOUNT}" 2>/dev/null || true
wait ${PID} 2>/dev/null || true

# === Summary ===
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║               BENCHMARK RESULTS (10 MB)                    ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  %-25s %8s ms  %8s MB/s  ║\n" "Raw I/O (No Encryption)" "${RAW_MS}" "${RAW_MBS}"
printf "║  %-25s %8s ms  %8s MB/s  ║\n" "CPU PQC (ML-KEM-512)" "${CPU_MS}" "${CPU_MBS}"
printf "║  %-25s %8s ms  %8s MB/s  ║\n" "GPU PQC (Zero-copy)" "${GPU_MS}" "${GPU_MBS}"
echo "╠══════════════════════════════════════════════════════════════╣"
CPU_OVERHEAD=$(echo "scale=1; (${CPU_MS} - ${RAW_MS}) * 100 / ${RAW_MS}" | bc)
GPU_OVERHEAD=$(echo "scale=1; (${GPU_MS} - ${RAW_MS}) * 100 / ${RAW_MS}" | bc)
SPEEDUP=$(echo "scale=1; ${CPU_MS}.0 / ${GPU_MS}.0" | bc)
printf "║  CPU PQC overhead vs Raw:  +%s%%                       ║\n" "${CPU_OVERHEAD}"
printf "║  GPU PQC overhead vs Raw:  +%s%%                       ║\n" "${GPU_OVERHEAD}"
printf "║  GPU speedup vs CPU PQC:   %sx faster                  ║\n" "${SPEEDUP}"
echo "╚══════════════════════════════════════════════════════════════╝"
