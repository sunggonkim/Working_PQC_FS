#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  run_experiment.sh — PQC-FUSE Edge AI Bottleneck Profiling Script
# ═══════════════════════════════════════════════════════════════════════════════
#
#  This script:
#    1. Builds the PQC-FUSE filesystem
#    2. Sets up directory structure
#    3. Mounts the FUSE filesystem
#    4. Runs I/O benchmarks (dd / fio) against the encrypted mount
#    5. Simultaneously logs CPU/GPU usage via tegrastats (Jetson only)
#    6. Collects all results for analysis
#
#  Usage:
#    chmod +x run_experiment.sh
#    ./run_experiment.sh [--skip-build] [--fio] [--size SIZE_MB]
#
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Color Output ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "\n${CYAN}${BOLD}═══ $* ═══${NC}\n"; }

# ── Configuration ──
WORKSPACE="${HOME}/pqc_edge_workspace"
MOUNT_POINT="${WORKSPACE}/mnt_secure"
STORAGE_DIR="${WORKSPACE}/storage_physical"
BUILD_DIR="${WORKSPACE}/build"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${WORKSPACE}/results"
LOG_FILE="${WORKSPACE}/pqc_fuse_latency.log"

# Experiment parameters
TEST_SIZE_MB=${SIZE_MB:-256}            # Default 256 MB (change to 1024 for 1GB)
DD_BS="4K"                              # Block size for dd
TEGRASTATS_INTERVAL=200                 # ms between tegrastats samples
SKIP_BUILD=false
USE_FIO=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build) SKIP_BUILD=true; shift ;;
        --fio)        USE_FIO=true; shift ;;
        --size)       TEST_SIZE_MB="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--skip-build] [--fio] [--size SIZE_MB]"
            echo "  --skip-build : Skip cmake/make build step"
            echo "  --fio        : Use fio instead of dd for benchmarking"
            echo "  --size N     : Test file size in MB (default: 256)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 0: Cleanup function
# ═══════════════════════════════════════════════════════════════════════════════

cleanup() {
    log_step "Cleanup"
    
    # Stop tegrastats if running
    if [[ -n "${TEGRA_PID:-}" ]] && kill -0 "$TEGRA_PID" 2>/dev/null; then
        log_info "Stopping tegrastats (PID: $TEGRA_PID)..."
        sudo kill "$TEGRA_PID" 2>/dev/null || true
    fi

    # Stop mpstat if running
    if [[ -n "${MPSTAT_PID:-}" ]] && kill -0 "$MPSTAT_PID" 2>/dev/null; then
        log_info "Stopping mpstat (PID: $MPSTAT_PID)..."
        kill "$MPSTAT_PID" 2>/dev/null || true
    fi

    # Unmount FUSE
    if mountpoint -q "${MOUNT_POINT}" 2>/dev/null; then
        log_info "Unmounting FUSE at ${MOUNT_POINT}..."
        fusermount3 -u "${MOUNT_POINT}" 2>/dev/null || \
        fusermount -u "${MOUNT_POINT}" 2>/dev/null || \
        sudo umount "${MOUNT_POINT}" 2>/dev/null || true
    fi

    log_info "Cleanup complete."
}

trap cleanup EXIT

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1: Create directory structure
# ═══════════════════════════════════════════════════════════════════════════════

log_step "Step 1: Setting up workspace"

mkdir -p "${MOUNT_POINT}" "${STORAGE_DIR}" "${BUILD_DIR}" "${RESULTS_DIR}"
log_info "Workspace    : ${WORKSPACE}"
log_info "Mount point  : ${MOUNT_POINT}"
log_info "Storage dir  : ${STORAGE_DIR}"
log_info "Results dir  : ${RESULTS_DIR}"

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2: Build PQC-FUSE
# ═══════════════════════════════════════════════════════════════════════════════

if [[ "${SKIP_BUILD}" == "false" ]]; then
    log_step "Step 2: Building PQC-FUSE"
    
    cd "${BUILD_DIR}"
    cmake "${SOURCE_DIR}" -DCMAKE_BUILD_TYPE=Release
    make -j"$(nproc)"
    
    log_info "Build successful: ${BUILD_DIR}/pqc_fuse"
else
    log_step "Step 2: Skipping build (--skip-build)"
fi

FUSE_BIN="${BUILD_DIR}/pqc_fuse"
if [[ ! -x "${FUSE_BIN}" ]]; then
    log_error "pqc_fuse binary not found at ${FUSE_BIN}"
    log_error "Run without --skip-build or check build errors."
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3: Unmount any existing FUSE mount
# ═══════════════════════════════════════════════════════════════════════════════

if mountpoint -q "${MOUNT_POINT}" 2>/dev/null; then
    log_warn "Mount point already mounted, unmounting..."
    fusermount3 -u "${MOUNT_POINT}" 2>/dev/null || \
    fusermount -u "${MOUNT_POINT}" 2>/dev/null || true
    sleep 1
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4: Start PQC-FUSE in background
# ═══════════════════════════════════════════════════════════════════════════════

log_step "Step 3: Mounting PQC-FUSE filesystem"

# Run FUSE in foreground mode but backgrounded, so we get logs on stderr
"${FUSE_BIN}" "${STORAGE_DIR}" "${MOUNT_POINT}" -f &
FUSE_PID=$!

# Wait for mount to be ready
sleep 2

if ! mountpoint -q "${MOUNT_POINT}" 2>/dev/null; then
    log_error "FUSE mount failed! Check error messages above."
    exit 1
fi

log_info "PQC-FUSE mounted successfully (PID: ${FUSE_PID})"
log_info "  Virtual mount : ${MOUNT_POINT}"
log_info "  Physical store: ${STORAGE_DIR}"

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 5: Start system monitoring
# ═══════════════════════════════════════════════════════════════════════════════

log_step "Step 4: Starting system monitors"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEGRA_LOG="${RESULTS_DIR}/tegrastats_${TIMESTAMP}.log"
MPSTAT_LOG="${RESULTS_DIR}/mpstat_${TIMESTAMP}.log"

# Try tegrastats (Jetson-specific), fall back to mpstat
if command -v tegrastats &>/dev/null; then
    log_info "Starting tegrastats (Jetson detected)..."
    sudo tegrastats --interval "${TEGRASTATS_INTERVAL}" --logfile "${TEGRA_LOG}" &
    TEGRA_PID=$!
    log_info "tegrastats logging to: ${TEGRA_LOG} (PID: ${TEGRA_PID})"
else
    log_warn "tegrastats not found (not a Jetson?). Using mpstat instead."
    if command -v mpstat &>/dev/null; then
        mpstat -P ALL 1 > "${MPSTAT_LOG}" &
        MPSTAT_PID=$!
        log_info "mpstat logging to: ${MPSTAT_LOG} (PID: ${MPSTAT_PID})"
    else
        log_warn "mpstat not found. Install sysstat: sudo apt install sysstat"
        log_warn "Continuing without CPU monitoring."
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 6: Run I/O benchmark
# ═══════════════════════════════════════════════════════════════════════════════

log_step "Step 5: Running I/O benchmark (${TEST_SIZE_MB} MB)"

DD_COUNT=$((TEST_SIZE_MB * 1024 / 4))    # Number of 4K blocks
BENCH_FILE="${MOUNT_POINT}/benchmark_test_${TIMESTAMP}.bin"
BENCH_RESULT="${RESULTS_DIR}/benchmark_${TIMESTAMP}.log"

echo "═══════════════════════════════════════════════════════" | tee "${BENCH_RESULT}"
echo "  PQC-FUSE I/O Benchmark Results"                       | tee -a "${BENCH_RESULT}"
echo "  Date     : $(date)"                                    | tee -a "${BENCH_RESULT}"
echo "  Size     : ${TEST_SIZE_MB} MB"                         | tee -a "${BENCH_RESULT}"
echo "  Block sz : ${DD_BS}"                                   | tee -a "${BENCH_RESULT}"
echo "  Platform : $(uname -m)"                                | tee -a "${BENCH_RESULT}"
echo "═══════════════════════════════════════════════════════" | tee -a "${BENCH_RESULT}"

if [[ "${USE_FIO}" == "true" ]] && command -v fio &>/dev/null; then
    log_info "Using fio for benchmark..."
    
    FIO_RESULT="${RESULTS_DIR}/fio_${TIMESTAMP}.json"
    
    fio --name=pqc_write_test \
        --directory="${MOUNT_POINT}" \
        --rw=write \
        --bs=4k \
        --size="${TEST_SIZE_MB}m" \
        --numjobs=1 \
        --ioengine=sync \
        --direct=0 \
        --output-format=json \
        --output="${FIO_RESULT}" \
        2>&1 | tee -a "${BENCH_RESULT}"
    
    log_info "fio results saved to: ${FIO_RESULT}"

else
    log_info "Using dd for benchmark..."
    
    # ── Sequential Write Test ──
    echo "" | tee -a "${BENCH_RESULT}"
    echo "── Sequential Write (dd, bs=${DD_BS}) ──" | tee -a "${BENCH_RESULT}"
    
    T_START=$(date +%s%N)
    
    dd if=/dev/urandom of="${BENCH_FILE}" \
       bs="${DD_BS}" count="${DD_COUNT}" \
       conv=fdatasync status=progress 2>&1 | tee -a "${BENCH_RESULT}"
    
    T_END=$(date +%s%N)
    ELAPSED_MS=$(( (T_END - T_START) / 1000000 ))
    ELAPSED_SEC=$(echo "scale=2; ${ELAPSED_MS} / 1000" | bc)
    THROUGHPUT=$(echo "scale=2; ${TEST_SIZE_MB} / ${ELAPSED_SEC}" | bc)
    
    echo "" | tee -a "${BENCH_RESULT}"
    echo "  Total time  : ${ELAPSED_SEC} seconds" | tee -a "${BENCH_RESULT}"
    echo "  Throughput  : ${THROUGHPUT} MB/s"      | tee -a "${BENCH_RESULT}"
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 7: Collect results
# ═══════════════════════════════════════════════════════════════════════════════

log_step "Step 6: Collecting results"

# Copy FUSE latency log
if [[ -f "${LOG_FILE}" ]]; then
    cp "${LOG_FILE}" "${RESULTS_DIR}/fuse_latency_${TIMESTAMP}.log"
    log_info "FUSE latency log copied to results/"
fi

# Clean up benchmark file
rm -f "${BENCH_FILE}" 2>/dev/null || true

# Stop monitoring
if [[ -n "${TEGRA_PID:-}" ]] && kill -0 "$TEGRA_PID" 2>/dev/null; then
    sudo kill "$TEGRA_PID" 2>/dev/null || true
    log_info "tegrastats stopped."
fi

if [[ -n "${MPSTAT_PID:-}" ]] && kill -0 "$MPSTAT_PID" 2>/dev/null; then
    kill "$MPSTAT_PID" 2>/dev/null || true
    log_info "mpstat stopped."
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 8: Print summary
# ═══════════════════════════════════════════════════════════════════════════════

log_step "Experiment Complete!"

echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    EXPERIMENT SUMMARY                          ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║                                                                ║"
echo "║  Results directory: ${RESULTS_DIR}"
echo "║                                                                ║"
echo "║  Files generated:                                              ║"
echo "║    • benchmark_${TIMESTAMP}.log  (I/O throughput)              ║"

if [[ -f "${TEGRA_LOG}" ]]; then
echo "║    • tegrastats_${TIMESTAMP}.log (CPU/GPU usage)               ║"
fi

if [[ -f "${MPSTAT_LOG}" ]]; then
echo "║    • mpstat_${TIMESTAMP}.log     (CPU usage)                   ║"
fi

echo "║    • fuse_latency_${TIMESTAMP}.log (per-write PQC latency)     ║"
echo "║                                                                ║"
echo "║  Next steps:                                                   ║"
echo "║    1. Open another terminal and run YOLO inference             ║"
echo "║    2. Re-run this script while YOLO is running                 ║"
echo "║    3. Observe YOLO FPS drop → Problem Statement proven!        ║"
echo "║                                                                ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo ""
log_info "FUSE is still mounted at ${MOUNT_POINT}"
log_info "To unmount: fusermount3 -u ${MOUNT_POINT}"
log_info "Or just exit this script (auto-cleanup on Ctrl+C)."
echo ""

# Wait for user to finish or Ctrl+C
log_info "Press Ctrl+C to unmount and exit."
wait "${FUSE_PID}" 2>/dev/null || true
