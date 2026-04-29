#!/bin/bash
# run_camera_benchmark.sh - PQC-FUSE v3 카메라 워크로드 3-way 벤치마크
#
# 사용법:
#   bash run_camera_benchmark.sh [--fps N] [--duration N] [--real-camera]
#
# 결과:
#   CPU PQC v3 vs GPU PQC v3 vs NVMe 기준선 (카메라 JPEG 워크로드)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$HOME/pqc_edge_workspace"
BUILD="$WORKSPACE/build"
STORAGE="$WORKSPACE/storage_physical"
MOUNT="$WORKSPACE/mnt_secure"
CAMERA_TEST="$SCRIPT_DIR/camera_capture_test.py"

CPU_BIN="$BUILD/pqc_fuse"
GPU_BIN="$BUILD/pqc_fuse_gpu"

# --- 기본값 ---
FPS=30
DURATION=10
EXTRA_ARGS=""

# --- 인자 파싱 ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fps)      FPS="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --real-camera) EXTRA_ARGS="$EXTRA_ARGS --real-camera"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- 유틸 함수 ---
cleanup_mount() {
    fusermount3 -u "$MOUNT" 2>/dev/null || true
    sleep 1
    rm -rf "$STORAGE"/* 2>/dev/null || true
}

run_camera_test() {
    local label="$1"
    python3 "$CAMERA_TEST" \
        --mount "$MOUNT" \
        --fps "$FPS" \
        --duration "$DURATION" \
        --label "$label" \
        $EXTRA_ARGS
}

run_raw_nvme_camera_test() {
    # NVMe 기준선: FUSE 없이 직접 쓰기
    local out_dir="$STORAGE/camera_frames_raw"
    mkdir -p "$out_dir"
    python3 "$CAMERA_TEST" \
        --mount "$STORAGE" \
        --fps "$FPS" \
        --duration "$DURATION" \
        --label "NVMe-Raw" \
        $EXTRA_ARGS
}

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║      PQC-FUSE v3 카메라 워크로드 3-way 벤치마크                  ║"
echo "║      Physical AI 보안 - 실시간 카메라 암호화 성능                 ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  목표 FPS    : $FPS                                               ║"
echo "║  테스트 시간  : ${DURATION}초                                      ║"
echo "║  프레임 크기  : 1280x720 JPEG (~50-80 KB/프레임)                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# ──────────────────────────────────────────────────────────────────────────
# [1/3] NVMe Raw I/O 기준선
# ──────────────────────────────────────────────────────────────────────────
echo "━━━ [1/3] NVMe Raw I/O 기준선 (암호화 없음) ━━━"
cleanup_mount
run_raw_nvme_camera_test
echo ""

# ──────────────────────────────────────────────────────────────────────────
# [2/3] CPU PQC v3
# ──────────────────────────────────────────────────────────────────────────
echo "━━━ [2/3] CPU PQC v3 (ML-KEM-512 + SHAKE128 + 512KB 코얼레싱) ━━━"
cleanup_mount
"$CPU_BIN" "$STORAGE" "$MOUNT" -f 2>/dev/null &
CPU_PID=$!
sleep 2
if mountpoint -q "$MOUNT"; then
    echo "[INFO] CPU PQC v3 마운트 완료"
    run_camera_test "CPU-PQC-v3"
else
    echo "[ERROR] CPU PQC 마운트 실패"
fi
fusermount3 -u "$MOUNT" 2>/dev/null || true
wait $CPU_PID 2>/dev/null || true
echo ""

# ──────────────────────────────────────────────────────────────────────────
# [3/3] GPU PQC v3
# ──────────────────────────────────────────────────────────────────────────
echo "━━━ [3/3] GPU PQC v3 (ML-KEM-512 + CUDA XOR + 512KB 코얼레싱) ━━━"
cleanup_mount
"$GPU_BIN" "$STORAGE" "$MOUNT" -f 2>/dev/null &
GPU_PID=$!
sleep 3
if mountpoint -q "$MOUNT"; then
    echo "[INFO] GPU PQC v3 마운트 완료"
    run_camera_test "GPU-PQC-v3"
else
    echo "[ERROR] GPU PQC 마운트 실패"
fi
fusermount3 -u "$MOUNT" 2>/dev/null || true
wait $GPU_PID 2>/dev/null || true
echo ""

# ──────────────────────────────────────────────────────────────────────────
# 요약 비교표
# ──────────────────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                   카메라 워크로드 성능 비교                       ║"
echo "║           (30fps, 1280x720 JPEG, ${DURATION}초 지속)              ║"
echo "╠══════════════════════════════════════════════════════════════════╣"

# JSON 결과 파싱 (python3으로)
python3 - << 'PYEOF'
import json, os, sys

files = {
    'NVMe Raw': '/tmp/camera_bench_NVMe-Raw.json',
    'CPU PQC v3': '/tmp/camera_bench_CPU-PQC-v3.json',
    'GPU PQC v3': '/tmp/camera_bench_GPU-PQC-v3.json',
}

results = {}
for label, path in files.items():
    if os.path.exists(path):
        with open(path) as f:
            results[label] = json.load(f)

if not results:
    print("║  결과 파일을 찾을 수 없음                                      ║")
else:
    print(f"║  {'항목':<14} {'FPS':>6} {'처리량':>10} {'P50':>8} {'P95':>8} {'드롭':>6} ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    for label, r in results.items():
        fps_ok = '✅' if r['actual_fps'] >= r['target_fps'] * 0.95 else '⚠️ '
        print(f"║  {label:<14} {r['actual_fps']:>5.1f}{fps_ok} "
              f"{r['throughput_mbs']:>7.1f} MB/s "
              f"{r['latency_p50_ms']:>7.1f}ms "
              f"{r['latency_p95_ms']:>7.1f}ms "
              f"{r['dropped_frames']:>5} ║")

PYEOF

echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "상세 결과: /tmp/camera_bench_*.json"
echo "완료!"
