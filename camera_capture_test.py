#!/usr/bin/env python3
"""
camera_capture_test.py - PQC-FUSE 카메라 워크로드 시뮬레이션 테스트

실제 Physical AI 카메라 파이프라인을 시뮬레이션:
  - 실제 V4L2 카메라 (--real-camera 플래그)
  - 또는 현실적인 센서 노이즈 패턴으로 합성 JPEG 프레임 생성

Usage:
  python3 camera_capture_test.py --mount /path/to/mount [options]

Options:
  --mount PATH       FUSE 마운트 경로 (기본: ~/pqc_edge_workspace/mnt_secure)
  --fps N            목표 FPS (기본: 30)
  --duration N       테스트 지속 시간 초 (기본: 10)
  --width N          프레임 폭 (기본: 1280)
  --height N         프레임 높이 (기본: 720)
  --real-camera      실제 V4L2 카메라 사용 시도
  --device PATH      V4L2 장치 경로 (기본: /dev/video0)
  --output-dir PATH  프레임 저장 디렉토리 (기본: mount/camera_frames/)
  --label NAME       결과 레이블 (기본: PQC)
"""

import argparse
import os
import sys
import time
import struct
import random
import statistics
import math

# 선택적 imports
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ---------------------------------------------------------------------------
# 합성 카메라 프레임 생성
# ---------------------------------------------------------------------------

def make_synthetic_jpeg_frame(width: int, height: int, frame_idx: int) -> bytes:
    """현실적인 센서 노이즈 패턴을 가진 합성 JPEG 프레임 생성."""
    if HAS_NUMPY and HAS_CV2:
        # OpenCV + NumPy: 실제 JPEG 인코딩
        # 그라디언트 배경 + 가우시안 노이즈 (카메라 센서 특성 시뮬레이션)
        y_coords = np.linspace(0, 255, height, dtype=np.uint8)
        x_coords = np.linspace(0, 255, width, dtype=np.uint8)
        gradient = np.outer(y_coords, np.ones(width, dtype=np.uint8))

        # 프레임별 노이즈 시드로 시간적 변화 시뮬레이션
        rng = np.random.default_rng(frame_idx)
        noise = rng.integers(0, 30, (height, width), dtype=np.uint8)

        # BGR 채널 (카메라 raw 패턴 시뮬레이션)
        b = np.clip(gradient.astype(np.int32) + noise - 15, 0, 255).astype(np.uint8)
        g = np.clip((gradient // 2).astype(np.int32) + noise, 0, 255).astype(np.uint8)
        r = np.clip((255 - gradient).astype(np.int32) + noise - 10, 0, 255).astype(np.uint8)
        frame = np.stack([b, g, r], axis=2)

        # JPEG 인코딩 (품질 85 = 전형적인 카메라 JPEG)
        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret:
            return buf.tobytes()

    if HAS_NUMPY:
        # NumPy만 있는 경우: 합성 JPEG 헤더 + 노이즈 데이터
        rng = np.random.default_rng(frame_idx)
        # 720p JPEG: ~50-80KB 예상
        target_size = int(55000 + rng.integers(0, 25000))
        payload = rng.bytes(target_size - 20)
    else:
        # 순수 Python 폴백: pseudo-random 데이터
        random.seed(frame_idx)
        target_size = random.randint(50000, 80000)
        # lcg로 빠른 pseudo-random 생성
        data = []
        state = frame_idx ^ 0xDEADBEEF
        for _ in range(target_size - 20):
            state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            data.append((state >> 33) & 0xFF)
        payload = bytes(data)

    # 최소한의 JPEG 헤더 + EOI 마커 삽입
    jpeg_soi  = b'\xFF\xD8'
    jpeg_app0 = b'\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
    jpeg_eoi  = b'\xFF\xD9'
    # 프레임 인덱스 메타데이터 삽입 (실제 카메라 EXIF 시뮬레이션)
    meta = struct.pack('>II', frame_idx, width * height)
    return jpeg_soi + jpeg_app0 + meta + payload[:target_size - 24] + jpeg_eoi


def open_real_camera(device: str, width: int, height: int):
    """V4L2 카메라 열기. 실패 시 None 반환."""
    if not HAS_CV2:
        return None
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    return cap


def capture_real_frame(cap) -> bytes:
    """V4L2 카메라에서 JPEG 프레임 캡처."""
    ret, frame = cap.read()
    if not ret:
        return None
    ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ret else None


# ---------------------------------------------------------------------------
# 벤치마크 실행
# ---------------------------------------------------------------------------

def run_benchmark(mount: str, fps: int, duration: int, width: int, height: int,
                  use_real_camera: bool, device: str, label: str) -> dict:
    """카메라 캡처 워크로드 벤치마크 실행."""

    output_dir = os.path.join(mount, 'camera_frames')
    os.makedirs(output_dir, exist_ok=True)

    cap = None
    if use_real_camera:
        cap = open_real_camera(device, width, height)
        if cap:
            print(f"[INFO] 실제 카메라 사용: {device}")
        else:
            print(f"[WARN] {device} 열기 실패, 합성 프레임으로 전환")

    frame_interval = 1.0 / fps
    frame_latencies = []  # ms
    frame_sizes = []       # bytes
    dropped_frames = 0
    total_written = 0
    frame_idx = 0

    print(f"[{label}] 시작: {fps}fps 목표, {duration}초, {width}x{height}")
    print(f"         Mount: {mount}")
    print(f"         프레임 모드: {'실제 V4L2' if cap else '합성 JPEG'}")
    if not HAS_NUMPY:
        print(f"[WARN] NumPy 없음 - 순수 Python 폴백 사용 (느림)")
    print()

    bench_start = time.perf_counter()
    next_frame_time = bench_start

    while True:
        now = time.perf_counter()
        elapsed = now - bench_start
        if elapsed >= duration:
            break

        # 다음 프레임 시각까지 대기 (busy-wait 대신 sleep)
        wait = next_frame_time - now
        if wait > 0.001:
            time.sleep(wait - 0.0005)

        # 프레임 시각 보정
        capture_time = time.perf_counter()
        if capture_time > next_frame_time + frame_interval:
            # 프레임 드롭
            skipped = int((capture_time - next_frame_time) / frame_interval)
            dropped_frames += skipped
            next_frame_time += skipped * frame_interval

        # 프레임 생성/캡처
        if cap:
            frame_data = capture_real_frame(cap)
            if frame_data is None:
                frame_data = make_synthetic_jpeg_frame(width, height, frame_idx)
        else:
            frame_data = make_synthetic_jpeg_frame(width, height, frame_idx)

        # FUSE 마운트에 쓰기
        frame_path = os.path.join(output_dir, f'frame_{frame_idx:06d}.jpg')
        write_start = time.perf_counter()
        try:
            with open(frame_path, 'wb') as f:
                f.write(frame_data)
                f.flush()
                os.fsync(f.fileno())
        except OSError as e:
            print(f"[ERROR] 쓰기 실패: {e}")
            dropped_frames += 1
            next_frame_time += frame_interval
            continue

        write_end = time.perf_counter()
        write_ms = (write_end - write_start) * 1000.0

        frame_latencies.append(write_ms)
        frame_sizes.append(len(frame_data))
        total_written += len(frame_data)
        frame_idx += 1

        next_frame_time += frame_interval

    total_elapsed = time.perf_counter() - bench_start

    if cap:
        cap.release()

    # 결과 집계
    if not frame_latencies:
        print("[ERROR] 기록된 프레임 없음")
        return {}

    latencies_sorted = sorted(frame_latencies)
    p50 = latencies_sorted[len(latencies_sorted) // 2]
    p95_idx = min(int(len(latencies_sorted) * 0.95), len(latencies_sorted) - 1)
    p95 = latencies_sorted[p95_idx]
    p99_idx = min(int(len(latencies_sorted) * 0.99), len(latencies_sorted) - 1)
    p99 = latencies_sorted[p99_idx]
    mean_lat = statistics.mean(frame_latencies)

    actual_fps = frame_idx / total_elapsed
    throughput_mbs = (total_written / 1024 / 1024) / total_elapsed
    avg_frame_kb = statistics.mean(frame_sizes) / 1024

    results = {
        'label': label,
        'frames_written': frame_idx,
        'dropped_frames': dropped_frames,
        'total_elapsed_s': total_elapsed,
        'actual_fps': actual_fps,
        'target_fps': fps,
        'throughput_mbs': throughput_mbs,
        'total_written_mb': total_written / 1024 / 1024,
        'avg_frame_kb': avg_frame_kb,
        'latency_mean_ms': mean_lat,
        'latency_p50_ms': p50,
        'latency_p95_ms': p95,
        'latency_p99_ms': p99,
        'latency_max_ms': max(frame_latencies),
    }

    print(f"┌─────────────────────────────────────────────────────────────┐")
    print(f"│  [{label}] 카메라 워크로드 결과                               ")
    print(f"├─────────────────────────────────────────────────────────────┤")
    print(f"│  프레임: {frame_idx} 기록 / {dropped_frames} 드롭              ")
    print(f"│  실제 FPS: {actual_fps:.1f} (목표: {fps})                      ")
    print(f"│  처리량: {throughput_mbs:.1f} MB/s                             ")
    print(f"│  평균 프레임 크기: {avg_frame_kb:.1f} KB                        ")
    print(f"│  쓰기 레이턴시 (fsync 포함):                                  ")
    print(f"│    평균: {mean_lat:.1f}ms                                       ")
    print(f"│    P50:  {p50:.1f}ms                                            ")
    print(f"│    P95:  {p95:.1f}ms                                            ")
    print(f"│    P99:  {p99:.1f}ms                                            ")
    print(f"│    Max:  {max(frame_latencies):.1f}ms                           ")
    print(f"└─────────────────────────────────────────────────────────────┘")

    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='PQC-FUSE 카메라 워크로드 벤치마크')
    parser.add_argument('--mount',
                        default=os.path.expanduser('~/pqc_edge_workspace/mnt_secure'),
                        help='FUSE 마운트 경로')
    parser.add_argument('--fps', type=int, default=30, help='목표 FPS')
    parser.add_argument('--duration', type=int, default=10, help='테스트 시간(초)')
    parser.add_argument('--width', type=int, default=1280, help='프레임 폭')
    parser.add_argument('--height', type=int, default=720, help='프레임 높이')
    parser.add_argument('--real-camera', action='store_true',
                        help='실제 V4L2 카메라 사용')
    parser.add_argument('--device', default='/dev/video0',
                        help='V4L2 장치 경로')
    parser.add_argument('--label', default='PQC',
                        help='결과 레이블')
    args = parser.parse_args()

    if not os.path.ismount(args.mount) and not os.path.isdir(args.mount):
        print(f"[ERROR] 마운트 경로를 찾을 수 없음: {args.mount}")
        sys.exit(1)

    results = run_benchmark(
        mount=args.mount,
        fps=args.fps,
        duration=args.duration,
        width=args.width,
        height=args.height,
        use_real_camera=args.real_camera,
        device=args.device,
        label=args.label,
    )

    # 결과를 JSON으로 저장
    if results:
        import json
        out_json = f'/tmp/camera_bench_{args.label.replace(" ", "_")}.json'
        with open(out_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n결과 저장: {out_json}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
