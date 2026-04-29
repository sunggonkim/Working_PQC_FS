# 🔐 PQC-FUSE: Physical AI를 위한 양자내성암호 파일시스템

<div align="center">

**디바이스 탈취 위협에 대응하는 전체-디스크 PQC 암호화 FUSE 파일시스템 연구 프로토타입**

`ML-KEM-512 (NIST PQC 표준)` · `SHAKE128 XOF 스트림 암호` · `CUDA Pinned Memory` · `FUSE3` · `ARM64`

---

</div>

## 🎯 연구 목적 — Physical AI 보안

자율주행차, 로봇, 드론 등 **Physical AI 디바이스**는 현장에 배치되며 **물리적 탈취 공격**에 노출됩니다. 탈취된 디바이스에서 NVMe SSD를 추출하면 모든 센서 데이터, 모델 가중치, 운행 기록이 그대로 노출됩니다.

### 위협 모델

```
┌─────────────────────────────────────────────────────┐
│           Physical AI 디바이스 탈취 시나리오           │
│                                                     │
│  공격자가 디바이스 탈취 → NVMe SSD 추출 → 데이터 덤프  │
│                                                     │
│  보호해야 할 것:                                       │
│  • 센서 데이터 (카메라/라이다 로그)                      │
│  • AI 모델 가중치                                     │
│  • 위치 기록 / 운행 이력                               │
│  • 개인정보 (탑승자 얼굴, 음성 등)                      │
└─────────────────────────────────────────────────────┘
```

### 왜 양자내성암호(PQC)인가?

기존 RSA/ECC는 양자 컴퓨터 `Shor` 알고리즘으로 해독 가능합니다. 지금 수집한 데이터를 나중에 양자 컴퓨터로 해독하는 **"Harvest Now, Decrypt Later"** 공격에 취약합니다. NIST PQC 표준인 **ML-KEM-512**는 격자 기반으로 양자내성을 제공합니다.

### 핵심 기여

> ❌ **Naive CPU PQC는 실사용 불가** — 쓰기마다 KEM 반복 호출 시 2.1 MB/s (Raw 대비 57×)
>
> ✅ **올바른 Hybrid 설계** — KEM은 파일 생성 시 1회, SHAKE128 XOF 스트림 암호로 벌크 처리 → CPU 147 MB/s (Raw 대비 6.7×)
>
> 🔬 **GPU 가속 연구** — Pinned Memory + CUDA 커널 → 84 MB/s (향후 파이프라인 최적화 가능)

---

## 📊 벤치마크 결과 (NVIDIA Thor, Blackwell GPU)

> **테스트 조건**: 100 MB 순차 쓰기 (4 KB 블록 × 25,600회), `dd if=/dev/zero conv=fdatasync`
>
> **기준 I/O**: `/dev/zero` 사용 (CPU 엔트로피 병목 제거) → NVMe 실속도 측정

### 3-way 비교표 (v2 — 올바른 설계)

| 조건 | 시간 | Throughput | Raw 대비 | 비고 |
|:-----|-----:|----------:|:-------:|:-----|
| 🟢 **Raw NVMe I/O** | 101 ms | 990 MB/s | — 기준 | WD SN5000S, 4K 블록 |
| 🔵 **CPU PQC v2** (ML-KEM + SHAKE128) | 680 ms | 147 MB/s | **6.7× slow** | KEM 1회/파일, XOF 스트림 |
| 🟡 **GPU PQC v2** (CUDA + Pinned) | 1,187 ms | 84 MB/s | **11.8× slow** | 커널 launch 오버헤드 |

### 구 버전 vs 신 버전 (설계 오류 수정 효과)

| 버전 | CPU PQC | GPU PQC | 원인 |
|:-----|--------:|--------:|:-----|
| **v1 (버그 있음)** | 4,973 ms / **2.1 MB/s** | 883 ms / **11.9 MB/s** | CPU: 32바이트마다 KEM 호출; GPU: cudaMallocManaged page fault |
| **v2 (수정됨)** | 680 ms / **147 MB/s** | 1,187 ms / **84 MB/s** | CPU: KEM 1회/파일 + SHAKE128 XOF; GPU: cudaHostAlloc pinned |
| **개선** | **7.3× 빨라짐** | 벤치마크 정상화 | — |

### I/O 시간 시각화 (100 MB 기준)

```
  101 ms |=                                           | Raw NVMe    (990 MB/s)
  680 ms |=======                                     | CPU PQC v2  (147 MB/s)
1,187 ms |============                                | GPU PQC v2  ( 84 MB/s)
         0 ms                                 1,200 ms
```

### GPU가 CPU보다 느린 이유

GPU PQC v2가 CPU v2보다 느린 것은 **FUSE의 4K write 분할** 때문입니다:

```
100 MB / 4 KB = 25,600번의 FUSE write() 호출
         |
각 호출마다 GPU 커널 launch = ~20-50 µs overhead
         |
25,600 × ~45 µs = ~1,150 ms kernel launch 오버헤드만 발생

반면 CPU SHAKE128 XOF는 메모리 연속 처리 → I/O 바운드
```

> 대용량 배치 처리(파일 단위 암호화) 시나리오에서는 GPU가 유리하며, 비동기 I/O 파이프라인으로 개선 가능합니다.

---

## 🏗️ 시스템 아키텍처

### Hybrid 암호화 설계 (v2)

핵심 원칙: **KEM은 비싸지만 1회만. 스트림 암호는 싸고 빠름.**

```
파일 create() 시:
┌─────────────────────────────────────────────────────┐
│  ML-KEM-512.Encaps(pk)  →  shared_secret (32B)      │
│  (1회 실행, ~15 µs)         파일 컨텍스트에 저장        │
└─────────────────────────────────────────────────────┘

파일 write() 시 (핫 패스):
┌─────────────────────────────────────────────────────┐
│  seed = shared_secret || file_id || write_offset    │
│  keystream = SHAKE128_XOF(seed, len)  ← ~1 GB/s    │
│  ciphertext = plaintext XOR keystream               │
└─────────────────────────────────────────────────────┘
```

### CPU 버전 (pqc_fuse.c)

```
App → FUSE write() → pqc_stream_encrypt()
                          |
                    ctx_get(fd)          <- per-fd shared_secret 조회 (mutex)
                          |
                    SHAKE128_XOF(seed)   <- OpenSSL EVP_DigestFinalXOF()
                          |
                    plaintext XOR ks     <- 메모리 XOR
                          |
                    pwrite(storage_fd)   <- NVMe 저장
```

### GPU 버전 (pqc_fuse.cu)

```
App → FUSE write() → gpu_pqc_encrypt()
                          |
                    ctx_get(fd)            <- per-fd shared_secret 조회
                          |
                    FUSE buf → g_pinned_buf <- cudaHostAlloc (DMA 접근 가능)
                          |
                    xor_encrypt_kernel<<<>>>  <- GPU in-place 처리
                    ntt_butterfly_kernel<<<>>> <- GPU 추가 변환
                          |
                    pwrite(g_pinned_buf)      <- 추가 복사 없이 NVMe 저장
```

**핵심**: `cudaHostAlloc` (Pinned Memory)는 DMA-capable이므로 별도 복사 없이 NVMe 직접 쓰기 가능.

---

## 🐛 수정된 설계 버그

### CPU v1 버그: KEM per-chunk 무한 반복

```c
// ❌ 구버전 (v1): 쓰기마다 KEM 반복 호출
for (offset = 0; offset < size; offset += ss_len) {  // size/32 번 반복!
    OQS_KEM_encaps(kem, ct, ss, pk);                  // ~15 µs × 327,680 = 5초!
    xor_chunk(buf+offset, ss, ss_len);
}

// ✅ 신버전 (v2): KEM은 파일 생성 시 1회
// pqc_create(): OQS_KEM_encaps() 단 1회 → ctx 저장
// pqc_write():  SHAKE128 XOF로 keystream 생성 → ~1 GB/s
```

**영향**: 10 MB 쓰기 = 327,680 KEM 호출 × 15 µs = **4,915 ms** 순수 KEM 오버헤드 → v2에서 완전 제거

### GPU v1 버그: cudaMallocManaged page fault

```c
// ❌ 구버전 (v1): Managed memory = page fault on non-Jetson GPU
cudaMallocManaged(&buf, size);  // 비-Jetson에서 page migration 오버헤드

// ✅ 신버전 (v2): Pinned memory = DMA 직접 접근, no page fault
cudaHostAlloc(&g_pinned_buf, PQC_MAX_WRITE, cudaHostAllocDefault);
```

---

## 📁 프로젝트 구조

```
.
├── CMakeLists.txt        # 빌드 시스템 (C + CUDA, OpenSSL, liboqs)
├── pqc_fuse.c            # CPU 버전: ML-KEM-512 + SHAKE128 XOF
├── pqc_fuse.cu           # GPU 버전: CUDA Pinned Memory + per-file KEM
├── run_experiment.sh     # 자동화 실험 스크립트
├── run_benchmark_3way.sh # 3-way 벤치마크 스크립트
└── README.md
```

### 런타임 디렉토리 (자동 생성)

```
~/pqc_edge_workspace/
├── mnt_secure/           # FUSE 마운트 포인트 (앱이 파일을 쓰는 곳)
├── storage_physical/     # 암호화된 데이터 실제 저장 위치
├── results/              # 벤치마크 결과 로그
└── build/
    ├── pqc_fuse          # CPU 버전 바이너리
    └── pqc_fuse_gpu      # GPU 버전 바이너리
```

---

## 🔧 설치 및 빌드

### 사전 요구사항

| 패키지 | 용도 | 설치 |
|--------|------|------|
| libfuse3-dev | FUSE 3 개발 헤더 | `sudo apt install libfuse3-dev fuse3` |
| liboqs | ML-KEM-512 (NIST PQC) | 소스 빌드 (아래 참조) |
| libssl-dev | SHAKE128 XOF (OpenSSL EVP) | `sudo apt install libssl-dev` |
| CUDA Toolkit | GPU 가속 | JetPack 또는 CUDA Toolkit |
| build-essential, cmake | 빌드 도구 | `sudo apt install build-essential cmake` |

### liboqs 설치 (소스 빌드)

```bash
git clone -b main --depth 1 https://github.com/open-quantum-safe/liboqs.git
cd liboqs && mkdir build && cd build
cmake -GNinja -DBUILD_SHARED_LIBS=ON ..
ninja -j$(nproc)
sudo ninja install && sudo ldconfig
```

### 프로젝트 빌드

```bash
mkdir -p ~/pqc_edge_workspace/{mnt_secure,storage_physical,results,build}

cd ~/pqc_edge_workspace/build
cmake /path/to/pqc_encrpyted_fs -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

---

## 🚀 실행 방법

```bash
# CPU 버전
./pqc_fuse ~/pqc_edge_workspace/storage_physical ~/pqc_edge_workspace/mnt_secure -f &

# GPU 버전
./pqc_fuse_gpu ~/pqc_edge_workspace/storage_physical ~/pqc_edge_workspace/mnt_secure -f &

# 벤치마크 (100 MB)
T0=$(date +%s%N)
dd if=/dev/zero of=~/pqc_edge_workspace/mnt_secure/test.bin bs=4K count=25600 conv=fdatasync
T1=$(date +%s%N)
echo "$(( (T1-T0)/1000000 )) ms"

# 마운트 해제
fusermount3 -u ~/pqc_edge_workspace/mnt_secure
```

---

## 🔮 로드맵

- [x] **v1 — 문제 증명**: Naive CPU PQC (KEM per-chunk → 2.1 MB/s, 57× 느림)
- [x] **v2 — 올바른 설계**: Hybrid (KEM-once + SHAKE128 XOF → CPU 147 MB/s, GPU 84 MB/s)
- [ ] **v3 — GPU 파이프라인**: CUDA Streams 비동기 I/O + 멀티 버퍼 → GPU 병목 해소
- [ ] **v4 — 실제 Kyber NTT**: 커스텀 CUDA NTT/INTT 커널 (현재 XOR+NTT dummy)
- [ ] **v5 — 전체 암호화**: 읽기 경로 복호화 + 키 관리 (TPM 통합)

---

## ⚠️ 참고사항

- 이 프로토타입의 스트림 암호는 **연구 목적**입니다. GPU 커널은 XOR + NTT butterfly 구조이며, 인증(AEAD) 없이 기밀성만 제공합니다.
- 실제 배포 시 인증 암호화(AES-256-GCM 또는 ChaCha20-Poly1305)와 키 관리 시스템이 필요합니다.
- 벤치마크는 NVIDIA Thor (Blackwell, sm_110), WD SN5000S NVMe 1TB 환경에서 측정되었습니다.

---

## 📄 라이선스

MIT License

## 👥 연구팀

PQC Edge Research Team — Physical AI Security
