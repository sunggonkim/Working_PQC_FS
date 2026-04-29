/**
 * ============================================================================
 *  pqc_fuse.cu — GPU-Accelerated PQC FUSE Filesystem (Phase 2)
 * ============================================================================
 *
 *  Upgrade from pqc_fuse.c (CPU-only) to CUDA C++ with:
 *    - Unified Memory (cudaMallocManaged) buffer pool → Zero-copy on Jetson
 *    - Dummy PQC CUDA kernel for GPU offloading pipeline validation
 *    - Separate CPU vs GPU latency logging
 *
 *  Architecture (Zero-copy on Jetson):
 *    ┌──────────────────────────────────────────────────────────────────┐
 *    │  Jetson SoC (Shared Physical DRAM)                              │
 *    │  ┌─────────────┐    cudaMallocManaged    ┌─────────────┐       │
 *    │  │  ARM CPU     │ ◄═══════════════════► │  GPU Cores   │       │
 *    │  │  (FUSE ops)  │   No PCIe copy needed  │  (PQC kernel)│       │
 *    │  └─────────────┘                         └─────────────┘       │
 *    └──────────────────────────────────────────────────────────────────┘
 *
 *  On Jetson, CPU and GPU share the same physical memory (DRAM).
 *  cudaMallocManaged() allocates memory accessible by both without any
 *  DMA transfer — this is TRUE zero-copy, unlike discrete GPUs where
 *  Unified Memory still triggers page migration over PCIe.
 *
 *  Build:
 *    mkdir build && cd build && cmake .. && make
 *
 *  Usage:
 *    ./pqc_fuse_gpu <storage_dir> <mountpoint> -f
 * ============================================================================
 */

#define FUSE_USE_VERSION 31

#include <fuse.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <dirent.h>
#include <time.h>
#include <stdarg.h>

#include <cuda_runtime.h>
#include <oqs/oqs.h>   /* CPU-side KEM (encaps/decaps) */
#include <openssl/evp.h> /* SHAKE128 XOF keystream generation */
#include <pthread.h>
#include <sys/xattr.h>

/* ════════════════════════════════════════════════════════════════════════════
 *  Configuration & Globals
 * ════════════════════════════════════════════════════════════════════════════ */

static char g_storage_dir[4096] = {0};
static FILE *g_logfile = NULL;
static const char *LOG_FILENAME = "pqc_fuse_gpu_latency.log";

#define GPU_BUFFER_SIZE (8 * 1024 * 1024)   /* 8 MB pool per stream */
#define COALESCE_SIZE   (512 * 1024)          /* 512 KB per-fd write-coalescing buffer */

#define NUM_STREAMS 16

typedef struct {
    cudaStream_t stream;
    uint8_t *pinned_buf;
    uint8_t *pinned_keystream;
    pthread_mutex_t lock;
} gpu_stream_ctx_t;

static gpu_stream_ctx_t g_streams[NUM_STREAMS];

/** XOR key for stream cipher kernel (pinned memory) */
#define PQC_KEY_SIZE 256
static uint8_t *g_pqc_key = NULL;  /* pinned, accessible by both CPU and GPU */

/* Per-fd encryption context (same design as CPU version).
 * KEM runs once per file; the 32-byte shared secret seeds the GPU stream cipher.
 */
#define PQC_MAX_FD 4096

#define PQC_TIER_FULL  1
#define PQC_TIER_NONE  2
#define PQC_XATTR_TIER "user.pqc_tier"
#define KEY_ROTATION_INTERVAL_S  1
#define MAX_EPOCHS 32

typedef struct {
    uint64_t offset_start;
    size_t   ss_len;
    uint8_t  ss[64];
} pqc_epoch_info_t;

typedef struct {
    int      valid;
    uint64_t file_id;
    /* write coalescing */
    uint8_t *wbuf;           /* heap-allocated COALESCE_SIZE bytes */
    size_t   wbuf_used;
    off_t    wbuf_base_off;
    /* Strategy 2: Selective Encryption tier */
    int      tier;           /* PQC_TIER_FULL or PQC_TIER_NONE */
    /* Strategy 1: Forward-Secure KEM rotation */
    pqc_epoch_info_t epochs[MAX_EPOCHS];
    int      num_epochs;
    uint64_t key_epoch;      /* current epoch index */
    time_t   last_rekey;     /* wall-clock time of last KEM call */
} pqc_fd_ctx_t;

static pqc_fd_ctx_t    g_fd_ctx[PQC_MAX_FD];
static pthread_mutex_t g_fd_locks[PQC_MAX_FD];
static pthread_mutex_t g_global_lock = PTHREAD_MUTEX_INITIALIZER;
static uint64_t        g_file_id_ctr = 0;

static int acquire_stream() {
    while(1) {
        for (int i=0; i<NUM_STREAMS; i++) {
            if (pthread_mutex_trylock(&g_streams[i].lock) == 0) return i;
        }
        usleep(10);
    }
}

static void release_stream(int i) {
    pthread_mutex_unlock(&g_streams[i].lock);
}

/* KEM state (CPU-side, same keypair for all files) */
static OQS_KEM *g_kem        = NULL;
static uint8_t *g_public_key = NULL;
static uint8_t *g_secret_key = NULL;

/* ════════════════════════════════════════════════════════════════════════════
 *  Utility Functions
 * ════════════════════════════════════════════════════════════════════════════ */

static inline double get_time_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

static void pqc_log(const char *fmt, ...)
{
    va_list args;
    char timebuf[64];
    struct timeval tv;
    struct tm tm_info;

    gettimeofday(&tv, NULL);
    localtime_r(&tv.tv_sec, &tm_info);
    strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S", &tm_info);

    fprintf(stderr, "[PQC-GPU %s.%03ld] ", timebuf, tv.tv_usec / 1000);
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");

    if (g_logfile) {
        fprintf(g_logfile, "[%s.%03ld] ", timebuf, tv.tv_usec / 1000);
        va_start(args, fmt);
        vfprintf(g_logfile, fmt, args);
        va_end(args);
        fprintf(g_logfile, "\n");
        fflush(g_logfile);
    }
}

static void resolve_physical_path(char *dest, size_t dest_size, const char *path)
{
    snprintf(dest, dest_size, "%s%s", g_storage_dir, path);
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        pqc_log("CUDA ERROR at %s:%d — %s", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
    } \
} while(0)

/* ════════════════════════════════════════════════════════════════════════════
 *  CUDA Kernels — Dummy PQC Operations
 * ════════════════════════════════════════════════════════════════════════════
 *
 *  These kernels simulate PQC workload on the GPU.  In a real implementation,
 *  they would perform NTT (Number Theoretic Transform), polynomial
 *  multiplication, and other lattice operations used in Kyber/ML-KEM.
 *
 *  Current implementation: Multi-round XOR + byte permutation to exercise
 *  GPU cores and demonstrate the zero-copy pipeline.
 * ════════════════════════════════════════════════════════════════════════════ */

/**
 * Kernel 1: XOR encryption with key stream (simulates symmetric enc after KEM)
 * Each thread processes one byte, applying multi-round XOR with the PQC key.
 */
__global__ void pqc_xor_encrypt_kernel(uint8_t *data, const uint8_t *key,
                                        size_t data_size, int key_size,
                                        int rounds)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= data_size) return;

    uint8_t val = data[idx];

    /* Multi-round transformation to increase GPU workload */
    for (int r = 0; r < rounds; r++) {
        /* XOR with rotating key */
        val ^= key[(idx + r) % key_size];

        /* Simulate lattice polynomial operation (byte-level substitution) */
        val = (val * 0x9D + 0x5F) & 0xFF;  /* Galois field-like multiply */

        /* Bit rotation */
        val = (val << 3) | (val >> 5);
    }

    data[idx] = val;
}

#define KYBER_Q 3329
#define KYBER_N 256

__device__ int16_t montgomery_reduce(int32_t a) {
    int32_t t;
    int16_t u = (int16_t)(a * 62209); // q^-1 mod 2^16
    t = (int32_t)u * KYBER_Q;
    t = a - t;
    t >>= 16;
    return (int16_t)t;
}

/**
 * Kernel 2: Kyber Radix-2 NTT Butterfly (Phase 3)
 * Simulates heavy polynomial multiplication over Z_q[X]/(X^256 + 1)
 */
__global__ void pqc_kyber_ntt_kernel(int16_t *data, size_t data_size)
{
    // Each block processes one Kyber polynomial (256 int16_t = 512 bytes)
    size_t poly_idx = blockIdx.x;
    if (poly_idx * KYBER_N * sizeof(int16_t) >= data_size) return;

    int16_t *poly = data + poly_idx * KYBER_N;
    int tid = threadIdx.x; // 128 threads per block

    // 7 layers of Cooley-Tukey butterfly
    int len = 128;
    for (int step = 0; step < 7; step++) {
        int start = (tid / len) * (2 * len) + (tid % len);
        int16_t zeta = 1729 + step; // Dummy twiddle factor for profiling

        int16_t a = poly[start];
        int16_t b = poly[start + len];

        int16_t t = montgomery_reduce((int32_t)b * zeta);
        poly[start + len] = a - t;
        poly[start]       = a + t;

        __syncthreads();
        len >>= 1;
    }
}

/**
 * Kernel 3 (NEW): Pure SHAKE128-derived stream XOR
 * data[i] ^= keystream[i] in parallel — reversible (decrypt == encrypt)
 */
__global__ void pqc_stream_xor_kernel(uint8_t *data, const uint8_t *keystream, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        data[idx] ^= keystream[idx];
}

/* ════════════════════════════════════════════════════════════════════════════
 *  GPU Buffer Pool Management
 * ════════════════════════════════════════════════════════════════════════════ */

static int gpu_init(void)
{
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        pqc_log("FATAL: No CUDA devices found!");
        return -1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    pqc_log("GPU Device    : %s", prop.name);
    pqc_log("  SM count    : %d", prop.multiProcessorCount);
    pqc_log("  Memory      : %zu MB", prop.totalGlobalMem / (1024 * 1024));
    pqc_log("  Unified Mem : %s", prop.managedMemory ? "YES (Zero-copy!)" : "NO");
    pqc_log("  Integrated  : %s", prop.integrated ? "YES (SoC shared DRAM)" : "NO (discrete)");

    /* Create Stream Pool and allocate pinned buffers for Multi-buffering */
    double t0 = get_time_us();
    CUDA_CHECK(cudaHostAlloc(&g_pqc_key, PQC_KEY_SIZE, cudaHostAllocDefault));
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&g_streams[i].stream));
        CUDA_CHECK(cudaHostAlloc(&g_streams[i].pinned_buf,       GPU_BUFFER_SIZE, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&g_streams[i].pinned_keystream, GPU_BUFFER_SIZE, cudaHostAllocDefault));
        pthread_mutex_init(&g_streams[i].lock, NULL);
    }
    double t1 = get_time_us();

    pqc_log("Stream Pool created: %d streams × (8MB buf + 8MB key) (%.2f µs)",
            NUM_STREAMS, t1 - t0);

    /* Generate fallback key */
    srand((unsigned)time(NULL));
    for (int i = 0; i < PQC_KEY_SIZE; i++)
        g_pqc_key[i] = (uint8_t)(rand() & 0xFF);

    /* Warm up GPU */
    pqc_xor_encrypt_kernel<<<1, 32, 0, g_streams[0].stream>>>(g_streams[0].pinned_buf, g_pqc_key, 32, PQC_KEY_SIZE, 1);
    CUDA_CHECK(cudaStreamSynchronize(g_streams[0].stream));

    /* Init per-FD locks */
    for (int i = 0; i < PQC_MAX_FD; i++) {
        pthread_mutex_init(&g_fd_locks[i], NULL);
    }

    /* ── Init CPU-side KEM (generates keypair for all files) ── */
    g_kem = OQS_KEM_new(OQS_KEM_alg_ml_kem_512);
    if (!g_kem) g_kem = OQS_KEM_new(OQS_KEM_alg_kyber_512);
    if (g_kem) {
        g_public_key = (uint8_t *)malloc(g_kem->length_public_key);
        g_secret_key = (uint8_t *)malloc(g_kem->length_secret_key);
        double tk0 = get_time_us();
        OQS_KEM_keypair(g_kem, g_public_key, g_secret_key);
        pqc_log("KEM keypair (%s) in %.1fµs", g_kem->method_name, get_time_us() - tk0);
    } else {
        pqc_log("WARNING: KEM init failed — writes will use fallback XOR key");
    }

    pqc_log("GPU init complete — pinned-memory pipeline ready");
    return 0;
}

static void gpu_cleanup(void)
{
    for (int i = 0; i < NUM_STREAMS; i++) {
        if (g_streams[i].pinned_buf)       cudaFreeHost(g_streams[i].pinned_buf);
        if (g_streams[i].pinned_keystream) cudaFreeHost(g_streams[i].pinned_keystream);
        cudaStreamDestroy(g_streams[i].stream);
        pthread_mutex_destroy(&g_streams[i].lock);
    }
    if (g_pqc_key) cudaFreeHost(g_pqc_key);
    if (g_kem) {
        OQS_MEM_cleanse(g_secret_key, g_kem->length_secret_key);
        free(g_public_key); free(g_secret_key);
        OQS_KEM_free(g_kem);
        g_kem = NULL;
    }
    pqc_log("GPU resources released");
}

/* ────────────────────────────────────────────────────────────────────────────
Per-fd context helpers (identical logic to CPU version)
──────────────────────────────────────────────────────────────────────────── */

static void ctx_set(int fd, const uint8_t *ss, size_t ss_len, uint64_t fid)
{
    int idx = fd % PQC_MAX_FD;
    pthread_mutex_lock(&g_fd_locks[idx]);
    g_fd_ctx[idx].file_id    = fid;
    g_fd_ctx[idx].valid      = 1;
    g_fd_ctx[idx].tier       = PQC_TIER_FULL;
    g_fd_ctx[idx].key_epoch  = 0;
    g_fd_ctx[idx].last_rekey = time(NULL);
    g_fd_ctx[idx].num_epochs = 1;
    g_fd_ctx[idx].epochs[0].offset_start = 0;
    g_fd_ctx[idx].epochs[0].ss_len = ss_len;
    memcpy(g_fd_ctx[idx].epochs[0].ss, ss, ss_len);
    if (!g_fd_ctx[idx].wbuf)
        g_fd_ctx[idx].wbuf = (uint8_t *)malloc(COALESCE_SIZE);
    g_fd_ctx[idx].wbuf_used     = 0;
    g_fd_ctx[idx].wbuf_base_off = 0;
    pthread_mutex_unlock(&g_fd_locks[idx]);
}

static int ctx_get(int fd, pqc_fd_ctx_t *out)
{
    int idx = fd % PQC_MAX_FD;
    pthread_mutex_lock(&g_fd_locks[idx]);
    if (!g_fd_ctx[idx].valid) { pthread_mutex_unlock(&g_fd_locks[idx]); return -1; }
    *out = g_fd_ctx[idx];
    pthread_mutex_unlock(&g_fd_locks[idx]);
    return 0;
}

static void ctx_clear(int fd)
{
    int idx = fd % PQC_MAX_FD;
    pthread_mutex_lock(&g_fd_locks[idx]);
    OQS_MEM_cleanse(g_fd_ctx[idx].epochs, sizeof(g_fd_ctx[idx].epochs));
    if (g_fd_ctx[idx].wbuf) {
        OQS_MEM_cleanse(g_fd_ctx[idx].wbuf, COALESCE_SIZE);
        free(g_fd_ctx[idx].wbuf);
        g_fd_ctx[idx].wbuf = NULL;
    }
    g_fd_ctx[idx].wbuf_used = 0;
    g_fd_ctx[idx].valid = 0;
    pthread_mutex_unlock(&g_fd_locks[idx]);
}

/* gpu_pqc_encrypt removed (replaced by stream pool pipeline) */

/* ────────────────────────────────────────────────────────────────────────────
 *  GPU Write-coalescing flush
 *  CPU generates SHAKE128 keystream → GPU XOR (parallel) → pwrite.
 *  MUST be called with g_fd_lock held.  Resets wbuf_used = 0 on success.
 * -------------------------------------------------------------------------- */

/* Helper: generate SHAKE128 keystream into g_pinned_keystream (locked context) */
/*
 * do_rekey_locked() — Strategy 1: Forward-Secure Key Rotation
 * Runs ML-KEM-512.Encaps() to produce a fresh shared_secret and appends
 * a new epoch entry to the .pqckey sidecar.  MUST be called with g_fd_lock.
 */
static void do_rekey_locked(int storage_fd, int idx, const char *phys_path)
{
    (void)storage_fd;
    if (!g_kem || !phys_path) return;

    uint8_t *ct = (uint8_t *)malloc(g_kem->length_ciphertext);
    uint8_t *ss = (uint8_t *)malloc(g_kem->length_shared_secret);
    if (!ct || !ss) { free(ct); free(ss); return; }

    if (OQS_KEM_encaps(g_kem, ct, ss, g_public_key) != OQS_SUCCESS) {
        free(ct); free(ss); return;
    }

    pqc_fd_ctx_t *ctx = &g_fd_ctx[idx];
    uint64_t off_start = (uint64_t)(ctx->wbuf_base_off + ctx->wbuf_used);
    
    if (ctx->num_epochs < MAX_EPOCHS) {
        int e = ctx->num_epochs++;
        ctx->epochs[e].offset_start = off_start;
        ctx->epochs[e].ss_len = g_kem->length_shared_secret;
        memcpy(ctx->epochs[e].ss, ss, g_kem->length_shared_secret);
    }
    
    ctx->key_epoch++;
    ctx->last_rekey = time(NULL);

    /* Append epoch record to sidecar: epoch(8B)|offset(8B)|ss_len(8B)|ss */
    char key_path[4096 + 8];
    snprintf(key_path, sizeof(key_path), "%s.pqckey", phys_path);
    int kfd = open(key_path, O_WRONLY | O_CREAT | O_APPEND, 0600);
    if (kfd >= 0) {
        uint64_t ep       = ctx->key_epoch;
        uint64_t ss_len64 = (uint64_t)g_kem->length_shared_secret;
        (void)(write(kfd, &ep,       8) +
               write(kfd, &off_start, 8) +
               write(kfd, &ss_len64, 8) +
               write(kfd, ss, (size_t)ss_len64));
        close(kfd);
    }

    pqc_log("REKEY epoch=%llu fid=%llu offset=%llu",
            (unsigned long long)ctx->key_epoch,
            (unsigned long long)ctx->file_id,
            (unsigned long long)off_start);
    OQS_MEM_cleanse(ss, g_kem->length_shared_secret);
    free(ct); free(ss);
}

static void gpu_gen_keystream(const pqc_fd_ctx_t *ctx, off_t base, size_t sz, uint8_t *keystream_out)
{
    uint64_t off = (uint64_t)base;
    int e_idx = 0;
    for (int i = 0; i < ctx->num_epochs; i++) {
        if (ctx->epochs[i].offset_start <= off) {
            e_idx = i;
        } else {
            break;
        }
    }
    const pqc_epoch_info_t *ep = &ctx->epochs[e_idx];

    uint8_t seed[80];
    size_t  seed_len = ep->ss_len + 16;
    memcpy(seed, ep->ss, ep->ss_len);
    uint64_t fid = ctx->file_id;
    memcpy(seed + ep->ss_len,     &fid, 8);
    memcpy(seed + ep->ss_len + 8, &off, 8);
    EVP_MD_CTX *md = EVP_MD_CTX_new();
    if (!md) return;
    EVP_DigestInit_ex(md, EVP_shake128(), NULL);
    EVP_DigestUpdate(md, seed, seed_len);
    EVP_DigestFinalXOF(md, keystream_out, sz);
    EVP_MD_CTX_free(md);
}

static int do_flush_wbuf_locked(int storage_fd, int idx)
{
    pqc_fd_ctx_t *ctx = &g_fd_ctx[idx];
    if (ctx->wbuf_used == 0) return 0;

    if (ctx->tier == PQC_TIER_NONE) {
        int res = (int)pwrite(storage_fd, ctx->wbuf, ctx->wbuf_used, ctx->wbuf_base_off);
        ctx->wbuf_used = 0;
        return res == -1 ? -errno : 0;
    }

    size_t sz   = ctx->wbuf_used;
    off_t  base = ctx->wbuf_base_off;

    /* Phase 4: Async Pipeline — Acquire dedicated stream and buffers */
    int stream_id = acquire_stream();
    gpu_stream_ctx_t *sctx = &g_streams[stream_id];

    /* Step 1: CPU → SHAKE128 keystream */
    gpu_gen_keystream(ctx, base, sz, sctx->pinned_keystream);

    /* Step 2: memcpy to stream's pinned buffer */
    memcpy(sctx->pinned_buf, ctx->wbuf, sz);

    /* Phase 3: GPU NTT Workload (Simulate heavy Kyber polynomial ops first) */
    int ntt_blocks = (sz / (KYBER_N * sizeof(int16_t)));
    if (ntt_blocks > 0) {
        pqc_kyber_ntt_kernel<<<ntt_blocks, 128, 0, sctx->stream>>>((int16_t*)sctx->pinned_buf, sz);
    }

    /* Step 3: GPU XOR (plaintext XOR keystream) */
    int tpb    = 256;
    int blocks = (int)((sz + tpb - 1) / tpb);
    pqc_stream_xor_kernel<<<blocks, tpb, 0, sctx->stream>>>(
        sctx->pinned_buf, sctx->pinned_keystream, sz);
    
    CUDA_CHECK(cudaStreamSynchronize(sctx->stream));

    /* Step 4: write ciphertext from pinned buffer → NVMe */
    int res = (int)pwrite(storage_fd, sctx->pinned_buf, sz, base);
    
    release_stream(stream_id);
    
    ctx->wbuf_used = 0;
    return res == -1 ? -errno : 0;
}

/* ════════════════════════════════════════════════════════════════════════════
 *  FUSE Operations
 * ════════════════════════════════════════════════════════════════════════════ */

static int pqc_getattr(const char *path, struct stat *stbuf, struct fuse_file_info *fi)
{
    (void)fi;
    char phys[4096];
    resolve_physical_path(phys, sizeof(phys), path);
    return lstat(phys, stbuf) == -1 ? -errno : 0;
}

static int pqc_readdir(const char *path, void *buf, fuse_fill_dir_t filler,
                        off_t offset, struct fuse_file_info *fi,
                        enum fuse_readdir_flags flags)
{
    (void)offset; (void)fi; (void)flags;
    char phys[4096];
    resolve_physical_path(phys, sizeof(phys), path);

    DIR *dp = opendir(phys);
    if (!dp) return -errno;

    struct dirent *de;
    while ((de = readdir(dp)) != NULL) {
        struct stat st;
        memset(&st, 0, sizeof(st));
        st.st_ino  = de->d_ino;
        st.st_mode = de->d_type << 12;
        if (filler(buf, de->d_name, &st, 0, (enum fuse_fill_dir_flags)0)) break;
    }
    closedir(dp);
    return 0;
}

static int pqc_open(const char *path, struct fuse_file_info *fi)
{
    char phys[4096];
    resolve_physical_path(phys, sizeof(phys), path);
    int fd = open(phys, fi->flags);
    if (fd == -1) return -errno;
    fi->fh = (uint64_t)fd;

    /* Load per-file key sidecar (.pqckey) for read-decryption if it exists */
    char key_path[4096 + 8];
    snprintf(key_path, sizeof(key_path), "%s.pqckey", phys);
    int kfd = open(key_path, O_RDONLY);
    if (kfd >= 0) {
        uint64_t fid = 0, ss_len = 0;
        uint8_t  ss[64] = {0};
        if (read(kfd, &fid,    8) == 8 &&
            read(kfd, &ss_len, 8) == 8 &&
            ss_len <= sizeof(ss) &&
            (size_t)read(kfd, ss, ss_len) == ss_len) {
            ctx_set(fd, ss, (size_t)ss_len, fid);
            /* Advance through rotation epoch records to load latest key.
             * Format: epoch(8)|offset_start(8)|ss_len(8)|ss(ss_len) */
            uint64_t ep_hdr[3];
            while (read(kfd, ep_hdr, 24) == 24) {
                uint64_t ep_ss_len = ep_hdr[2];
                if (ep_ss_len == 0 || ep_ss_len > sizeof(ss)) break;
                uint8_t ep_ss[64] = {0};
                if ((size_t)read(kfd, ep_ss, ep_ss_len) == ep_ss_len) {
                    int idx2 = fd % PQC_MAX_FD;
                    pthread_mutex_lock(&g_fd_locks[idx2]);
                    if (g_fd_ctx[idx2].valid && g_fd_ctx[idx2].num_epochs < MAX_EPOCHS) {
                        int e = g_fd_ctx[idx2].num_epochs++;
                        g_fd_ctx[idx2].epochs[e].offset_start = ep_hdr[1];
                        g_fd_ctx[idx2].epochs[e].ss_len = ep_ss_len;
                        memcpy(g_fd_ctx[idx2].epochs[e].ss, ep_ss, ep_ss_len);
                        g_fd_ctx[idx2].key_epoch = ep_hdr[0];
                    }
                    pthread_mutex_unlock(&g_fd_locks[idx2]);
                    OQS_MEM_cleanse(ep_ss, sizeof(ep_ss));
                }
            }
        }
        OQS_MEM_cleanse(ss, sizeof(ss));
        close(kfd);
    }
    /* Strategy 2: restore tier from physical xattr if present */
    {
        char xval[8] = {0};
        ssize_t xlen = getxattr(phys, PQC_XATTR_TIER, xval, sizeof(xval) - 1);
        if (xlen > 0) {
            int t = atoi(xval);
            if (t == PQC_TIER_NONE) {
                int idx2 = fd % PQC_MAX_FD;
                pthread_mutex_lock(&g_fd_locks[idx2]);
                if (g_fd_ctx[idx2].valid)
                    g_fd_ctx[idx2].tier = PQC_TIER_NONE;
                pthread_mutex_unlock(&g_fd_locks[idx2]);
            }
        }
    }
    return 0;
}

static int pqc_read(const char *path, char *buf, size_t size, off_t offset,
                     struct fuse_file_info *fi)
{
    (void)path;
    int fd  = (int)fi->fh;
    int idx = fd % PQC_MAX_FD;

    int res = (int)pread(fd, buf, size, offset);
    if (res <= 0) return res == -1 ? -errno : 0;

    /* Strategy 2: tier NONE → plaintext, no decrypt needed */
    pthread_mutex_lock(&g_fd_locks[idx]);
    if (g_fd_ctx[idx].valid && g_fd_ctx[idx].tier != PQC_TIER_NONE) {
        pqc_fd_ctx_t snap = g_fd_ctx[idx];
        pthread_mutex_unlock(&g_fd_locks[idx]);
        uint8_t *ks = (uint8_t *)malloc((size_t)res);
        if (ks) {
            gpu_gen_keystream(&snap, offset, (size_t)res, ks);
            for (int i = 0; i < res; i++)
                buf[i] ^= ks[i];
            free(ks);
        }
        return res;
    }
    pthread_mutex_unlock(&g_fd_locks[idx]);
    return res;
}

static int pqc_create(const char *path, mode_t mode, struct fuse_file_info *fi)
{
    char phys[4096];
    resolve_physical_path(phys, sizeof(phys), path);
    int fd = open(phys, fi->flags, mode);
    if (fd == -1) return -errno;
    fi->fh = (uint64_t)fd;

    /* Run KEM once per file on CPU — shared secret seeds the GPU stream cipher */
    if (g_kem) {
        uint8_t *ct = (uint8_t *)malloc(g_kem->length_ciphertext);
        uint8_t *ss = (uint8_t *)malloc(g_kem->length_shared_secret);
        if (OQS_KEM_encaps(g_kem, ct, ss, g_public_key) == OQS_SUCCESS) {
            pthread_mutex_lock(&g_global_lock);
            uint64_t fid = ++g_file_id_ctr;
            pthread_mutex_unlock(&g_global_lock);
            ctx_set(fd, ss, g_kem->length_shared_secret, fid);
            pqc_log("CREATE %s fd=%d fid=%llu", path, fd, (unsigned long long)fid);
            /* Persist key sidecar for read-back decryption across mounts */
            char key_path[4096 + 8];
            snprintf(key_path, sizeof(key_path), "%s.pqckey", phys);
            int kfd = open(key_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
            if (kfd >= 0) {
                uint64_t ss_len64 = (uint64_t)g_kem->length_shared_secret;
                (void)(write(kfd, &fid,     8) +
                       write(kfd, &ss_len64, 8) +
                       write(kfd, ss, (size_t)ss_len64));
                close(kfd);
            }
        }
        OQS_MEM_cleanse(ss, g_kem->length_shared_secret);
        free(ct); free(ss);
    }
    return 0;
}

/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║  WRITE — GPU pipeline (pinned memory, no malloc per write)             ║
 * ║                                                                          ║
 * ║  1. memcpy FUSE buf → g_pinned_buf              (1 CPU copy)            ║
 * ║  2. GPU kernel: in-place XOR stream cipher on pinned buf               ║
 * ║  3. pwrite(g_pinned_buf) directly — no extra copy                      ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 */
static int pqc_write(const char *path, const char *buf, size_t size,
                      off_t offset, struct fuse_file_info *fi)
{
    (void)path;
    int fd  = (int)fi->fh;
    int idx = fd % PQC_MAX_FD;

    pthread_mutex_lock(&g_fd_locks[idx]);
    pqc_fd_ctx_t *ctx = &g_fd_ctx[idx];

    /* No context → passthrough (file opened without create, e.g. sidecar) */
    if (!ctx->valid) {
        pthread_mutex_unlock(&g_fd_locks[idx]);
        int res = (int)pwrite(fd, buf, size, offset);
        return res == -1 ? -errno : (int)size;
    }

    /* Strategy 2: Tier NONE → coalesce plaintext, no GPU encryption */
    if (ctx->tier == PQC_TIER_NONE) {
        int is_c = (ctx->wbuf_used == 0) ||
                   (offset == ctx->wbuf_base_off + (off_t)ctx->wbuf_used);
        if (!is_c) do_flush_wbuf_locked(fd, idx);
        if (ctx->wbuf_used == 0) ctx->wbuf_base_off = offset;
        if (ctx->wbuf && size < COALESCE_SIZE) {
            memcpy(ctx->wbuf + ctx->wbuf_used, buf, size);
            ctx->wbuf_used += size;
            if (ctx->wbuf_used >= COALESCE_SIZE)
                do_flush_wbuf_locked(fd, idx);
        } else {
            do_flush_wbuf_locked(fd, idx);
            int _r = (int)pwrite(fd, buf, size, offset);
            (void)_r;
        }
        pthread_mutex_unlock(&g_fd_locks[idx]);
        return (int)size;
    }

    /* Strategy 1: Key Rotation — re-key if interval expired */
    {
        time_t now = time(NULL);
        if (now - ctx->last_rekey >= KEY_ROTATION_INTERVAL_S) {
            char phys_r[4096];
            resolve_physical_path(phys_r, sizeof(phys_r), path);
            if (ctx->wbuf_used > 0)
                do_flush_wbuf_locked(fd, idx);
            do_rekey_locked(fd, idx, phys_r);
        }
    }

    /* No coalescing buffer (malloc failed) → encrypt directly */
    if (!ctx->wbuf) {
        pqc_fd_ctx_t snap = *ctx;
        pthread_mutex_unlock(&g_fd_locks[idx]);
        
        int sid = acquire_stream();
        gpu_stream_ctx_t *sctx = &g_streams[sid];
        size_t proc = (size <= (size_t)GPU_BUFFER_SIZE) ? size : (size_t)GPU_BUFFER_SIZE;
        
        gpu_gen_keystream(&snap, offset, proc, sctx->pinned_keystream);
        memcpy(sctx->pinned_buf, buf, proc);
        
        int tpb = 256, blocks = (int)((proc + tpb - 1) / tpb);
        pqc_stream_xor_kernel<<<blocks, tpb, 0, sctx->stream>>>(
            sctx->pinned_buf, sctx->pinned_keystream, proc);
        CUDA_CHECK(cudaStreamSynchronize(sctx->stream));
        
        int res = (int)pwrite(fd, sctx->pinned_buf, proc, offset);
        release_stream(sid);
        return res == -1 ? -errno : (int)proc;
    }

    /* Non-contiguous write: flush pending data */
    int is_cont = (ctx->wbuf_used == 0) ||
                  (offset == ctx->wbuf_base_off + (off_t)ctx->wbuf_used);
    if (!is_cont) {
        int fr = do_flush_wbuf_locked(fd, idx);
        if (fr < 0) { pthread_mutex_unlock(&g_fd_locks[idx]); return fr; }
    }

    /* Large write (≥ COALESCE_SIZE): encrypt directly, bypass buffering */
    if (size >= COALESCE_SIZE) {
        do_flush_wbuf_locked(fd, idx);
        pqc_fd_ctx_t snap = *ctx;
        pthread_mutex_unlock(&g_fd_locks[idx]);
        
        int sid = acquire_stream();
        gpu_stream_ctx_t *sctx = &g_streams[sid];
        size_t proc = (size <= (size_t)GPU_BUFFER_SIZE) ? size : (size_t)GPU_BUFFER_SIZE;
        
        gpu_gen_keystream(&snap, offset, proc, sctx->pinned_keystream);
        memcpy(sctx->pinned_buf, buf, proc);
        
        int tpb = 256, blocks = (int)((proc + tpb - 1) / tpb);
        pqc_stream_xor_kernel<<<blocks, tpb, 0, sctx->stream>>>(
            sctx->pinned_buf, sctx->pinned_keystream, proc);
        CUDA_CHECK(cudaStreamSynchronize(sctx->stream));
        
        int res = (int)pwrite(fd, sctx->pinned_buf, proc, offset);
        release_stream(sid);
        return res == -1 ? -errno : (int)proc;
    }

    /* Initialize base offset when buffer is freshly empty */
    if (ctx->wbuf_used == 0)
        ctx->wbuf_base_off = offset;

    /* Append to coalescing buffer */
    memcpy(ctx->wbuf + ctx->wbuf_used, buf, size);
    ctx->wbuf_used += size;

    /* Auto-flush when buffer fills up */
    if (ctx->wbuf_used >= COALESCE_SIZE) {
        int fr = do_flush_wbuf_locked(fd, idx);
        if (fr < 0) { pthread_mutex_unlock(&g_fd_locks[idx]); return fr; }
    }

    pthread_mutex_unlock(&g_fd_locks[idx]);
    return (int)size;
}

static int pqc_fsync(const char *path, int datasync, struct fuse_file_info *fi)
{
    (void)path; (void)datasync;
    int fd  = (int)fi->fh;
    int idx = fd % PQC_MAX_FD;

    pthread_mutex_lock(&g_fd_locks[idx]);
    if (g_fd_ctx[idx].valid && g_fd_ctx[idx].wbuf)
        do_flush_wbuf_locked(fd, idx);
    pthread_mutex_unlock(&g_fd_locks[idx]);

    return fdatasync(fd) == -1 ? -errno : 0;
}

static int pqc_truncate(const char *path, off_t size, struct fuse_file_info *fi)
{
    (void)fi;
    char phys[4096];
    resolve_physical_path(phys, sizeof(phys), path);
    return truncate(phys, size) == -1 ? -errno : 0;
}

static int pqc_unlink(const char *path)
{
    char phys[4096];
    resolve_physical_path(phys, sizeof(phys), path);
    /* Remove key sidecar if it exists */
    char key_path[4096 + 8];
    snprintf(key_path, sizeof(key_path), "%s.pqckey", phys);
    unlink(key_path);
    return unlink(phys) == -1 ? -errno : 0;
}

static int pqc_mkdir(const char *path, mode_t mode)
{
    char phys[4096];
    resolve_physical_path(phys, sizeof(phys), path);
    return mkdir(phys, mode) == -1 ? -errno : 0;
}

static int pqc_rmdir(const char *path)
{
    char phys[4096];
    resolve_physical_path(phys, sizeof(phys), path);
    return rmdir(phys) == -1 ? -errno : 0;
}

static int pqc_release(const char *path, struct fuse_file_info *fi)
{
    (void)path;
    int fd  = (int)fi->fh;
    int idx = fd % PQC_MAX_FD;

    /* Flush any remaining coalesced data before closing */
    pthread_mutex_lock(&g_fd_locks[idx]);
    if (g_fd_ctx[idx].valid && g_fd_ctx[idx].wbuf && g_fd_ctx[idx].wbuf_used > 0)
        do_flush_wbuf_locked(fd, idx);
    pthread_mutex_unlock(&g_fd_locks[idx]);

    ctx_clear(fd);
    close(fd);
    return 0;
}

static int pqc_utimens(const char *path, const struct timespec tv[2],
                        struct fuse_file_info *fi)
{
    (void)fi;
    char phys[4096];
    resolve_physical_path(phys, sizeof(phys), path);
    return utimensat(AT_FDCWD, phys, tv, AT_SYMLINK_NOFOLLOW) == -1 ? -errno : 0;
}

static void pqc_destroy(void *private_data)
{
    (void)private_data;
    pqc_log("Shutting down PQC-FUSE (GPU) ...");
    gpu_cleanup();
    if (g_logfile) { fclose(g_logfile); g_logfile = NULL; }
}

/* ════════════════════════════════════════════════════════════════════════════
 *  Strategy 2: xattr handlers — Semantic-Aware Selective Encryption
 *  Mirrors the CPU version; stored on physical backing file for persistence.
 * ════════════════════════════════════════════════════════════════════════════ */

static int pqc_setxattr(const char *path, const char *name,
                         const char *value, size_t size, int flags)
{
    char phys[4096];
    resolve_physical_path(phys, sizeof(phys), path);
    if (setxattr(phys, name, value, size, flags) == -1)
        return -errno;
    if (strcmp(name, PQC_XATTR_TIER) == 0 && size > 0) {
        char tmp[8] = {0};
        memcpy(tmp, value, size < 7 ? size : 7);
        int new_tier = atoi(tmp);
        if (new_tier != PQC_TIER_FULL && new_tier != PQC_TIER_NONE)
            return -EINVAL;
        /* Scan open fd contexts for this phys_path */
        for (int i = 0; i < PQC_MAX_FD; i++) {
            pthread_mutex_lock(&g_fd_locks[i]);
            if (g_fd_ctx[i].valid)
                g_fd_ctx[i].tier = new_tier;  /* best-effort; path not stored */
            pthread_mutex_unlock(&g_fd_locks[i]);
        }
        pqc_log("SETXATTR %s tier=%d", path, new_tier);
    }
    return 0;
}

static int pqc_getxattr(const char *path, const char *name,
                         char *value, size_t size)
{
    char phys[4096];
    resolve_physical_path(phys, sizeof(phys), path);
    ssize_t res = getxattr(phys, name, value, size);
    return res == -1 ? -errno : (int)res;
}

static int pqc_listxattr(const char *path, char *list, size_t size)
{
    char phys[4096];
    resolve_physical_path(phys, sizeof(phys), path);
    ssize_t res = listxattr(phys, list, size);
    return res == -1 ? -errno : (int)res;
}

static struct fuse_operations make_pqc_oper() {
    struct fuse_operations ops = {};
    ops.getattr  = pqc_getattr;
    ops.readdir  = pqc_readdir;
    ops.mkdir    = pqc_mkdir;
    ops.unlink   = pqc_unlink;
    ops.rmdir    = pqc_rmdir;
    ops.truncate = pqc_truncate;
    ops.open     = pqc_open;
    ops.read     = pqc_read;
    ops.write    = pqc_write;
    ops.fsync    = pqc_fsync;
    ops.release  = pqc_release;
    ops.utimens  = pqc_utimens;
    ops.create    = pqc_create;
    ops.destroy   = pqc_destroy;
    ops.setxattr  = pqc_setxattr;
    ops.getxattr  = pqc_getxattr;
    ops.listxattr = pqc_listxattr;
    return ops;
}

/* ════════════════════════════════════════════════════════════════════════════
 *  Main
 * ════════════════════════════════════════════════════════════════════════════ */

int main(int argc, char *argv[])
{
    fprintf(stderr,
        "\n"
        "  ┌─────────────────────────────────────────────────────┐\n"
        "  │  PQC-FUSE v0.2 — GPU-Accelerated (Zero-copy)       │\n"
        "  │  Jetson Unified Memory Pipeline                     │\n"
        "  └─────────────────────────────────────────────────────┘\n\n");

    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <storage_dir> <mountpoint> [FUSE options]\n"
            "  -f : Run in foreground (recommended)\n\n", argv[0]);
        return EXIT_FAILURE;
    }

    char *storage_dir = realpath(argv[1], NULL);
    if (!storage_dir) {
        fprintf(stderr, "[PQC-GPU] ERROR: Storage dir '%s' not found\n", argv[1]);
        return EXIT_FAILURE;
    }
    strncpy(g_storage_dir, storage_dir, sizeof(g_storage_dir) - 1);
    free(storage_dir);

    char log_path[4096];
    snprintf(log_path, sizeof(log_path), "%s/../%s", g_storage_dir, LOG_FILENAME);
    g_logfile = fopen(log_path, "a");
    if (!g_logfile) g_logfile = fopen(LOG_FILENAME, "a");

    pqc_log("═══════════════════════════════════════════════════════");
    pqc_log("  PQC-FUSE GPU Session Started");
    pqc_log("  Storage : %s", g_storage_dir);
    pqc_log("  Mount   : %s", argv[2]);
    pqc_log("═══════════════════════════════════════════════════════");

    if (gpu_init() != 0) {
        fprintf(stderr, "[PQC-GPU] FATAL: GPU init failed\n");
        return EXIT_FAILURE;
    }

    int fuse_argc = argc - 1;
    char **fuse_argv = (char **)malloc(sizeof(char *) * (size_t)fuse_argc);
    fuse_argv[0] = argv[0];
    for (int i = 2; i < argc; i++) fuse_argv[i - 1] = argv[i];

    pqc_log("Starting FUSE main loop (GPU mode)...");
    struct fuse_operations pqc_oper = make_pqc_oper();
    int ret = fuse_main(fuse_argc, fuse_argv, &pqc_oper, NULL);

    free(fuse_argv);
    gpu_cleanup();
    if (g_logfile) { fclose(g_logfile); g_logfile = NULL; }
    return ret;
}
