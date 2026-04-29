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
#include <pthread.h>

/* ════════════════════════════════════════════════════════════════════════════
 *  Configuration & Globals
 * ════════════════════════════════════════════════════════════════════════════ */

static char g_storage_dir[4096] = {0};
static FILE *g_logfile = NULL;
static const char *LOG_FILENAME = "pqc_fuse_gpu_latency.log";

/*
 * Pinned (page-locked) host memory pool.
 * Unlike cudaMallocManaged, pinned memory never triggers GPU page faults —
 * the GPU accesses it via DMA directly, giving deterministic low latency.
 */
#define GPU_BUFFER_SIZE (4 * 1024 * 1024)  /* 4 MB pool */
static uint8_t *g_pinned_buf = NULL;  /* single in-place pinned buffer */

/** CUDA stream for async operations */
static cudaStream_t g_cuda_stream;

/** XOR key for stream cipher kernel (pinned memory) */
#define PQC_KEY_SIZE 256
static uint8_t *g_pqc_key = NULL;  /* pinned, accessible by both CPU and GPU */

/*
 * Per-fd encryption context (same design as CPU version).
 * KEM runs once per file; the 32-byte shared secret seeds the GPU stream cipher.
 */
#define PQC_MAX_FD 4096

typedef struct {
    int      valid;
    uint8_t  ss[64];
    size_t   ss_len;
    uint64_t file_id;
} pqc_fd_ctx_t;

static pqc_fd_ctx_t    g_fd_ctx[PQC_MAX_FD];
static pthread_mutex_t g_fd_lock     = PTHREAD_MUTEX_INITIALIZER;
static uint64_t        g_file_id_ctr = 0;

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

/**
 * Kernel 2: Dummy NTT-like butterfly operation on 32-bit words
 * Simulates the Number Theoretic Transform used in Kyber.
 */
__global__ void pqc_ntt_butterfly_kernel(uint8_t *data, size_t data_size,
                                          int iterations)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    /* Process as 32-bit words */
    size_t word_count = data_size / 4;
    if (idx >= word_count) return;

    uint32_t *words = (uint32_t *)data;
    uint32_t val = words[idx];

    for (int i = 0; i < iterations; i++) {
        /* Butterfly-like operation */
        uint32_t partner_idx = idx ^ (1 << (i % 12));
        if (partner_idx < word_count) {
            uint32_t partner = words[partner_idx];
            val = val + partner * 3329;  /* 3329 = Kyber prime q */
            val ^= (val >> 16);
        }
    }

    words[idx] = val;
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

    /* Create CUDA stream */
    CUDA_CHECK(cudaStreamCreate(&g_cuda_stream));

    /*
     * Allocate PINNED host memory (cudaHostAlloc).
     * Pinned memory is directly accessible by the GPU via DMA without
     * page migration — eliminates the ~100µs page-fault overhead per write
     * that cudaMallocManaged causes on discrete/hybrid GPUs.
     */
    double t0 = get_time_us();
    CUDA_CHECK(cudaHostAlloc(&g_pinned_buf, GPU_BUFFER_SIZE, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&g_pqc_key,   PQC_KEY_SIZE,    cudaHostAllocDefault));
    double t1 = get_time_us();

    pqc_log("Pinned memory allocated: %d MB buf + %d B key (%.2f µs)",
            GPU_BUFFER_SIZE / (1024 * 1024), PQC_KEY_SIZE, t1 - t0);

    /* Generate stream-cipher key (will be overridden per-file by KEM secret) */
    srand((unsigned)time(NULL));
    for (int i = 0; i < PQC_KEY_SIZE; i++)
        g_pqc_key[i] = (uint8_t)(rand() & 0xFF);

    /* Warm up GPU: one small kernel launch to amortise first-launch overhead */
    pqc_xor_encrypt_kernel<<<1, 32, 0, g_cuda_stream>>>(g_pinned_buf, g_pqc_key, 32, PQC_KEY_SIZE, 1);
    CUDA_CHECK(cudaStreamSynchronize(g_cuda_stream));

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
    if (g_pinned_buf) { cudaFreeHost(g_pinned_buf); g_pinned_buf = NULL; }
    if (g_pqc_key)    { cudaFreeHost(g_pqc_key);    g_pqc_key    = NULL; }
    cudaStreamDestroy(g_cuda_stream);
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
    pthread_mutex_lock(&g_fd_lock);
    memcpy(g_fd_ctx[idx].ss, ss, ss_len);
    g_fd_ctx[idx].ss_len  = ss_len;
    g_fd_ctx[idx].file_id = fid;
    g_fd_ctx[idx].valid   = 1;
    pthread_mutex_unlock(&g_fd_lock);
}

static int ctx_get(int fd, pqc_fd_ctx_t *out)
{
    int idx = fd % PQC_MAX_FD;
    pthread_mutex_lock(&g_fd_lock);
    if (!g_fd_ctx[idx].valid) { pthread_mutex_unlock(&g_fd_lock); return -1; }
    *out = g_fd_ctx[idx];
    pthread_mutex_unlock(&g_fd_lock);
    return 0;
}

static void ctx_clear(int fd)
{
    int idx = fd % PQC_MAX_FD;
    pthread_mutex_lock(&g_fd_lock);
    OQS_MEM_cleanse(g_fd_ctx[idx].ss, sizeof(g_fd_ctx[idx].ss));
    g_fd_ctx[idx].valid = 0;
    pthread_mutex_unlock(&g_fd_lock);
}

/**
 * GPU-accelerated stream-cipher encryption.
 *
 * Optimised pipeline (vs old version):
 *   OLD: FUSE buf → malloc → unified_in → unified_out → [kernel] → malloc → pwrite
 *        = 4 data copies + page-fault overhead per write
 *   NEW: FUSE buf → pinned_buf → [kernel in-place] → pwrite(pinned_buf)
 *        = 1 data copy + DMA access (no page faults)
 *
 * The kernel uses the per-file KEM-derived shared secret (loaded into g_pqc_key
 * before the launch) so every file has a unique keystream.
 */
static int gpu_pqc_encrypt(int fd, const char *in_buf, size_t in_size,
                            off_t write_offset,
                            double *cpu_us, double *gpu_us, double *total_us)
{
    size_t process_size = (in_size < (size_t)GPU_BUFFER_SIZE)
                          ? in_size : (size_t)GPU_BUFFER_SIZE;

    double t_total = get_time_us();

    /* ── Step 1: one memcpy into pinned buffer ── */
    double t_cpu = get_time_us();
    memcpy(g_pinned_buf, in_buf, process_size);

    /* Load per-file secret into key slot (first PQC_KEY_SIZE bytes of ss, padded) */
    pqc_fd_ctx_t ctx;
    if (ctx_get(fd, &ctx) == 0) {
        size_t klen = ctx.ss_len < (size_t)PQC_KEY_SIZE ? ctx.ss_len : (size_t)PQC_KEY_SIZE;
        memcpy(g_pqc_key, ctx.ss, klen);
        /* Mix in offset as nonce to avoid keystream reuse across writes */
        uint64_t off = (uint64_t)write_offset;
        for (size_t i = 0; i < 8 && i < (size_t)PQC_KEY_SIZE; i++)
            g_pqc_key[i] ^= ((uint8_t *)&off)[i];
    }
    *cpu_us = get_time_us() - t_cpu;

    /* ── Step 2: launch GPU kernel on pinned buffer ── */
    double t_gpu = get_time_us();

    int tpb    = 256;
    int blocks = (int)((process_size + tpb - 1) / tpb);
    pqc_xor_encrypt_kernel<<<blocks, tpb, 0, g_cuda_stream>>>(
        g_pinned_buf, g_pqc_key, process_size, PQC_KEY_SIZE, 8);

    int ntt_blocks = (int)((process_size / 4 + tpb - 1) / tpb);
    if (ntt_blocks > 0)
        pqc_ntt_butterfly_kernel<<<ntt_blocks, tpb, 0, g_cuda_stream>>>(
            g_pinned_buf, process_size, 4);

    /* ── Step 3: wait for GPU ── */
    CUDA_CHECK(cudaStreamSynchronize(g_cuda_stream));
    *gpu_us = get_time_us() - t_gpu;

    /* Result is already in g_pinned_buf — no extra copy needed */
    *total_us = get_time_us() - t_total;
    return (int)process_size;
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
    return 0;
}

static int pqc_read(const char *path, char *buf, size_t size, off_t offset,
                     struct fuse_file_info *fi)
{
    (void)path;
    int res = (int)pread((int)fi->fh, buf, size, offset);
    return res == -1 ? -errno : res;
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
        if (ct && ss && OQS_KEM_encaps(g_kem, ct, ss, g_public_key) == OQS_SUCCESS) {
            pthread_mutex_lock(&g_fd_lock);
            uint64_t fid = ++g_file_id_ctr;
            pthread_mutex_unlock(&g_fd_lock);
            ctx_set(fd, ss, g_kem->length_shared_secret, fid);
            pqc_log("CREATE %s fd=%d fid=%llu", path, fd, (unsigned long long)fid);
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
    int fd = (int)fi->fh;

    double cpu_us = 0, gpu_us = 0, total_us = 0;
    int enc_len = gpu_pqc_encrypt(fd, buf, size, offset,
                                   &cpu_us, &gpu_us, &total_us);
    if (enc_len < 0) {
        pqc_log("WRITE ERROR: GPU encrypt failed %zu bytes", size);
        int res = (int)pwrite(fd, buf, size, offset);
        return res == -1 ? -errno : res;
    }

    /* pwrite directly from pinned buffer — no extra memcpy! */
    int res = (int)pwrite(fd, g_pinned_buf, (size_t)enc_len, offset);
    return (res == -1) ? -errno : res;
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
    ctx_clear((int)fi->fh);
    close((int)fi->fh);
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
    ops.release  = pqc_release;
    ops.utimens  = pqc_utimens;
    ops.create   = pqc_create;
    ops.destroy  = pqc_destroy;
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
