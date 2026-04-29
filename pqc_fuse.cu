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

/* ════════════════════════════════════════════════════════════════════════════
 *  Configuration & Globals
 * ════════════════════════════════════════════════════════════════════════════ */

static char g_storage_dir[4096] = {0};
static FILE *g_logfile = NULL;
static const char *LOG_FILENAME = "pqc_fuse_gpu_latency.log";

/** Unified Memory buffer pool (allocated once at startup) */
#define GPU_BUFFER_SIZE (4 * 1024 * 1024)  /* 4 MB pool */
static uint8_t *g_unified_buf_in  = NULL;  /* Input buffer (plaintext) */
static uint8_t *g_unified_buf_out = NULL;  /* Output buffer (encrypted) */

/** CUDA stream for async operations */
static cudaStream_t g_cuda_stream;

/** XOR key for dummy PQC (generated at startup) */
#define PQC_KEY_SIZE 256
static uint8_t *g_pqc_key = NULL;  /* Also in Unified Memory */

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

    /* Allocate Unified Memory buffers — TRUE zero-copy on Jetson! */
    double t0 = get_time_us();
    CUDA_CHECK(cudaMallocManaged(&g_unified_buf_in,  GPU_BUFFER_SIZE));
    CUDA_CHECK(cudaMallocManaged(&g_unified_buf_out, GPU_BUFFER_SIZE));
    CUDA_CHECK(cudaMallocManaged(&g_pqc_key, PQC_KEY_SIZE));
    double t1 = get_time_us();

    pqc_log("Unified Memory allocated: %d MB × 2 buffers + %d B key (%.2f µs)",
            GPU_BUFFER_SIZE / (1024 * 1024), PQC_KEY_SIZE, t1 - t0);

    /* Generate dummy PQC key */
    srand((unsigned)time(NULL));
    for (int i = 0; i < PQC_KEY_SIZE; i++) {
        g_pqc_key[i] = (uint8_t)(rand() & 0xFF);
    }

    /* Warm up: touch the key from GPU side to ensure caching */
    CUDA_CHECK(cudaStreamSynchronize(g_cuda_stream));

    pqc_log("GPU initialization complete — Zero-copy pipeline ready");
    return 0;
}

static void gpu_cleanup(void)
{
    if (g_unified_buf_in)  { cudaFree(g_unified_buf_in);  g_unified_buf_in  = NULL; }
    if (g_unified_buf_out) { cudaFree(g_unified_buf_out); g_unified_buf_out = NULL; }
    if (g_pqc_key)         { cudaFree(g_pqc_key);         g_pqc_key         = NULL; }
    cudaStreamDestroy(g_cuda_stream);
    pqc_log("GPU resources released");
}

/**
 * GPU-accelerated PQC encryption via Unified Memory (Zero-copy on Jetson).
 *
 * Pipeline:
 *   1. memcpy plaintext → Unified Memory buffer (CPU writes, no DMA needed)
 *   2. Launch CUDA kernels (GPU reads same physical memory — zero copy!)
 *   3. cudaDeviceSynchronize (wait for GPU)
 *   4. Read result from Unified Memory (CPU reads, still zero copy!)
 *
 * Returns bytes written, or -1 on error.
 */
static int gpu_pqc_encrypt(const char *in_buf, size_t in_size,
                            char *out_buf, size_t out_buf_size,
                            double *cpu_latency_us,
                            double *gpu_latency_us,
                            double *total_latency_us)
{
    size_t process_size = (in_size < (size_t)GPU_BUFFER_SIZE) ? in_size : (size_t)GPU_BUFFER_SIZE;
    if (process_size > out_buf_size) process_size = out_buf_size;

    double t_total_start = get_time_us();

    /* ── Step 1: CPU copies data to Unified Memory (no DMA on Jetson!) ── */
    double t_cpu_start = get_time_us();
    memcpy(g_unified_buf_in, in_buf, process_size);
    /* Copy input to output buffer for in-place kernel processing */
    memcpy(g_unified_buf_out, in_buf, process_size);
    double t_cpu_end = get_time_us();
    *cpu_latency_us = t_cpu_end - t_cpu_start;

    /* ── Step 2: Launch GPU kernels ── */
    double t_gpu_start = get_time_us();

    int threads_per_block = 256;
    int blocks = (int)((process_size + threads_per_block - 1) / threads_per_block);
    int ntt_blocks = (int)((process_size / 4 + threads_per_block - 1) / threads_per_block);

    /* Kernel 1: XOR encryption (16 rounds for visible GPU workload) */
    pqc_xor_encrypt_kernel<<<blocks, threads_per_block, 0, g_cuda_stream>>>(
        g_unified_buf_out, g_pqc_key, process_size, PQC_KEY_SIZE, 16
    );

    /* Kernel 2: NTT butterfly (simulates Kyber polynomial math) */
    if (ntt_blocks > 0) {
        pqc_ntt_butterfly_kernel<<<ntt_blocks, threads_per_block, 0, g_cuda_stream>>>(
            g_unified_buf_out, process_size, 8
        );
    }

    /* ── Step 3: Wait for GPU completion ── */
    CUDA_CHECK(cudaStreamSynchronize(g_cuda_stream));
    double t_gpu_end = get_time_us();
    *gpu_latency_us = t_gpu_end - t_gpu_start;

    /* ── Step 4: Read result from Unified Memory (zero-copy read on Jetson) ── */
    memcpy(out_buf, g_unified_buf_out, process_size);

    double t_total_end = get_time_us();
    *total_latency_us = t_total_end - t_total_start;

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
    return 0;
}

/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║  WRITE — GPU-Accelerated PQC Encryption (Zero-copy on Jetson)           ║
 * ║                                                                          ║
 * ║  Pipeline:                                                               ║
 * ║    1. Copy data to Unified Memory (CPU → shared DRAM, no DMA)           ║
 * ║    2. Launch CUDA PQC kernels (GPU reads same physical pages)           ║
 * ║    3. cudaStreamSynchronize (wait)                                       ║
 * ║    4. Write encrypted data to disk                                       ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 */
static int pqc_write(const char *path, const char *buf, size_t size,
                      off_t offset, struct fuse_file_info *fi)
{
    (void)path;
    int fd = (int)fi->fh;

    double cpu_us = 0, gpu_us = 0, total_us = 0;
    char *enc_buf = (char *)malloc(size);
    if (!enc_buf) return -ENOMEM;

    int enc_len = gpu_pqc_encrypt(buf, size, enc_buf, size,
                                   &cpu_us, &gpu_us, &total_us);
    if (enc_len < 0) {
        pqc_log("WRITE ERROR: GPU encryption failed for %zu bytes", size);
        free(enc_buf);
        int res = (int)pwrite(fd, buf, size, offset);
        return res == -1 ? -errno : res;
    }

    double throughput = (size > 0 && total_us > 0)
        ? ((double)size / (1024.0 * 1024.0)) / (total_us / 1e6) : 0.0;

    pqc_log("WRITE: %zu B | CPU: %.1f µs | GPU: %.1f µs | Total: %.1f µs (%.2f ms) | %.2f MB/s",
            size, cpu_us, gpu_us, total_us, total_us / 1000.0, throughput);

    double t_io = get_time_us();
    int res = (int)pwrite(fd, enc_buf, (size_t)enc_len, offset);
    double t_io_end = get_time_us();

    if (res == -1) { free(enc_buf); return -errno; }

    pqc_log("WRITE I/O: %d B disk | disk: %.1f µs", res, t_io_end - t_io);
    free(enc_buf);
    return res;
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
