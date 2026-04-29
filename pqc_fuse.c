/**
 * ============================================================================
 *  pqc_fuse.c — PQC-based FUSE Filesystem for Edge AI Bottleneck Profiling
 * ============================================================================
 *
 *  Purpose:
 *    Transparent FUSE filesystem that intercepts write operations and performs
 *    Kyber-512 (ML-KEM) key encapsulation on every data buffer, deliberately
 *    generating CPU-bound PQC workload.  This creates a measurable bottleneck
 *    that degrades concurrent AI inference (e.g., YOLO FPS drop), proving the
 *    need for GPU-accelerated PQC offloading on edge devices.
 *
 *  Architecture:
 *    mnt_secure/  (FUSE mount)  ──write()──►  [Kyber-512 KEM on CPU]  ──►  storage_physical/
 *
 *  Build:
 *    mkdir build && cd build && cmake .. && make
 *
 *  Usage:
 *    ./pqc_fuse  -f  <mountpoint>
 *    (the storage_physical directory is set via -o storage_dir=<path> or defaults
 *     to ../storage_physical relative to the mountpoint)
 *
 *  Author : PQC Edge Research Team
 *  License: MIT
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

#include <oqs/oqs.h>

/* ════════════════════════════════════════════════════════════════════════════
 *  Configuration & Globals
 * ════════════════════════════════════════════════════════════════════════════ */

/** Physical storage backend directory (where encrypted files are stored) */
static char g_storage_dir[4096] = {0};

/** Log file for latency measurements */
static FILE *g_logfile = NULL;
static const char *LOG_FILENAME = "pqc_fuse_latency.log";

/** PQC KEM pre-generated keypair (generated once at startup) */
static OQS_KEM *g_kem = NULL;
static uint8_t *g_public_key  = NULL;
static uint8_t *g_secret_key  = NULL;

/** Number of KEM operations per write (increase for heavier CPU load) */
#define KEM_OPS_PER_WRITE 1

/* ════════════════════════════════════════════════════════════════════════════
 *  Utility Functions
 * ════════════════════════════════════════════════════════════════════════════ */

/**
 * Get monotonic timestamp in microseconds.
 */
static inline double get_time_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

/**
 * Log a timestamped message to both stderr and the log file.
 */
static void pqc_log(const char *fmt, ...)
{
    va_list args;
    char timebuf[64];
    struct timeval tv;
    struct tm tm_info;

    gettimeofday(&tv, NULL);
    localtime_r(&tv.tv_sec, &tm_info);
    strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S", &tm_info);

    /* stderr */
    fprintf(stderr, "[PQC-FUSE %s.%03ld] ", timebuf, tv.tv_usec / 1000);
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");

    /* log file */
    if (g_logfile) {
        fprintf(g_logfile, "[%s.%03ld] ", timebuf, tv.tv_usec / 1000);
        va_start(args, fmt);
        vfprintf(g_logfile, fmt, args);
        va_end(args);
        fprintf(g_logfile, "\n");
        fflush(g_logfile);
    }
}

/**
 * Resolve a FUSE virtual path to the physical storage path.
 */
static void resolve_physical_path(char *dest, size_t dest_size, const char *path)
{
    snprintf(dest, dest_size, "%s%s", g_storage_dir, path);
}

/* ════════════════════════════════════════════════════════════════════════════
 *  PQC (Kyber-512 / ML-KEM) Engine
 * ════════════════════════════════════════════════════════════════════════════
 *
 *  Design Note for GPU Offloading (Phase 2):
 *  ──────────────────────────────────────────
 *  The PQC workload is isolated in pqc_encrypt_buffer().  To offload to GPU:
 *
 *  1. Replace the buffer with cudaMallocManaged() (Unified Memory / Zero-copy)
 *     so that CPU and GPU share the same address space without explicit memcpy.
 *
 *  2. Implement a CUDA kernel that performs the lattice polynomial multiplication
 *     (NTT + point-wise multiply + INTT) which is the core of Kyber.
 *
 *  3. Call the CUDA kernel from pqc_encrypt_buffer() instead of OQS_KEM_encaps().
 *
 *  4. Use cudaStreamSynchronize() to ensure the GPU finishes before FUSE
 *     returns from write().  For async I/O, use FUSE's multithreaded mode
 *     and let the GPU operate on a separate CUDA stream per file descriptor.
 *
 *  Key design constraints:
 *    - Keep the buffer in Unified Memory to avoid PCIe/MMIO copy overhead.
 *    - Batch small writes into larger chunks before launching GPU kernels
 *      (kernel launch overhead ~5-10µs, amortize over ≥4KB blocks).
 *    - Use pinned memory (cudaHostAlloc) for DMA if not using Unified Memory.
 * ════════════════════════════════════════════════════════════════════════════ */

/**
 * Initialize the PQC subsystem.  Generates a Kyber-512 keypair.
 */
static int pqc_init(void)
{
    /* Try ML-KEM-512 first (NIST standardized name), fallback to Kyber512 */
    g_kem = OQS_KEM_new(OQS_KEM_alg_ml_kem_512);
    if (!g_kem) {
        g_kem = OQS_KEM_new(OQS_KEM_alg_kyber_512);
    }
    if (!g_kem) {
        fprintf(stderr, "[PQC-FUSE] FATAL: Neither ML-KEM-512 nor Kyber-512 available in liboqs!\n");
        fprintf(stderr, "[PQC-FUSE] Rebuild liboqs with KEM_ml_kem_512 or KEM_kyber_512 enabled.\n");
        return -1;
    }

    pqc_log("KEM algorithm : %s", g_kem->method_name);
    pqc_log("  public key  : %zu bytes", g_kem->length_public_key);
    pqc_log("  secret key  : %zu bytes", g_kem->length_secret_key);
    pqc_log("  ciphertext  : %zu bytes", g_kem->length_ciphertext);
    pqc_log("  shared secret: %zu bytes", g_kem->length_shared_secret);

    g_public_key = malloc(g_kem->length_public_key);
    g_secret_key = malloc(g_kem->length_secret_key);
    if (!g_public_key || !g_secret_key) {
        fprintf(stderr, "[PQC-FUSE] FATAL: Failed to allocate key memory\n");
        return -1;
    }

    double t0 = get_time_us();
    OQS_STATUS rc = OQS_KEM_keypair(g_kem, g_public_key, g_secret_key);
    double t1 = get_time_us();

    if (rc != OQS_SUCCESS) {
        fprintf(stderr, "[PQC-FUSE] FATAL: Key generation failed\n");
        return -1;
    }

    pqc_log("Keypair generated in %.2f µs", t1 - t0);
    return 0;
}

/**
 * Perform PQC encryption workload on a data buffer.
 *
 * This function deliberately creates CPU-bound PQC load by running
 * KEM encapsulation on each chunk of the write buffer.
 *
 * The "encrypted" output is: ciphertext || shared_secret XOR plaintext
 * (This is NOT a secure construction — it's for profiling only!)
 *
 * Returns: number of bytes written to out_buf, or -1 on error.
 *
 * ┌─────────────────────────────────────────────────────────────────┐
 * │  GPU OFFLOADING CHANGE POINT:                                   │
 * │  Replace the inner loop with a CUDA kernel call.  The buffer    │
 * │  should be allocated via cudaMallocManaged() in the caller.     │
 * └─────────────────────────────────────────────────────────────────┘
 */
static int pqc_encrypt_buffer(const char *in_buf, size_t in_size,
                               char *out_buf, size_t out_buf_size,
                               double *latency_us)
{
    if (!g_kem) return -1;

    double t_start = get_time_us();
    size_t total_written = 0;

    /* Allocate temporary KEM buffers */
    uint8_t *ciphertext    = malloc(g_kem->length_ciphertext);
    uint8_t *shared_secret = malloc(g_kem->length_shared_secret);

    if (!ciphertext || !shared_secret) {
        free(ciphertext);
        free(shared_secret);
        return -1;
    }

    /*
     * Process the buffer in chunks of shared_secret length.
     * For each chunk, perform a full KEM encapsulation (the expensive part)
     * and XOR the plaintext with the shared secret.
     */
    size_t ss_len = g_kem->length_shared_secret;
    size_t offset = 0;

    while (offset < in_size) {
        /* ── KEM Encapsulation (CPU-intensive!) ── */
        for (int k = 0; k < KEM_OPS_PER_WRITE; k++) {
            OQS_STATUS rc = OQS_KEM_encaps(g_kem, ciphertext, shared_secret,
                                            g_public_key);
            if (rc != OQS_SUCCESS) {
                pqc_log("ERROR: KEM encapsulation failed at offset %zu", offset);
                free(ciphertext);
                free(shared_secret);
                return -1;
            }
        }

        /* ── XOR plaintext with shared secret ── */
        size_t chunk_len = (in_size - offset < ss_len) ? (in_size - offset) : ss_len;

        if (total_written + chunk_len > out_buf_size) {
            chunk_len = out_buf_size - total_written;
        }

        for (size_t i = 0; i < chunk_len; i++) {
            out_buf[total_written + i] = in_buf[offset + i] ^ shared_secret[i % ss_len];
        }

        total_written += chunk_len;
        offset += chunk_len;
    }

    double t_end = get_time_us();
    *latency_us = t_end - t_start;

    free(ciphertext);
    free(shared_secret);

    return (int)total_written;
}

/**
 * Cleanup PQC resources.
 */
static void pqc_cleanup(void)
{
    if (g_kem) {
        OQS_MEM_cleanse(g_secret_key, g_kem->length_secret_key);
        free(g_public_key);
        free(g_secret_key);
        OQS_KEM_free(g_kem);
        g_public_key = NULL;
        g_secret_key = NULL;
        g_kem = NULL;
    }
}

/* ════════════════════════════════════════════════════════════════════════════
 *  FUSE Operations
 * ════════════════════════════════════════════════════════════════════════════ */

static int pqc_getattr(const char *path, struct stat *stbuf,
                        struct fuse_file_info *fi)
{
    (void)fi;
    char phys_path[4096];
    resolve_physical_path(phys_path, sizeof(phys_path), path);

    int res = lstat(phys_path, stbuf);
    if (res == -1)
        return -errno;

    return 0;
}

static int pqc_readdir(const char *path, void *buf, fuse_fill_dir_t filler,
                        off_t offset, struct fuse_file_info *fi,
                        enum fuse_readdir_flags flags)
{
    (void)offset;
    (void)fi;
    (void)flags;

    char phys_path[4096];
    resolve_physical_path(phys_path, sizeof(phys_path), path);

    DIR *dp = opendir(phys_path);
    if (!dp)
        return -errno;

    struct dirent *de;
    while ((de = readdir(dp)) != NULL) {
        struct stat st;
        memset(&st, 0, sizeof(st));
        st.st_ino  = de->d_ino;
        st.st_mode = de->d_type << 12;

        if (filler(buf, de->d_name, &st, 0, 0))
            break;
    }

    closedir(dp);
    return 0;
}

static int pqc_open(const char *path, struct fuse_file_info *fi)
{
    char phys_path[4096];
    resolve_physical_path(phys_path, sizeof(phys_path), path);

    int fd = open(phys_path, fi->flags);
    if (fd == -1)
        return -errno;

    fi->fh = (uint64_t)fd;
    return 0;
}

static int pqc_read(const char *path, char *buf, size_t size, off_t offset,
                     struct fuse_file_info *fi)
{
    (void)path;
    int fd = (int)fi->fh;

    int res = (int)pread(fd, buf, size, offset);
    if (res == -1)
        return -errno;

    return res;
}

static int pqc_create(const char *path, mode_t mode,
                       struct fuse_file_info *fi)
{
    char phys_path[4096];
    resolve_physical_path(phys_path, sizeof(phys_path), path);

    int fd = open(phys_path, fi->flags, mode);
    if (fd == -1)
        return -errno;

    fi->fh = (uint64_t)fd;
    return 0;
}

/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║  WRITE — The core profiling function                                     ║
 * ║                                                                          ║
 * ║  Every write to mnt_secure/ goes through here.  We:                      ║
 * ║    1. Run Kyber-512 KEM encapsulation on the data (CPU-heavy)            ║
 * ║    2. Log the encryption latency                                         ║
 * ║    3. Write the "encrypted" data to storage_physical/                    ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 */
static int pqc_write(const char *path, const char *buf, size_t size,
                      off_t offset, struct fuse_file_info *fi)
{
    (void)path;
    int fd = (int)fi->fh;

    /* ── Encrypt the buffer with PQC (CPU-bound workload) ── */
    double latency_us = 0.0;
    char *enc_buf = malloc(size);
    if (!enc_buf)
        return -ENOMEM;

    int enc_len = pqc_encrypt_buffer(buf, size, enc_buf, size, &latency_us);
    if (enc_len < 0) {
        pqc_log("WRITE ERROR: PQC encryption failed for %zu bytes", size);
        free(enc_buf);
        /* Fallback: write plaintext */
        int res = (int)pwrite(fd, buf, size, offset);
        return (res == -1) ? -errno : res;
    }

    /* ── Log the latency ── */
    double throughput_mbps = (size > 0 && latency_us > 0)
        ? ((double)size / (1024.0 * 1024.0)) / (latency_us / 1e6)
        : 0.0;

    pqc_log("WRITE: %zu bytes | PQC latency: %.2f µs (%.2f ms) | "
            "throughput: %.2f MB/s | KEM ops: %d",
            size, latency_us, latency_us / 1000.0,
            throughput_mbps, KEM_OPS_PER_WRITE);

    /* ── Write encrypted data to physical storage ── */
    double t_io_start = get_time_us();
    int res = (int)pwrite(fd, enc_buf, (size_t)enc_len, offset);
    double t_io_end = get_time_us();

    if (res == -1) {
        free(enc_buf);
        return -errno;
    }

    pqc_log("WRITE I/O: %d bytes written to disk | disk latency: %.2f µs",
            res, t_io_end - t_io_start);

    free(enc_buf);
    return res;
}

static int pqc_truncate(const char *path, off_t size,
                         struct fuse_file_info *fi)
{
    (void)fi;
    char phys_path[4096];
    resolve_physical_path(phys_path, sizeof(phys_path), path);

    int res = truncate(phys_path, size);
    if (res == -1)
        return -errno;

    return 0;
}

static int pqc_unlink(const char *path)
{
    char phys_path[4096];
    resolve_physical_path(phys_path, sizeof(phys_path), path);

    int res = unlink(phys_path);
    if (res == -1)
        return -errno;

    return 0;
}

static int pqc_mkdir(const char *path, mode_t mode)
{
    char phys_path[4096];
    resolve_physical_path(phys_path, sizeof(phys_path), path);

    int res = mkdir(phys_path, mode);
    if (res == -1)
        return -errno;

    return 0;
}

static int pqc_rmdir(const char *path)
{
    char phys_path[4096];
    resolve_physical_path(phys_path, sizeof(phys_path), path);

    int res = rmdir(phys_path);
    if (res == -1)
        return -errno;

    return 0;
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
    char phys_path[4096];
    resolve_physical_path(phys_path, sizeof(phys_path), path);

    int res = utimensat(AT_FDCWD, phys_path, tv, AT_SYMLINK_NOFOLLOW);
    if (res == -1)
        return -errno;

    return 0;
}

static void pqc_destroy(void *private_data)
{
    (void)private_data;
    pqc_log("Shutting down PQC-FUSE filesystem...");
    pqc_cleanup();
    if (g_logfile) {
        fclose(g_logfile);
        g_logfile = NULL;
    }
}

/* ── FUSE operations table ── */
static const struct fuse_operations pqc_oper = {
    .getattr  = pqc_getattr,
    .readdir  = pqc_readdir,
    .open     = pqc_open,
    .read     = pqc_read,
    .write    = pqc_write,
    .create   = pqc_create,
    .truncate = pqc_truncate,
    .unlink   = pqc_unlink,
    .mkdir    = pqc_mkdir,
    .rmdir    = pqc_rmdir,
    .release  = pqc_release,
    .utimens  = pqc_utimens,
    .destroy  = pqc_destroy,
};

/* ════════════════════════════════════════════════════════════════════════════
 *  Main Entry Point
 * ════════════════════════════════════════════════════════════════════════════ */

static void print_usage(const char *progname)
{
    fprintf(stderr,
        "\n"
        "╔══════════════════════════════════════════════════════════════════╗\n"
        "║  PQC-FUSE: Post-Quantum Cryptography FUSE Filesystem          ║\n"
        "║  For Edge AI Bottleneck Profiling (Kyber-512 / ML-KEM)        ║\n"
        "╚══════════════════════════════════════════════════════════════════╝\n"
        "\n"
        "Usage: %s <storage_dir> <mountpoint> [FUSE options]\n"
        "\n"
        "  <storage_dir>  : Physical directory where encrypted files are stored\n"
        "  <mountpoint>   : Virtual mount point (where apps write data)\n"
        "\n"
        "Example:\n"
        "  mkdir -p ~/pqc_edge_workspace/{mnt_secure,storage_physical}\n"
        "  %s ~/pqc_edge_workspace/storage_physical ~/pqc_edge_workspace/mnt_secure -f\n"
        "\n"
        "  -f : Run in foreground (recommended for profiling)\n"
        "\n",
        progname, progname);
}

int main(int argc, char *argv[])
{
    fprintf(stderr,
        "\n"
        "  ┌─────────────────────────────────────────────────────┐\n"
        "  │  PQC-FUSE v0.1 — Kyber-512 Encrypted Filesystem    │\n"
        "  │  Edge AI Bottleneck Profiling Tool                  │\n"
        "  └─────────────────────────────────────────────────────┘\n\n");

    if (argc < 3) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    /* First argument: storage directory (physical backend) */
    char *storage_dir = realpath(argv[1], NULL);
    if (!storage_dir) {
        fprintf(stderr, "[PQC-FUSE] ERROR: Storage directory '%s' does not exist.\n", argv[1]);
        fprintf(stderr, "[PQC-FUSE] Create it with: mkdir -p %s\n", argv[1]);
        return EXIT_FAILURE;
    }
    strncpy(g_storage_dir, storage_dir, sizeof(g_storage_dir) - 1);
    free(storage_dir);

    /* ── Open log file ── */
    char log_path[4096];
    snprintf(log_path, sizeof(log_path), "%s/../%s", g_storage_dir, LOG_FILENAME);
    g_logfile = fopen(log_path, "a");
    if (!g_logfile) {
        fprintf(stderr, "[PQC-FUSE] WARNING: Cannot open log file '%s': %s\n",
                log_path, strerror(errno));
        /* Try current directory */
        g_logfile = fopen(LOG_FILENAME, "a");
    }

    pqc_log("════════════════════════════════════════════════════════════");
    pqc_log("  PQC-FUSE Session Started");
    pqc_log("  Storage dir : %s", g_storage_dir);
    pqc_log("  Mount point : %s", argv[2]);
    pqc_log("════════════════════════════════════════════════════════════");

    /* ── Initialize PQC subsystem ── */
    if (pqc_init() != 0) {
        fprintf(stderr, "[PQC-FUSE] FATAL: PQC initialization failed.\n");
        return EXIT_FAILURE;
    }

    /* ── Shift argv to remove storage_dir argument before passing to FUSE ── */
    /* Original: argv[0] argv[1=storage] argv[2=mount] argv[3..N=fuse_opts] */
    /* Shifted:  argv[0] argv[2=mount]   argv[3..N=fuse_opts]               */
    int fuse_argc = argc - 1;
    char **fuse_argv = malloc(sizeof(char *) * (size_t)fuse_argc);
    fuse_argv[0] = argv[0];
    for (int i = 2; i < argc; i++) {
        fuse_argv[i - 1] = argv[i];
    }

    pqc_log("Starting FUSE main loop...");
    int ret = fuse_main(fuse_argc, fuse_argv, &pqc_oper, NULL);

    free(fuse_argv);
    pqc_cleanup();

    if (g_logfile) {
        fclose(g_logfile);
        g_logfile = NULL;
    }

    return ret;
}
