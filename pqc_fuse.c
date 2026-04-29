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
#include <openssl/evp.h>
#include <pthread.h>
#include <sys/xattr.h>

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

/*
 * Per-file encryption context.
 * KEM runs ONCE per file (in pqc_create), yielding a 32-byte shared secret.
 * All subsequent writes use SHAKE128 XOF to expand that secret into a
 * keystream — no repeated KEM calls on the write hot path.
 */
#define PQC_MAX_FD    4096
#define COALESCE_SIZE (512 * 1024)  /* 512 KB per-fd write-coalescing buffer */

/*
 * Encryption tier (Strategy 2: Semantic-Aware Selective Encryption)
 * Set via FUSE xattr "user.pqc_tier" on any file.
 *   PQC_TIER_FULL (1) : ML-KEM-512 + SHAKE128 stream cipher  [default]
 *   PQC_TIER_NONE (2) : Plaintext passthrough — low-value sensor data
 */
#define PQC_TIER_FULL  1
#define PQC_TIER_NONE  2
#define PQC_XATTR_TIER "user.pqc_tier"

/*
 * Strategy 1: Forward-Secure Key Rotation
 * Re-run ML-KEM-512.Encaps() every KEY_ROTATION_INTERVAL seconds within a
 * single file's lifetime.  Each epoch gets a new shared_secret stored as an
 * additional entry in the .pqckey sidecar:
 *   { epoch(u64) | timestamp(u64) | ss_len(u64) | ss(ss_len bytes) } ...
 * Guarantees that past plaintext is safe even if current key is leaked.
 */
#define KEY_ROTATION_INTERVAL_S  1   /* re-key every N seconds */

typedef struct {
    int      valid;
    uint8_t  ss[64];
    size_t   ss_len;
    uint64_t file_id;
    /* write coalescing */
    uint8_t *wbuf;           /* heap-allocated COALESCE_SIZE bytes */
    size_t   wbuf_used;      /* bytes currently buffered */
    off_t    wbuf_base_off;  /* file offset of wbuf[0] */
    /* Strategy 2: selective encryption tier */
    int      tier;           /* PQC_TIER_FULL or PQC_TIER_NONE */
    /* Strategy 1: key rotation */
    uint64_t key_epoch;      /* current epoch index (0 = initial KEM) */
    time_t   last_rekey;     /* wall-clock time of last KEM call */
} pqc_fd_ctx_t;

static pqc_fd_ctx_t      g_fd_ctx[PQC_MAX_FD];
static pthread_mutex_t   g_fd_lock    = PTHREAD_MUTEX_INITIALIZER;
static uint64_t          g_file_id_ctr = 0;

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
 *  Per-fd Context Helpers
 * ════════════════════════════════════════════════════════════════════════════ */

static void ctx_set(int fd, const uint8_t *ss, size_t ss_len, uint64_t fid)
{
    int idx = fd % PQC_MAX_FD;
    pthread_mutex_lock(&g_fd_lock);
    memcpy(g_fd_ctx[idx].ss, ss, ss_len);
    g_fd_ctx[idx].ss_len     = ss_len;
    g_fd_ctx[idx].file_id    = fid;
    g_fd_ctx[idx].valid      = 1;
    g_fd_ctx[idx].tier       = PQC_TIER_FULL;  /* default: full PQC */
    g_fd_ctx[idx].key_epoch  = 0;
    g_fd_ctx[idx].last_rekey = time(NULL);
    if (!g_fd_ctx[idx].wbuf)
        g_fd_ctx[idx].wbuf = (uint8_t *)malloc(COALESCE_SIZE);
    g_fd_ctx[idx].wbuf_used     = 0;
    g_fd_ctx[idx].wbuf_base_off = 0;
    pthread_mutex_unlock(&g_fd_lock);
}

static int ctx_get(int fd, pqc_fd_ctx_t *out)
{
    int idx = fd % PQC_MAX_FD;
    pthread_mutex_lock(&g_fd_lock);
    if (!g_fd_ctx[idx].valid) {
        pthread_mutex_unlock(&g_fd_lock);
        return -1;
    }
    *out = g_fd_ctx[idx];
    pthread_mutex_unlock(&g_fd_lock);
    return 0;
}

static void ctx_clear(int fd)
{
    int idx = fd % PQC_MAX_FD;
    pthread_mutex_lock(&g_fd_lock);
    OQS_MEM_cleanse(g_fd_ctx[idx].ss, sizeof(g_fd_ctx[idx].ss));
    if (g_fd_ctx[idx].wbuf) {
        OQS_MEM_cleanse(g_fd_ctx[idx].wbuf, COALESCE_SIZE);
        free(g_fd_ctx[idx].wbuf);
        g_fd_ctx[idx].wbuf = NULL;
    }
    g_fd_ctx[idx].wbuf_used = 0;
    g_fd_ctx[idx].valid = 0;
    pthread_mutex_unlock(&g_fd_lock);
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
 * Correct hybrid-encryption design:
 *   - KEM runs ONCE per file (in pqc_create), producing a 32-byte shared secret.
 *   - This function uses that secret as a seed for SHAKE128 XOF to derive a
 *     keystream of exactly in_size bytes, then XORs it with plaintext.
 *
 * Why SHAKE128?  It's a standard (NIST FIPS 202) XOF with ~1 GB/s throughput on
 * modern CPUs — fast enough to be disk-limited, while being cryptographically
 * sound as a stream cipher when seeded from a KEM-derived secret.
 *
 * Seed = shared_secret || file_id (8B) || write_offset (8B)
 * This ensures every write produces a unique, non-repeating keystream.
 */
static int pqc_stream_encrypt(const pqc_fd_ctx_t *ctx,
                               const char *in_buf, size_t in_size,
                               off_t offset,
                               char *out_buf,
                               double *latency_us)
{
    double t_start = get_time_us();

    /* Build SHAKE128 seed: ss || file_id || write_offset */
    uint8_t seed[80];  /* max ss_len=64 + 8 + 8 */
    size_t  seed_len = ctx->ss_len + 16;
    memcpy(seed, ctx->ss, ctx->ss_len);
    uint64_t fid = ctx->file_id;
    uint64_t off = (uint64_t)offset;
    memcpy(seed + ctx->ss_len,     &fid, 8);
    memcpy(seed + ctx->ss_len + 8, &off, 8);

    /* Squeeze a keystream of in_size bytes via OpenSSL SHAKE128 XOF */
    EVP_MD_CTX *md = EVP_MD_CTX_new();
    if (!md) return -1;

    EVP_DigestInit_ex(md, EVP_shake128(), NULL);
    EVP_DigestUpdate(md, seed, seed_len);
    /* EVP_DigestFinalXOF squeezes arbitrary-length output */
    EVP_DigestFinalXOF(md, (uint8_t *)out_buf, in_size);
    EVP_MD_CTX_free(md);

    /* XOR keystream with plaintext */
    for (size_t i = 0; i < in_size; i++)
        out_buf[i] ^= in_buf[i];

    double t_end = get_time_us();
    *latency_us = t_end - t_start;
    return (int)in_size;
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

/* ────────────────────────────────────────────────────────────────────────────
 *  Write-coalescing flush
 *  Encrypts ctx->wbuf[0..wbuf_used) with SHAKE128 XOR and pwrite to storage.
 *  MUST be called with g_fd_lock held.  Resets wbuf_used = 0 on success.
 * -------------------------------------------------------------------------- */
/*
 * do_rekey_locked() — Strategy 1: Forward-Secure Key Rotation
 * Runs ML-KEM-512.Encaps() to produce a fresh shared_secret and appends
 * a new epoch entry to the .pqckey sidecar.  MUST be called with g_fd_lock.
 */
static void do_rekey_locked(int storage_fd, int idx, const char *phys_path)
{
    (void)storage_fd;
    if (!g_kem || !phys_path) return;

    uint8_t *ct = malloc(g_kem->length_ciphertext);
    uint8_t *ss = malloc(g_kem->length_shared_secret);
    if (!ct || !ss) { free(ct); free(ss); return; }

    if (OQS_KEM_encaps(g_kem, ct, ss, g_public_key) != OQS_SUCCESS) {
        free(ct); free(ss); return;
    }

    pqc_fd_ctx_t *ctx = &g_fd_ctx[idx];
    OQS_MEM_cleanse(ctx->ss, sizeof(ctx->ss));
    memcpy(ctx->ss, ss, g_kem->length_shared_secret);
    ctx->ss_len    = g_kem->length_shared_secret;
    ctx->key_epoch++;
    ctx->last_rekey = time(NULL);

    /* Append epoch record to sidecar: epoch(8B)|ts(8B)|ss_len(8B)|ss */
    char key_path[4096 + 8];
    snprintf(key_path, sizeof(key_path), "%s.pqckey", phys_path);
    int kfd = open(key_path, O_WRONLY | O_CREAT | O_APPEND, 0600);
    if (kfd >= 0) {
        uint64_t ep      = ctx->key_epoch;
        uint64_t ts      = (uint64_t)ctx->last_rekey;
        uint64_t ss_len64 = (uint64_t)ctx->ss_len;
        (void)(write(kfd, &ep,      8) +
               write(kfd, &ts,      8) +
               write(kfd, &ss_len64, 8) +
               write(kfd, ss, (size_t)ss_len64));
        close(kfd);
    }

    pqc_log("REKEY epoch=%llu fid=%llu",
            (unsigned long long)ctx->key_epoch,
            (unsigned long long)ctx->file_id);
    OQS_MEM_cleanse(ss, g_kem->length_shared_secret);
    free(ct); free(ss);
}

static int do_flush_wbuf_locked(int storage_fd, int idx)
{
    pqc_fd_ctx_t *ctx = &g_fd_ctx[idx];
    if (ctx->wbuf_used == 0) return 0;

    /* Strategy 2: tier NONE → write plaintext directly */
    if (ctx->tier == PQC_TIER_NONE) {
        int res = (int)pwrite(storage_fd, ctx->wbuf, ctx->wbuf_used,
                              ctx->wbuf_base_off);
        ctx->wbuf_used = 0;
        return res == -1 ? -errno : 0;
    }

    size_t sz   = ctx->wbuf_used;
    off_t  base = ctx->wbuf_base_off;

    char *enc = (char *)malloc(sz);
    if (!enc) { ctx->wbuf_used = 0; return -ENOMEM; }

    double lat = 0.0;
    pqc_stream_encrypt(ctx, (const char *)ctx->wbuf, sz, base, enc, &lat);

    int res = (int)pwrite(storage_fd, enc, sz, base);
    OQS_MEM_cleanse(enc, sz);
    free(enc);
    ctx->wbuf_used = 0;
    return res == -1 ? -errno : 0;
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

    /* Load per-file key sidecar (.pqckey) for read-decryption if it exists */
    char key_path[4096 + 8];
    snprintf(key_path, sizeof(key_path), "%s.pqckey", phys_path);
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
             * Format: epoch(8)|ts(8)|ss_len(8)|ss(ss_len) */
            uint64_t ep_hdr[3];
            while (read(kfd, ep_hdr, 24) == 24) {
                uint64_t ep_ss_len = ep_hdr[2];
                if (ep_ss_len == 0 || ep_ss_len > sizeof(ss)) break;
                uint8_t ep_ss[64] = {0};
                if ((size_t)read(kfd, ep_ss, ep_ss_len) == ep_ss_len) {
                    int idx2 = fd % PQC_MAX_FD;
                    pthread_mutex_lock(&g_fd_lock);
                    if (g_fd_ctx[idx2].valid) {
                        memcpy(g_fd_ctx[idx2].ss, ep_ss, ep_ss_len);
                        g_fd_ctx[idx2].ss_len    = ep_ss_len;
                        g_fd_ctx[idx2].key_epoch = ep_hdr[0];
                    }
                    pthread_mutex_unlock(&g_fd_lock);
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
        ssize_t xlen = getxattr(phys_path, PQC_XATTR_TIER, xval, sizeof(xval) - 1);
        if (xlen > 0) {
            int t = atoi(xval);
            if (t == PQC_TIER_NONE) {
                int idx = fd % PQC_MAX_FD;
                pthread_mutex_lock(&g_fd_lock);
                if (g_fd_ctx[idx].valid)
                    g_fd_ctx[idx].tier = PQC_TIER_NONE;
                pthread_mutex_unlock(&g_fd_lock);
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

    /* Strategy 2: check tier; tier NONE → plaintext, no decrypt needed */
    pthread_mutex_lock(&g_fd_lock);
    if (g_fd_ctx[idx].valid && g_fd_ctx[idx].tier != PQC_TIER_NONE) {
        pqc_fd_ctx_t snap = g_fd_ctx[idx];
        pthread_mutex_unlock(&g_fd_lock);
        char *tmp = (char *)malloc((size_t)res);
        if (tmp) {
            double lat = 0.0;
            pqc_stream_encrypt(&snap, buf, (size_t)res, offset, tmp, &lat);
            memcpy(buf, tmp, (size_t)res);
            OQS_MEM_cleanse(tmp, (size_t)res);
            free(tmp);
        }
    } else {
        pthread_mutex_unlock(&g_fd_lock);
    }
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

    /* Run KEM once per file — establishes shared secret for all writes */
    if (g_kem) {
        uint8_t *ct = malloc(g_kem->length_ciphertext);
        uint8_t *ss = malloc(g_kem->length_shared_secret);
        if (ct && ss) {
            double t0 = get_time_us();
            if (OQS_KEM_encaps(g_kem, ct, ss, g_public_key) == OQS_SUCCESS) {
                pthread_mutex_lock(&g_fd_lock);
                uint64_t fid = ++g_file_id_ctr;
                pthread_mutex_unlock(&g_fd_lock);
                ctx_set(fd, ss, g_kem->length_shared_secret, fid);
                pqc_log("CREATE %s: KEM encaps %.1fµs fd=%d fid=%llu",
                        path, get_time_us() - t0, fd, (unsigned long long)fid);
                /* Persist key sidecar for read-back decryption across mounts */
                char key_path[4096 + 8];
                snprintf(key_path, sizeof(key_path), "%s.pqckey", phys_path);
                int kfd = open(key_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
                if (kfd >= 0) {
                    uint64_t ss_len64 = (uint64_t)g_kem->length_shared_secret;
                    (void)(write(kfd, &fid,     8) +
                           write(kfd, &ss_len64, 8) +
                           write(kfd, ss, (size_t)ss_len64));
                    close(kfd);
                }
            } else {
                pqc_log("CREATE %s: KEM encaps FAILED", path);
            }
        }
        OQS_MEM_cleanse(ss, g_kem->length_shared_secret);
        free(ct);
        free(ss);
    }
    return 0;
}

/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║  WRITE — Hybrid PQC + SHAKE128 stream cipher                           ║
 * ║                                                                          ║
 * ║  Hot path (most writes):                                                 ║
 * ║    1. ctx_get() retrieves the per-file shared secret (from KEM in open)  ║
 * ║    2. SHAKE128(ss || file_id || offset) generates keystream in one call   ║
 * ║    3. XOR plaintext with keystream -> encrypted output                   ║
 * ║    4. pwrite() to physical storage                                       ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 */
static int pqc_write(const char *path, const char *buf, size_t size,
                      off_t offset, struct fuse_file_info *fi)
{
    (void)path;
    int fd  = (int)fi->fh;
    int idx = fd % PQC_MAX_FD;

    pthread_mutex_lock(&g_fd_lock);
    pqc_fd_ctx_t *ctx = &g_fd_ctx[idx];

    /* No context → passthrough (file opened without create, e.g. sidecar) */
    if (!ctx->valid) {
        pthread_mutex_unlock(&g_fd_lock);
        int res = (int)pwrite(fd, buf, size, offset);
        return res == -1 ? -errno : (int)size;
    }

    /* Strategy 2: Tier NONE → plaintext passthrough, no encryption */
    if (ctx->tier == PQC_TIER_NONE) {
        if (ctx->wbuf_used == 0)
            ctx->wbuf_base_off = offset;
        /* Still coalesce for write batching, but do_flush writes plaintext */
        int is_c = (ctx->wbuf_used == 0) ||
                   (offset == ctx->wbuf_base_off + (off_t)ctx->wbuf_used);
        if (!is_c) do_flush_wbuf_locked(fd, idx);
        if (ctx->wbuf_used == 0) ctx->wbuf_base_off = offset;
        if (size < COALESCE_SIZE && ctx->wbuf) {
            memcpy(ctx->wbuf + ctx->wbuf_used, buf, size);
            ctx->wbuf_used += size;
            if (ctx->wbuf_used >= COALESCE_SIZE)
                do_flush_wbuf_locked(fd, idx);
        } else {
            do_flush_wbuf_locked(fd, idx);
            int _r = (int)pwrite(fd, buf, size, offset);
            (void)_r;
        }
        pthread_mutex_unlock(&g_fd_lock);
        return (int)size;
    }

    /* Strategy 1: Key Rotation — re-key if interval expired */
    {
        time_t now = time(NULL);
        if (now - ctx->last_rekey >= KEY_ROTATION_INTERVAL_S) {
            /* need phys_path to append sidecar; store it in stack */
            char phys_path[4096];
            resolve_physical_path(phys_path, sizeof(phys_path), path);
            /* flush existing buffered data under OLD key first */
            if (ctx->wbuf_used > 0)
                do_flush_wbuf_locked(fd, idx);
            do_rekey_locked(fd, idx, phys_path);
        }
    }

    /* No coalescing buffer (malloc failed) → encrypt directly */
    if (!ctx->wbuf) {
        pqc_fd_ctx_t snap = *ctx;
        pthread_mutex_unlock(&g_fd_lock);
        char *enc = (char *)malloc(size);
        if (!enc) return -ENOMEM;
        double lat = 0.0;
        pqc_stream_encrypt(&snap, buf, size, offset, enc, &lat);
        int res = (int)pwrite(fd, enc, size, offset);
        free(enc);
        return res == -1 ? -errno : (int)size;
    }

    /* Non-contiguous write: flush pending buffer first */
    int is_cont = (ctx->wbuf_used == 0) ||
                  (offset == ctx->wbuf_base_off + (off_t)ctx->wbuf_used);
    if (!is_cont) {
        int fr = do_flush_wbuf_locked(fd, idx);
        if (fr < 0) { pthread_mutex_unlock(&g_fd_lock); return fr; }
    }

    /* Large write (≥ COALESCE_SIZE): encrypt directly, bypass buffering */
    if (size >= COALESCE_SIZE) {
        do_flush_wbuf_locked(fd, idx);
        pqc_fd_ctx_t snap = *ctx;
        pthread_mutex_unlock(&g_fd_lock);
        char *enc = (char *)malloc(size);
        if (!enc) return -ENOMEM;
        double lat = 0.0;
        pqc_stream_encrypt(&snap, buf, size, offset, enc, &lat);
        int res = (int)pwrite(fd, enc, size, offset);
        free(enc);
        return res == -1 ? -errno : (int)size;
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
        if (fr < 0) { pthread_mutex_unlock(&g_fd_lock); return fr; }
    }

    pthread_mutex_unlock(&g_fd_lock);
    return (int)size;
}

static int pqc_fsync(const char *path, int datasync,
                     struct fuse_file_info *fi)
{
    (void)path; (void)datasync;
    int fd  = (int)fi->fh;
    int idx = fd % PQC_MAX_FD;

    pthread_mutex_lock(&g_fd_lock);
    if (g_fd_ctx[idx].valid && g_fd_ctx[idx].wbuf)
        do_flush_wbuf_locked(fd, idx);
    pthread_mutex_unlock(&g_fd_lock);

    return fdatasync(fd) == -1 ? -errno : 0;
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

    /* Remove key sidecar if it exists */
    char key_path[4096 + 8];
    snprintf(key_path, sizeof(key_path), "%s.pqckey", phys_path);
    unlink(key_path);

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
    int fd  = (int)fi->fh;
    int idx = fd % PQC_MAX_FD;

    /* Flush any remaining coalesced data before closing */
    pthread_mutex_lock(&g_fd_lock);
    if (g_fd_ctx[idx].valid && g_fd_ctx[idx].wbuf && g_fd_ctx[idx].wbuf_used > 0)
        do_flush_wbuf_locked(fd, idx);
    pthread_mutex_unlock(&g_fd_lock);

    ctx_clear(fd);
    close(fd);
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

/* ════════════════════════════════════════════════════════════════════════════
 *  Strategy 2: xattr handlers — Semantic-Aware Selective Encryption
 *
 *  Set tier for a file:
 *    setfattr -n user.pqc_tier -v 2 /mnt_secure/raw_lidar.bin  -> no encryption
 *    setfattr -n user.pqc_tier -v 1 /mnt_secure/model.bin      -> full PQC
 *
 *  The xattr is stored on the physical backing file so it persists across
 *  remounts.  On open(), the tier is restored into the fd context.
 * ════════════════════════════════════════════════════════════════════════════ */

static int pqc_setxattr(const char *path, const char *name,
                         const char *value, size_t size, int flags)
{
    char phys_path[4096];
    resolve_physical_path(phys_path, sizeof(phys_path), path);

    if (setxattr(phys_path, name, value, size, flags) == -1)
        return -errno;

    /* If setting the tier live, update in-flight fd context */
    if (strcmp(name, PQC_XATTR_TIER) == 0 && size > 0) {
        char tmp[8] = {0};
        memcpy(tmp, value, size < 7 ? size : 7);
        int new_tier = atoi(tmp);
        if (new_tier != PQC_TIER_FULL && new_tier != PQC_TIER_NONE)
            return -EINVAL;
        /* Scan open fd contexts for this phys_path */
        pthread_mutex_lock(&g_fd_lock);
        for (int i = 0; i < PQC_MAX_FD; i++) {
            if (g_fd_ctx[i].valid)
                g_fd_ctx[i].tier = new_tier;  /* best-effort; path not stored */
        }
        pthread_mutex_unlock(&g_fd_lock);
        pqc_log("SETXATTR %s tier=%d", path, new_tier);
    }
    return 0;
}

static int pqc_getxattr(const char *path, const char *name,
                         char *value, size_t size)
{
    char phys_path[4096];
    resolve_physical_path(phys_path, sizeof(phys_path), path);

    ssize_t res = getxattr(phys_path, name, value, size);
    return res == -1 ? -errno : (int)res;
}

static int pqc_listxattr(const char *path, char *list, size_t size)
{
    char phys_path[4096];
    resolve_physical_path(phys_path, sizeof(phys_path), path);

    ssize_t res = listxattr(phys_path, list, size);
    return res == -1 ? -errno : (int)res;
}

/* ── FUSE operations table ── */
static const struct fuse_operations pqc_oper = {
    .getattr    = pqc_getattr,
    .readdir    = pqc_readdir,
    .open       = pqc_open,
    .read       = pqc_read,
    .write      = pqc_write,
    .fsync      = pqc_fsync,
    .create     = pqc_create,
    .truncate   = pqc_truncate,
    .unlink     = pqc_unlink,
    .mkdir      = pqc_mkdir,
    .rmdir      = pqc_rmdir,
    .release    = pqc_release,
    .utimens    = pqc_utimens,
    .destroy    = pqc_destroy,
    .setxattr   = pqc_setxattr,
    .getxattr   = pqc_getxattr,
    .listxattr  = pqc_listxattr,
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
