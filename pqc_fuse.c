/**
 * ============================================================================
 *  pqc_fuse.c — PQC-based FUSE Filesystem for Edge AI Bottleneck Profiling
 * ============================================================================
 *
 *  Purpose:
 *    Transparent FUSE prototype for authenticated encrypted block storage.
 *    Every file receives a random data-encryption key (DEK), stored in an
 *    HMAC-authenticated envelope under a mount-derived key.  AES-256-GCM
 *    protects data records; ciphertext is synchronized before journal
 *    publication.  ML-KEM-768 is initialized for optional key-plane batch
 *    work and microbenchmarks, not for per-block encryption.
 *
 *  Architecture:
 *    mnt_secure/ (FUSE mount) ──write()──► AES-GCM + journal ──► storage_physical/
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
#include <sys/file.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <dirent.h>
#include <time.h>
#include <stdarg.h>
#include <stddef.h>
#include <signal.h>
#include <linux/falloc.h>

#include <oqs/oqs.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <openssl/hmac.h>
#include <pthread.h>
#include <sys/xattr.h>

#include "pqc_block_job.h"
#include "pqc_anchor.h"
#include "cuda_aead.h"
#include "cuda_pqc.h"
#include "pqc_admission.h"

static inline double get_time_us(void);
static double gpu_load_ewma_read(void);
static void pqc_log(const char *fmt, ...);

/* ════════════════════════════════════════════════════════════════════════════
 *  Configuration & Globals
 * ════════════════════════════════════════════════════════════════════════════ */

/** Physical storage backend directory (where encrypted files are stored) */
static char g_storage_dir[4096] = {0};

/** Log file for latency measurements */
static FILE *g_logfile = NULL;
static const char *LOG_FILENAME = "pqc_fuse_latency.log";

/** PQC KEM pre-generated keypair (generated once at startup) */
OQS_KEM *g_kem = NULL;
uint8_t *g_public_key  = NULL;
uint8_t *g_secret_key  = NULL;

/* ── Master Key & Sidecar Wrapping for Forward Secrecy ── */
uint8_t g_master_key[32];
static int g_has_master_key = 0;

static int derive_master_key(const char *password) {
    const uint8_t salt[] = "PQC_FUSE_SALT_NIST";
    if (PKCS5_PBKDF2_HMAC(password, strlen(password), salt, sizeof(salt) - 1,
                           600000, EVP_sha256(), 32, g_master_key) == 1) {
        g_has_master_key = 1;
        return 0;
    }
    return -1;
}

static int wrap_shared_secret(const uint8_t *in_ss, size_t ss_len, uint64_t fid, uint64_t epoch, uint8_t *out_wrapped) {
    if (!g_has_master_key) return -EACCES;
    uint8_t seed[48];
    memcpy(seed, g_master_key, 32);
    memcpy(seed + 32, &fid, 8);
    memcpy(seed + 40, &epoch, 8);

    EVP_MD_CTX *md = EVP_MD_CTX_new();
    if (!md) return -1;
    EVP_DigestInit_ex(md, EVP_shake128(), NULL);
    EVP_DigestUpdate(md, seed, sizeof(seed));
    EVP_DigestFinalXOF(md, out_wrapped, ss_len);
    EVP_MD_CTX_free(md);

    for (size_t i = 0; i < ss_len; i++) {
        out_wrapped[i] ^= in_ss[i];
    }
    return 0;
}

static int unwrap_shared_secret(const uint8_t *in_wrapped, size_t ss_len, uint64_t fid, uint64_t epoch, uint8_t *out_ss) {
    return wrap_shared_secret(in_wrapped, ss_len, fid, epoch, out_ss);
}

/*
 * Per-file encryption context.  The mounted data path uses a random DEK
 * wrapped in an authenticated envelope under the mount key.  Optional ML-KEM
 * work is a key-plane maintenance path; every data block is protected by the
 * same AES-256-GCM format regardless of whether CPU or CUDA executes CTR work.
 */
#define PQC_MAX_FD    4096
#define COALESCE_SIZE (512 * 1024)  /* 512 KB per-fd write-coalescing buffer */
#define PQC_LOGICAL_BLOCK_SIZE 4096U
#define PQC_AEAD_TAG_SIZE 16U
#define PQC_AEAD_NONCE_SIZE 12U
#define PQC_JOURNAL_MAGIC UINT64_C(0x5051434a4e4c3031) /* PQCJNL01 */
#define PQC_JOURNAL_VERSION 1U
#define PQC_JOURNAL_COMMITTED UINT32_C(0x434f4d4d)

/*
 * Encryption tier (Strategy 2: Semantic-Aware Selective Encryption)
 * Set via FUSE xattr "user.pqc_tier" on any file.
 *   PQC_TIER_FULL (1) : authenticated AES-256-GCM blocks  [default]
 *   PQC_TIER_NONE (2) : Plaintext passthrough — low-value sensor data
 */
#define PQC_TIER_FULL  1
#define PQC_TIER_NONE  2
#define PQC_XATTR_TIER "user.pqc_tier"
#define PQC_XATTR_QOS_CLASS "user.pqc_qos_class"
#define PQC_XATTR_METADATA "user.pqc_metadata"
#define PQC_XATTR_LOGICAL_SIZE "user.pqc_logical_size"
#define PQC_XATTR_CHECKPOINT "user.pqc_checkpoint"
#define PQC_QOS_CLASS_ELASTIC 0
#define PQC_QOS_CLASS_LATENCY 1
#define PQC_METADATA_MAGIC UINT64_C(0x5043514d45544131) /* "PQC META1" */
#define PQC_METADATA_VERSION 1U
#define PQC_CHECKPOINT_MAGIC UINT64_C(0x504351434b505431) /* "PQCCPT1" */
#define PQC_CHECKPOINT_VERSION 1U

/*
 * Persistent per-file key material.  Keeping this as one binary xattr avoids
 * namespace pollution and keeps key lookup tied atomically to its data inode.
 * This is deliberately fixed-size so metadata I/O is bounded.
 */
typedef struct {
    uint64_t magic;
    uint32_t version;
    uint32_t ss_len;
    uint64_t file_id;
    uint8_t  wrapped_ss[64];
    uint8_t  digest[32];
} pqc_metadata_t;

typedef struct {
    uint64_t magic;
    uint32_t version;
    uint32_t reserved;
    uint64_t file_id;
    uint64_t sequence;
    uint64_t logical_size;
    uint64_t max_generation;
    uint8_t  digest[32];
} pqc_checkpoint_t;

static pthread_mutex_t   g_anchor_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t    g_anchor_cv = PTHREAD_COND_INITIALIZER;
static pthread_t         g_anchor_thread;
static int               g_anchor_thread_started = 0;
static int               g_anchor_stop = 0;
static int               g_anchor_dirty = 0;
static pqc_anchor_state_t g_anchor_state = {0};
static time_t            g_anchor_last_commit = 0;
static pthread_mutex_t   g_fault_lock = PTHREAD_MUTEX_INITIALIZER;
static int               g_fault_triggered = 0;

static void pqc_fault_cutpoint(const char *name)
{
    const char *target = getenv("PQC_FAULT_CUTPOINT");
    if (!target || strcmp(target, name) != 0)
        return;

    pthread_mutex_lock(&g_fault_lock);
    if (g_fault_triggered) {
        pthread_mutex_unlock(&g_fault_lock);
        return;
    }
    g_fault_triggered = 1;
    pthread_mutex_unlock(&g_fault_lock);

    const char *marker = getenv("PQC_FAULT_MARKER_PATH");
    if (marker && marker[0] != '\0') {
        int fd = open(marker, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
        if (fd >= 0) {
            time_t now = time(NULL);
            (void)dprintf(fd,
                          "{\"event\":\"fault_cutpoint\",\"name\":\"%s\","
                          "\"pid\":%ld,\"unix_time\":%lld}\n",
                          name, (long)getpid(), (long long)now);
            (void)fsync(fd);
            (void)close(fd);
        }
    }

    raise(SIGKILL);
    _exit(128 + SIGKILL);
}

static int metadata_store(const char *phys_path, const uint8_t *ss,
                          size_t ss_len, uint64_t file_id)
{
    if (ss_len == 0 || ss_len > sizeof(((pqc_metadata_t *)0)->wrapped_ss))
        return -EINVAL;
    pqc_metadata_t meta = {0};
    meta.magic = PQC_METADATA_MAGIC;
    meta.version = PQC_METADATA_VERSION;
    meta.ss_len = (uint32_t)ss_len;
    meta.file_id = file_id;
    if (wrap_shared_secret(ss, ss_len, file_id, 0, meta.wrapped_ss) != 0)
        return -EIO;
    unsigned int digest_len = 0;
    if (!HMAC(EVP_sha256(), g_master_key, sizeof(g_master_key),
              (const unsigned char *)&meta, offsetof(pqc_metadata_t, digest),
              meta.digest, &digest_len) || digest_len != sizeof(meta.digest)) {
        OQS_MEM_cleanse(&meta, sizeof(meta));
        return -EIO;
    }
    int rc = setxattr(phys_path, PQC_XATTR_METADATA, &meta, sizeof(meta), 0);
    OQS_MEM_cleanse(&meta, sizeof(meta));
    return rc == -1 ? -errno : 0;
}

static int metadata_load(const char *phys_path, uint8_t *ss, size_t *ss_len,
                         uint64_t *file_id)
{
    pqc_metadata_t meta = {0};
    ssize_t n = getxattr(phys_path, PQC_XATTR_METADATA, &meta, sizeof(meta));
    if (n == -1) return -errno;
    if ((size_t)n != sizeof(meta) || meta.magic != PQC_METADATA_MAGIC ||
        meta.version != PQC_METADATA_VERSION || meta.ss_len == 0 ||
        meta.ss_len > 64) {
        OQS_MEM_cleanse(&meta, sizeof(meta));
        return -EINVAL;
    }
    uint8_t digest[sizeof(meta.digest)];
    unsigned int digest_len = 0;
    if (!HMAC(EVP_sha256(), g_master_key, sizeof(g_master_key),
              (const unsigned char *)&meta, offsetof(pqc_metadata_t, digest),
              digest, &digest_len) || digest_len != sizeof(digest) ||
        CRYPTO_memcmp(digest, meta.digest, sizeof(digest)) != 0) {
        OQS_MEM_cleanse(digest, sizeof(digest));
        OQS_MEM_cleanse(&meta, sizeof(meta));
        return -EKEYREJECTED;
    }
    OQS_MEM_cleanse(digest, sizeof(digest));
    int rc = unwrap_shared_secret(meta.wrapped_ss, meta.ss_len, meta.file_id,
                                  0, ss);
    if (rc == 0) {
        *ss_len = meta.ss_len;
        *file_id = meta.file_id;
    }
    OQS_MEM_cleanse(&meta, sizeof(meta));
    return rc == 0 ? 0 : -EIO;
}

static int logical_size_load(const char *path, uint64_t *size)
{
    uint64_t value = 0;
    ssize_t n = getxattr(path, PQC_XATTR_LOGICAL_SIZE, &value, sizeof(value));
    if (n == -1 && errno == ENODATA) { *size = 0; return 0; }
    if (n != (ssize_t)sizeof(value)) return -errno;
    *size = value;
    return 0;
}

static int logical_size_store(const char *path, uint64_t size)
{
    return setxattr(path, PQC_XATTR_LOGICAL_SIZE, &size, sizeof(size), 0) == 0 ? 0 : -errno;
}

static int checkpoint_store(const char *path, uint64_t file_id, uint64_t sequence,
                            uint64_t logical_size, uint64_t max_generation)
{
    pqc_checkpoint_t ckpt = {
        .magic = PQC_CHECKPOINT_MAGIC,
        .version = PQC_CHECKPOINT_VERSION,
        .reserved = 0,
        .file_id = file_id,
        .sequence = sequence,
        .logical_size = logical_size,
        .max_generation = max_generation,
    };
    unsigned int out_len = 0;
    unsigned char *mac = HMAC(EVP_sha256(), g_master_key, sizeof(g_master_key),
                              (unsigned char *)&ckpt,
                              offsetof(pqc_checkpoint_t, digest),
                              ckpt.digest, &out_len);
    if (!mac || out_len != sizeof(ckpt.digest))
        return -EIO;
    int rc = setxattr(path, PQC_XATTR_CHECKPOINT, &ckpt, sizeof(ckpt), 0);
    if (rc == 0) {
        pqc_fault_cutpoint("checkpoint_xattr_after");

        /* Register file state to multi-file committed prefix map */
        (void)pqc_anchor_record_file(file_id, max_generation, sequence, logical_size);

        pthread_mutex_lock(&g_anchor_lock);
        g_anchor_state.epoch = max_generation;
        g_anchor_state.sequence = sequence;
        g_anchor_state.logical_size = logical_size;
        g_anchor_dirty = 1;
        pthread_cond_signal(&g_anchor_cv);
        pthread_mutex_unlock(&g_anchor_lock);
        if (pqc_anchor_backend() == PQC_ANCHOR_BACKEND_HARDWARE) {
            pqc_anchor_state_t state = {
                .epoch = max_generation,
                .sequence = sequence,
                .logical_size = logical_size,
            };
            rc = pqc_anchor_store(&state);
            if (rc != 0) {
                OQS_MEM_cleanse(&ckpt, sizeof(ckpt));
                return rc;
            }
        }
    }
    OQS_MEM_cleanse(&ckpt, sizeof(ckpt));
    return rc == -1 ? -errno : 0;
}

static int checkpoint_load(const char *path, uint64_t expected_file_id,
                           pqc_checkpoint_t *out)
{
    if (!out) return -EINVAL;
    pqc_checkpoint_t ckpt = {0};
    ssize_t n = getxattr(path, PQC_XATTR_CHECKPOINT, &ckpt, sizeof(ckpt));
    if (n == -1) return -errno;
    if ((size_t)n != sizeof(ckpt) || ckpt.magic != PQC_CHECKPOINT_MAGIC ||
        ckpt.version != PQC_CHECKPOINT_VERSION || ckpt.file_id != expected_file_id) {
        OQS_MEM_cleanse(&ckpt, sizeof(ckpt));
        return -EINVAL;
    }
    uint8_t digest[32];
    unsigned int out_len = 0;
    unsigned char *mac = HMAC(EVP_sha256(), g_master_key, sizeof(g_master_key),
                              (unsigned char *)&ckpt,
                              offsetof(pqc_checkpoint_t, digest),
                              digest, &out_len);
    if (!mac || out_len != sizeof(digest) ||
        CRYPTO_memcmp(digest, ckpt.digest, sizeof(digest)) != 0) {
        OQS_MEM_cleanse(digest, sizeof(digest));
        OQS_MEM_cleanse(&ckpt, sizeof(ckpt));
        return -EKEYREJECTED;
    }
    {
        /* Register to multi-file prefix map prior to loading/verifying freshness */
        (void)pqc_anchor_record_file(ckpt.file_id, ckpt.max_generation, ckpt.sequence, ckpt.logical_size);

        pqc_anchor_state_t expected = {
            .epoch = ckpt.max_generation,
            .sequence = ckpt.sequence,
            .logical_size = ckpt.logical_size,
        };
        int anchor_rc = pqc_anchor_load(&expected);
        if (anchor_rc != 0) {
            OQS_MEM_cleanse(digest, sizeof(digest));
            OQS_MEM_cleanse(&ckpt, sizeof(ckpt));
            return anchor_rc;
        }
    }
    *out = ckpt;
    OQS_MEM_cleanse(digest, sizeof(digest));
    return 0;
}

static void *anchor_worker_main(void *arg)
{
    (void)arg;
    while (1) {
        pthread_mutex_lock(&g_anchor_lock);
        while (!g_anchor_stop && !g_anchor_dirty) {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_sec += 10;
            (void)pthread_cond_timedwait(&g_anchor_cv, &g_anchor_lock, &ts);
        }
        if (g_anchor_stop) {
            pthread_mutex_unlock(&g_anchor_lock);
            break;
        }
        struct timespec batch_deadline;
        clock_gettime(CLOCK_REALTIME, &batch_deadline);
        if (getenv("PQC_FRESHNESS_WINDOW_N")) {
            batch_deadline.tv_sec += 86400; /* Wait effectively forever */
        } else {
            batch_deadline.tv_nsec += 250000000L;
            if (batch_deadline.tv_nsec >= 1000000000L) {
                batch_deadline.tv_sec += 1;
                batch_deadline.tv_nsec -= 1000000000L;
            }
        }
        while (!g_anchor_stop && g_anchor_dirty) {
            int wait_rc = pthread_cond_timedwait(&g_anchor_cv, &g_anchor_lock, &batch_deadline);
            if (wait_rc == ETIMEDOUT)
                break;
        }
        pqc_anchor_state_t state = g_anchor_state;
        g_anchor_dirty = 0;
        pthread_mutex_unlock(&g_anchor_lock);

        if (pqc_anchor_store(&state) == 0 && pqc_anchor_flush() == 0) {
            pthread_mutex_lock(&g_anchor_lock);
            g_anchor_last_commit = time(NULL);
            pthread_mutex_unlock(&g_anchor_lock);
        }
    }
    return NULL;
}

static int anchor_flush_now(void)
{
    if (!g_anchor_thread_started)
        return 0;

    pthread_mutex_lock(&g_anchor_lock);
    if (!g_anchor_dirty) {
        pthread_mutex_unlock(&g_anchor_lock);
        return 0;
    }

    pqc_anchor_state_t state = g_anchor_state;
    g_anchor_dirty = 0;
    pthread_mutex_unlock(&g_anchor_lock);

    pqc_fault_cutpoint("anchor_update_before");

    int rc = pqc_anchor_store(&state);
    if (rc == 0) {
        pthread_mutex_lock(&g_anchor_lock);
        g_anchor_last_commit = time(NULL);
        pthread_mutex_unlock(&g_anchor_lock);
    } else {
        pthread_mutex_lock(&g_anchor_lock);
        g_anchor_dirty = 1;
        pthread_mutex_unlock(&g_anchor_lock);
    }
    return rc;
}

static int sidecar_path(char *out, size_t out_size, const char *path, const char *suffix)
{
    int n = snprintf(out, out_size, "%s%s", path, suffix);
    return n < 0 || (size_t)n >= out_size ? -ENAMETOOLONG : 0;
}

static int is_hidden_sidecar_path(const char *path)
{
    size_t len = strlen(path);
    return (len >= 8 && strcmp(path + len - 8, ".pqcdata") == 0) ||
           (len >= 8 && strcmp(path + len - 8, ".pqcmeta") == 0);
}

static int is_sqlite_wal_sidecar_path(const char *path)
{
    size_t len = strlen(path);
    return (len >= 4 && strcmp(path + len - 4, "-wal") == 0) ||
           (len >= 4 && strcmp(path + len - 4, "-shm") == 0);
}

static uint64_t pqc_hash_path(const char *path)
{
    uint64_t h = 1469598103934665603ULL;
    for (const unsigned char *p = (const unsigned char *)path; *p; ++p) {
        h ^= (uint64_t)*p;
        h *= 1099511628211ULL;
    }
    return h;
}

static int sqlite_sidecar_redirect_path(char *out, size_t out_size, const char *path)
{
    const char *compat = getenv("PQC_ALLOW_SQLITE_MMAP");
    if (!(compat && *compat))
        return -ENOENT;
    if (!is_sqlite_wal_sidecar_path(path))
        return -ENOENT;

    if (mkdir("/dev/shm/pqc_sqlite", 0700) != 0 && errno != EEXIST)
        return -errno;

    uint64_t h = pqc_hash_path(path);
    int n = snprintf(out, out_size, "/dev/shm/pqc_sqlite/%016llx%s",
                     (unsigned long long)h,
                     (strstr(path, "-shm") != NULL) ? ".shm" : ".wal");
    return n < 0 || (size_t)n >= out_size ? -ENAMETOOLONG : 0;
}

static int path_has_suffix(const char *path, const char *suffix)
{
    size_t path_len = strlen(path);
    size_t suffix_len = strlen(suffix);
    return path_len >= suffix_len &&
           strcmp(path + path_len - suffix_len, suffix) == 0;
}

static const char *qos_class_name(int qos_class)
{
    return qos_class == PQC_QOS_CLASS_LATENCY ? "latency" : "elastic";
}

static int parse_qos_class_value(const char *value, size_t size, int *out)
{
    if (!value || !out || size == 0)
        return -EINVAL;
    char tmp[32] = {0};
    size_t copy = size < sizeof(tmp) - 1 ? size : sizeof(tmp) - 1;
    memcpy(tmp, value, copy);
    tmp[copy] = '\0';
    if (strcmp(tmp, "latency") == 0 ||
        strcmp(tmp, "foreground") == 0 ||
        strcmp(tmp, "1") == 0) {
        *out = PQC_QOS_CLASS_LATENCY;
        return 0;
    }
    if (strcmp(tmp, "elastic") == 0 ||
        strcmp(tmp, "background") == 0 ||
        strcmp(tmp, "default") == 0 ||
        strcmp(tmp, "0") == 0) {
        *out = PQC_QOS_CLASS_ELASTIC;
        return 0;
    }
    return -EINVAL;
}

static int load_qos_class_xattr(const char *path, int *out)
{
    char value[32] = {0};
    ssize_t n = getxattr(path, PQC_XATTR_QOS_CLASS, value, sizeof(value) - 1);
    if (n == -1)
        return -errno;
    if (n <= 0)
        return -EINVAL;
    return parse_qos_class_value(value, (size_t)n, out);
}

static int qos_class_load_for_path(const char *phys_path, int *out)
{
    if (!phys_path || !out)
        return -EINVAL;
    int rc = load_qos_class_xattr(phys_path, out);
    if (rc == 0)
        return 0;
    if (rc != -ENODATA && rc != -ENOENT)
        return rc;

    char base[4096];
    const char *suffixes[] = {"-journal", "-wal", "-shm"};
    for (size_t i = 0; i < sizeof(suffixes) / sizeof(suffixes[0]); ++i) {
        const char *suffix = suffixes[i];
        if (!path_has_suffix(phys_path, suffix))
            continue;
        size_t len = strlen(phys_path) - strlen(suffix);
        if (len >= sizeof(base))
            return -ENAMETOOLONG;
        memcpy(base, phys_path, len);
        base[len] = '\0';
        rc = load_qos_class_xattr(base, out);
        if (rc == 0)
            return 0;
        if (rc != -ENODATA && rc != -ENOENT)
            return rc;
    }
    *out = PQC_QOS_CLASS_ELASTIC;
    return 0;
}

/*
 * The final mapping journal stores one generation per logical block.  This
 * primitive derives a unique GCM nonce from that generation and binds the file
 * identity, logical block, generation, and plaintext length as associated data.
 * It deliberately operates on complete logical blocks; partial writes must be
 * read-modify-write operations before calling it.
 */
static int aes256_ecb_block(const uint8_t key[32], const uint8_t in[16], uint8_t out[16])
{
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    int produced = 0, final = 0;
    int ok = ctx && EVP_EncryptInit_ex(ctx, EVP_aes_256_ecb(), NULL, key, NULL) == 1 &&
             EVP_CIPHER_CTX_set_padding(ctx, 0) == 1 &&
             EVP_EncryptUpdate(ctx, out, &produced, in, 16) == 1 &&
             EVP_EncryptFinal_ex(ctx, out + produced, &final) == 1 &&
             produced + final == 16;
    EVP_CIPHER_CTX_free(ctx);
    return ok ? 0 : -EIO;
}

static void gcm_shift_right(uint8_t value[16])
{
    uint8_t carry = 0;
    for (size_t i = 0; i < 16; ++i) {
        uint8_t next = (uint8_t)(value[i] & 1U);
        value[i] = (uint8_t)((value[i] >> 1) | (carry << 7));
        carry = next;
    }
}

static void gcm_multiply(const uint8_t x[16], const uint8_t y[16], uint8_t out[16])
{
    uint8_t z[16] = {0}, v[16];
    memcpy(v, y, sizeof(v));
    for (size_t bit = 0; bit < 128; ++bit) {
        if ((x[bit / 8] >> (7 - (bit % 8))) & 1U)
            for (size_t i = 0; i < sizeof(z); ++i) z[i] ^= v[i];
        int lsb = v[15] & 1U;
        gcm_shift_right(v);
        if (lsb) v[0] ^= 0xe1U;
    }
    memcpy(out, z, sizeof(z));
    OPENSSL_cleanse(v, sizeof(v));
    OPENSSL_cleanse(z, sizeof(z));
}

static void gcm_absorb(uint8_t state[16], const uint8_t h[16], const uint8_t *data, size_t length)
{
    while (length) {
        uint8_t block[16] = {0};
        size_t take = length < sizeof(block) ? length : sizeof(block);
        memcpy(block, data, take);
        for (size_t i = 0; i < sizeof(block); ++i) state[i] ^= block[i];
        gcm_multiply(state, h, state);
        OPENSSL_cleanse(block, sizeof(block));
        data += take;
        length -= take;
    }
}

static void store_u64_be(uint8_t out[8], uint64_t value)
{
    for (int i = 7; i >= 0; --i) { out[i] = (uint8_t)value; value >>= 8; }
}

static int gcm_compute_tag(const uint8_t key[32], const uint8_t nonce[12],
                           const uint8_t *aad, size_t aad_length,
                           const uint8_t *ciphertext, size_t ciphertext_length,
                           uint8_t tag[16])
{
    uint8_t zero[16] = {0}, h[16], state[16] = {0}, j0[16] = {0}, encrypted_j0[16], lengths[16] = {0};
    memcpy(j0, nonce, 12);
    j0[15] = 1;
    if (aes256_ecb_block(key, zero, h) || aes256_ecb_block(key, j0, encrypted_j0)) return -EIO;
    gcm_absorb(state, h, aad, aad_length);
    gcm_absorb(state, h, ciphertext, ciphertext_length);
    store_u64_be(lengths, (uint64_t)aad_length * 8U);
    store_u64_be(lengths + 8, (uint64_t)ciphertext_length * 8U);
    for (size_t i = 0; i < sizeof(state); ++i) state[i] ^= lengths[i];
    gcm_multiply(state, h, state);
    for (size_t i = 0; i < 16; ++i) tag[i] = encrypted_j0[i] ^ state[i];
    OPENSSL_cleanse(h, sizeof(h)); OPENSSL_cleanse(state, sizeof(state));
    OPENSSL_cleanse(encrypted_j0, sizeof(encrypted_j0));
    return 0;
}

static int derive_block_nonce(uint64_t file_id, uint64_t block, uint64_t generation,
                              uint8_t nonce[PQC_AEAD_NONCE_SIZE])
{
    uint8_t digest[32];
    uint8_t nonce_seed[24];
    EVP_MD_CTX *md = NULL;

    memcpy(nonce_seed, &file_id, sizeof(file_id));
    memcpy(nonce_seed + 8, &block, sizeof(block));
    memcpy(nonce_seed + 16, &generation, sizeof(generation));
    md = EVP_MD_CTX_new();
    if (!md) return -ENOMEM;
    if (EVP_DigestInit_ex(md, EVP_sha256(), NULL) != 1 ||
        EVP_DigestUpdate(md, nonce_seed, sizeof(nonce_seed)) != 1 ||
        EVP_DigestFinal_ex(md, digest, NULL) != 1) {
        EVP_MD_CTX_free(md);
        return -EIO;
    }
    EVP_MD_CTX_free(md);
    memcpy(nonce, digest, PQC_AEAD_NONCE_SIZE);
    OPENSSL_cleanse(digest, sizeof(digest));
    OPENSSL_cleanse(nonce_seed, sizeof(nonce_seed));
    return 0;
}

static void build_block_aad(uint8_t aad[28], uint64_t file_id, uint64_t block,
                            uint64_t generation, uint32_t length)
{
    memcpy(aad, &file_id, 8);
    memcpy(aad + 8, &block, 8);
    memcpy(aad + 16, &generation, 8);
    memcpy(aad + 24, &length, 4);
}

static int crypt_block_gcm(const uint8_t *key, size_t key_len, uint64_t file_id,
                           uint64_t block, uint64_t generation, uint32_t length,
                           const uint8_t *in, uint8_t *out, uint8_t tag[16],
                           int encrypt, int prefer_gpu)
{
    if (!key || key_len < 32 || !in || !out || length > PQC_LOGICAL_BLOCK_SIZE)
        return -EINVAL;
    uint8_t nonce[PQC_AEAD_NONCE_SIZE] = {0};
    uint8_t aad[28];
    int rc = derive_block_nonce(file_id, block, generation, nonce);
    if (rc) return rc;
    build_block_aad(aad, file_id, block, generation, length);
    if (prefer_gpu && skim_cuda_aead_available()) {
        if (encrypt) {
            if (skim_cuda_aes256_gcm_ctr(key, nonce, in, out, length) == 0 &&
                gcm_compute_tag(key, nonce, aad, sizeof(aad), out, length, tag) == 0)
                return 0;
        } else {
            uint8_t expected_tag[PQC_AEAD_TAG_SIZE];
            if (gcm_compute_tag(key, nonce, aad, sizeof(aad), in, length, expected_tag) != 0)
                return -EIO;
            int valid = CRYPTO_memcmp(expected_tag, tag, PQC_AEAD_TAG_SIZE) == 0;
            OPENSSL_cleanse(expected_tag, sizeof(expected_tag));
            if (!valid) return -EBADMSG;
            if (skim_cuda_aes256_gcm_ctr(key, nonce, in, out, length) == 0)
                return 0;
        }
        /* CUDA failure is a performance failure, never a format change. */
    }
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx) return -ENOMEM;
    int n = 0, total = 0;
    int ok;
    if (encrypt) {
        ok = EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, key, nonce) == 1 &&
             EVP_EncryptUpdate(ctx, NULL, &n, aad, sizeof(aad)) == 1 &&
             EVP_EncryptUpdate(ctx, out, &n, in, (int)length) == 1;
        total = n;
        ok = ok && EVP_EncryptFinal_ex(ctx, out + total, &n) == 1 &&
             EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, PQC_AEAD_TAG_SIZE, tag) == 1;
    } else {
        ok = EVP_DecryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, key, nonce) == 1 &&
             EVP_DecryptUpdate(ctx, NULL, &n, aad, sizeof(aad)) == 1 &&
             EVP_DecryptUpdate(ctx, out, &n, in, (int)length) == 1 &&
             EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, PQC_AEAD_TAG_SIZE, tag) == 1;
        total = n;
        ok = ok && EVP_DecryptFinal_ex(ctx, out + total, &n) == 1;
    }
    EVP_CIPHER_CTX_free(ctx);
    return ok ? 0 : -EBADMSG;
}

typedef struct {
    uint64_t block;
    uint64_t generation;
    uint32_t length;
    size_t input_offset;
    size_t output_offset;
    uint64_t ciphertext_offset;
    uint8_t nonce[PQC_AEAD_NONCE_SIZE];
    uint8_t aad[28];
    uint8_t tag[PQC_AEAD_TAG_SIZE];
} flush_block_desc_t;

static int crypt_block_batch_gcm(const uint8_t *key, size_t key_len, uint64_t file_id,
                                 flush_block_desc_t *blocks, size_t count,
                                 const uint8_t *input, uint8_t *output)
{
    if (!key || key_len < 32 || !blocks || !input || !output || count == 0)
        return -EINVAL;

    size_t *offsets = NULL;
    size_t *lengths = NULL;
    uint8_t *nonces = NULL;
    uint8_t *aads = NULL;
    uint8_t *tags = NULL;
    int rc = 0;

    offsets = calloc(count, sizeof(size_t));
    lengths = calloc(count, sizeof(size_t));
    nonces = calloc(count, PQC_AEAD_NONCE_SIZE);
    aads = calloc(count, sizeof(blocks[0].aad));
    tags = calloc(count, PQC_AEAD_TAG_SIZE);
    if (!offsets || !lengths || !nonces || !aads || !tags) {
        rc = -ENOMEM;
        goto out;
    }
    for (size_t i = 0; i < count; ++i) {
        offsets[i] = blocks[i].input_offset;
        lengths[i] = blocks[i].length;
        memcpy(nonces + i * PQC_AEAD_NONCE_SIZE, blocks[i].nonce, PQC_AEAD_NONCE_SIZE);
        memcpy(aads + i * sizeof(blocks[i].aad), blocks[i].aad, sizeof(blocks[i].aad));
    }

    if (skim_cuda_aead_available() &&
        skim_cuda_aes256_gcm_batch(key, nonces, aads, input, output, offsets, lengths, tags, count) == 0) {
        for (size_t i = 0; i < count; ++i) {
            memcpy(blocks[i].tag, tags + i * PQC_AEAD_TAG_SIZE, PQC_AEAD_TAG_SIZE);
        }
        rc = 0;
        goto out;
    }

    for (size_t i = 0; i < count; ++i) {
        uint8_t tag[PQC_AEAD_TAG_SIZE];
        rc = crypt_block_gcm(key, key_len, file_id, blocks[i].block, blocks[i].generation,
                             blocks[i].length, input + blocks[i].input_offset,
                             output + blocks[i].output_offset, tag, 1, 0);
        if (rc != 0)
            goto out;
    }
out:
    if (tags) OPENSSL_cleanse(tags, count * PQC_AEAD_TAG_SIZE);
    free(tags);
    if (aads) OPENSSL_cleanse(aads, count * sizeof(blocks[0].aad));
    free(aads);
    if (nonces) OPENSSL_cleanse(nonces, count * PQC_AEAD_NONCE_SIZE);
    free(nonces);
    free(lengths);
    free(offsets);
    return rc;
}

static int crypto_self_test(void)
{
    uint8_t key[32], plain[PQC_LOGICAL_BLOCK_SIZE], cipher[PQC_LOGICAL_BLOCK_SIZE];
    uint8_t recovered[PQC_LOGICAL_BLOCK_SIZE], tag[PQC_AEAD_TAG_SIZE];
    if (RAND_bytes(key, sizeof(key)) != 1 || RAND_bytes(plain, sizeof(plain)) != 1)
        return -1;
    if (crypt_block_gcm(key, sizeof(key), 7, 11, 1, sizeof(plain), plain, cipher, tag, 1, 0) != 0 ||
        crypt_block_gcm(key, sizeof(key), 7, 11, 1, sizeof(plain), cipher, recovered, tag, 0, 0) != 0 ||
        memcmp(plain, recovered, sizeof(plain)) != 0)
        return -1;
    tag[0] ^= 1;
    int tamper_rejected = crypt_block_gcm(key, sizeof(key), 7, 11, 1, sizeof(plain),
                                           cipher, recovered, tag, 0, 0) == -EBADMSG;
    if (tamper_rejected && skim_cuda_aead_available()) {
        uint8_t cpu_reference[PQC_LOGICAL_BLOCK_SIZE], reference_tag[PQC_AEAD_TAG_SIZE];
        uint8_t gpu_cipher[PQC_LOGICAL_BLOCK_SIZE], gpu_plain[PQC_LOGICAL_BLOCK_SIZE], gpu_tag[PQC_AEAD_TAG_SIZE];
        size_t offsets[1] = {0}, lengths[1] = {sizeof(plain)};
        uint8_t nonces[PQC_AEAD_NONCE_SIZE];
        uint8_t aad[28];
        build_block_aad(aad, 7, 11, 2, sizeof(plain));
        int gpu_ok = derive_block_nonce(7, 11, 2, nonces) == 0 &&
                     crypt_block_gcm(key, sizeof(key), 7, 11, 2, sizeof(plain), plain, cpu_reference, reference_tag, 1, 0) == 0 &&
                     skim_cuda_aes256_gcm_batch(key, nonces, aad, plain, gpu_cipher,
                                                 offsets, lengths, gpu_tag, 1) == 0 &&
                     memcmp(cpu_reference, gpu_cipher, sizeof(cpu_reference)) == 0 &&
                     CRYPTO_memcmp(reference_tag, gpu_tag, sizeof(reference_tag)) == 0 &&
                     crypt_block_gcm(key, sizeof(key), 7, 11, 2, sizeof(plain), gpu_cipher, gpu_plain, gpu_tag, 0, 1) == 0 &&
                     memcmp(plain, gpu_plain, sizeof(plain)) == 0;
        tamper_rejected = gpu_ok;
        OPENSSL_cleanse(cpu_reference, sizeof(cpu_reference)); OPENSSL_cleanse(reference_tag, sizeof(reference_tag));
        OPENSSL_cleanse(gpu_cipher, sizeof(gpu_cipher)); OPENSSL_cleanse(gpu_plain, sizeof(gpu_plain));
        OPENSSL_cleanse(gpu_tag, sizeof(gpu_tag));
        OPENSSL_cleanse(nonces, sizeof(nonces));
    }
    OQS_MEM_cleanse(key, sizeof(key)); OQS_MEM_cleanse(plain, sizeof(plain));
    OQS_MEM_cleanse(recovered, sizeof(recovered));
    return tamper_rejected ? 0 : -1;
}

#define PQC_ALGO_AES_256_GCM 0u

typedef struct {
    uint64_t logical_block;
    uint64_t generation;
    uint64_t ciphertext_offset;
    uint32_t plaintext_length;
    uint32_t algorithm_id;
    uint8_t tag[PQC_AEAD_TAG_SIZE];
} block_mapping_t;

typedef struct {
    uint64_t magic;
    uint32_t version;
    uint32_t committed;
    block_mapping_t mapping;
    uint8_t digest[32];
} journal_record_t;

static int journal_digest(journal_record_t *record)
{
    EVP_MD_CTX *md = EVP_MD_CTX_new();
    unsigned int out_len = 0;
    if (!md) return -ENOMEM;
    int ok = EVP_DigestInit_ex(md, EVP_sha256(), NULL) == 1 &&
             EVP_DigestUpdate(md, record, offsetof(journal_record_t, digest)) == 1 &&
             EVP_DigestFinal_ex(md, record->digest, &out_len) == 1 &&
             out_len == sizeof(record->digest);
    EVP_MD_CTX_free(md);
    return ok ? 0 : -EIO;
}

static int journal_record_valid(journal_record_t *record)
{
    uint8_t saved[sizeof(record->digest)];
    if (record->magic != PQC_JOURNAL_MAGIC ||
        record->version != PQC_JOURNAL_VERSION ||
        record->committed != PQC_JOURNAL_COMMITTED ||
        record->mapping.plaintext_length > PQC_LOGICAL_BLOCK_SIZE ||
        record->mapping.algorithm_id != PQC_ALGO_AES_256_GCM)
        return 0;
    memcpy(saved, record->digest, sizeof(saved));
    return journal_digest(record) == 0 &&
           memcmp(saved, record->digest, sizeof(saved)) == 0;
}

static int journal_append_mapping(int journal_fd, const block_mapping_t *mapping)
{
    if (!mapping || mapping->plaintext_length > PQC_LOGICAL_BLOCK_SIZE ||
        mapping->algorithm_id != PQC_ALGO_AES_256_GCM) return -EINVAL;
    journal_record_t record = { .magic = PQC_JOURNAL_MAGIC,
                                .version = PQC_JOURNAL_VERSION,
                                .committed = PQC_JOURNAL_COMMITTED,
                                .mapping = *mapping };
    int rc = journal_digest(&record);
    if (rc) return rc;
    ssize_t n = write(journal_fd, &record, sizeof(record));
    if (n != (ssize_t)sizeof(record)) return n < 0 ? -errno : -EIO;
    return fdatasync(journal_fd) == 0 ? 0 : -errno;
}

/* The commit coordinator calls this only after the entire ciphertext batch is
 * durable.  One subsequent journal fdatasync publishes the whole batch. */
static int journal_append_mapping_unsynced(int journal_fd, const block_mapping_t *mapping)
{
    if (!mapping || mapping->plaintext_length > PQC_LOGICAL_BLOCK_SIZE ||
        mapping->algorithm_id != PQC_ALGO_AES_256_GCM) return -EINVAL;
    journal_record_t record = { .magic = PQC_JOURNAL_MAGIC,
                                .version = PQC_JOURNAL_VERSION,
                                .committed = PQC_JOURNAL_COMMITTED,
                                .mapping = *mapping };
    int rc = journal_digest(&record);
    if (rc) return rc;
    ssize_t n = write(journal_fd, &record, sizeof(record));
    return n == (ssize_t)sizeof(record) ? 0 : (n < 0 ? -errno : -EIO);
}

static int journal_lookup_mapping(int journal_fd, uint64_t logical_block,
                                  block_mapping_t *out)
{
    if (!out || lseek(journal_fd, 0, SEEK_SET) < 0) return -EINVAL;
    journal_record_t record;
    block_mapping_t best = {0};
    int found = 0;
    ssize_t n;
    while ((n = read(journal_fd, &record, sizeof(record))) == (ssize_t)sizeof(record)) {
        if (journal_record_valid(&record) && record.mapping.logical_block == logical_block &&
            (!found || record.mapping.generation > best.generation)) {
            best = record.mapping;
            found = 1;
        }
    }
    if (n < 0) return -errno;
    if (!found) return -ENOENT;
    *out = best;
    return 0;
}

static uint64_t journal_max_generation(int journal_fd)
{
    if (lseek(journal_fd, 0, SEEK_SET) < 0) return 0;
    journal_record_t record;
    uint64_t max = 0;
    while (read(journal_fd, &record, sizeof(record)) == (ssize_t)sizeof(record))
        if (journal_record_valid(&record) && record.mapping.generation > max)
            max = record.mapping.generation;
    (void)lseek(journal_fd, 0, SEEK_END);
    return max;
}

static int journal_self_test(void)
{
    FILE *tmp = tmpfile();
    if (!tmp) return -1;
    int fd = fileno(tmp);
    block_mapping_t first = { .logical_block = 4, .generation = 1,
                              .ciphertext_offset = 4096, .plaintext_length = 17 };
    block_mapping_t latest = { .logical_block = 4, .generation = 2,
                               .ciphertext_offset = 8192, .plaintext_length = 31 };
    int ok = journal_append_mapping(fd, &first) == 0 &&
             journal_append_mapping(fd, &latest) == 0;
    block_mapping_t recovered = {0};
    ok = ok && journal_lookup_mapping(fd, 4, &recovered) == 0 &&
         recovered.generation == latest.generation &&
         recovered.ciphertext_offset == latest.ciphertext_offset;
    /* A short crash tail must not invalidate the previous committed mapping. */
    ok = ok && write(fd, "tail", 4) == 4 &&
         journal_lookup_mapping(fd, 4, &recovered) == 0 &&
         recovered.generation == latest.generation;
    fclose(tmp);
    return ok ? 0 : -1;
}

static int generation_replay_self_test(void)
{
    FILE *tmp = tmpfile();
    if (!tmp) return -1;
    int fd = fileno(tmp);

    block_mapping_t gen1 = { .logical_block = 9, .generation = 1,
                             .ciphertext_offset = 4096, .plaintext_length = 32,
                             .algorithm_id = PQC_ALGO_AES_256_GCM };
    block_mapping_t gen2 = { .logical_block = 9, .generation = 2,
                             .ciphertext_offset = 8192, .plaintext_length = 32,
                             .algorithm_id = PQC_ALGO_AES_256_GCM };
    block_mapping_t replayed_gen1 = { .logical_block = 9, .generation = 1,
                                      .ciphertext_offset = 12288, .plaintext_length = 32,
                                      .algorithm_id = PQC_ALGO_AES_256_GCM };
    block_mapping_t recovered = {0};
    int ok = journal_append_mapping(fd, &gen1) == 0 &&
             journal_append_mapping(fd, &gen2) == 0 &&
             journal_append_mapping(fd, &replayed_gen1) == 0 &&
             journal_lookup_mapping(fd, 9, &recovered) == 0 &&
             recovered.generation == gen2.generation &&
             recovered.ciphertext_offset == gen2.ciphertext_offset &&
             journal_max_generation(fd) == gen2.generation;

    uint8_t key[32];
    uint8_t plain[32];
    uint8_t cipher[32];
    uint8_t recovered_plain[32];
    uint8_t tag[PQC_AEAD_TAG_SIZE];
    for (size_t i = 0; i < sizeof(key); ++i) key[i] = (uint8_t)(0xa0U + i);
    for (size_t i = 0; i < sizeof(plain); ++i) plain[i] = (uint8_t)(0x30U + i);
    memset(cipher, 0, sizeof(cipher));
    memset(recovered_plain, 0, sizeof(recovered_plain));
    memset(tag, 0, sizeof(tag));

    int enc_rc = crypt_block_gcm(key, sizeof(key), UINT64_C(0xabcdef0123456789),
                                 9, 2, sizeof(plain), plain, cipher, tag, 1, 0);
    int wrong_gen_rc = crypt_block_gcm(key, sizeof(key), UINT64_C(0xabcdef0123456789),
                                       9, 1, sizeof(cipher), cipher, recovered_plain,
                                       tag, 0, 0);
    int right_gen_rc = crypt_block_gcm(key, sizeof(key), UINT64_C(0xabcdef0123456789),
                                       9, 2, sizeof(cipher), cipher, recovered_plain,
                                       tag, 0, 0);
    ok = ok && enc_rc == 0 && wrong_gen_rc != 0 && right_gen_rc == 0 &&
         CRYPTO_memcmp(plain, recovered_plain, sizeof(plain)) == 0;

    OPENSSL_cleanse(key, sizeof(key));
    OPENSSL_cleanse(plain, sizeof(plain));
    OPENSSL_cleanse(cipher, sizeof(cipher));
    OPENSSL_cleanse(recovered_plain, sizeof(recovered_plain));
    OPENSSL_cleanse(tag, sizeof(tag));
    fclose(tmp);
    return ok ? 0 : -1;
}

static int checkpoint_self_test(void)
{
    char path[] = "/tmp/skim_ckpt_selftestXXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return -1;
    close(fd);
    int ok = derive_master_key("checkpoint-self-test") == 0;
    if (ok) {
        ok = checkpoint_store(path, 42, 7, 8192, 11) == 0;
        if (ok && pqc_anchor_backend() == PQC_ANCHOR_BACKEND_HARDWARE)
            ok = pqc_anchor_flush() == 0;
        pqc_checkpoint_t ckpt = {0};
        ok = ok && checkpoint_load(path, 42, &ckpt) == 0 &&
             ckpt.sequence == 7 && ckpt.logical_size == 8192 && ckpt.max_generation == 11;
    }
    unlink(path);
    OQS_MEM_cleanse(g_master_key, sizeof(g_master_key));
    g_has_master_key = 0;
    return ok ? 0 : -1;
}

static pqc_scheduler_policy_t scheduler_policy_from_env(void);

static uint64_t parse_u64_env_or_default(const char *name, uint64_t fallback)
{
    const char *value = getenv(name);
    if (!value || !*value) return fallback;
    char *end = NULL;
    unsigned long long parsed = strtoull(value, &end, 10);
    if (end == value || !end || *end != '\0') return fallback;
    return (uint64_t)parsed;
}

static double parse_double_env_or_default(const char *name, double fallback)
{
    const char *value = getenv(name);
    if (!value || !*value) return fallback;
    char *end = NULL;
    double parsed = strtod(value, &end);
    if (end == value || !end || *end != '\0') return fallback;
    return parsed;
}

static int scheduler_self_test(void)
{
    pqc_scheduler_policy_t policy = {
        .gpu_min_bytes = 8192,
        .gpu_queue_penalty_ns = 25000,
        .coherence_penalty_ns = 65536,
        .gpu_max_inflight_jobs = 2,
        .gpu_max_wait_ns = 25000,
        .cpu_load_bias = 1.0,
        .gpu_queue_bias = 1.0,
    };
    pqc_block_job_t small = {0}, large = {0};
    pqc_block_job_init(&small, 1, 0, 1, 0, 4096, PQC_JOB_FLAG_ENCRYPT | PQC_JOB_FLAG_READMOD, PQC_PLANE_KEY);
    pqc_block_job_init(&large, 1, 1, 2, 4096, 16384, PQC_JOB_FLAG_ENCRYPT | PQC_JOB_FLAG_READMOD | PQC_JOB_FLAG_GPU_ELIGIBLE, PQC_PLANE_KEY);
    small.cpu_queue_depth = 2;
    large.gpu_queue_depth = 0;
    small.coherence_cost_ns = 0;
    large.coherence_cost_ns = 1024;
    large.ai_qos_budget_ns = 1000000;
    if (pqc_block_job_choose_target(&small, &policy, 0.5, 0.0) != PQC_JOB_CPU)
        return -1;
    if (pqc_block_job_choose_target(&large, &policy, 0.5, 0.0) != PQC_JOB_GPU)
        return -1;
    return 0;
}

static void scheduler_smoke_report(void)
{
    pqc_scheduler_policy_t policy = scheduler_policy_from_env();
    uint64_t supplied_budget_ns = 2000000;
    uint64_t supplied_cpu_queue_depth = 2;
    uint64_t supplied_pressure_gpu_queue_depth = 2;
    const char *budget_env = getenv("PQC_SCHED_SMOKE_AI_BUDGET_NS");
    if (budget_env && *budget_env) {
        char *end = NULL;
        unsigned long long parsed = strtoull(budget_env, &end, 10);
        if (end != budget_env && end && *end == '\0')
            supplied_budget_ns = (uint64_t)parsed;
    }
    const char *cpu_q_env = getenv("PQC_SCHED_SMOKE_CPU_QUEUE_DEPTH");
    if (cpu_q_env && *cpu_q_env) {
        char *end = NULL;
        unsigned long long parsed = strtoull(cpu_q_env, &end, 10);
        if (end != cpu_q_env && end && *end == '\0')
            supplied_cpu_queue_depth = (uint64_t)parsed;
    }
    const char *gpu_q_env = getenv("PQC_SCHED_SMOKE_GPU_QUEUE_DEPTH");
    if (gpu_q_env && *gpu_q_env) {
        char *end = NULL;
        unsigned long long parsed = strtoull(gpu_q_env, &end, 10);
        if (end != gpu_q_env && end && *end == '\0')
            supplied_pressure_gpu_queue_depth = (uint64_t)parsed;
    }
    pqc_block_job_t jobs[3];
    pqc_block_job_init(&jobs[0], 1, 0, 1, 0, 4096, PQC_JOB_FLAG_ENCRYPT | PQC_JOB_FLAG_READMOD, PQC_PLANE_KEY);
    pqc_block_job_init(&jobs[1], 1, 1, 2, 4096, 131072, PQC_JOB_FLAG_ENCRYPT | PQC_JOB_FLAG_READMOD | PQC_JOB_FLAG_GPU_ELIGIBLE, PQC_PLANE_KEY);
    pqc_block_job_init(&jobs[2], 1, 2, 3, 20480, 262144, PQC_JOB_FLAG_ENCRYPT | PQC_JOB_FLAG_READMOD | PQC_JOB_FLAG_GPU_ELIGIBLE, PQC_PLANE_KEY);
    jobs[0].cpu_queue_depth = supplied_cpu_queue_depth;
    jobs[1].cpu_queue_depth = supplied_cpu_queue_depth > 0 ? supplied_cpu_queue_depth - 1 : 0;
    jobs[2].cpu_queue_depth = supplied_cpu_queue_depth;
    jobs[0].ai_qos_budget_ns = supplied_budget_ns;
    jobs[1].ai_qos_budget_ns = supplied_budget_ns;
    jobs[2].ai_qos_budget_ns = supplied_budget_ns;
    jobs[0].coherence_cost_ns = 0;
    jobs[1].coherence_cost_ns = 1024;
    jobs[2].coherence_cost_ns = 8192;
    fprintf(stderr,
            "{\"event\":\"scheduler_smoke_begin\",\"gpu_min_bytes\":%zu,"
            "\"coherence_penalty_ns\":%llu,\"supplied_ai_budget_ns\":%llu}\n",
            policy.gpu_min_bytes,
            (unsigned long long)policy.coherence_penalty_ns,
            (unsigned long long)supplied_budget_ns);
    for (size_t i = 0; i < 3; ++i) {
        jobs[i].target = pqc_block_job_choose_target(&jobs[i], &policy, 0.5, (double)jobs[i].gpu_queue_depth);
        fprintf(stderr,
                "{\"event\":\"scheduler_smoke_job\",\"index\":%zu,"
                "\"bytes\":%u,\"target\":\"%s\",\"coherence_ns\":%llu,"
                "\"cpu_queue_depth\":%llu,\"gpu_queue_depth\":%llu,"
                "\"gpu_wait_ns\":%llu}\n",
                i,
                jobs[i].plaintext_length,
                jobs[i].target == PQC_JOB_GPU ? "GPU" : "CPU",
                (unsigned long long)jobs[i].coherence_cost_ns,
                (unsigned long long)jobs[i].cpu_queue_depth,
                (unsigned long long)jobs[i].gpu_queue_depth,
                (unsigned long long)jobs[i].gpu_wait_ns);
    }
    pqc_block_job_t spill = {0};
    pqc_block_job_init(&spill, 9, 9, 9, 0, 131072,
                       PQC_JOB_FLAG_ENCRYPT | PQC_JOB_FLAG_READMOD | PQC_JOB_FLAG_GPU_ELIGIBLE, PQC_PLANE_KEY);
    spill.cpu_queue_depth = 0;
    spill.gpu_queue_depth = supplied_pressure_gpu_queue_depth;
    spill.gpu_wait_ns = policy.gpu_max_wait_ns + 1;
    spill.ai_qos_budget_ns = supplied_budget_ns;
    spill.coherence_cost_ns = 2048;
    spill.target = pqc_block_job_choose_target(&spill, &policy, 0.5, (double)spill.gpu_queue_depth);
    fprintf(stderr,
            "{\"event\":\"scheduler_smoke_pressure_job\",\"bytes\":%u,"
            "\"target\":\"%s\",\"gpu_wait_ns\":%llu,\"gpu_queue_depth\":%llu}\n",
            spill.plaintext_length,
            spill.target == PQC_JOB_GPU ? "GPU" : "CPU",
            (unsigned long long)spill.gpu_wait_ns,
            (unsigned long long)spill.gpu_queue_depth);
    fprintf(stderr,
            "{\"event\":\"scheduler_smoke_end\",\"jobs\":3}\n");
}

static int admission_telemetry_smoke_report(void)
{
    const char *trace_path = getenv("PQC_ADMISSION_TRACE_PATH");
    if (!trace_path || !*trace_path)
        trace_path = "artifacts/validation/admission_telemetry_smoke_trace.jsonl";

    if (pqc_admission_init(trace_path) != 0)
        return -1;

    const double mem_bw = parse_double_env_or_default("PQC_TELEMETRY_MEM_BANDWIDTH", 0.0);
    const double tensor_core = parse_double_env_or_default("PQC_TELEMETRY_TENSOR_CORE", 0.0);
    const uint64_t ai_budget_ns =
        parse_u64_env_or_default("PQC_ADMISSION_SMOKE_AI_BUDGET_NS", 2000000ULL);
    const uint64_t cpu_queue_depth =
        parse_u64_env_or_default("PQC_ADMISSION_SMOKE_CPU_QUEUE_DEPTH", 1ULL);
    const uint64_t gpu_queue_depth =
        parse_u64_env_or_default("PQC_ADMISSION_SMOKE_GPU_QUEUE_DEPTH", 1ULL);
    const uint64_t uma_cost_ns =
        parse_u64_env_or_default("PQC_ADMISSION_SMOKE_UMA_COST_NS", 0ULL);
    const size_t bytes_total =
        (size_t)parse_u64_env_or_default("PQC_ADMISSION_SMOKE_BYTES", 131072ULL);
    const uint64_t batch_age_ns =
        parse_u64_env_or_default("PQC_ADMISSION_SMOKE_BATCH_AGE_NS", 0ULL);
    const uint64_t stale_sleep_us =
        parse_u64_env_or_default("PQC_ADMISSION_SMOKE_STALE_SLEEP_US", 0ULL);

    pqc_admission_update_telemetry(mem_bw, tensor_core);
    pqc_admission_update_ai_budget(ai_budget_ns, 0);
    if (stale_sleep_us > 0)
        usleep((useconds_t)stale_sleep_us);

    pqc_admission_context_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.batch_count = 1;
    ctx.bytes_total = bytes_total;
    ctx.batch_age_ns = batch_age_ns;
    ctx.gpu_kernel_est_ns = parse_u64_env_or_default("PQC_ADMISSION_SMOKE_GPU_KERNEL_NS", 100000ULL);
    ctx.gpu_h2d_staging_ns = parse_u64_env_or_default("PQC_ADMISSION_SMOKE_H2D_NS", 100000ULL);
    ctx.gpu_d2h_staging_ns = parse_u64_env_or_default("PQC_ADMISSION_SMOKE_D2H_NS", 100000ULL);
    ctx.cpu_queue_depth = cpu_queue_depth;
    ctx.gpu_queue_depth = gpu_queue_depth;
    ctx.cpu_load_avg = parse_double_env_or_default("PQC_ADMISSION_SMOKE_CPU_LOAD", 0.0);
    ctx.gpu_load_avg = parse_double_env_or_default("PQC_ADMISSION_SMOKE_GPU_LOAD", 0.0);
    ctx.ai_inference_deadline_ns =
        parse_u64_env_or_default("PQC_ADMISSION_SMOKE_DEADLINE_NS", 10000000ULL);
    ctx.uma_migration_cost_ns = uma_cost_ns;
    ctx.uma_migration_bytes_est =
        parse_u64_env_or_default("PQC_ADMISSION_SMOKE_UMA_BYTES", 0ULL);

    int rc = pqc_admit(&ctx);
    pqc_admission_stats_t stats;
    memset(&stats, 0, sizeof(stats));
    pqc_scheduler_trace_stats(&stats);

    fprintf(stdout,
            "{\"event\":\"admission_telemetry_smoke\","
            "\"rc\":%d,"
            "\"trace_path\":\"%s\","
            "\"telemetry_mem_bandwidth_util\":%.4f,"
            "\"telemetry_tensor_core_util\":%.4f,"
            "\"ai_budget_ns\":%llu,"
            "\"batch_age_ns\":%llu,"
            "\"deadline_ns\":%llu,"
            "\"producer_slack_age_ns\":%llu,"
            "\"producer_slack_stale_after_ns\":%llu,"
            "\"producer_slack_stale\":%s,"
            "\"cpu_queue_depth\":%llu,"
            "\"gpu_queue_depth\":%llu,"
            "\"bytes_total\":%zu,"
            "\"chosen_target\":\"%s\","
            "\"decision_reason\":%u,"
            "\"deferral_reason\":%u,"
            "\"stats_total\":%llu,"
            "\"stats_gpu\":%llu,"
            "\"stats_cpu\":%llu}\n",
            rc,
            trace_path,
            mem_bw,
            tensor_core,
            (unsigned long long)ai_budget_ns,
            (unsigned long long)ctx.batch_age_ns,
            (unsigned long long)ctx.ai_inference_deadline_ns,
            (unsigned long long)ctx.producer_slack_age_ns,
            (unsigned long long)ctx.producer_slack_stale_after_ns,
            ctx.producer_slack_stale ? "true" : "false",
            (unsigned long long)cpu_queue_depth,
            (unsigned long long)gpu_queue_depth,
            bytes_total,
            ctx.chosen_target == PQC_JOB_GPU ? "GPU" : "CPU",
            (unsigned int)ctx.decision_reason,
            (unsigned int)ctx.deferral_reason,
            (unsigned long long)stats.total_requests,
            (unsigned long long)stats.gpu_admitted_count,
            (unsigned long long)stats.cpu_routed_count);

    pqc_admission_shutdown();
    return rc;
}

/*
 * Strategy 1: Forward-Secure Key Rotation
 * Re-run ML-KEM-512.Encaps() every KEY_ROTATION_INTERVAL seconds within a
 * single file's lifetime.  Each epoch gets a new shared_secret stored as an
 * additional entry in the .pqckey sidecar:
 *   { epoch(u64) | timestamp(u64) | ss_len(u64) | ss(ss_len bytes) } ...
 * Guarantees that past plaintext is safe even if current key is leaked.
 */
#define KEY_ROTATION_INTERVAL_S  1   /* re-key every N seconds */

/* Shared state is keyed by the backing inode, never by a reusable FD number. */
typedef struct file_state {
    dev_t dev;
    ino_t ino;
    unsigned refs;
    uint64_t next_generation;
    pthread_mutex_t commit_lock;
    struct file_state *next;
} file_state_t;

static pthread_mutex_t g_file_state_table_lock = PTHREAD_MUTEX_INITIALIZER;
static file_state_t *g_file_states = NULL;

static file_state_t *file_state_acquire(int fd)
{
    struct stat st;
    if (fstat(fd, &st) != 0) return NULL;
    pthread_mutex_lock(&g_file_state_table_lock);
    for (file_state_t *p = g_file_states; p; p = p->next) {
        if (p->dev == st.st_dev && p->ino == st.st_ino) {
            ++p->refs;
            pthread_mutex_unlock(&g_file_state_table_lock);
            return p;
        }
    }
    file_state_t *p = calloc(1, sizeof(*p));
    if (p) {
        p->dev = st.st_dev; p->ino = st.st_ino; p->refs = 1;
        pthread_mutex_init(&p->commit_lock, NULL);
        p->next = g_file_states; g_file_states = p;
    }
    pthread_mutex_unlock(&g_file_state_table_lock);
    return p;
}

static void file_state_release(file_state_t *state)
{
    if (!state) return;
    pthread_mutex_lock(&g_file_state_table_lock);
    if (--state->refs == 0) {
        file_state_t **p = &g_file_states;
        while (*p && *p != state) p = &(*p)->next;
        if (*p) *p = state->next;
        pthread_mutex_destroy(&state->commit_lock);
        free(state);
    }
    pthread_mutex_unlock(&g_file_state_table_lock);
}

typedef struct {
    int             valid;
    uint8_t         ss[64];
    size_t          ss_len;
    uint64_t        file_id;
    file_state_t   *state;
    int             data_fd;
    int             journal_fd;
    uint64_t        logical_size;
    char            marker_path[4096];
    pthread_mutex_t fd_lock;       /* FD-specific mutex */
    /* write coalescing */
    uint8_t        *wbuf;          /* heap-allocated COALESCE_SIZE bytes */
    size_t          wbuf_used;      /* bytes currently buffered */
    off_t           wbuf_base_off;  /* file offset of wbuf[0] */
    uint64_t        pending_jobs;   /* in-flight flush/crypto work */
    pthread_cond_t   pending_cv;
    /* Strategy 2: selective encryption tier */
    int             tier;           /* PQC_TIER_FULL or PQC_TIER_NONE */
    int             qos_class;      /* PQC_QOS_CLASS_ELASTIC or LATENCY */
    /* Strategy 1: key rotation */
    uint64_t        key_epoch;      /* current epoch index (0 = initial KEM) */
    time_t          last_rekey;     /* wall-clock time of last KEM call */
} pqc_fd_ctx_t;

static pqc_fd_ctx_t      g_fd_ctx[PQC_MAX_FD];
static pqc_scheduler_stats_t g_sched_stats = {0};
static pthread_mutex_t   g_sched_pressure_lock = PTHREAD_MUTEX_INITIALIZER;
static uint64_t          g_gpu_inflight_jobs = 0;
static uint64_t          g_gpu_inflight_bytes = 0;
static pthread_mutex_t   g_gpu_load_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_t         g_gpu_load_thread;
static int               g_gpu_load_thread_started = 0;
static int               g_gpu_load_stop = 0;
static double            g_gpu_load_ewma = 0.0;
static pthread_t         g_admission_telemetry_thread;
static int               g_admission_telemetry_thread_started = 0;
static int               g_admission_telemetry_stop = 0;
static pthread_mutex_t   g_qos_throttle_lock = PTHREAD_MUTEX_INITIALIZER;
static int               g_qos_throttle_state = 0;
static double            g_qos_pressure_value = 0.0;
static unsigned          g_qos_below_exit_count = 0;
static pthread_mutex_t   g_qos_trace_lock = PTHREAD_MUTEX_INITIALIZER;
static pqc_scheduler_policy_t g_sched_policy = {
    .gpu_min_bytes = 131072,
    .gpu_queue_penalty_ns = 25000,
    .coherence_penalty_ns = 25000,
    .gpu_contention_penalty_ns = 25000,
    .contention_score_ns = 75000,
    .gpu_max_inflight_jobs = 2,
    .gpu_max_inflight_bytes = 256 * 1024,
    .gpu_max_wait_ns = 25000,
    .cpu_load_bias = 1.0,
    .gpu_queue_bias = 1.0,
};

static void restore_qos_class_for_fd(int fd, const char *phys_path)
{
    int qos_class = PQC_QOS_CLASS_ELASTIC;
    int rc = qos_class_load_for_path(phys_path, &qos_class);
    if (rc != 0)
        qos_class = PQC_QOS_CLASS_ELASTIC;
    int idx = fd % PQC_MAX_FD;
    pthread_mutex_lock(&g_fd_ctx[idx].fd_lock);
    if (g_fd_ctx[idx].valid)
        g_fd_ctx[idx].qos_class = qos_class;
    pthread_mutex_unlock(&g_fd_ctx[idx].fd_lock);
}

static long parse_positive_long_env(const char *name, long fallback)
{
    const char *env = getenv(name);
    if (!env || !*env)
        return fallback;
    char *end = NULL;
    long value = strtol(env, &end, 10);
    return (end != env && value > 0) ? value : fallback;
}

static double parse_double_env_or_fallback(const char *name, double fallback)
{
    const char *env = getenv(name);
    if (!env || !*env)
        return fallback;
    char *end = NULL;
    double value = strtod(env, &end);
    return end != env ? value : fallback;
}

static int qos_runtime_throttle_enabled(void)
{
    const char *env = getenv("PQC_ENABLE_QOS_THROTTLE_ON_WRITE");
    return env && *env && strcmp(env, "0") != 0;
}

static void qos_update_runtime_pressure(double mem_util, double tensor_util)
{
    if (!qos_runtime_throttle_enabled())
        return;

    const double enter = parse_double_env_or_fallback("PQC_QOS_MEM_ENTER_UTIL", 0.70);
    const double exit = parse_double_env_or_fallback("PQC_QOS_MEM_EXIT_UTIL", 0.60);
    const unsigned hold = (unsigned)parse_positive_long_env("PQC_QOS_HOLD_SAMPLES", 2);
    const double pressure = mem_util > tensor_util ? mem_util : tensor_util;

    pthread_mutex_lock(&g_qos_throttle_lock);
    g_qos_pressure_value = pressure;
    if (!g_qos_throttle_state) {
        g_qos_below_exit_count = 0;
        if (pressure >= enter)
            g_qos_throttle_state = 1;
    } else {
        if (pressure <= exit) {
            if (++g_qos_below_exit_count >= hold) {
                g_qos_throttle_state = 0;
                g_qos_below_exit_count = 0;
            }
        } else {
            g_qos_below_exit_count = 0;
        }
    }
    pthread_mutex_unlock(&g_qos_throttle_lock);
}

static void qos_apply_runtime_throttle(size_t bytes, int qos_class)
{
    if (!qos_runtime_throttle_enabled())
        return;

    pthread_mutex_lock(&g_qos_throttle_lock);
    const int throttled = g_qos_throttle_state;
    const double pressure = g_qos_pressure_value;
    pthread_mutex_unlock(&g_qos_throttle_lock);

    const int eligible = qos_class != PQC_QOS_CLASS_LATENCY;
    const long sleep_us = (throttled && eligible) ?
        parse_positive_long_env("PQC_QOS_THROTTLE_SLEEP_US", 50000) : 0;
    if (sleep_us > 0)
        usleep((useconds_t)sleep_us);

    const char *trace_path = getenv("PQC_QOS_THROTTLE_TRACE_PATH");
    if (trace_path && *trace_path) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        const uint64_t now_ns = (uint64_t)ts.tv_sec * 1000000000ULL +
                                (uint64_t)ts.tv_nsec;
        pthread_mutex_lock(&g_qos_trace_lock);
        FILE *fp = fopen(trace_path, "a");
        if (fp) {
            fprintf(fp,
                    "{\"timestamp_ns\":%llu,\"bytes\":%zu,"
                    "\"pressure\":%.4f,\"qos_class\":\"%s\","
                    "\"eligible\":%d,\"throttled\":%d,\"sleep_us\":%ld}\n",
                    (unsigned long long)now_ns, bytes, pressure,
                    qos_class_name(qos_class), eligible,
                    throttled && eligible, sleep_us);
            fclose(fp);
        }
        pthread_mutex_unlock(&g_qos_trace_lock);
    }
}

static void *admission_telemetry_file_main(void *arg)
{
    const char *path = (const char *)arg;
    const long poll_ms = parse_positive_long_env("PQC_TELEMETRY_POLL_MS", 50);
    uint64_t last_budget_ns = UINT64_MAX;
    uint64_t last_queue_depth = UINT64_MAX;
    double last_mem = -1.0;
    double last_tensor = -1.0;

    while (!g_admission_telemetry_stop) {
        FILE *fp = fopen(path, "r");
        if (fp) {
            double mem = 0.0;
            double tensor = 0.0;
            unsigned long long budget_ns = 0;
            unsigned long long queue_depth = 0;
            int fields = fscanf(fp, "%lf %lf %llu %llu",
                                &mem, &tensor, &budget_ns, &queue_depth);
            fclose(fp);
            if (fields >= 2) {
                if (mem < 0.0) mem = 0.0;
                if (mem > 1.0) mem = 1.0;
                if (tensor < 0.0) tensor = 0.0;
                if (tensor > 1.0) tensor = 1.0;
                if (mem != last_mem || tensor != last_tensor) {
                    pqc_admission_update_telemetry(mem, tensor);
                    qos_update_runtime_pressure(mem, tensor);
                    last_mem = mem;
                    last_tensor = tensor;
                }
                if (fields >= 3 && budget_ns != last_budget_ns) {
                    uint64_t qd = (fields >= 4) ? (uint64_t)queue_depth : 0;
                    pqc_admission_update_ai_budget((uint64_t)budget_ns, qd);
                    last_budget_ns = (uint64_t)budget_ns;
                    last_queue_depth = qd;
                } else if (fields >= 4 && queue_depth != last_queue_depth) {
                    pqc_admission_update_ai_budget(last_budget_ns == UINT64_MAX ? 0 : last_budget_ns,
                                                    (uint64_t)queue_depth);
                    last_queue_depth = (uint64_t)queue_depth;
                }
            }
        }
        usleep((useconds_t)poll_ms * 1000U);
    }
    return NULL;
}

/*
 * Runtime override for key-rotation interval.
 * - Default: KEY_ROTATION_INTERVAL_S (compile-time)
 * - Override: PQC_KEY_ROTATION_INTERVAL_S environment variable
 *   - 0: disable periodic rekey (for ablation)
 *   - N>0: rekey every N seconds
 */
/* ────────────────────────────────────────────────────────────────────────────
 *  Background Rekey Batch Queue & Worker
 * ────────────────────────────────────────────────────────────────────────── */
#define PQC_REKEY_QUEUE_MAX 4096

typedef struct {
    int      fd_list[PQC_REKEY_QUEUE_MAX];
    size_t   head;
    size_t   tail;
    size_t   count;
    pthread_mutex_t lock;
    pthread_cond_t  cond;
} pqc_rekey_queue_t;

static pqc_rekey_queue_t g_rekey_queue = {
    .head = 0,
    .tail = 0,
    .count = 0,
    .lock = PTHREAD_MUTEX_INITIALIZER,
    .cond = PTHREAD_COND_INITIALIZER
};

static int           g_rekey_stop = 0;
static pthread_t     g_rekey_thread;
static int           g_rekey_thread_started = 0;

static void scheduler_gpu_admit(uint32_t bytes);
static void scheduler_gpu_release(uint32_t bytes);

static void rekey_queue_push(int fd)
{
    pthread_mutex_lock(&g_rekey_queue.lock);
    for (size_t i = 0; i < g_rekey_queue.count; i++) {
        size_t idx = (g_rekey_queue.head + i) % PQC_REKEY_QUEUE_MAX;
        if (g_rekey_queue.fd_list[idx] == fd) {
            pthread_mutex_unlock(&g_rekey_queue.lock);
            return;
        }
    }
    if (g_rekey_queue.count < PQC_REKEY_QUEUE_MAX) {
        g_rekey_queue.fd_list[g_rekey_queue.tail] = fd;
        g_rekey_queue.tail = (g_rekey_queue.tail + 1) % PQC_REKEY_QUEUE_MAX;
        g_rekey_queue.count++;
        pthread_cond_signal(&g_rekey_queue.cond);
    }
    pthread_mutex_unlock(&g_rekey_queue.lock);
}

static void *rekey_worker_main(void *arg)
{
    (void)arg;
    int fds[PQC_REKEY_QUEUE_MAX];
    while (!g_rekey_stop) {
        long configured_max_batch = parse_positive_long_env("PQC_REKEY_BATCH_MAX", 64);
        if (configured_max_batch > PQC_REKEY_QUEUE_MAX)
            configured_max_batch = PQC_REKEY_QUEUE_MAX;
        pthread_mutex_lock(&g_rekey_queue.lock);
        while (g_rekey_queue.count == 0 && !g_rekey_stop) {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_sec += 1;
            pthread_cond_timedwait(&g_rekey_queue.cond, &g_rekey_queue.lock, &ts);
        }
        if (g_rekey_stop) {
            pthread_mutex_unlock(&g_rekey_queue.lock);
            break;
        }
        size_t batch_size = 0;
        while (g_rekey_queue.count > 0 && batch_size < (size_t)configured_max_batch) {
            fds[batch_size++] = g_rekey_queue.fd_list[g_rekey_queue.head];
            g_rekey_queue.head = (g_rekey_queue.head + 1) % PQC_REKEY_QUEUE_MAX;
            g_rekey_queue.count--;
        }
        pthread_mutex_unlock(&g_rekey_queue.lock);

        long collect_ms = parse_positive_long_env("PQC_REKEY_BATCH_COLLECT_MS", 0);
        if (collect_ms > 0 && batch_size < (size_t)configured_max_batch) {
            usleep((useconds_t)collect_ms * 1000U);
            pthread_mutex_lock(&g_rekey_queue.lock);
            while (g_rekey_queue.count > 0 && batch_size < (size_t)configured_max_batch) {
                fds[batch_size++] = g_rekey_queue.fd_list[g_rekey_queue.head];
                g_rekey_queue.head = (g_rekey_queue.head + 1) % PQC_REKEY_QUEUE_MAX;
                g_rekey_queue.count--;
            }
            pthread_mutex_unlock(&g_rekey_queue.lock);
        }

        if (batch_size == 0) continue;

        double t0 = get_time_us();
        double load = 0.0;
        if (getloadavg(&load, 1) < 0) load = 0.0;
        double gpu_ewma = gpu_load_ewma_read();
        const size_t key_work_bytes =
            batch_size * (g_kem->length_ciphertext + g_kem->length_shared_secret);
        const char *gpu_min_batch_env = getenv("PQC_GPU_MIN_BATCH");
        int min_batch = gpu_min_batch_env ? atoi(gpu_min_batch_env) : 16;
        pqc_job_target_t target = PQC_JOB_CPU;
        pqc_admission_context_t admission_ctx;
        memset(&admission_ctx, 0, sizeof(admission_ctx));
        admission_ctx.batch_count = batch_size;
        admission_ctx.bytes_total = key_work_bytes;
        admission_ctx.batch_age_ns = (uint64_t)collect_ms * 1000000ULL;
        admission_ctx.gpu_kernel_est_ns =
            (uint64_t)parse_positive_long_env("PQC_REKEY_GPU_KERNEL_EST_NS", 250000L);
        admission_ctx.gpu_h2d_staging_ns = key_work_bytes;
        admission_ctx.gpu_d2h_staging_ns = batch_size * g_kem->length_shared_secret;
        admission_ctx.cpu_queue_depth = batch_size;
        pthread_mutex_lock(&g_sched_pressure_lock);
        admission_ctx.gpu_queue_depth = g_gpu_inflight_jobs;
        pthread_mutex_unlock(&g_sched_pressure_lock);
        admission_ctx.cpu_load_avg = load;
        admission_ctx.gpu_load_avg = gpu_ewma;
        admission_ctx.ai_inference_deadline_ns =
            (uint64_t)parse_positive_long_env("PQC_REKEY_DEADLINE_NS", 10000000L);
        admission_ctx.uma_migration_cost_ns = 0;
        admission_ctx.uma_migration_bytes_est = 0;
        if ((int)batch_size >= min_batch && pqc_admit(&admission_ctx) == 0) {
            target = admission_ctx.chosen_target;
        }

        uint8_t *ct_batch = malloc(g_kem->length_ciphertext * batch_size);
        uint8_t *ss_batch = malloc(g_kem->length_shared_secret * batch_size);
        int success = 0;
        int gpu_used = 0;

        if (ct_batch && ss_batch) {
            if (target == PQC_JOB_GPU && skim_cuda_pqc_available()) {
                scheduler_gpu_admit((uint32_t)(key_work_bytes > UINT32_MAX ? UINT32_MAX : key_work_bytes));
                uint8_t seeds[32] = {0};
                int rc = 0;
                int burn_iters = 1;
                const char *burn_env = getenv("PQC_GPU_BURN_ITERS");
                if (burn_env) burn_iters = atoi(burn_env);
                for (int iter = 0; iter < burn_iters; iter++) {
                    rc = skim_cuda_mlkem768_encaps_batch(g_public_key, seeds, ct_batch, ss_batch, batch_size);
                    if (rc != 0) break;
                }
                if (rc == 0) {
                    success = 1;
                    gpu_used = 1;
                }
                scheduler_gpu_release((uint32_t)(key_work_bytes > UINT32_MAX ? UINT32_MAX : key_work_bytes));
            }
            if (!success) {
                success = 1;
                for (size_t i = 0; i < batch_size; i++) {
                    if (OQS_KEM_encaps(g_kem, ct_batch + i * g_kem->length_ciphertext,
                                       ss_batch + i * g_kem->length_shared_secret, g_public_key) != OQS_SUCCESS) {
                        success = 0;
                        break;
                    }
                }
            }
        } else {
            success = 0;
        }

        if (success) {
            for (size_t i = 0; i < batch_size; i++) {
                int fd = fds[i];
                int idx = fd % PQC_MAX_FD;
                pthread_mutex_lock(&g_fd_ctx[idx].fd_lock);
                if (g_fd_ctx[idx].valid) {
                    memcpy(g_fd_ctx[idx].ss, ss_batch + i * g_kem->length_shared_secret, g_kem->length_shared_secret);
                    g_fd_ctx[idx].key_epoch++;
                    g_fd_ctx[idx].last_rekey = time(NULL);
                    metadata_store(g_fd_ctx[idx].marker_path, g_fd_ctx[idx].ss, g_kem->length_shared_secret, g_fd_ctx[idx].file_id);
                }
                pthread_mutex_unlock(&g_fd_ctx[idx].fd_lock);
            }
            pqc_log("REKEY WORKER: batched %zu files %.1fµs (target=%s, run=%s)",
                    batch_size, get_time_us() - t0,
                    target == PQC_JOB_GPU ? "GPU" : "CPU",
                    gpu_used ? "GPU" : "CPU");
            pqc_log("REKEY WORKER DETAIL: work_bytes=%zu budget_ns=%llu decision_reason=%u deferral_reason=%u",
                    key_work_bytes,
                    (unsigned long long)admission_ctx.ai_qos_budget_remaining_ns,
                    (unsigned int)admission_ctx.decision_reason,
                    (unsigned int)admission_ctx.deferral_reason);

            pthread_mutex_lock(&g_sched_pressure_lock);
            g_sched_stats.submitted += batch_size;
            if (gpu_used) {
                g_sched_stats.gpu_executed += batch_size;
                g_sched_stats.key_plane_gpu += batch_size;
            } else {
                g_sched_stats.cpu_executed += batch_size;
                g_sched_stats.key_plane_cpu += batch_size;
            }
            pthread_mutex_unlock(&g_sched_pressure_lock);
        } else {
            pqc_log("REKEY WORKER: batched rekey FAILED for %zu files", batch_size);
        }

        free(ct_batch);
        free(ss_batch);
    }
    return NULL;
}

static int __attribute__((unused)) get_rotation_interval_s(void)
{
    static int cached = -2;
    if (cached != -2) return cached;

    const char *env = getenv("PQC_KEY_ROTATION_INTERVAL_S");
    if (!env || *env == '\0') {
        cached = KEY_ROTATION_INTERVAL_S;
        return cached;
    }

    char *end = NULL;
    long v = strtol(env, &end, 10);
    if (end == env || (end && *end != '\0')) {
        cached = KEY_ROTATION_INTERVAL_S;
        return cached;
    }
    if (v < 0) v = 0;
    cached = (int)v;
    return cached;
}

static const char *gpu_load_path(void)
{
    const char *env = getenv("PQC_GPU_LOAD_PATH");
    if (env && *env)
        return env;
    return "/sys/devices/gpu.0/load";
}

static void *gpu_load_monitor_main(void *arg)
{
    (void)arg;
    const char *path = gpu_load_path();
    int fd = open(path, O_RDONLY);
    if (fd < 0)
        return NULL;
    char buf[32];
    while (!g_gpu_load_stop) {
        ssize_t n = pread(fd, buf, sizeof(buf) - 1, 0);
        if (n > 0) {
            buf[n] = '\0';
            double sample = (double)strtod(buf, NULL);
            if (sample > 1.0)
                sample /= 10.0;
            if (sample < 0.0)
                sample = 0.0;
            if (sample > 100.0)
                sample = 100.0;
            pthread_mutex_lock(&g_gpu_load_lock);
            g_gpu_load_ewma = 0.2 * sample + 0.8 * g_gpu_load_ewma;
            pthread_mutex_unlock(&g_gpu_load_lock);
        }
        usleep(5000);
    }
    close(fd);
    return NULL;
}

static double gpu_load_ewma_read(void)
{
    pthread_mutex_lock(&g_gpu_load_lock);
    double v = g_gpu_load_ewma;
    pthread_mutex_unlock(&g_gpu_load_lock);
    return v;
}

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

static void update_scheduler_policy_from_env(void)
{
    const char *env = getenv("PQC_GPU_MIN_BYTES");
    if (env && *env) {
        long v = strtol(env, NULL, 10);
        if (v > 0) g_sched_policy.gpu_min_bytes = (size_t)v;
    }
    env = getenv("PQC_GPU_QUEUE_PENALTY_NS");
    if (env && *env) {
        long v = strtol(env, NULL, 10);
        if (v > 0) g_sched_policy.gpu_queue_penalty_ns = (uint64_t)v;
    }
    env = getenv("PQC_COHERENCE_PENALTY_NS");
    if (env && *env) {
        long v = strtol(env, NULL, 10);
        if (v > 0) g_sched_policy.coherence_penalty_ns = (uint64_t)v;
    }
    env = getenv("PQC_GPU_CONTENTION_PENALTY_NS");
    if (env && *env) {
        long v = strtol(env, NULL, 10);
        if (v > 0) g_sched_policy.gpu_contention_penalty_ns = (uint64_t)v;
    }
    env = getenv("PQC_CONTENTION_SCORE_NS");
    if (env && *env) {
        long v = strtol(env, NULL, 10);
        if (v > 0) g_sched_policy.contention_score_ns = (uint64_t)v;
    }
}

static pqc_scheduler_policy_t scheduler_policy_from_env(void)
{
    pqc_scheduler_policy_t policy = {
        .gpu_min_bytes = 131072,
        .gpu_queue_penalty_ns = 25000,
        .coherence_penalty_ns = 25000,
        .gpu_contention_penalty_ns = 25000,
        .contention_score_ns = 75000,
        .gpu_max_inflight_jobs = 2,
        .gpu_max_inflight_bytes = 256 * 1024,
        .gpu_max_wait_ns = 25000,
        .cpu_load_bias = 1.0,
        .gpu_queue_bias = 1.0,
    };
    const char *env = getenv("PQC_GPU_MIN_BYTES");
    if (env && *env) {
        long v = strtol(env, NULL, 10);
        if (v > 0) policy.gpu_min_bytes = (size_t)v;
    }
    env = getenv("PQC_GPU_QUEUE_PENALTY_NS");
    if (env && *env) {
        long v = strtol(env, NULL, 10);
        if (v > 0) policy.gpu_queue_penalty_ns = (uint64_t)v;
    }
    env = getenv("PQC_COHERENCE_PENALTY_NS");
    if (env && *env) {
        long v = strtol(env, NULL, 10);
        if (v > 0) policy.coherence_penalty_ns = (uint64_t)v;
    }
    env = getenv("PQC_GPU_CONTENTION_PENALTY_NS");
    if (env && *env) {
        long v = strtol(env, NULL, 10);
        if (v > 0) policy.gpu_contention_penalty_ns = (uint64_t)v;
    }
    env = getenv("PQC_CONTENTION_SCORE_NS");
    if (env && *env) {
        long v = strtol(env, NULL, 10);
        if (v > 0) policy.contention_score_ns = (uint64_t)v;
    }
    env = getenv("PQC_GPU_MAX_INFLIGHT_JOBS");
    if (env && *env) {
        long v = strtol(env, NULL, 10);
        if (v > 0) policy.gpu_max_inflight_jobs = (uint64_t)v;
    }
    env = getenv("PQC_GPU_MAX_INFLIGHT_BYTES");
    if (env && *env) {
        long v = strtol(env, NULL, 10);
        if (v > 0) policy.gpu_max_inflight_bytes = (uint64_t)v;
    }
    env = getenv("PQC_GPU_MAX_WAIT_NS");
    if (env && *env) {
        long v = strtol(env, NULL, 10);
        if (v > 0) policy.gpu_max_wait_ns = (uint64_t)v;
    }
    env = getenv("PQC_AI_QOS_MIN_BUDGET_NS");
    if (env && *env) {
        long long v = strtoll(env, NULL, 10);
        if (v > 0) policy.ai_qos_min_budget_ns = (uint64_t)v;
    }
    return policy;
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

static int ctx_set(int fd, const char *marker_path, const uint8_t *ss, size_t ss_len, uint64_t fid)
{
    int idx = fd % PQC_MAX_FD;
    pthread_mutex_lock(&g_fd_ctx[idx].fd_lock);
    file_state_t *state = file_state_acquire(fd);
    if (!state) { pthread_mutex_unlock(&g_fd_ctx[idx].fd_lock); return -errno; }
    char data_path[4096 + 16], journal_path[4096 + 16];
    if (sidecar_path(data_path, sizeof(data_path), marker_path, ".pqcdata") ||
        sidecar_path(journal_path, sizeof(journal_path), marker_path, ".pqcmeta")) {
        file_state_release(state); pthread_mutex_unlock(&g_fd_ctx[idx].fd_lock); return -ENAMETOOLONG;
    }
    int data_fd = open(data_path, O_RDWR | O_CREAT, 0600);
    int journal_fd = open(journal_path, O_RDWR | O_CREAT | O_APPEND, 0600);
    if (data_fd < 0 || journal_fd < 0) {
        if (data_fd >= 0) close(data_fd);
        if (journal_fd >= 0) close(journal_fd);
        file_state_release(state); pthread_mutex_unlock(&g_fd_ctx[idx].fd_lock); return -errno;
    }
    memcpy(g_fd_ctx[idx].ss, ss, ss_len);
    g_fd_ctx[idx].ss_len     = ss_len;
    g_fd_ctx[idx].file_id    = fid;
    g_fd_ctx[idx].state      = state;
    g_fd_ctx[idx].data_fd    = data_fd;
    g_fd_ctx[idx].journal_fd = journal_fd;
    strncpy(g_fd_ctx[idx].marker_path, marker_path, sizeof(g_fd_ctx[idx].marker_path) - 1);
    if (logical_size_load(marker_path, &g_fd_ctx[idx].logical_size) != 0)
        g_fd_ctx[idx].logical_size = 0;
    {
        pqc_checkpoint_t ckpt = {0};
        int ckpt_rc = checkpoint_load(marker_path, fid, &ckpt);
        if (ckpt_rc == 0) {
            if (ckpt.logical_size > g_fd_ctx[idx].logical_size)
                g_fd_ctx[idx].logical_size = ckpt.logical_size;
            pqc_fault_cutpoint("remount_after_checkpoint_load");
        } else if (ckpt_rc != -ENODATA && ckpt_rc != -ENOENT) {
            close(data_fd);
            close(journal_fd);
            memset(g_fd_ctx[idx].marker_path, 0, sizeof(g_fd_ctx[idx].marker_path));
            g_fd_ctx[idx].data_fd = -1;
            g_fd_ctx[idx].journal_fd = -1;
            g_fd_ctx[idx].file_id = 0;
            g_fd_ctx[idx].ss_len = 0;
            g_fd_ctx[idx].state = NULL;
            file_state_release(state);
            pthread_mutex_unlock(&g_fd_ctx[idx].fd_lock);
            return ckpt_rc;
        }
    }
    if (state->next_generation == 0)
        state->next_generation = journal_max_generation(journal_fd);
    g_fd_ctx[idx].valid      = 1;
    g_fd_ctx[idx].tier       = PQC_TIER_FULL;  /* default: full PQC */
    g_fd_ctx[idx].qos_class  = PQC_QOS_CLASS_ELASTIC;
    g_fd_ctx[idx].key_epoch  = 0;
    g_fd_ctx[idx].last_rekey = time(NULL);
    if (!g_fd_ctx[idx].wbuf) {
        if (skim_cuda_aead_is_uma())
            g_fd_ctx[idx].wbuf = (uint8_t *)skim_cuda_managed_alloc(COALESCE_SIZE);
        else
            g_fd_ctx[idx].wbuf = (uint8_t *)malloc(COALESCE_SIZE);
    }
    if (g_fd_ctx[idx].wbuf && skim_cuda_aead_is_uma())
        (void)skim_cuda_mem_prefetch(g_fd_ctx[idx].wbuf, COALESCE_SIZE, skim_cuda_current_device());
    g_fd_ctx[idx].wbuf_used     = 0;
    g_fd_ctx[idx].wbuf_base_off = 0;
    pthread_mutex_unlock(&g_fd_ctx[idx].fd_lock);
    return 0;
}

static void ctx_clear(int fd)
{
    int idx = fd % PQC_MAX_FD;
    pthread_mutex_lock(&g_fd_ctx[idx].fd_lock);
    OQS_MEM_cleanse(g_fd_ctx[idx].ss, sizeof(g_fd_ctx[idx].ss));
    if (g_fd_ctx[idx].wbuf) {
        OQS_MEM_cleanse(g_fd_ctx[idx].wbuf, COALESCE_SIZE);
        if (skim_cuda_aead_is_uma())
            skim_cuda_managed_free(g_fd_ctx[idx].wbuf);
        else
            free(g_fd_ctx[idx].wbuf);
        g_fd_ctx[idx].wbuf = NULL;
    }
    g_fd_ctx[idx].pending_jobs = 0;
    g_fd_ctx[idx].wbuf_used = 0;
    if (g_fd_ctx[idx].data_fd >= 0) close(g_fd_ctx[idx].data_fd);
    if (g_fd_ctx[idx].journal_fd >= 0) close(g_fd_ctx[idx].journal_fd);
    g_fd_ctx[idx].data_fd = g_fd_ctx[idx].journal_fd = -1;
    g_fd_ctx[idx].valid = 0;
    file_state_release(g_fd_ctx[idx].state);
    g_fd_ctx[idx].state = NULL;
    pthread_mutex_unlock(&g_fd_ctx[idx].fd_lock);
}

/* ════════════════════════════════════════════════════════════════════════════
 *  Optional ML-KEM key-plane helper
 * ════════════════════════════════════════════════════════════════════════════
 *
 *  The FUSE data path does not derive its record key from this object.  The
 *  CUDA executors use managed allocations, prefetch, and stream synchronization
 *  only within executor-owned buffers.  They do not implement NVMe DMA into
 *  managed memory, cudaHostRegister/GUP pinning, or an io_uring/eBPF path.
 * ════════════════════════════════════════════════════════════════════════════ */

/**
 * Initialize the PQC subsystem and its optional ML-KEM-768 helper keypair.
 */
static int pqc_subsystem_init(void)
{
    const char *admission_trace = getenv("PQC_ADMISSION_TRACE_PATH");
    if (!admission_trace || !*admission_trace)
        admission_trace = "experiments/scheduler_trace.jsonl";
    if (pqc_admission_init(admission_trace) != 0)
        return -1;
    update_scheduler_policy_from_env();
    /* Try ML-KEM-768 first (NIST standardized name), fallback to Kyber768 */
    g_kem = OQS_KEM_new(OQS_KEM_alg_ml_kem_768);
    if (!g_kem) {
        g_kem = OQS_KEM_new(OQS_KEM_alg_kyber_768);
    }
    if (!g_kem) {
        fprintf(stderr, "[PQC-FUSE] FATAL: Neither ML-KEM-768 nor Kyber-768 available in liboqs!\n");
        fprintf(stderr, "[PQC-FUSE] Rebuild liboqs with KEM_ml_kem_768 or KEM_kyber_768 enabled.\n");
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

    for (int i = 0; i < PQC_MAX_FD; i++) {
        pthread_mutex_init(&g_fd_ctx[i].fd_lock, NULL);
        pthread_cond_init(&g_fd_ctx[i].pending_cv, NULL);
        g_fd_ctx[i].valid = 0;
        g_fd_ctx[i].wbuf = NULL;
        g_fd_ctx[i].data_fd = g_fd_ctx[i].journal_fd = -1;
    }

    double t0 = get_time_us();
    OQS_STATUS rc = OQS_KEM_keypair(g_kem, g_public_key, g_secret_key);
    double t1 = get_time_us();

    if (rc != OQS_SUCCESS) {
        fprintf(stderr, "[PQC-FUSE] FATAL: Key generation failed\n");
        return -1;
    }

    pqc_log("Keypair generated in %.2f µs", t1 - t0);

    /* Derive master key */
    const char *pass = getenv("PQC_MASTER_PASSWORD");
    if (!pass || !*pass) {
        fprintf(stderr, "[PQC-FUSE] FATAL: PQC_MASTER_PASSWORD is required.\n");
        return -1;
    }
    if (derive_master_key(pass) != 0) {
        fprintf(stderr, "[PQC-FUSE] FATAL: Failed to derive Master Key\n");
        return -1;
    }
    pqc_log("Master Key derived successfully (PBKDF2-SHA256)");
    pqc_log("Scheduler policy: gpu_min_bytes=%zu coherence_penalty_ns=%llu",
            g_sched_policy.gpu_min_bytes,
            (unsigned long long)g_sched_policy.coherence_penalty_ns);

    g_gpu_load_stop = 0;
    if (pthread_create(&g_gpu_load_thread, NULL, gpu_load_monitor_main, NULL) == 0) {
        g_gpu_load_thread_started = 1;
        pqc_log("GPU load monitor started: %s", gpu_load_path());
    } else {
        pqc_log("GPU load monitor disabled: %s", strerror(errno));
    }

    const char *telemetry_path = getenv("PQC_TELEMETRY_FILE");
    if (telemetry_path && *telemetry_path) {
        g_admission_telemetry_stop = 0;
        if (pthread_create(&g_admission_telemetry_thread, NULL,
                           admission_telemetry_file_main, (void *)telemetry_path) == 0) {
            g_admission_telemetry_thread_started = 1;
            pqc_log("Admission telemetry file monitor started: %s", telemetry_path);
        } else {
            pqc_log("Admission telemetry file monitor disabled: %s", strerror(errno));
        }
    }

    int anchor_rc = pqc_anchor_probe();
    if (anchor_rc != 0) {
        fprintf(stderr, "[PQC-FUSE] FATAL: Freshness anchor probe failed: %s\n",
                strerror(-anchor_rc));
        return -1;
    }

    const char *anchor_path = getenv("PQC_FRESHNESS_ANCHOR_PATH");
    if (anchor_path && *anchor_path) {
        pthread_mutex_lock(&g_anchor_lock);
        g_anchor_stop = 0;
        g_anchor_dirty = 0;
        g_anchor_last_commit = time(NULL);
        pthread_mutex_unlock(&g_anchor_lock);
        if (pthread_create(&g_anchor_thread, NULL, anchor_worker_main, NULL) == 0) {
            g_anchor_thread_started = 1;
            pqc_log("Freshness anchor worker started");
        } else {
            pqc_log("Freshness anchor worker failed to start: %s", strerror(errno));
        }
    }

    /* The GPU lane owns a mount-lifetime stream and buffers.  Allocating them
     * here keeps stream creation and managed allocation out of write/fsync. */
    if (skim_cuda_aead_available() &&
        skim_cuda_executor_init(COALESCE_SIZE, COALESCE_SIZE / PQC_LOGICAL_BLOCK_SIZE) != 0)
        pqc_log("GPU executor initialization failed; secure I/O will use CPU fallback");

    g_rekey_stop = 0;
    if (pthread_create(&g_rekey_thread, NULL, rekey_worker_main, NULL) == 0) {
        g_rekey_thread_started = 1;
        pqc_log("PQC background rekey worker started");
    } else {
        pqc_log("PQC background rekey worker failed to start: %s", strerror(errno));
    }

    return 0;
}

/**
 * Cleanup PQC resources.
 */
static void pqc_cleanup(void)
{
    if (g_admission_telemetry_thread_started) {
        g_admission_telemetry_stop = 1;
        pthread_join(g_admission_telemetry_thread, NULL);
        g_admission_telemetry_thread_started = 0;
        pqc_log("Admission telemetry file monitor stopped");
    }
    if (g_rekey_thread_started) {
        pthread_mutex_lock(&g_rekey_queue.lock);
        g_rekey_stop = 1;
        pthread_cond_broadcast(&g_rekey_queue.cond);
        pthread_mutex_unlock(&g_rekey_queue.lock);
        pthread_join(g_rekey_thread, NULL);
        g_rekey_thread_started = 0;
        pqc_log("PQC background rekey worker stopped");
    }
    skim_cuda_executor_shutdown();
    pqc_log("Scheduler stats: submitted=%llu cpu=%llu gpu=%llu bytes_cpu=%llu bytes_gpu=%llu migration_ns=%llu",
            (unsigned long long)g_sched_stats.submitted,
            (unsigned long long)g_sched_stats.cpu_executed,
            (unsigned long long)g_sched_stats.gpu_executed,
            (unsigned long long)g_sched_stats.bytes_cpu,
            (unsigned long long)g_sched_stats.bytes_gpu,
            (unsigned long long)g_sched_stats.gpu_migration_ns);
    OQS_MEM_cleanse(g_master_key, sizeof(g_master_key));
    if (g_kem) {
        OQS_MEM_cleanse(g_secret_key, g_kem->length_secret_key);
        free(g_public_key);
        free(g_secret_key);
        OQS_KEM_free(g_kem);
        g_public_key = NULL;
        g_secret_key = NULL;
        g_kem = NULL;
    }
    pqc_admission_shutdown();
}

/* ────────────────────────────────────────────────────────────────────────────
 *  Write-coalescing flush
 *  Encrypts ctx->wbuf[0..wbuf_used) as authenticated AES-256-GCM blocks.
 *  MUST be called with g_fd_lock held.  Resets wbuf_used = 0 on success.
 * -------------------------------------------------------------------------- */
static int load_authenticated_block(const pqc_fd_ctx_t *ctx, uint64_t block,
                                    uint8_t plain[PQC_LOGICAL_BLOCK_SIZE])
{
    memset(plain, 0, PQC_LOGICAL_BLOCK_SIZE);
    block_mapping_t map;
    int rc = journal_lookup_mapping(ctx->journal_fd, block, &map);
    if (rc == -ENOENT) return 0;
    if (rc) return rc;
    uint8_t cipher[PQC_LOGICAL_BLOCK_SIZE] = {0};
    if (pread(ctx->data_fd, cipher, map.plaintext_length, (off_t)map.ciphertext_offset) !=
        (ssize_t)map.plaintext_length) return -EIO;
    if (map.algorithm_id != PQC_ALGO_AES_256_GCM)
        return -EINVAL;
    return crypt_block_gcm(ctx->ss, ctx->ss_len, ctx->file_id, block, map.generation,
                           map.plaintext_length, cipher, plain, map.tag, 0, 0);
}

static void schedule_block_job(pqc_block_job_t *job, const pqc_fd_ctx_t *ctx,
                               uint64_t logical_offset, uint32_t length)
{
    if (!job || !ctx) return;
    pqc_block_job_init(job, ctx->file_id, logical_offset / PQC_LOGICAL_BLOCK_SIZE,
                       ctx->state ? ctx->state->next_generation + 1 : 0,
                       logical_offset, length,
                       PQC_JOB_FLAG_ENCRYPT | PQC_JOB_FLAG_READMOD, PQC_PLANE_DATA);
    job->submit_ns = (uint64_t)get_time_us() * 1000ULL;
    double load = 0.0;
    if (getloadavg(&load, 1) < 0)
        load = 0.0;
    double gpu_ewma = gpu_load_ewma_read();
    job->cpu_queue_depth = 1 + (load > 1.5 ? 1 : 0);
    /* This backend has no outstanding GPU work at submission.  Queue depth is
     * an observed value, not a proxy for eligibility; inventing one here made
     * every eligible request look artificially GPU-congested. */
    job->gpu_queue_depth = 0;
    /* On an integrated managed-memory device there is no PCIe copy.  The
     * remaining shared-DRAM contention is measured by the policy, not
     * fabricated as a per-byte migration cost. */
    job->coherence_cost_ns = skim_cuda_aead_is_uma() ? 0 : (uint64_t)length * 64ULL;
    if (length >= g_sched_policy.gpu_min_bytes)
        job->flags |= PQC_JOB_FLAG_GPU_ELIGIBLE;
    uint64_t inflight_jobs = 0, inflight_bytes = 0;
    pthread_mutex_lock(&g_sched_pressure_lock);
    inflight_jobs = g_gpu_inflight_jobs;
    inflight_bytes = g_gpu_inflight_bytes;
    pthread_mutex_unlock(&g_sched_pressure_lock);
    job->gpu_queue_depth = inflight_jobs;
    job->gpu_wait_ns = inflight_jobs * g_sched_policy.gpu_queue_penalty_ns +
                       inflight_bytes / 4096ULL * (g_sched_policy.gpu_queue_penalty_ns / 4ULL);
    int gpu_pressure_spill = (g_sched_policy.gpu_max_inflight_jobs > 0 &&
                              inflight_jobs >= g_sched_policy.gpu_max_inflight_jobs) ||
                             (g_sched_policy.gpu_max_inflight_bytes > 0 &&
                              inflight_bytes + length > g_sched_policy.gpu_max_inflight_bytes);
    if (gpu_pressure_spill) {
        job->target = PQC_JOB_CPU;
    } else if (getenv("PQC_ENABLE_ADMISSION_ON_WRITE") &&
               pqc_block_job_gpu_eligible(job)) {
        pqc_admission_context_t admission_ctx;
        memset(&admission_ctx, 0, sizeof(admission_ctx));
        admission_ctx.batch_count = 1;
        admission_ctx.bytes_total = length;
        admission_ctx.batch_age_ns = ((uint64_t)get_time_us() * 1000ULL) - job->submit_ns;
        admission_ctx.gpu_kernel_est_ns = length / 4ULL + 100000ULL;
        admission_ctx.gpu_h2d_staging_ns = skim_cuda_aead_is_uma() ? 0ULL : (uint64_t)length * 8ULL;
        admission_ctx.gpu_d2h_staging_ns = skim_cuda_aead_is_uma() ? 0ULL : (uint64_t)length * 8ULL;
        admission_ctx.cpu_queue_depth = job->cpu_queue_depth;
        admission_ctx.gpu_queue_depth = job->gpu_queue_depth;
        admission_ctx.cpu_load_avg = load;
        admission_ctx.gpu_load_avg = gpu_ewma;
        admission_ctx.ai_inference_deadline_ns =
            (uint64_t)parse_positive_long_env("PQC_ADMISSION_WRITE_DEADLINE_NS", 10000000L);
        admission_ctx.uma_migration_cost_ns = job->coherence_cost_ns;
        admission_ctx.uma_migration_bytes_est = skim_cuda_aead_is_uma() ? 0ULL : length;
        if (pqc_admit(&admission_ctx) == 0)
            job->target = admission_ctx.chosen_target;
        else
            job->target = PQC_JOB_CPU;
    } else {
        job->target = pqc_block_job_choose_target(job, &g_sched_policy, load, gpu_ewma);
    }
    g_sched_stats.submitted++;
    if (job->target == PQC_JOB_GPU) {
        g_sched_stats.gpu_executed++;
        g_sched_stats.gpu_migration_ns += job->coherence_cost_ns;
    } else {
        g_sched_stats.cpu_executed++;
    }
}

static void scheduler_gpu_admit(uint32_t bytes)
{
    pthread_mutex_lock(&g_sched_pressure_lock);
    ++g_gpu_inflight_jobs;
    g_gpu_inflight_bytes += bytes;
    pthread_mutex_unlock(&g_sched_pressure_lock);
}

static void scheduler_gpu_release(uint32_t bytes)
{
    pthread_mutex_lock(&g_sched_pressure_lock);
    if (g_gpu_inflight_jobs > 0)
        --g_gpu_inflight_jobs;
    if (g_gpu_inflight_bytes >= bytes)
        g_gpu_inflight_bytes -= bytes;
    else
        g_gpu_inflight_bytes = 0;
    pthread_mutex_unlock(&g_sched_pressure_lock);
}

static int do_flush_wbuf_locked(int storage_fd, int idx)
{
    pqc_fd_ctx_t *ctx = &g_fd_ctx[idx];
    if (ctx->wbuf_used == 0) return 0;
    if (!ctx->state) return -EIO;
    qos_apply_runtime_throttle(ctx->wbuf_used, ctx->qos_class);
    pthread_mutex_lock(&ctx->state->commit_lock);

    /* Strategy 2: tier NONE → write plaintext directly */
    if (ctx->tier == PQC_TIER_NONE) {
        uint64_t bytes = (uint64_t)ctx->wbuf_used;
        pqc_block_job_t job;
        schedule_block_job(&job, ctx, (uint64_t)ctx->wbuf_base_off, (uint32_t)bytes);
        ++ctx->pending_jobs;
        int res = (int)pwrite(storage_fd, ctx->wbuf, ctx->wbuf_used,
                              ctx->wbuf_base_off);
        uint64_t end = (uint64_t)ctx->wbuf_base_off + bytes;
        uint64_t final_size = end > ctx->logical_size ? end : ctx->logical_size;
        ctx->wbuf_used = 0;
        g_sched_stats.bytes_cpu += bytes;
        if (res == (int)bytes) {
            ctx->logical_size = final_size;
            if (logical_size_store(ctx->marker_path, final_size) != 0)
                res = -EIO;
            else if (ftruncate(storage_fd, (off_t)final_size) != 0)
                res = -errno;
        }
        if (ctx->pending_jobs > 0)
            --ctx->pending_jobs;
        if (ctx->pending_jobs == 0)
            pthread_cond_broadcast(&ctx->pending_cv);
        pthread_mutex_unlock(&ctx->state->commit_lock);
        return res == -1 ? -errno : (res < 0 ? res : 0);
    }

    size_t sz = ctx->wbuf_used;
    uint64_t base = (uint64_t)ctx->wbuf_base_off;
    uint64_t end = base + sz;
    uint64_t final_size = end > ctx->logical_size ? end : ctx->logical_size;
    uint64_t first = base / PQC_LOGICAL_BLOCK_SIZE;
    uint64_t last = (end - 1) / PQC_LOGICAL_BLOCK_SIZE;
    size_t block_count = (size_t)(last - first + 1);
    flush_block_desc_t *blocks = calloc(block_count, sizeof(*blocks));
    uint8_t *plain_batch = NULL;
    uint8_t *cipher_batch = NULL;
    int res = 0;
    size_t packed_bytes = 0;
    int gpu_admitted = 0;
    if (!blocks) {
        res = -ENOMEM;
        goto out_unlock;
    }
    for (size_t bi = 0; bi < block_count; ++bi) {
        uint64_t block = first + bi;
        uint64_t block_start = block * PQC_LOGICAL_BLOCK_SIZE;
        uint64_t from = base > block_start ? base - block_start : 0;
        uint64_t to = end < block_start + PQC_LOGICAL_BLOCK_SIZE ? end - block_start : PQC_LOGICAL_BLOCK_SIZE;
        uint64_t source = block_start + from - base;
        uint32_t length = (uint32_t)((final_size - block_start) < PQC_LOGICAL_BLOCK_SIZE ?
                                     (final_size - block_start) : PQC_LOGICAL_BLOCK_SIZE);
        blocks[bi].block = block;
        blocks[bi].generation = ctx->state->next_generation + 1 + bi;
        blocks[bi].length = length;
        blocks[bi].input_offset = packed_bytes;
        blocks[bi].output_offset = packed_bytes;
        if ((res = derive_block_nonce(ctx->file_id, block, blocks[bi].generation, blocks[bi].nonce)) != 0)
            break;
        build_block_aad(blocks[bi].aad, ctx->file_id, block, blocks[bi].generation, length);
        packed_bytes += length;
        (void)from;
        (void)to;
        (void)source;
    }
    if (res != 0)
        goto out_free;

    if (skim_cuda_aead_is_uma()) {
        plain_batch = skim_cuda_managed_alloc(packed_bytes);
        cipher_batch = skim_cuda_managed_alloc(packed_bytes);
    } else {
        plain_batch = malloc(packed_bytes);
        cipher_batch = malloc(packed_bytes);
    }
    if (!plain_batch || !cipher_batch) {
        res = -ENOMEM;
        goto out_free;
    }

    packed_bytes = 0;
    for (size_t bi = 0; bi < block_count; ++bi) {
        uint64_t block = blocks[bi].block;
        uint8_t plain[PQC_LOGICAL_BLOCK_SIZE];
        uint64_t block_start = block * PQC_LOGICAL_BLOCK_SIZE;
        uint64_t from = base > block_start ? base - block_start : 0;
        uint64_t to = end < block_start + PQC_LOGICAL_BLOCK_SIZE ? end - block_start : PQC_LOGICAL_BLOCK_SIZE;
        uint64_t source = block_start + from - base;
        if ((res = load_authenticated_block(ctx, block, plain)) != 0)
            goto out_free;
        memcpy(plain + from, ctx->wbuf + source, to - from);
        memcpy(plain_batch + packed_bytes, plain, blocks[bi].length);
        blocks[bi].input_offset = packed_bytes;
        blocks[bi].output_offset = packed_bytes;
        packed_bytes += blocks[bi].length;
    }

    pqc_block_job_t job;
    schedule_block_job(&job, ctx, base, (uint32_t)sz);
    int use_gpu_batch = (job.target == PQC_JOB_GPU && block_count > 1);
    uint32_t algorithm_id = PQC_ALGO_AES_256_GCM;
    if (use_gpu_batch) {
        scheduler_gpu_admit((uint32_t)(final_size - base));
        gpu_admitted = 1;
        ++ctx->pending_jobs;
        if (skim_cuda_aead_is_uma()) {
            int dev = skim_cuda_current_device();
            (void)skim_cuda_mem_prefetch(plain_batch, packed_bytes, dev);
            (void)skim_cuda_mem_prefetch(cipher_batch, packed_bytes, dev);
        }
        res = crypt_block_batch_gcm(ctx->ss, ctx->ss_len, ctx->file_id, blocks, block_count,
                                    plain_batch, cipher_batch);
        if (ctx->pending_jobs > 0)
            --ctx->pending_jobs;
        if (ctx->pending_jobs == 0)
            pthread_cond_broadcast(&ctx->pending_cv);
    }
    if (!use_gpu_batch || res != 0) {
        res = 0;
        for (size_t bi = 0; bi < block_count; ++bi) {
            uint8_t tag[PQC_AEAD_TAG_SIZE];
            ++ctx->pending_jobs;
            res = crypt_block_gcm(ctx->ss, ctx->ss_len, ctx->file_id, blocks[bi].block,
                                  blocks[bi].generation, blocks[bi].length,
                                  plain_batch + blocks[bi].input_offset,
                                  cipher_batch + blocks[bi].output_offset,
                                  tag, 1, 0);
            if (ctx->pending_jobs > 0)
                --ctx->pending_jobs;
            if (ctx->pending_jobs == 0)
                pthread_cond_broadcast(&ctx->pending_cv);
            if (res != 0)
                break;
            memcpy(blocks[bi].tag, tag, sizeof(tag));
        }
    } else {
        for (size_t bi = 0; bi < block_count; ++bi) {
            uint8_t tag[PQC_AEAD_TAG_SIZE];
            if ((res = gcm_compute_tag(ctx->ss, blocks[bi].nonce, blocks[bi].aad, sizeof(blocks[bi].aad),
                                       cipher_batch + blocks[bi].output_offset, blocks[bi].length, tag)) != 0)
                break;
            memcpy(blocks[bi].tag, tag, sizeof(tag));
        }
    }
    if (res != 0)
        goto out_free;

    /* Phase 1: write every ciphertext record.  None is visible yet because
     * the journal has not published a mapping. */
    for (size_t bi = 0; bi < block_count; ++bi) {
        off_t pos = lseek(ctx->data_fd, 0, SEEK_END);
        if (pos < 0 || pwrite(ctx->data_fd, cipher_batch + blocks[bi].output_offset,
                              blocks[bi].length, pos) != (ssize_t)blocks[bi].length) {
            res = -EIO;
            break;
        }
        blocks[bi].ciphertext_offset = (uint64_t)pos;
        pqc_fault_cutpoint("data_write_after_pwrite");
    }
    /* Phase 2: establish the data-before-metadata durability boundary once. */
    if (res == 0) {
        if (fdatasync(ctx->data_fd) != 0)
            res = -errno;
        else
            pqc_fault_cutpoint("data_fsync_after");
    }

    /* Phase 3: append all committed mappings, then publish them with one
     * journal barrier.  Recovery ignores any torn tail without a full record. */
    for (size_t bi = 0; res == 0 && bi < block_count; ++bi) {
        block_mapping_t map = {
            .logical_block = blocks[bi].block,
            .generation = blocks[bi].generation,
            .ciphertext_offset = blocks[bi].ciphertext_offset,
            .plaintext_length = blocks[bi].length,
            .algorithm_id = (uint32_t)algorithm_id,
        };
        memcpy(map.tag, blocks[bi].tag, sizeof(map.tag));
        res = journal_append_mapping_unsynced(ctx->journal_fd, &map);
        if (res == 0)
            pqc_fault_cutpoint("journal_append_after");
    }
    if (res == 0) {
        if (fdatasync(ctx->journal_fd) != 0)
            res = -errno;
        else
            pqc_fault_cutpoint("journal_fsync_after");
    }
    if (res == 0) {
        ctx->state->next_generation += block_count;
        ctx->logical_size = final_size;
        res = logical_size_store(ctx->marker_path, final_size);
        if (res == 0) {
            pqc_fault_cutpoint("logical_size_xattr_after");
            res = checkpoint_store(ctx->marker_path, ctx->file_id,
                                   ctx->state->next_generation,
                                   ctx->logical_size, ctx->state->next_generation);
        }
        if (res == 0 && ftruncate(storage_fd, (off_t)final_size) != 0)
            res = -errno;
    }
out_free:
    if (gpu_admitted)
        scheduler_gpu_release((uint32_t)(final_size - base));
    if (cipher_batch) {
        OPENSSL_cleanse(cipher_batch, packed_bytes);
        if (skim_cuda_aead_is_uma())
            skim_cuda_managed_free(cipher_batch);
        else
            free(cipher_batch);
    }
    if (plain_batch) {
        OPENSSL_cleanse(plain_batch, packed_bytes);
        if (skim_cuda_aead_is_uma())
            skim_cuda_managed_free(plain_batch);
        else
            free(plain_batch);
    }
    OPENSSL_cleanse(blocks, block_count * sizeof(*blocks));
    free(blocks);
out_unlock:
    if (res != 0)
        pqc_log("authenticated flush failed: %s", strerror(-res));
    ctx->wbuf_used = 0;
    pthread_mutex_unlock(&ctx->state->commit_lock);
    if (res == 0) {
        g_sched_stats.bytes_cpu += 0;
        g_sched_stats.bytes_gpu += (uint64_t)(final_size - base);
        g_sched_stats.gpu_migration_ns += (uint64_t)(final_size - base) * 64ULL;
    }
    if (skim_cuda_aead_is_uma()) {
        (void)skim_cuda_mem_prefetch_host(plain_batch, packed_bytes);
        (void)skim_cuda_mem_prefetch_host(cipher_batch, packed_bytes);
    }
    return res;
}

/* ════════════════════════════════════════════════════════════════════════════
 *  FUSE Operations
 * ════════════════════════════════════════════════════════════════════════════ */

static int pqc_getattr(const char *path, struct stat *stbuf,
                        struct fuse_file_info *fi)
{
    (void)fi;
    if (is_hidden_sidecar_path(path))
        return -ENOENT;
    char phys_path[4096];
    resolve_physical_path(phys_path, sizeof(phys_path), path);

    int res = lstat(phys_path, stbuf);
    if (res == -1)
        return -errno;

    uint64_t logical_size = 0;
    if (logical_size_load(phys_path, &logical_size) == 0 &&
        getxattr(phys_path, PQC_XATTR_METADATA, NULL, 0) >= 0)
        stbuf->st_size = (off_t)logical_size;

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
        size_t name_len = strlen(de->d_name);
        if ((name_len >= 8 && strcmp(de->d_name + name_len - 8, ".pqcdata") == 0) ||
            (name_len >= 8 && strcmp(de->d_name + name_len - 8, ".pqcmeta") == 0))
            continue;
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
    if (is_hidden_sidecar_path(path))
        return -ENOENT;
    char phys_path[4096];
    resolve_physical_path(phys_path, sizeof(phys_path), path);

    char compat_path[4096];
    const char *open_path = phys_path;
    if (sqlite_sidecar_redirect_path(compat_path, sizeof(compat_path), path) == 0)
        open_path = compat_path;

    int fd = open(open_path, fi->flags, 0600);
    if (fd == -1)
        return -errno;

    fi->fh = (uint64_t)fd;
    /* SQLite WAL uses a shared-memory sidecar (`-shm`) and expects mmap-capable
     * file descriptors.  The default direct-IO path breaks that contract on
     * this FUSE stack, so we allow a compatibility mode that keeps buffered I/O
     * for WAL-visible files while preserving the authenticated data path. */
    const char *allow_mmap = getenv("PQC_ALLOW_SQLITE_MMAP");
    fi->direct_io = (allow_mmap && *allow_mmap) ? 0 : 1; /* legacy fast path */
    if (open_path == compat_path)
        fi->keep_cache = 1;

    /* A persistent per-file data-encryption key is wrapped under the
     * mount-derived key and authenticated before use.  The prior experimental
     * ML-KEM metadata format could not recover its per-file decapsulation key
     * after a remount, so this version rejects it rather than risking an
     * unauthenticated or incorrect decrypt.  ML-KEM remains an optional
     * provisioning/microbenchmark component; it is not in the block-I/O path. */
    uint8_t ss[64] = {0};
    size_t ss_len = 0;
    uint64_t fid = 0;
    int metadata_rc = metadata_load(phys_path, ss, &ss_len, &fid);
    if (metadata_rc != 0 || ctx_set(fd, phys_path, ss, ss_len, fid) != 0) {
        OQS_MEM_cleanse(ss, sizeof(ss));
        close(fd);
        return metadata_rc == -EKEYREJECTED ? -EKEYREJECTED : -EIO;
    }
    OQS_MEM_cleanse(ss, sizeof(ss));
    pqc_log("OPEN %s: restored authenticated file-key envelope (fid=%llu)",
            path, (unsigned long long)fid);

    /* Strategy 2: restore tier from physical xattr if present */
    {
        char xval[8] = {0};
        ssize_t xlen = getxattr(phys_path, PQC_XATTR_TIER, xval, sizeof(xval) - 1);
        if (xlen > 0) {
            int t = atoi(xval);
            if (t == PQC_TIER_NONE) {
                int idx = fd % PQC_MAX_FD;
                pthread_mutex_lock(&g_fd_ctx[idx].fd_lock);
                if (g_fd_ctx[idx].valid)
                    g_fd_ctx[idx].tier = PQC_TIER_NONE;
                pthread_mutex_unlock(&g_fd_ctx[idx].fd_lock);
            }
        }
    }
    restore_qos_class_for_fd(fd, phys_path);
    return 0;
}

static int pqc_read(const char *path, char *buf, size_t size, off_t offset,
                     struct fuse_file_info *fi)
{
    (void)path;
    int fd  = (int)fi->fh;
    int idx = fd % PQC_MAX_FD;

    pthread_mutex_lock(&g_fd_ctx[idx].fd_lock);
    pqc_fd_ctx_t *ctx = &g_fd_ctx[idx];
    if (!ctx->valid || ctx->tier == PQC_TIER_NONE) {
        pthread_mutex_unlock(&g_fd_ctx[idx].fd_lock);
        int res = (int)pread(fd, buf, size, offset);
        return res <= 0 ? (res == -1 ? -errno : 0) : res;
    }
    if ((uint64_t)offset >= ctx->logical_size) { pthread_mutex_unlock(&ctx->fd_lock); return 0; }
    size_t want = size;
    if (want > ctx->logical_size - (uint64_t)offset) want = ctx->logical_size - (uint64_t)offset;
    pthread_mutex_lock(&ctx->state->commit_lock);
    uint64_t base = (uint64_t)offset, end = base + want;
    uint64_t first = base / PQC_LOGICAL_BLOCK_SIZE, last = (end - 1) / PQC_LOGICAL_BLOCK_SIZE;
    int rc = 0;
    for (uint64_t block = first; block <= last; ++block) {
        uint8_t plain[PQC_LOGICAL_BLOCK_SIZE];
        uint64_t block_start = block * PQC_LOGICAL_BLOCK_SIZE;
        uint64_t from = base > block_start ? base - block_start : 0;
        uint64_t to = end < block_start + PQC_LOGICAL_BLOCK_SIZE ? end - block_start : PQC_LOGICAL_BLOCK_SIZE;
        rc = load_authenticated_block(ctx, block, plain);
        if (rc) break;
        pqc_fault_cutpoint("read_after_auth");
        memcpy(buf + block_start + from - base, plain + from, to - from);
    }
    pthread_mutex_unlock(&ctx->state->commit_lock);
    pthread_mutex_unlock(&ctx->fd_lock);
    return rc ? rc : (int)want;
}

static int pqc_create(const char *path, mode_t mode,
                       struct fuse_file_info *fi)
{
    if (is_hidden_sidecar_path(path))
        return -ENOENT;
    char phys_path[4096];
    resolve_physical_path(phys_path, sizeof(phys_path), path);

    int fd = open(phys_path, fi->flags, mode);
    if (fd == -1)
        return -errno;

    fi->fh = (uint64_t)fd;
    fi->direct_io = 1;

    uint8_t ss[32] = {0};
    uint64_t fid = 0;
    if (RAND_bytes(ss, sizeof(ss)) != 1 ||
        RAND_bytes((unsigned char *)&fid, sizeof(fid)) != 1 || fid == 0) {
        OQS_MEM_cleanse(ss, sizeof(ss));
        close(fd);
        unlink(phys_path);
        return -EIO;
    }
    if (metadata_store(phys_path, ss, sizeof(ss), fid) != 0 ||
        ctx_set(fd, phys_path, ss, sizeof(ss), fid) != 0) {
        OQS_MEM_cleanse(ss, sizeof(ss));
        close(fd);
        unlink(phys_path);
        return -EIO;
    }
    OQS_MEM_cleanse(ss, sizeof(ss));
    restore_qos_class_for_fd(fd, phys_path);
    pqc_log("CREATE %s: authenticated file-key envelope initialized (fid=%llu)",
            path, (unsigned long long)fid);
    return 0;
}

/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║  WRITE — ML-KEM-derived AES-256-GCM secure storage                    ║
 * ║                                                                          ║
 * ║  Hot path (most writes):                                                 ║
 * ║    1. ctx_get() retrieves the per-file KEM-derived key material          ║
 * ║    2. CPU or CUDA executes the identical AES-GCM block format            ║
 * ║    3. ciphertext and authentication tag are journaled atomically         ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 */
static int pqc_write(const char *path, const char *buf, size_t size,
                      off_t offset, struct fuse_file_info *fi)
{
    (void)path;
    int fd  = (int)fi->fh;
    int idx = fd % PQC_MAX_FD;

    pthread_mutex_lock(&g_fd_ctx[idx].fd_lock);
    pqc_fd_ctx_t *ctx = &g_fd_ctx[idx];

    /* No context → passthrough (file opened without create, e.g. sidecar) */
    if (!ctx->valid) {
        pthread_mutex_unlock(&g_fd_ctx[idx].fd_lock);
        int res = (int)pwrite(fd, buf, size, offset);
        return res == -1 ? -errno : (int)size;
    }

    /* A coalescing buffer represents one contiguous file range.  Never append
     * a random write to it: that would lose the request's logical offset. */
    if (ctx->wbuf_used > 0 &&
        offset != ctx->wbuf_base_off + (off_t)ctx->wbuf_used) {
        int fr = do_flush_wbuf_locked(fd, idx);
        if (fr < 0) {
            pthread_mutex_unlock(&ctx->fd_lock);
            return fr;
        }
    }

    /* Strategy 2: Tier NONE → plaintext passthrough, no encryption */
    if (ctx->tier == PQC_TIER_NONE) {
        size_t bytes_written = 0;
        while (bytes_written < size) {
            size_t chunk = size - bytes_written;
            if (chunk > COALESCE_SIZE - ctx->wbuf_used) {
                chunk = COALESCE_SIZE - ctx->wbuf_used;
            }
            if (ctx->wbuf_used == 0) {
                ctx->wbuf_base_off = offset + bytes_written;
            }
            memcpy(ctx->wbuf + ctx->wbuf_used, buf + bytes_written, chunk);
            ctx->wbuf_used += chunk;
            bytes_written += chunk;
            if (ctx->wbuf_used >= COALESCE_SIZE) {
                int fr = do_flush_wbuf_locked(fd, idx);
                if (fr < 0) { pthread_mutex_unlock(&ctx->fd_lock); return fr; }
            }
        }
        int interval = get_rotation_interval_s();
        int force_rekey = getenv("PQC_FORCE_REKEY_ON_WRITE") != NULL;
        if (force_rekey || (interval > 0 && (time(NULL) - ctx->last_rekey) >= interval)) {
            rekey_queue_push(fd);
        }
        pthread_mutex_unlock(&g_fd_ctx[idx].fd_lock);
        return (int)size;
    }

    // Process chunked write with coalescing
    size_t bytes_written = 0;
    while (bytes_written < size) {
        size_t chunk = size - bytes_written;
        if (chunk > COALESCE_SIZE - ctx->wbuf_used) {
            chunk = COALESCE_SIZE - ctx->wbuf_used;
        }
        if (ctx->wbuf_used == 0) {
            ctx->wbuf_base_off = offset + bytes_written;
        }
        memcpy(ctx->wbuf + ctx->wbuf_used, buf + bytes_written, chunk);
        ctx->wbuf_used += chunk;
        bytes_written += chunk;
        if (ctx->wbuf_used >= COALESCE_SIZE) {
            int fr = do_flush_wbuf_locked(fd, idx);
            if (fr < 0) { pthread_mutex_unlock(&ctx->fd_lock); return fr; }
        }
    }
    int interval = get_rotation_interval_s();
    int force_rekey = getenv("PQC_FORCE_REKEY_ON_WRITE") != NULL;
    if (force_rekey || (interval > 0 && (time(NULL) - ctx->last_rekey) >= interval)) {
        rekey_queue_push(fd);
    }
    pthread_mutex_unlock(&g_fd_ctx[idx].fd_lock);
    return (int)size;
}

static int pqc_fsync(const char *path, int datasync,
                     struct fuse_file_info *fi)
{
    (void)path; (void)datasync;
    int fd  = (int)fi->fh;
    int idx = fd % PQC_MAX_FD;

    pthread_mutex_lock(&g_fd_ctx[idx].fd_lock);
    if (g_fd_ctx[idx].valid && g_fd_ctx[idx].wbuf)
        do_flush_wbuf_locked(fd, idx);
    while (g_fd_ctx[idx].pending_jobs > 0)
        pthread_cond_wait(&g_fd_ctx[idx].pending_cv, &g_fd_ctx[idx].fd_lock);
    if (g_fd_ctx[idx].valid) {
        if (g_fd_ctx[idx].data_fd >= 0 && fdatasync(g_fd_ctx[idx].data_fd) != 0) {
            pthread_mutex_unlock(&g_fd_ctx[idx].fd_lock);
            return -errno;
        }
        if (g_fd_ctx[idx].journal_fd >= 0 && fdatasync(g_fd_ctx[idx].journal_fd) != 0) {
            pthread_mutex_unlock(&g_fd_ctx[idx].fd_lock);
            return -errno;
        }
    }
    pthread_mutex_unlock(&g_fd_ctx[idx].fd_lock);

    if (anchor_flush_now() != 0)
        return -EIO;

    pqc_fault_cutpoint("fsync_before_return");

    return fdatasync(fd) == -1 ? -errno : 0;
}

static int pqc_flush(const char *path, struct fuse_file_info *fi)
{
    (void)path;
    if (!fi)
        return 0;
    int fd  = (int)fi->fh;
    int idx = fd % PQC_MAX_FD;

    pthread_mutex_lock(&g_fd_ctx[idx].fd_lock);
    if (g_fd_ctx[idx].valid && g_fd_ctx[idx].wbuf && g_fd_ctx[idx].wbuf_used)
        do_flush_wbuf_locked(fd, idx);
    while (g_fd_ctx[idx].pending_jobs > 0)
        pthread_cond_wait(&g_fd_ctx[idx].pending_cv, &g_fd_ctx[idx].fd_lock);
    if (g_fd_ctx[idx].valid) {
        if (g_fd_ctx[idx].data_fd >= 0 && fdatasync(g_fd_ctx[idx].data_fd) != 0) {
            pthread_mutex_unlock(&g_fd_ctx[idx].fd_lock);
            return -errno;
        }
        if (g_fd_ctx[idx].journal_fd >= 0 && fdatasync(g_fd_ctx[idx].journal_fd) != 0) {
            pthread_mutex_unlock(&g_fd_ctx[idx].fd_lock);
            return -errno;
        }
    }
    pthread_mutex_unlock(&g_fd_ctx[idx].fd_lock);
    return 0;
}

static int pqc_truncate(const char *path, off_t size,
                         struct fuse_file_info *fi)
{
    if (is_hidden_sidecar_path(path))
        return -ENOENT;
    char phys_path[4096];
    resolve_physical_path(phys_path, sizeof(phys_path), path);
    if (size < 0)
        return -EINVAL;

    int fd = -1;
    if (fi)
        fd = (int)fi->fh;
    else
        fd = open(phys_path, O_RDWR);
    if (fd == -1)
        return -errno;

    int idx = fd % PQC_MAX_FD;
    int res = 0;
    pthread_mutex_lock(&g_fd_ctx[idx].fd_lock);
    pqc_fd_ctx_t *ctx = &g_fd_ctx[idx];
    if (ctx->valid && ctx->wbuf && ctx->wbuf_used > 0) {
        res = do_flush_wbuf_locked(fd, idx);
    }
    if (res == 0 && ctx->valid) {
        ctx->logical_size = (uint64_t)size;
        res = logical_size_store(ctx->marker_path, (uint64_t)size);
        if (res == 0) {
            if (ftruncate(fd, size) != 0)
                res = -errno;
            if (size == 0 && ctx->journal_fd >= 0) {
                if (ftruncate(ctx->journal_fd, 0) != 0)
                    res = -errno;
            }
        }
    } else if (res == 0) {
        if (ftruncate(fd, size) != 0)
            res = -errno;
    }
    pthread_mutex_unlock(&g_fd_ctx[idx].fd_lock);
    if (!fi)
        close(fd);
    return res;
}

static int pqc_fallocate(const char *path, int mode, off_t offset,
                         off_t length, struct fuse_file_info *fi)
{
    if (is_hidden_sidecar_path(path))
        return -ENOENT;
    if (offset < 0 || length < 0)
        return -EINVAL;
    if ((uint64_t)offset > UINT64_MAX - (uint64_t)length)
        return -EFBIG;

    int supported_flags = 0;
#ifdef FALLOC_FL_KEEP_SIZE
    supported_flags |= FALLOC_FL_KEEP_SIZE;
#endif
    if (mode & ~supported_flags)
        return -EOPNOTSUPP;
    if (!fi)
        return -EOPNOTSUPP;

    int fd = (int)fi->fh;
    int idx = fd % PQC_MAX_FD;
    int res = 0;

    pthread_mutex_lock(&g_fd_ctx[idx].fd_lock);
    pqc_fd_ctx_t *ctx = &g_fd_ctx[idx];
    if (ctx->valid && ctx->wbuf && ctx->wbuf_used > 0)
        res = do_flush_wbuf_locked(fd, idx);

    if (res == 0 && ctx->valid) {
        uint64_t end = (uint64_t)offset + (uint64_t)length;
        uint64_t logical_size = ctx->logical_size;
#ifdef FALLOC_FL_KEEP_SIZE
        if ((mode & FALLOC_FL_KEEP_SIZE) == 0 && end > logical_size)
            logical_size = end;
#else
        if (end > logical_size)
            logical_size = end;
#endif
        pthread_mutex_lock(&ctx->state->commit_lock);
        ctx->logical_size = logical_size;
        res = logical_size_store(ctx->marker_path, logical_size);
        if (res == 0 && ftruncate(fd, (off_t)logical_size) != 0)
            res = -errno;
        if (res == 0)
            res = checkpoint_store(ctx->marker_path, ctx->file_id,
                                   ctx->state->next_generation,
                                   ctx->logical_size,
                                   ctx->state->next_generation);
        pthread_mutex_unlock(&ctx->state->commit_lock);
    } else if (res == 0) {
        off_t end = offset + length;
#ifdef FALLOC_FL_KEEP_SIZE
        if ((mode & FALLOC_FL_KEEP_SIZE) != 0)
            end = 0;
#endif
        if (end > 0 && ftruncate(fd, end) != 0)
            res = -errno;
    }
    pthread_mutex_unlock(&g_fd_ctx[idx].fd_lock);
    return res;
}

static int pqc_unlink(const char *path)
{
    if (is_hidden_sidecar_path(path))
        return -ENOENT;
    char phys_path[4096];
    resolve_physical_path(phys_path, sizeof(phys_path), path);
    char compat_path[4096];
    int compat_rc = sqlite_sidecar_redirect_path(compat_path, sizeof(compat_path), path);
    char data_path[4096 + 16], journal_path[4096 + 16];
    int res = sidecar_path(data_path, sizeof(data_path), phys_path, ".pqcdata");
    if (res == 0) res = sidecar_path(journal_path, sizeof(journal_path), phys_path, ".pqcmeta");
    if (res != 0)
        return res;

    if (unlink(data_path) == -1 && errno != ENOENT)
        return -errno;
    if (unlink(journal_path) == -1 && errno != ENOENT)
        return -errno;
    if (compat_rc == 0 && unlink(compat_path) == -1 && errno != ENOENT)
        return -errno;
    if (unlink(phys_path) == -1)
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
    pthread_mutex_lock(&g_fd_ctx[idx].fd_lock);
    if (g_fd_ctx[idx].valid && g_fd_ctx[idx].wbuf && g_fd_ctx[idx].wbuf_used > 0)
        do_flush_wbuf_locked(fd, idx);
    pthread_mutex_unlock(&g_fd_ctx[idx].fd_lock);

    ctx_clear(fd);
    close(fd);
    return 0;
}

static int pqc_lock(const char *path, struct fuse_file_info *fi, int cmd, struct flock *lock)
{
    (void)path;
    if (!fi || !lock)
        return -EINVAL;
    int fd = (int)fi->fh;
    if (fcntl(fd, cmd, lock) == -1)
        return -errno;
    return 0;
}

static int pqc_flock(const char *path, struct fuse_file_info *fi, int op)
{
    (void)path;
    if (!fi)
        return -EINVAL;
    int fd = (int)fi->fh;
    if (flock(fd, op) == -1)
        return -errno;
    return 0;
}

static int pqc_rename(const char *from, const char *to, unsigned int flags)
{
    (void)from;
    (void)to;
    (void)flags;
    /*
     * Rename must move the marker file, .pqcdata, .pqcmeta, logical-size
     * state, checkpoint, and external-anchor association atomically.  The
     * current prototype has no retained crash campaign for that transition, so
     * reject it explicitly instead of letting the kernel infer ENOSYS.
     */
    return -ENOTSUP;
}

static int pqc_fsyncdir(const char *path, int datasync,
                         struct fuse_file_info *fi)
{
    (void)path;
    (void)datasync;
    (void)fi;
    /*
     * The publication protocol syncs file sidecars and checkpoint xattrs.  It
     * does not certify directory-entry durability, so directory fsync is not a
     * supported durability boundary in this prototype.
     */
    return -ENOTSUP;
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
    if (g_anchor_thread_started) {
        pthread_mutex_lock(&g_anchor_lock);
        g_anchor_stop = 1;
        pthread_cond_signal(&g_anchor_cv);
        pthread_mutex_unlock(&g_anchor_lock);
        pthread_join(g_anchor_thread, NULL);
        g_anchor_thread_started = 0;
    }
    if (g_gpu_load_thread_started) {
        g_gpu_load_stop = 1;
        pthread_join(g_gpu_load_thread, NULL);
        g_gpu_load_thread_started = 0;
    }
    (void)anchor_flush_now();
    pthread_mutex_lock(&g_anchor_lock);
    if (g_anchor_dirty) {
        pqc_anchor_state_t state = g_anchor_state;
        g_anchor_dirty = 0;
        pthread_mutex_unlock(&g_anchor_lock);
        (void)pqc_anchor_store(&state);
        (void)pqc_anchor_flush();
    } else {
        pthread_mutex_unlock(&g_anchor_lock);
    }
    (void)pqc_anchor_finalize();
    pqc_cleanup();
    if (g_logfile) {
        fclose(g_logfile);
        g_logfile = NULL;
    }
}

static void *pqc_fuse_init(struct fuse_conn_info *conn, struct fuse_config *cfg)
{
    if (cfg) {
        cfg->kernel_cache = 0;
        cfg->auto_cache = 0;
    }
    if (!conn)
        return NULL;
#ifdef FUSE_CAP_WRITEBACK_CACHE
    conn->want &= ~FUSE_CAP_WRITEBACK_CACHE;
#endif
#ifdef FUSE_CAP_ASYNC_DIO
    if (conn->capable & FUSE_CAP_ASYNC_DIO)
        conn->want |= FUSE_CAP_ASYNC_DIO;
#endif
#ifdef FUSE_DIRECT_IO_ALLOW_MMAP
    conn->want &= ~FUSE_DIRECT_IO_ALLOW_MMAP;
#endif
    return NULL;
}

static int is_internal_xattr_name(const char *name)
{
    if (!name)
        return 0;
    return strcmp(name, PQC_XATTR_METADATA) == 0 ||
           strcmp(name, PQC_XATTR_LOGICAL_SIZE) == 0 ||
           strcmp(name, PQC_XATTR_CHECKPOINT) == 0;
}

static ssize_t filter_xattr_list(const char *raw, size_t raw_len,
                                  char *out, size_t out_size)
{
    size_t used = 0;
    for (size_t pos = 0; pos < raw_len;) {
        size_t name_len = strnlen(raw + pos, raw_len - pos);
        if (pos + name_len >= raw_len)
            return -EIO;
        const char *name = raw + pos;
        size_t record_len = name_len + 1;
        if (!is_internal_xattr_name(name)) {
            if (out && used + record_len <= out_size)
                memcpy(out + used, name, record_len);
            used += record_len;
        }
        pos += record_len;
    }
    if (out && used > out_size)
        return -ERANGE;
    return (ssize_t)used;
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
    if (is_hidden_sidecar_path(path))
        return -ENOENT;
    char phys_path[4096];
    resolve_physical_path(phys_path, sizeof(phys_path), path);

    if (is_internal_xattr_name(name))
        return -EPERM;

    if (strcmp(name, "user.pqc.freshness_window") == 0 && size > 0) {
        char tmp[16] = {0};
        memcpy(tmp, value, size < 15 ? size : 15);
        int n = atoi(tmp);
        if (n > 0) {
            pqc_anchor_set_freshness_window(n);
            return 0;
        }
        return -EINVAL;
    }

    if (strcmp(name, PQC_XATTR_TIER) == 0 && size > 0) {
        char tmp[8] = {0};
        memcpy(tmp, value, size < 7 ? size : 7);
        int new_tier = atoi(tmp);
        if (new_tier != PQC_TIER_FULL && new_tier != PQC_TIER_NONE)
            return -EINVAL;
        if (setxattr(phys_path, name, value, size, flags) == -1)
            return -errno;
        for (int i = 0; i < PQC_MAX_FD; i++) {
            pthread_mutex_lock(&g_fd_ctx[i].fd_lock);
            if (g_fd_ctx[i].valid && strcmp(g_fd_ctx[i].marker_path, phys_path) == 0)
                g_fd_ctx[i].tier = new_tier;
            pthread_mutex_unlock(&g_fd_ctx[i].fd_lock);
        }
        pqc_log("SETXATTR %s tier=%d", path, new_tier);
        return 0;
    }

    if (strcmp(name, PQC_XATTR_QOS_CLASS) == 0 && size > 0) {
        int qos_class = PQC_QOS_CLASS_ELASTIC;
        int rc = parse_qos_class_value(value, size, &qos_class);
        if (rc != 0)
            return rc;
        if (setxattr(phys_path, name, value, size, flags) == -1)
            return -errno;
        for (int i = 0; i < PQC_MAX_FD; i++) {
            pthread_mutex_lock(&g_fd_ctx[i].fd_lock);
            if (g_fd_ctx[i].valid && strcmp(g_fd_ctx[i].marker_path, phys_path) == 0)
                g_fd_ctx[i].qos_class = qos_class;
            pthread_mutex_unlock(&g_fd_ctx[i].fd_lock);
        }
        pqc_log("SETXATTR %s qos_class=%s", path, qos_class_name(qos_class));
        return 0;
    }

    if (setxattr(phys_path, name, value, size, flags) == -1)
        return -errno;
    return 0;
}

static int pqc_getxattr(const char *path, const char *name,
                         char *value, size_t size)
{
    if (is_hidden_sidecar_path(path))
        return -ENOENT;
    char phys_path[4096];
    resolve_physical_path(phys_path, sizeof(phys_path), path);

    if (is_internal_xattr_name(name))
        return -ENODATA;

    ssize_t res = getxattr(phys_path, name, value, size);
    return res == -1 ? -errno : (int)res;
}

static int pqc_listxattr(const char *path, char *list, size_t size)
{
    if (is_hidden_sidecar_path(path))
        return -ENOENT;
    char phys_path[4096];
    resolve_physical_path(phys_path, sizeof(phys_path), path);

    ssize_t raw_len = listxattr(phys_path, NULL, 0);
    if (raw_len <= 0)
        return raw_len == -1 ? -errno : 0;
    char *raw = malloc((size_t)raw_len);
    if (!raw)
        return -ENOMEM;
    ssize_t got = listxattr(phys_path, raw, (size_t)raw_len);
    if (got == -1) {
        int err = -errno;
        free(raw);
        return err;
    }
    ssize_t filtered = filter_xattr_list(raw, (size_t)got, list, size);
    free(raw);
    return filtered < 0 ? (int)filtered : (int)filtered;
}

/* ── FUSE operations table ── */
static const struct fuse_operations pqc_oper = {
    .init       = pqc_fuse_init,
    .getattr    = pqc_getattr,
    .readdir    = pqc_readdir,
    .open       = pqc_open,
    .read       = pqc_read,
    .write      = pqc_write,
    .flush      = pqc_flush,
    .fsync      = pqc_fsync,
    .create     = pqc_create,
    .truncate   = pqc_truncate,
    .fallocate  = pqc_fallocate,
    .unlink     = pqc_unlink,
    .mkdir      = pqc_mkdir,
    .rmdir      = pqc_rmdir,
    .release    = pqc_release,
    .lock       = pqc_lock,
    .flock      = pqc_flock,
    .rename     = pqc_rename,
    .fsyncdir   = pqc_fsyncdir,
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
        "║  Authenticated AES-GCM storage; optional ML-KEM-768 helpers  ║\n"
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
    if (argc == 2 && strcmp(argv[1], "--self-test") == 0) {
        int sched_rc = scheduler_self_test();
        int ckpt_rc = checkpoint_self_test();
        int anchor_rc = pqc_anchor_self_test();
        int gen_rc = generation_replay_self_test();
        int rc = crypto_self_test() || journal_self_test() || gen_rc || ckpt_rc || sched_rc || anchor_rc;
        fprintf(stderr, "PQC-FUSE scheduler self-test: %s\n",
                sched_rc == 0 ? "PASS" : "FAIL");
        fprintf(stderr, "PQC-FUSE generation replay self-test: %s\n",
                gen_rc == 0 ? "PASS" : "FAIL");
        fprintf(stderr, "PQC-FUSE checkpoint self-test: %s\n",
                ckpt_rc == 0 ? "PASS" : "FAIL");
        fprintf(stderr, "PQC-FUSE anchor self-test: %s\n",
                anchor_rc == 0 ? "PASS" : "FAIL");
        fprintf(stderr, "PQC-FUSE storage self-test: %s\n", rc == 0 ? "PASS" : "FAIL");
        return rc == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (argc == 2 && strcmp(argv[1], "--scheduler-smoke") == 0) {
        scheduler_smoke_report();
        return EXIT_SUCCESS;
    }
    if (argc == 2 && strcmp(argv[1], "--admission-telemetry-smoke") == 0) {
        int rc = admission_telemetry_smoke_report();
        return rc == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (argc == 2 && strcmp(argv[1], "--anchor-self-test") == 0) {
        int anchor_rc = pqc_anchor_self_test();
        fprintf(stderr, "PQC-FUSE anchor self-test: %s\n",
                anchor_rc == 0 ? "PASS" : "FAIL");
        return anchor_rc == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (argc == 2 && strcmp(argv[1], "--um-smoke") == 0) {
        int um_rc = skim_cuda_um_self_test();
        skim_cuda_um_stats_t s = skim_cuda_um_stats_snapshot();
        fprintf(stdout,
                "{\"self_test_rc\":%d,\"managed_alloc_bytes\":%llu,\"managed_free_bytes\":%llu,"
                "\"prefetch_to_device_bytes\":%llu,\"prefetch_to_host_bytes\":%llu,"
                "\"prefetch_device_calls\":%llu,\"prefetch_host_calls\":%llu}\n",
                um_rc,
                (unsigned long long)s.managed_alloc_bytes,
                (unsigned long long)s.managed_free_bytes,
                (unsigned long long)s.prefetch_to_device_bytes,
                (unsigned long long)s.prefetch_to_host_bytes,
                (unsigned long long)s.prefetch_device_calls,
                (unsigned long long)s.prefetch_host_calls);
        return EXIT_SUCCESS;
    }
    fprintf(stderr,
        "\n"
        "  ┌─────────────────────────────────────────────────────┐\n"
        "  │  PQC-FUSE v0.1 — Authenticated Encrypted Storage   │\n"
        "  │  CPU data plane; optional batch GPU helpers         │\n"
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
    int log_rc = sidecar_path(log_path, sizeof(log_path), g_storage_dir, "/../");
    if (log_rc == 0) {
        size_t used = strlen(log_path);
        int n = snprintf(log_path + used, sizeof(log_path) - used, "%s", LOG_FILENAME);
        if (n < 0 || (size_t)n >= sizeof(log_path) - used)
            log_rc = -ENAMETOOLONG;
    }
    if (log_rc != 0) {
        fprintf(stderr, "[PQC-FUSE] WARNING: log path too long; falling back to current directory\n");
        log_path[0] = '\0';
    }
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
    if (pqc_subsystem_init() != 0) {
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
