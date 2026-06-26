#include "pqc_anchor.h"

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stddef.h>
#include <sys/stat.h>
#include <pthread.h>

#ifdef SKIM_HAVE_CUDA_INTEGRITY
#include "cuda_integrity.h"
#endif

extern uint8_t g_master_key[32];

/* ════════════════════════════════════════════════════════════════════════════
 *  Committed-prefix root  (Gap 4)
 *
 *  Design: instead of anchoring a single file's state, we anchor the SHA-256
 *  root of ALL committed (file_id, max_generation) pairs known to this mount.
 *  The root is computed as:
 *
 *    prefix_root = SHA-256( ‖ᵢ (file_idᵢ LE64 ‖ max_genᵢ LE64) )
 *
 *  pairs are sorted by file_id for determinism.  On remount, if the stored
 *  anchor's global_sequence is ahead of any locally reconstructed sequence,
 *  the mount is rejected (fail-closed).
 *
 *  TPM NV stores a pqc_prefix_anchor_t (64 bytes) instead of the old
 *  pqc_freshness_anchor_t.  Both fit in the same NV index.
 * ════════════════════════════════════════════════════════════════════════════ */

#define PQC_PREFIX_ANCHOR_MAGIC   0x50524546u /* PREF */
#define PQC_PREFIX_ANCHOR_VERSION 3u
#define PQC_FRESHNESS_ANCHOR_MAGIC   0x46524553u /* FRES — legacy */
#define PQC_FRESHNESS_ANCHOR_VERSION  2u
#define PQC_TPM_NV_DEFAULT_INDEX      0x01500010u

/* On-disk / on-TPM anchor record (64 bytes) */
typedef struct {
    uint32_t magic;            /* PQC_PREFIX_ANCHOR_MAGIC */
    uint32_t version;          /* PQC_PREFIX_ANCHOR_VERSION */
    uint64_t global_sequence;  /* monotonically increasing batch counter */
    uint64_t file_count;       /* #files covered by this root */
    uint8_t  prefix_root[32];  /* SHA-256 of sorted (file_id||max_gen) pairs */
    uint8_t  digest[32];       /* HMAC-SHA256(master_key, above fields) */
} pqc_prefix_anchor_t;

/* Legacy on-disk format */
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t epoch;
    uint64_t sequence;
    uint64_t logical_size;
    uint8_t  digest[32];
} pqc_freshness_anchor_t;

/* In-memory committed-prefix map: (file_id → max_generation) */
#define COMMITTED_MAP_CAPACITY 512

typedef struct {
    uint64_t file_id;
    uint64_t max_generation;
} committed_entry_t;

static committed_entry_t g_committed_map[COMMITTED_MAP_CAPACITY];
static size_t            g_committed_count = 0;
static uint64_t          g_global_sequence = 0;
static pthread_mutex_t   g_committed_lock = PTHREAD_MUTEX_INITIALIZER;

/* Pending anchor for hardware backend (written by anchor worker) */
static pqc_prefix_anchor_t g_pending_anchor;
static int                  g_pending_anchor_valid = 0;
static pthread_mutex_t      g_pending_anchor_lock = PTHREAD_MUTEX_INITIALIZER;

static int s_freshness_window = -1;

int pqc_anchor_set_freshness_window(int n)
{
    if (n > 0) {
        s_freshness_window = n;
        return 0;
    }
    return -EINVAL;
}

/* ── Environment helpers ────────────────────────────────────────────────── */

static const char *anchor_path(void)
{
    const char *env = getenv("PQC_FRESHNESS_ANCHOR_PATH");
    return (env && *env) ? env : NULL;
}

static int anchor_path_is_hardware(void)
{
    const char *env = getenv("PQC_FRESHNESS_ANCHOR_BACKEND");
    return env && strcmp(env, "hardware") == 0;
}

static uint32_t tpm_nv_index(void)
{
    const char *env = getenv("PQC_TPM_NV_INDEX");
    if (!env || !*env) return PQC_TPM_NV_DEFAULT_INDEX;
    char *end = NULL;
    unsigned long v = strtoul(env, &end, 0);
    if (end == env || (end && *end != '\0')) return PQC_TPM_NV_DEFAULT_INDEX;
    return (uint32_t)v;
}

pqc_anchor_backend_t pqc_anchor_backend(void)
{
    if (anchor_path_is_hardware()) return PQC_ANCHOR_BACKEND_HARDWARE;
    return anchor_path() ? PQC_ANCHOR_BACKEND_FILE
                         : PQC_ANCHOR_BACKEND_DISABLED;
}

/* ── SHA-256 prefix root computation ────────────────────────────────────── */

/* int-sort comparator for committed_entry_t by file_id */
static int cmp_committed(const void *a, const void *b)
{
    const committed_entry_t *ea = (const committed_entry_t *)a;
    const committed_entry_t *eb = (const committed_entry_t *)b;
    if (ea->file_id < eb->file_id) return -1;
    if (ea->file_id > eb->file_id) return  1;
    return 0;
}

/* CPU-side helper to compute SHA-256 of arbitrary input buffer */
static int cpu_sha256(const uint8_t *in, size_t len, uint8_t out[32])
{
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    if (!ctx) return -ENOMEM;
    int ok = EVP_DigestInit_ex(ctx, EVP_sha256(), NULL) == 1 &&
             EVP_DigestUpdate(ctx, in, len) == 1 &&
             EVP_DigestFinal_ex(ctx, out, NULL) == 1;
    EVP_MD_CTX_free(ctx);
    return ok ? 0 : -EIO;
}

/* Recompute Merkle root of entries.
 * Leaf hashes are computed from little-endian 16-byte pairs: file_id || max_generation.
 * Merkle reduction tree uses O(log N) bottom-up reduction.
 */
static int compute_prefix_root(const committed_entry_t *entries, size_t count,
                                uint8_t out_root[32])
{
    if (count == 0) {
        /* Empty count yields hash of empty string */
        return cpu_sha256(NULL, 0, out_root);
    }

    /* Find next power of two >= count */
    size_t n = 1;
    while (n < count) { n <<= 1; }

    /* Allocate host memory for leaf digests */
    uint8_t *leaf_digests = (uint8_t *)calloc(n, 32);
    if (!leaf_digests) return -ENOMEM;

    /* Prepare leaves buffer (16-byte serializations) */
    uint8_t *leaves_buf = (uint8_t *)malloc(count * 16);
    if (!leaves_buf) {
        free(leaf_digests);
        return -ENOMEM;
    }
    for (size_t i = 0; i < count; ++i) {
        uint64_t fid = entries[i].file_id;
        uint64_t gen = entries[i].max_generation;
        for (int b = 0; b < 8; ++b) { leaves_buf[i * 16 + b]   = (uint8_t)(fid >> (8 * b)); }
        for (int b = 0; b < 8; ++b) { leaves_buf[i * 16 + 8 + b] = (uint8_t)(gen >> (8 * b)); }
    }

    int rc = -1;
    int gpu_success = 0;

#ifdef SKIM_HAVE_CUDA_INTEGRITY
    if (skim_cuda_integrity_available()) {
        /* Run GPU leaf batch hash */
        rc = skim_cuda_integrity_leaf_batch(leaves_buf, 16, leaf_digests, count, AEGISQ_HASH_SHA256_BATCH);
        if (rc == 0) {
            /* Run GPU Merkle tree reduction */
            rc = skim_cuda_integrity_merkle_root(leaf_digests, n, out_root);
            if (rc == 0) {
                gpu_success = 1;
            }
        }
    }
#endif

    if (!gpu_success) {
        /* CPU Fallback with Domain Separation (matching GPU) */
        /* 1. Compute leaf digests with 0x00 leaf prefix */
        for (size_t i = 0; i < count; ++i) {
            uint8_t leaf_buf[17];
            leaf_buf[0] = 0x00;
            memcpy(leaf_buf + 1, leaves_buf + i * 16, 16);
            rc = cpu_sha256(leaf_buf, 17, leaf_digests + i * 32);
            if (rc != 0) goto out;
        }
        /* Padding leaf digests are already 0 due to calloc */

        /* 2. CPU bottom-up Merkle reduction with 0x01 internal node prefix */
        size_t current_count = n;
        uint8_t *buf = (uint8_t *)malloc(n * 32);
        if (!buf) {
            rc = -ENOMEM;
            goto out;
        }
        memcpy(buf, leaf_digests, n * 32);

        while (current_count > 1) {
            size_t next_count = current_count / 2;
            for (size_t i = 0; i < next_count; ++i) {
                /* Hash child pairs (2 * 32 = 64 bytes) into parent digest with 0x01 prefix */
                uint8_t node_buf[65];
                node_buf[0] = 0x01;
                memcpy(node_buf + 1, buf + i * 64, 64);
                rc = cpu_sha256(node_buf, 65, buf + i * 32);
                if (rc != 0) {
                    free(buf);
                    goto out;
                }
            }
            current_count = next_count;
        }
        memcpy(out_root, buf, 32);
        free(buf);
        rc = 0;
    }

out:
    free(leaves_buf);
    free(leaf_digests);
    return rc;
}

static int digest_prefix_anchor(const pqc_prefix_anchor_t *anchor, uint8_t out[32])
{
    unsigned int out_len = 0;
    unsigned char *mac = HMAC(EVP_sha256(), g_master_key, sizeof(g_master_key),
                              (const unsigned char *)anchor,
                              offsetof(pqc_prefix_anchor_t, digest),
                              out, &out_len);
    return (!mac || out_len != 32) ? -EIO : 0;
}

/* ── Record a file's committed generation (called from checkpoint_store) ── */

int pqc_anchor_record_file(uint64_t file_id, uint64_t max_generation,
                           uint64_t sequence, uint64_t logical_size)
{
    (void)logical_size; /* not part of the root hash, kept for legacy compat */
    pthread_mutex_lock(&g_committed_lock);
    /* update or insert */
    for (size_t i = 0; i < g_committed_count; ++i) {
        if (g_committed_map[i].file_id == file_id) {
            if (max_generation > g_committed_map[i].max_generation)
                g_committed_map[i].max_generation = max_generation;
            if (sequence > g_global_sequence)
                g_global_sequence = sequence;
            pthread_mutex_unlock(&g_committed_lock);
            return 0;
        }
    }
    if (g_committed_count < COMMITTED_MAP_CAPACITY) {
        g_committed_map[g_committed_count].file_id        = file_id;
        g_committed_map[g_committed_count].max_generation = max_generation;
        ++g_committed_count;
    }
    if (sequence > g_global_sequence)
        g_global_sequence = sequence;
    pthread_mutex_unlock(&g_committed_lock);
    return 0;
}

/* ── Build and queue a prefix anchor ─────────────────────────────────────── */

static int build_prefix_anchor(pqc_prefix_anchor_t *out)
{
    pthread_mutex_lock(&g_committed_lock);
    size_t count = g_committed_count;
    uint64_t seq = g_global_sequence;
    /* work on a local sorted copy */
    committed_entry_t *copy = NULL;
    if (count > 0) {
        copy = (committed_entry_t *)malloc(count * sizeof(*copy));
        if (!copy) {
            pthread_mutex_unlock(&g_committed_lock);
            return -ENOMEM;
        }
        memcpy(copy, g_committed_map, count * sizeof(*copy));
    }
    pthread_mutex_unlock(&g_committed_lock);

    if (count > 1 && copy)
        qsort(copy, count, sizeof(*copy), cmp_committed);

    uint8_t root[32] = {0};
    int rc = compute_prefix_root(copy ? copy : NULL, count, root);
    free(copy);
    if (rc != 0) return rc;

    memset(out, 0, sizeof(*out));
    out->magic           = PQC_PREFIX_ANCHOR_MAGIC;
    out->version         = PQC_PREFIX_ANCHOR_VERSION;
    out->global_sequence = seq;
    out->file_count      = (uint64_t)count;
    memcpy(out->prefix_root, root, 32);
    return digest_prefix_anchor(out, out->digest);
}

/* ── TPM / file I/O ──────────────────────────────────────────────────────── */

static int ensure_tpm_nv_defined(void)
{
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "tpm2_nvreadpublic 0x%08x >/dev/null 2>&1", tpm_nv_index());
    /* NV provisioning is an administrative action.  Creating a persistent
     * index implicitly would make ownership, authorization, and lifecycle
     * opaque, so an unprovisioned device fails closed instead. */
    return system(cmd) == 0 ? 0 : -ENODEV;
}

static int write_anchor_file(const pqc_prefix_anchor_t *anchor)
{
    const char *path = anchor_path();
    if (!path) return 0;
    if (anchor_path_is_hardware()) return -ENOTSUP;
    char tmpname[4096];
    int n = snprintf(tmpname, sizeof(tmpname), "%s.tmp.XXXXXX", path);
    if (n < 0 || (size_t)n >= sizeof(tmpname)) return -ENAMETOOLONG;

    int fd = mkstemp(tmpname);
    if (fd < 0) return -errno;
    int rc = 0;
    if (fchmod(fd, 0600) != 0) {
        rc = -errno;
    } else {
        const uint8_t *cursor = (const uint8_t *)anchor;
        size_t remaining = sizeof(*anchor);
        while (remaining > 0) {
            ssize_t written = write(fd, cursor, remaining);
            if (written < 0) {
                if (errno == EINTR) continue;
                rc = -errno;
                break;
            }
            if (written == 0) {
                rc = -EIO;
                break;
            }
            cursor += (size_t)written;
            remaining -= (size_t)written;
        }
        if (rc == 0 && fdatasync(fd) != 0) rc = -errno;
    }
    if (close(fd) != 0 && rc == 0) rc = -errno;
    if (rc == 0 && rename(tmpname, path) != 0) rc = -errno;
    if (rc != 0) unlink(tmpname);
    return rc;
}

static int read_anchor_file(pqc_prefix_anchor_t *anchor)
{
    const char *path = anchor_path();
    if (!path) return 0;
    if (anchor_path_is_hardware()) return -ENOTSUP;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return (errno == ENOENT) ? -ENOENT : -errno;
    ssize_t rd = read(fd, anchor, sizeof(*anchor));
    close(fd);
    return rd == (ssize_t)sizeof(*anchor) ? 0 : -EIO;
}

static int write_anchor_tpm(const pqc_prefix_anchor_t *anchor)
{
    int rc = ensure_tpm_nv_defined();
    if (rc != 0) return rc;
    char tmpname[] = "/tmp/pqc_anchor_nv_XXXXXX";
    int fd = mkstemp(tmpname);
    if (fd < 0) return -errno;
    ssize_t wr = write(fd, anchor, sizeof(*anchor));
    close(fd);
    if (wr != (ssize_t)sizeof(*anchor)) { unlink(tmpname); return -EIO; }
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "tpm2_nvwrite -C o -i %s 0x%08x >/dev/null 2>&1",
             tmpname, tpm_nv_index());
    rc = system(cmd);
    unlink(tmpname);
    return rc == 0 ? 0 : -EIO;
}

static int read_anchor_tpm(pqc_prefix_anchor_t *anchor)
{
    int rc = ensure_tpm_nv_defined();
    if (rc != 0) return rc;
    char tmpname[] = "/tmp/pqc_anchor_nv_XXXXXX";
    int fd = mkstemp(tmpname);
    if (fd < 0) return -errno;
    close(fd);
    unlink(tmpname);
    char cmd[512];
    snprintf(cmd, sizeof(cmd),
             "tpm2_nvread -C o -s %zu 0x%08x >%s 2>/dev/null",
             sizeof(*anchor), tpm_nv_index(), tmpname);
    rc = system(cmd);
    if (rc != 0) { unlink(tmpname); return -EIO; }
    fd = open(tmpname, O_RDONLY);
    if (fd < 0) { unlink(tmpname); return -errno; }
    ssize_t rd = read(fd, anchor, sizeof(*anchor));
    close(fd);
    unlink(tmpname);
    return rd == (ssize_t)sizeof(*anchor) ? 0 : -EIO;
}

/* ── Public API ──────────────────────────────────────────────────────────── */

int pqc_anchor_probe(void)
{
    if (pqc_anchor_backend() == PQC_ANCHOR_BACKEND_DISABLED) return 0;
    if (pqc_anchor_backend() == PQC_ANCHOR_BACKEND_FILE) {
        pqc_prefix_anchor_t anchor = {0};
        int rc = read_anchor_file(&anchor);
        if (rc == -ENOENT) return 0;
        OPENSSL_cleanse(&anchor, sizeof(anchor));
        return (rc == 0) ? 0 : rc;
    }
    return ensure_tpm_nv_defined();
}

/*
 * pqc_anchor_store — legacy single-file compatibility shim.
 * Registers the file state in the committed map and, for file backends,
 * immediately rebuilds and writes the prefix anchor.
 * For hardware backend, defers to the background flush (pqc_anchor_flush).
 */
int pqc_anchor_store(const pqc_anchor_state_t *state)
{
    if (!state) return -EINVAL;
    if (pqc_anchor_backend() == PQC_ANCHOR_BACKEND_DISABLED) return 0;

    /* Register in the committed-prefix map with a synthetic file_id=0 for
     * single-file callers that have not yet adopted pqc_anchor_record_file. */
    (void)pqc_anchor_record_file(0, state->epoch, state->sequence,
                                  state->logical_size);

    /* Check freshness window N */
    static int s_blocks_since_commit = 0;
    if (s_freshness_window < 0) {
        const char *env = getenv("PQC_FRESHNESS_WINDOW_N");
        s_freshness_window = env ? atoi(env) : 100; // Default 100
        if (s_freshness_window <= 0) s_freshness_window = 1;
    }

    s_blocks_since_commit++;
    int should_flush = (s_blocks_since_commit >= s_freshness_window);

    if (pqc_anchor_backend() == PQC_ANCHOR_BACKEND_HARDWARE) {
        /* Build prefix anchor and stage for background flush */
        pqc_prefix_anchor_t anchor = {0};
        int rc = build_prefix_anchor(&anchor);
        if (rc != 0) return rc;
        /* monotonicity check against pending */
        pthread_mutex_lock(&g_pending_anchor_lock);
        if (g_pending_anchor_valid &&
            g_pending_anchor.global_sequence > anchor.global_sequence) {
            pthread_mutex_unlock(&g_pending_anchor_lock);
            OPENSSL_cleanse(&anchor, sizeof(anchor));
            return -ESTALE;
        }
        g_pending_anchor       = anchor;
        g_pending_anchor_valid = 1;
        pthread_mutex_unlock(&g_pending_anchor_lock);
        OPENSSL_cleanse(&anchor, sizeof(anchor));

        if (should_flush) {
            s_blocks_since_commit = 0;
            return pqc_anchor_flush();
        }
        return 0;
    }

    /* file backend: write when window is met */
    if (should_flush) {
        pqc_prefix_anchor_t anchor = {0};
        int rc = build_prefix_anchor(&anchor);
        if (rc == 0) rc = write_anchor_file(&anchor);
        OPENSSL_cleanse(&anchor, sizeof(anchor));
        s_blocks_since_commit = 0;
        return rc;
    }
    return 0;
}

int pqc_anchor_flush(void)
{
    if (pqc_anchor_backend() != PQC_ANCHOR_BACKEND_HARDWARE) return 0;
    pthread_mutex_lock(&g_pending_anchor_lock);
    if (!g_pending_anchor_valid) {
        pthread_mutex_unlock(&g_pending_anchor_lock);
        return 0;
    }
    pqc_prefix_anchor_t anchor = g_pending_anchor;
    pthread_mutex_unlock(&g_pending_anchor_lock);
    int rc = write_anchor_tpm(&anchor);
    if (rc == 0) {
        pthread_mutex_lock(&g_pending_anchor_lock);
        OPENSSL_cleanse(&g_pending_anchor, sizeof(g_pending_anchor));
        g_pending_anchor_valid = 0;
        pthread_mutex_unlock(&g_pending_anchor_lock);
    }
    OPENSSL_cleanse(&anchor, sizeof(anchor));
    return rc;
}

/*
 * pqc_anchor_load — called at mount time.
 *
 * Reads the stored prefix anchor and verifies:
 *   1. HMAC is valid (tamper detection).
 *   2. stored global_sequence ≤ locally reconstructed global_sequence.
 *      If stored > local: the on-disk data is OLDER than the anchor
 *      (disk rollback), so fail-closed (-ESTALE).
 *
 * The legacy pqc_anchor_state_t parameter is kept for API compatibility;
 * it is used to register the initial file state in the committed map.
 */
int pqc_anchor_load(const pqc_anchor_state_t *expected_state)
{
    if (!expected_state) return -EINVAL;
    if (pqc_anchor_backend() == PQC_ANCHOR_BACKEND_DISABLED) return 0;

    /* Register expected state so the local map is not empty */
    (void)pqc_anchor_record_file(0, expected_state->epoch,
                                  expected_state->sequence,
                                  expected_state->logical_size);

    pqc_prefix_anchor_t stored = {0};
    int rc;
    if (pqc_anchor_backend() == PQC_ANCHOR_BACKEND_HARDWARE)
        rc = read_anchor_tpm(&stored);
    else
        rc = read_anchor_file(&stored);

    if (rc == -ENOENT) {
        /* no anchor yet — first mount, nothing to check */
        return 0;
    }
    if (rc != 0) return rc;

    /* Handle both v2 (legacy single-file) and v3 (prefix root) on-disk */
    if (stored.magic == PQC_FRESHNESS_ANCHOR_MAGIC &&
        stored.version == PQC_FRESHNESS_ANCHOR_VERSION) {
        /* Legacy v2: verify HMAC over legacy struct, then check sequence */
        const pqc_freshness_anchor_t *legacy =
            (const pqc_freshness_anchor_t *)(const void *)&stored;
        uint8_t digest[32]; unsigned int dlen = 0;
        unsigned char *mac = HMAC(EVP_sha256(), g_master_key, sizeof(g_master_key),
                                  (const unsigned char *)legacy,
                                  offsetof(pqc_freshness_anchor_t, digest),
                                  digest, &dlen);
        if (!mac || dlen != 32 ||
            CRYPTO_memcmp(digest, legacy->digest, 32) != 0) {
            OPENSSL_cleanse(digest, sizeof(digest));
            OPENSSL_cleanse(&stored, sizeof(stored));
            return -EKEYREJECTED;
        }
        OPENSSL_cleanse(digest, sizeof(digest));
        /* If stored epoch > expected: disk was rolled back */
        if (legacy->epoch > expected_state->epoch ||
            legacy->sequence > expected_state->sequence) {
            OPENSSL_cleanse(&stored, sizeof(stored));
            return -ESTALE;
        }
        OPENSSL_cleanse(&stored, sizeof(stored));
        return 0;
    }

    if (stored.magic != PQC_PREFIX_ANCHOR_MAGIC ||
        stored.version != PQC_PREFIX_ANCHOR_VERSION) {
        OPENSSL_cleanse(&stored, sizeof(stored));
        return -EINVAL;
    }

    /* Verify HMAC */
    uint8_t digest[32];
    if (digest_prefix_anchor(&stored, digest) != 0 ||
        CRYPTO_memcmp(digest, stored.digest, 32) != 0) {
        OPENSSL_cleanse(digest, sizeof(digest));
        OPENSSL_cleanse(&stored, sizeof(stored));
        return -EKEYREJECTED;
    }
    OPENSSL_cleanse(digest, sizeof(digest));

    /* Fail-closed: if stored global_sequence > local, data was rolled back */
    uint64_t local_seq;
    pthread_mutex_lock(&g_committed_lock);
    local_seq = g_global_sequence;
    pthread_mutex_unlock(&g_committed_lock);
    if (stored.global_sequence > local_seq) {
        OPENSSL_cleanse(&stored, sizeof(stored));
        return -ESTALE;  /* disk rollback detected — fail-closed */
    }

    OPENSSL_cleanse(&stored, sizeof(stored));
    return 0;
}

int pqc_anchor_self_test(void)
{
    pqc_anchor_state_t state = {
        .epoch = 7,
        .sequence = 11,
        .logical_size = 8192,
    };
    const char *path = anchor_path();
    if (!path || !*path) return 0;
    if (pqc_anchor_backend() == PQC_ANCHOR_BACKEND_DISABLED) return 0;
    if (pqc_anchor_backend() == PQC_ANCHOR_BACKEND_FILE) {
        /* Register multi-file state and build prefix root */
        (void)pqc_anchor_record_file(1, 5, 10, 4096);
        (void)pqc_anchor_record_file(2, 7, 11, 8192);
        int rc = pqc_anchor_store(&state);
        if (rc == 0) rc = pqc_anchor_load(&state);
        return rc;
    }
    int rc = pqc_anchor_store(&state);
    if (rc == 0) rc = pqc_anchor_flush();
    if (rc == 0) rc = pqc_anchor_load(&state);
    return rc;
}

int pqc_anchor_finalize(void)
{
    (void)pqc_anchor_flush();
    return 0;
}
