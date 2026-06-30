#include "pqc_anchor.h"
#include "pqc_config.h"
#include "pqc_durability.h"
#include "pqc_format.h"
#include "pqc_keyring.h"
#include "pqc_lock_profile.h"
#include "pqc_metrics.h"
#include "pqc_plane_trace.h"
#include "pqc_trace_sink.h"

#include <dlfcn.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <oqs/oqs.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stddef.h>
#include <sys/stat.h>
#include <sys/xattr.h>
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>

#ifndef ENOATTR
#define ENOATTR ENODATA
#endif

#ifdef SKIM_HAVE_CUDA_INTEGRITY
#include "cuda_integrity.h"
#endif

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

static pqc_anchor_epoch_record_t g_epoch_record = {
    .version = 1,
    .status = PQC_ANCHOR_EPOCH_STATUS_NONE,
};
static pthread_mutex_t g_epoch_record_lock = PTHREAD_MUTEX_INITIALIZER;

static pthread_mutex_t   g_file_anchor_commit_lock = PTHREAD_MUTEX_INITIALIZER;
static int               g_file_anchor_last_valid = 0;
static char              g_file_anchor_last_path[4096];
static pqc_prefix_anchor_t g_file_anchor_last = {0};
static pqc_trace_sink_t g_anchor_trace_sink = PQC_TRACE_SINK_INITIALIZER;

static atomic_int s_freshness_window = ATOMIC_VAR_INIT(-1);
static atomic_int s_blocks_since_commit = ATOMIC_VAR_INIT(0);

static int anchor_freshness_window(void);

static int file_anchor_commit_lock(pqc_lock_profile_scope_t *scope,
                                   const char *site)
{
    return pqc_profiled_mutex_lock(&g_file_anchor_commit_lock,
                                   "file_anchor_commit_lock", site, scope);
}

static int file_anchor_commit_unlock(pqc_lock_profile_scope_t *scope,
                                     const char *site)
{
    return pqc_profiled_mutex_unlock(&g_file_anchor_commit_lock,
                                     "file_anchor_commit_lock", site, scope);
}

static const char *anchor_backend_name(pqc_anchor_backend_t backend)
{
    switch (backend) {
    case PQC_ANCHOR_BACKEND_DISABLED:
        return "disabled";
    case PQC_ANCHOR_BACKEND_FILE:
        return "file";
    case PQC_ANCHOR_BACKEND_HARDWARE:
        return "hardware";
    }
    return "unknown";
}

const char *pqc_anchor_epoch_status_name(uint32_t status)
{
    switch (status) {
    case PQC_ANCHOR_EPOCH_STATUS_NONE:
        return "none";
    case PQC_ANCHOR_EPOCH_STATUS_PENDING:
        return "pending";
    case PQC_ANCHOR_EPOCH_STATUS_COMMITTED:
        return "committed";
    case PQC_ANCHOR_EPOCH_STATUS_FAILED:
        return "failed";
    }
    return "unknown";
}

static const char *anchor_flush_policy_name(uint32_t policy)
{
    switch (policy) {
    case PQC_ANCHOR_EPOCH_FLUSH_DISABLED:
        return "disabled";
    case PQC_ANCHOR_EPOCH_FLUSH_FILE_WINDOW:
        return "file_window";
    case PQC_ANCHOR_EPOCH_FLUSH_FILE_FORCE:
        return "file_force";
    case PQC_ANCHOR_EPOCH_FLUSH_HARDWARE_STAGE:
        return "hardware_stage";
    case PQC_ANCHOR_EPOCH_FLUSH_HARDWARE_FORCE:
        return "hardware_force";
    case PQC_ANCHOR_EPOCH_FLUSH_FILE_EXTERNAL_SYNCFS:
        return "file_external_syncfs";
    }
    return "unknown";
}

int pqc_anchor_epoch_record_snapshot(pqc_anchor_epoch_record_t *out)
{
    if (!out)
        return -EINVAL;
    pqc_lock_profile_scope_t scope;
    (void)pqc_profiled_mutex_lock(&g_epoch_record_lock,
                                  "anchor_epoch_record_lock", __func__,
                                  &scope);
    *out = g_epoch_record;
    return pqc_profiled_mutex_unlock(&g_epoch_record_lock,
                                     "anchor_epoch_record_lock", __func__,
                                     &scope);
}

static uint64_t anchor_now_ns(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        return 0;
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) +
           (uint64_t)ts.tv_nsec;
}

static void anchor_hex32(const uint8_t in[32], char out[65])
{
    static const char hex[] = "0123456789abcdef";
    if (!in || !out)
        return;
    for (size_t i = 0; i < 32; ++i) {
        out[i * 2] = hex[in[i] >> 4];
        out[i * 2 + 1] = hex[in[i] & 0x0f];
    }
    out[64] = '\0';
}

static void anchor_trace_write_line(const char *line, size_t len)
{
    if (!line || len == 0)
        return;
    (void)pqc_trace_sink_write_env(&g_anchor_trace_sink,
                                   "PQC_ANCHOR_TRACE_PATH",
                                   line, len);
}

void pqc_anchor_trace_shutdown(void)
{
    pqc_trace_sink_close(&g_anchor_trace_sink);
}

static void anchor_trace_event(const char *event, const char *backend, int rc,
                               const pqc_prefix_anchor_t *anchor,
                               uint64_t start_ns, uint64_t end_ns)
{
    const char *trace_path =
        pqc_config_get_nonempty("PQC_ANCHOR_TRACE_PATH");
    if (!trace_path || !event || !backend)
        return;

    char root_hex[65] = {0};
    uint64_t sequence = 0;
    uint64_t file_count = 0;
    if (anchor) {
        anchor_hex32(anchor->prefix_root, root_hex);
        sequence = anchor->global_sequence;
        file_count = anchor->file_count;
    }

    char line[1024];
    uint64_t duration_ns = end_ns >= start_ns ? end_ns - start_ns : 0;
    int n = snprintf(
        line, sizeof(line),
        "{\"event\":\"%s\",\"backend\":\"%s\",\"rc\":%d,"
        "\"global_sequence\":%llu,\"file_count\":%llu,"
        "\"prefix_root\":\"%s\",\"duration_ns\":%llu,"
        "\"blocks_since_commit\":%d,\"freshness_window_cached\":%d}\n",
        event, backend, rc,
        (unsigned long long)sequence,
        (unsigned long long)file_count,
        root_hex,
        (unsigned long long)duration_ns,
        atomic_load_explicit(&s_blocks_since_commit, memory_order_relaxed),
        atomic_load_explicit(&s_freshness_window, memory_order_acquire));
    if (n > 0 && (size_t)n < sizeof(line))
        anchor_trace_write_line(line, (size_t)n);
}

static uint64_t anchor_epoch_interval_snapshot(void)
{
    int cached = atomic_load_explicit(&s_freshness_window,
                                      memory_order_acquire);
    if (cached > 0)
        return (uint64_t)cached;
    return (uint64_t)anchor_freshness_window();
}

static void anchor_trace_epoch_record(
    const char *cause, const pqc_anchor_epoch_record_t *record,
    uint64_t start_ns, uint64_t end_ns)
{
    const char *trace_path =
        pqc_config_get_nonempty("PQC_ANCHOR_TRACE_PATH");
    if (!trace_path || !cause || !record)
        return;

    char root_hex[65] = {0};
    anchor_hex32(record->prefix_root, root_hex);
    uint64_t duration_ns = end_ns >= start_ns ? end_ns - start_ns : 0;
    int pending = record->status == PQC_ANCHOR_EPOCH_STATUS_PENDING;
    int committed = record->status == PQC_ANCHOR_EPOCH_STATUS_COMMITTED;

    char line[1400];
    int n = snprintf(
        line, sizeof(line),
        "{\"event\":\"anchor_epoch_freshness_record\","
        "\"cause\":\"%s\",\"backend\":\"%s\",\"status\":\"%s\","
        "\"flush_policy\":\"%s\",\"rc\":%d,\"pending\":%s,"
        "\"committed\":%s,\"global_sequence\":%llu,"
        "\"file_count\":%llu,\"prefix_root\":\"%s\","
        "\"epoch_interval\":%llu,\"blocks_since_commit\":%llu,"
        "\"duration_ns\":%llu}\n",
        cause,
        anchor_backend_name((pqc_anchor_backend_t)record->backend),
        pqc_anchor_epoch_status_name(record->status),
        anchor_flush_policy_name(record->flush_policy),
        (int)record->last_rc,
        pending ? "true" : "false",
        committed ? "true" : "false",
        (unsigned long long)record->global_sequence,
        (unsigned long long)record->file_count,
        root_hex,
        (unsigned long long)record->epoch_interval,
        (unsigned long long)record->blocks_since_commit,
        (unsigned long long)duration_ns);
    if (n > 0 && (size_t)n < sizeof(line))
        anchor_trace_write_line(line, (size_t)n);
}

static void anchor_epoch_record_update(
    const char *cause, pqc_anchor_backend_t backend, uint32_t status,
    uint32_t flush_policy, int rc, const pqc_prefix_anchor_t *anchor,
    uint64_t start_ns, uint64_t end_ns)
{
    pqc_anchor_epoch_record_t record = {
        .version = 1,
        .backend = (uint32_t)backend,
        .status = status,
        .flush_policy = flush_policy,
        .epoch_interval = anchor_epoch_interval_snapshot(),
        .blocks_since_commit =
            (uint64_t)atomic_load_explicit(&s_blocks_since_commit,
                                           memory_order_relaxed),
        .last_rc = (int32_t)rc,
    };
    if (anchor) {
        record.global_sequence = anchor->global_sequence;
        record.file_count = anchor->file_count;
        memcpy(record.prefix_root, anchor->prefix_root,
               sizeof(record.prefix_root));
    }

    pqc_lock_profile_scope_t scope;
    (void)pqc_profiled_mutex_lock(&g_epoch_record_lock,
                                  "anchor_epoch_record_lock", __func__,
                                  &scope);
    g_epoch_record = record;
    (void)pqc_profiled_mutex_unlock(&g_epoch_record_lock,
                                    "anchor_epoch_record_lock", __func__,
                                    &scope);
    anchor_trace_epoch_record(cause, &record, start_ns, end_ns);
}

static void committed_map_lock(pqc_lock_profile_scope_t *scope,
                               const char *site)
{
    (void)pqc_profiled_mutex_lock(&g_committed_lock,
                                  "committed_map_lock", site, scope);
}

static void committed_map_unlock(pqc_lock_profile_scope_t *scope,
                                 const char *site)
{
    (void)pqc_profiled_mutex_unlock(&g_committed_lock,
                                    "committed_map_lock", site, scope);
}

int pqc_anchor_set_freshness_window(int n)
{
    if (n > 0) {
        atomic_store_explicit(&s_freshness_window, n, memory_order_release);
        return 0;
    }
    return -EINVAL;
}

/* ── Environment helpers ────────────────────────────────────────────────── */

static const char *anchor_path(void)
{
    return pqc_config_get_nonempty("PQC_FRESHNESS_ANCHOR_PATH");
}

static int anchor_path_is_hardware(void)
{
    const char *env = pqc_config_get_nonempty("PQC_FRESHNESS_ANCHOR_BACKEND");
    return env && strcmp(env, "hardware") == 0;
}

static uint32_t tpm_nv_index(void)
{
    return (uint32_t)pqc_config_u64_base_or_default("PQC_TPM_NV_INDEX",
                                                    PQC_TPM_NV_DEFAULT_INDEX, 0);
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
    if (!entries || !out_root)
        return -EINVAL;
    if (count > COMMITTED_MAP_CAPACITY)
        return -EOVERFLOW;

    /* Find next power of two >= count */
    size_t n = 1;
    while (n < count) { n <<= 1; }

    uint8_t leaf_digests[COMMITTED_MAP_CAPACITY * 32] = {0};
    uint8_t leaves_buf[COMMITTED_MAP_CAPACITY * 16];
    uint8_t reduction_buf[COMMITTED_MAP_CAPACITY * 32];

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
        /* Padding leaf digests are already 0 in the bounded scratch buffer. */

        /* 2. CPU bottom-up Merkle reduction with 0x01 internal node prefix */
        size_t current_count = n;
        memcpy(reduction_buf, leaf_digests, n * 32);

        while (current_count > 1) {
            size_t next_count = current_count / 2;
            for (size_t i = 0; i < next_count; ++i) {
                /* Hash child pairs (2 * 32 = 64 bytes) into parent digest with 0x01 prefix */
                uint8_t node_buf[65];
                node_buf[0] = 0x01;
                memcpy(node_buf + 1, reduction_buf + i * 64, 64);
                rc = cpu_sha256(node_buf, 65, reduction_buf + i * 32);
                if (rc != 0)
                    goto out;
            }
            current_count = next_count;
        }
        memcpy(out_root, reduction_buf, 32);
        rc = 0;
    }

out:
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
    pqc_lock_profile_scope_t scope;
    committed_map_lock(&scope, __func__);
    /* update or insert */
    for (size_t i = 0; i < g_committed_count; ++i) {
        if (g_committed_map[i].file_id == file_id) {
            if (max_generation > g_committed_map[i].max_generation)
                g_committed_map[i].max_generation = max_generation;
            if (sequence > g_global_sequence)
                g_global_sequence = sequence;
            committed_map_unlock(&scope, __func__);
            return 0;
        }
    }
    if (g_committed_count >= COMMITTED_MAP_CAPACITY) {
        committed_map_unlock(&scope, __func__);
        return -ENOSPC;
    }

    g_committed_map[g_committed_count].file_id        = file_id;
    g_committed_map[g_committed_count].max_generation = max_generation;
    ++g_committed_count;
    if (sequence > g_global_sequence)
        g_global_sequence = sequence;
    committed_map_unlock(&scope, __func__);
    return 0;
}

static int committed_map_reset(void)
{
    pqc_lock_profile_scope_t scope;
    committed_map_lock(&scope, __func__);
    memset(g_committed_map, 0, sizeof(g_committed_map));
    g_committed_count = 0;
    g_global_sequence = 0;
    committed_map_unlock(&scope, __func__);
    return 0;
}

static int anchor_checkpoint_digest(const pqc_checkpoint_t *ckpt,
                                    uint8_t out[32])
{
    unsigned int out_len = 0;
    unsigned char *mac = HMAC(EVP_sha256(), g_master_key, sizeof(g_master_key),
                              (const unsigned char *)ckpt,
                              offsetof(pqc_checkpoint_t, digest),
                              out, &out_len);
    return (!mac || out_len != sizeof(ckpt->digest)) ? -EIO : 0;
}

static int anchor_record_checkpoint_xattr(const char *path, size_t *recorded)
{
    pqc_checkpoint_t ckpt = {0};
    ssize_t n = getxattr(path, PQC_XATTR_CHECKPOINT, &ckpt, sizeof(ckpt));
    if (n < 0) {
        if (errno == ENODATA || errno == ENOATTR)
            return 0;
        return -errno;
    }
    if ((size_t)n != sizeof(ckpt) || ckpt.magic != PQC_CHECKPOINT_MAGIC ||
        ckpt.version != PQC_CHECKPOINT_VERSION) {
        OQS_MEM_cleanse(&ckpt, sizeof(ckpt));
        return -EINVAL;
    }

    uint8_t digest[32];
    int rc = anchor_checkpoint_digest(&ckpt, digest);
    if (rc == 0 &&
        CRYPTO_memcmp(digest, ckpt.digest, sizeof(digest)) != 0)
        rc = -EKEYREJECTED;
    OQS_MEM_cleanse(digest, sizeof(digest));
    if (rc == 0) {
        rc = pqc_anchor_record_file(ckpt.file_id, ckpt.max_generation,
                                    ckpt.sequence, ckpt.logical_size);
        if (rc == 0 && recorded)
            ++*recorded;
    }
    OQS_MEM_cleanse(&ckpt, sizeof(ckpt));
    return rc;
}

static int anchor_scan_storage_dir(const char *dir_path, size_t *recorded,
                                   size_t depth)
{
    if (!dir_path)
        return -EINVAL;
    if (depth > 64)
        return -ELOOP;

    DIR *dir = opendir(dir_path);
    if (!dir)
        return -errno;

    int rc = 0;
    struct dirent *de;
    while ((de = readdir(dir)) != NULL) {
        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;

        char child[4096];
        int n = snprintf(child, sizeof(child), "%s/%s", dir_path, de->d_name);
        if (n < 0 || (size_t)n >= sizeof(child)) {
            rc = -ENAMETOOLONG;
            break;
        }

        struct stat st;
        if (lstat(child, &st) != 0) {
            rc = -errno;
            break;
        }
        if (S_ISDIR(st.st_mode)) {
            rc = anchor_scan_storage_dir(child, recorded, depth + 1);
            if (rc != 0)
                break;
        } else if (S_ISREG(st.st_mode)) {
            rc = anchor_record_checkpoint_xattr(child, recorded);
            if (rc != 0)
                break;
        }
    }

    int saved = errno;
    closedir(dir);
    errno = saved;
    return rc;
}

int pqc_anchor_reconstruct_committed_map_from_storage(const char *root)
{
    if (pqc_anchor_backend() == PQC_ANCHOR_BACKEND_DISABLED)
        return 0;
    if (!root || !*root)
        return -EINVAL;

    committed_map_reset();
    size_t recorded = 0;
    int rc = anchor_scan_storage_dir(root, &recorded, 0);
    if (rc == 0 && recorded > 0)
        pqc_log("Freshness anchor reconstructed %zu committed checkpoint rows",
                recorded);
    return rc;
}

int pqc_anchor_committed_map_overflow_self_test(void)
{
    committed_entry_t saved_map[COMMITTED_MAP_CAPACITY];
    size_t saved_count;
    uint64_t saved_sequence;

    pqc_lock_profile_scope_t scope;
    committed_map_lock(&scope, __func__);
    saved_count = g_committed_count;
    saved_sequence = g_global_sequence;
    memcpy(saved_map, g_committed_map, sizeof(saved_map));

    g_committed_count = COMMITTED_MAP_CAPACITY;
    g_global_sequence = 17;
    for (size_t i = 0; i < COMMITTED_MAP_CAPACITY; i++) {
        g_committed_map[i].file_id = (uint64_t)i + 1;
        g_committed_map[i].max_generation = (uint64_t)i + 10;
    }
    committed_map_unlock(&scope, __func__);

    int overflow_rc = pqc_anchor_record_file(
        UINT64_C(0xf00d000000000001), 1, 18, 4096);
    int update_rc = pqc_anchor_record_file(1, 999, 999, 4096);

    committed_map_lock(&scope, __func__);
    memcpy(g_committed_map, saved_map, sizeof(saved_map));
    g_committed_count = saved_count;
    g_global_sequence = saved_sequence;
    committed_map_unlock(&scope, __func__);

    return overflow_rc == -ENOSPC && update_rc == 0 ? 0 : -1;
}

/* ── Build and queue a prefix anchor ─────────────────────────────────────── */

static int build_prefix_anchor(pqc_prefix_anchor_t *out)
{
    committed_entry_t copy[COMMITTED_MAP_CAPACITY];
    pqc_lock_profile_scope_t scope;
    committed_map_lock(&scope, __func__);
    size_t count = g_committed_count;
    uint64_t seq = g_global_sequence;
    if (count > COMMITTED_MAP_CAPACITY) {
        committed_map_unlock(&scope, __func__);
        return -EOVERFLOW;
    }
    if (count > 0)
        memcpy(copy, g_committed_map, count * sizeof(copy[0]));
    committed_map_unlock(&scope, __func__);

    if (count > 1)
        qsort(copy, count, sizeof(copy[0]), cmp_committed);

    uint8_t root[32] = {0};
    int rc = compute_prefix_root(count > 0 ? copy : NULL, count, root);
    if (rc != 0) return rc;

    memset(out, 0, sizeof(*out));
    out->magic           = PQC_PREFIX_ANCHOR_MAGIC;
    out->version         = PQC_PREFIX_ANCHOR_VERSION;
    out->global_sequence = seq;
    out->file_count      = (uint64_t)count;
    memcpy(out->prefix_root, root, 32);
    return digest_prefix_anchor(out, out->digest);
}

static int stage_pending_hardware_anchor(const pqc_prefix_anchor_t *anchor)
{
    if (!anchor)
        return -EINVAL;

    uint64_t start_ns = anchor_now_ns();
    int rc = 0;
    pqc_lock_profile_scope_t scope;
    (void)pqc_profiled_mutex_lock(&g_pending_anchor_lock,
                                  "anchor_pending_lock", __func__, &scope);
    if (g_pending_anchor_valid &&
        g_pending_anchor.global_sequence > anchor->global_sequence) {
        (void)pqc_profiled_mutex_unlock(&g_pending_anchor_lock,
                                        "anchor_pending_lock", __func__,
                                        &scope);
        rc = -ESTALE;
        anchor_epoch_record_update(
            "stage_pending", PQC_ANCHOR_BACKEND_HARDWARE,
            PQC_ANCHOR_EPOCH_STATUS_FAILED,
            PQC_ANCHOR_EPOCH_FLUSH_HARDWARE_STAGE, rc, anchor,
            start_ns, anchor_now_ns());
        anchor_trace_event("anchor_stage_pending", "hardware", rc, anchor,
                           start_ns, anchor_now_ns());
        return rc;
    }
    g_pending_anchor = *anchor;
    g_pending_anchor_valid = 1;
    (void)pqc_profiled_mutex_unlock(&g_pending_anchor_lock,
                                    "anchor_pending_lock", __func__, &scope);
    anchor_epoch_record_update(
        "stage_pending", PQC_ANCHOR_BACKEND_HARDWARE,
        PQC_ANCHOR_EPOCH_STATUS_PENDING,
        PQC_ANCHOR_EPOCH_FLUSH_HARDWARE_STAGE, rc, anchor,
        start_ns, anchor_now_ns());
    anchor_trace_event("anchor_stage_pending", "hardware", rc, anchor,
                       start_ns, anchor_now_ns());
    return 0;
}

static int pending_anchor_matches_snapshot(const pqc_prefix_anchor_t *snapshot)
{
    return snapshot &&
           memcmp(&g_pending_anchor, snapshot, sizeof(g_pending_anchor)) == 0;
}

static void clear_pending_hardware_anchor_if_current(
    const pqc_prefix_anchor_t *snapshot)
{
    pqc_lock_profile_scope_t scope;
    (void)pqc_profiled_mutex_lock(&g_pending_anchor_lock,
                                  "anchor_pending_lock", __func__, &scope);
    if (g_pending_anchor_valid && pending_anchor_matches_snapshot(snapshot)) {
        OPENSSL_cleanse(&g_pending_anchor, sizeof(g_pending_anchor));
        g_pending_anchor_valid = 0;
    }
    (void)pqc_profiled_mutex_unlock(&g_pending_anchor_lock,
                                    "anchor_pending_lock", __func__, &scope);
}

int pqc_anchor_pending_clear_self_test(void)
{
    pqc_prefix_anchor_t saved_anchor;
    int saved_valid;

    pqc_lock_profile_scope_t scope;
    (void)pqc_profiled_mutex_lock(&g_pending_anchor_lock,
                                  "anchor_pending_lock", __func__, &scope);
    saved_anchor = g_pending_anchor;
    saved_valid = g_pending_anchor_valid;
    OPENSSL_cleanse(&g_pending_anchor, sizeof(g_pending_anchor));
    g_pending_anchor_valid = 0;
    (void)pqc_profiled_mutex_unlock(&g_pending_anchor_lock,
                                    "anchor_pending_lock", __func__, &scope);

    pqc_prefix_anchor_t old_anchor = {
        .magic = PQC_PREFIX_ANCHOR_MAGIC,
        .version = PQC_PREFIX_ANCHOR_VERSION,
        .global_sequence = 10,
        .file_count = 1,
    };
    pqc_prefix_anchor_t new_anchor = old_anchor;
    old_anchor.prefix_root[0] = 0x10;
    old_anchor.digest[0] = 0x20;
    new_anchor.global_sequence = 11;
    new_anchor.prefix_root[0] = 0x11;
    new_anchor.digest[0] = 0x21;

    int ok = stage_pending_hardware_anchor(&old_anchor) == 0 &&
             stage_pending_hardware_anchor(&new_anchor) == 0;

    clear_pending_hardware_anchor_if_current(&old_anchor);
    (void)pqc_profiled_mutex_lock(&g_pending_anchor_lock,
                                  "anchor_pending_lock", __func__, &scope);
    ok = ok && g_pending_anchor_valid &&
         memcmp(&g_pending_anchor, &new_anchor, sizeof(new_anchor)) == 0;
    (void)pqc_profiled_mutex_unlock(&g_pending_anchor_lock,
                                    "anchor_pending_lock", __func__, &scope);

    clear_pending_hardware_anchor_if_current(&new_anchor);
    (void)pqc_profiled_mutex_lock(&g_pending_anchor_lock,
                                  "anchor_pending_lock", __func__, &scope);
    ok = ok && !g_pending_anchor_valid;
    g_pending_anchor = saved_anchor;
    g_pending_anchor_valid = saved_valid;
    (void)pqc_profiled_mutex_unlock(&g_pending_anchor_lock,
                                    "anchor_pending_lock", __func__, &scope);

    OPENSSL_cleanse(&old_anchor, sizeof(old_anchor));
    OPENSSL_cleanse(&new_anchor, sizeof(new_anchor));
    OPENSSL_cleanse(&saved_anchor, sizeof(saved_anchor));
    return ok ? 0 : -1;
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

static int sync_anchor_parent_dir(const char *path)
{
    if (!path || !*path)
        return -EINVAL;

    char dir_path[4096];
    const char *slash = strrchr(path, '/');
    if (!slash) {
        memcpy(dir_path, ".", 2);
    } else if (slash == path) {
        memcpy(dir_path, "/", 2);
    } else {
        size_t dir_len = (size_t)(slash - path);
        if (dir_len >= sizeof(dir_path))
            return -ENAMETOOLONG;
        memcpy(dir_path, path, dir_len);
        dir_path[dir_len] = '\0';
    }

    int dir_fd = open(dir_path, O_RDONLY | O_DIRECTORY);
    if (dir_fd < 0)
        return -errno;
    int rc = pqc_durability_fsync(dir_fd, PQC_DURABILITY_SITE_PARENT_DIR);
    if (close(dir_fd) != 0 && rc == 0)
        rc = -errno;
    return rc;
}

static int write_anchor_file(const pqc_prefix_anchor_t *anchor,
                             int external_syncfs)
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
    int renamed = 0;
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
        if (rc == 0 && !external_syncfs)
            rc = pqc_durability_fdatasync(
                fd, PQC_DURABILITY_SITE_ANCHOR_FILE);
    }
    if (close(fd) != 0 && rc == 0) rc = -errno;
    if (rc == 0) {
        if (rename(tmpname, path) != 0) {
            rc = -errno;
        } else {
            renamed = 1;
            if (!external_syncfs)
                rc = sync_anchor_parent_dir(path);
        }
    }
    if (rc != 0 && !renamed) unlink(tmpname);
    return rc;
}

static int file_anchor_already_committed(const char *path,
                                         const pqc_prefix_anchor_t *anchor)
{
    if (!path || !anchor)
        return 0;

    int same = 0;
    pqc_lock_profile_scope_t scope;
    if (file_anchor_commit_lock(&scope, "file_anchor_already_committed") != 0)
        return 0;
    same = g_file_anchor_last_valid &&
        strncmp(g_file_anchor_last_path, path,
                sizeof(g_file_anchor_last_path)) == 0 &&
        memcmp(&g_file_anchor_last, anchor, sizeof(*anchor)) == 0;
    (void)file_anchor_commit_unlock(&scope, "file_anchor_already_committed");
    return same;
}

static void file_anchor_note_committed(const char *path,
                                       const pqc_prefix_anchor_t *anchor)
{
    if (!path || !anchor)
        return;

    pqc_lock_profile_scope_t scope;
    if (file_anchor_commit_lock(&scope, "file_anchor_note_committed") != 0)
        return;
    memset(g_file_anchor_last_path, 0, sizeof(g_file_anchor_last_path));
    strncpy(g_file_anchor_last_path, path,
            sizeof(g_file_anchor_last_path) - 1U);
    g_file_anchor_last = *anchor;
    g_file_anchor_last_valid = 1;
    (void)file_anchor_commit_unlock(&scope, "file_anchor_note_committed");
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

static int anchor_freshness_window(void)
{
    int cached = atomic_load_explicit(&s_freshness_window,
                                      memory_order_acquire);
    if (cached > 0)
        return cached;

    int configured =
        (int)pqc_config_long_legacy_or_default("PQC_FRESHNESS_WINDOW_N",
                                               100);
    if (configured <= 0)
        configured = 1;

    int expected = -1;
    if (atomic_compare_exchange_strong_explicit(
            &s_freshness_window, &expected, configured,
            memory_order_release, memory_order_acquire)) {
        return configured;
    }

    cached = atomic_load_explicit(&s_freshness_window, memory_order_acquire);
    return cached > 0 ? cached : configured;
}

static int anchor_commit_current_prefix_sync(
    pqc_anchor_backend_t backend,
    int force_flush,
    pqc_anchor_external_sync_fn external_sync,
    void *external_sync_opaque)
{
    pqc_prefix_anchor_t anchor = {0};
    const char *backend_name = anchor_backend_name(backend);
    uint32_t flush_policy =
        backend == PQC_ANCHOR_BACKEND_HARDWARE
            ? (uint32_t)PQC_ANCHOR_EPOCH_FLUSH_HARDWARE_STAGE
            : backend == PQC_ANCHOR_BACKEND_FILE
                ? (external_sync
                       ? (uint32_t)PQC_ANCHOR_EPOCH_FLUSH_FILE_EXTERNAL_SYNCFS
                       : (force_flush
                              ? (uint32_t)PQC_ANCHOR_EPOCH_FLUSH_FILE_FORCE
                              : (uint32_t)PQC_ANCHOR_EPOCH_FLUSH_FILE_WINDOW))
                : (uint32_t)PQC_ANCHOR_EPOCH_FLUSH_DISABLED;
    uint64_t start_ns = anchor_now_ns();
    int rc = build_prefix_anchor(&anchor);
    if (rc != 0) {
        anchor_epoch_record_update(
            "commit_current_prefix", backend,
            PQC_ANCHOR_EPOCH_STATUS_FAILED, flush_policy, rc, NULL,
            start_ns, anchor_now_ns());
        anchor_trace_event("anchor_commit_current_prefix", backend_name, rc,
                           NULL, start_ns, anchor_now_ns());
        pqc_plane_trace_record_freshness_anchor((uint32_t)backend, rc);
        return rc;
    }

    if (backend == PQC_ANCHOR_BACKEND_HARDWARE) {
        rc = stage_pending_hardware_anchor(&anchor);
        if (rc == 0 && force_flush)
            rc = pqc_anchor_flush();
        if (rc == 0 && external_sync)
            rc = external_sync(external_sync_opaque);
    } else if (backend == PQC_ANCHOR_BACKEND_FILE) {
        const char *path = anchor_path();
        int already_committed = file_anchor_already_committed(path, &anchor);
        if (!already_committed) {
            rc = write_anchor_file(&anchor, external_sync != NULL);
        }
        if (rc == 0 && external_sync)
            rc = external_sync(external_sync_opaque);
        if (rc == 0 && !already_committed)
            file_anchor_note_committed(path, &anchor);
    } else if (external_sync) {
        rc = external_sync(external_sync_opaque);
    } else {
        rc = 0;
    }

    if (backend != PQC_ANCHOR_BACKEND_HARDWARE) {
        anchor_epoch_record_update(
            "commit_current_prefix", backend,
            rc == 0 ? PQC_ANCHOR_EPOCH_STATUS_COMMITTED
                    : PQC_ANCHOR_EPOCH_STATUS_FAILED,
            flush_policy, rc, &anchor, start_ns, anchor_now_ns());
    }
    anchor_trace_event("anchor_commit_current_prefix", backend_name, rc,
                       &anchor, start_ns, anchor_now_ns());
    pqc_plane_trace_record_freshness_anchor((uint32_t)backend, rc);
    OPENSSL_cleanse(&anchor, sizeof(anchor));
    return rc;
}

static int pqc_anchor_store_common_sync(
    const pqc_anchor_state_t *state,
    int force_flush,
    pqc_anchor_external_sync_fn external_sync,
    void *external_sync_opaque)
{
    if (!state) return -EINVAL;
    pqc_anchor_backend_t backend = pqc_anchor_backend();
    if (backend == PQC_ANCHOR_BACKEND_DISABLED)
        return external_sync ? external_sync(external_sync_opaque) : 0;

    /* Register in the committed-prefix map with a synthetic file_id=0 for
     * single-file callers that have not yet adopted pqc_anchor_record_file. */
    int record_rc = pqc_anchor_record_file(0, state->epoch,
                                           state->sequence,
                                           state->logical_size);
    if (record_rc != 0)
        return record_rc;

    int commits_since = atomic_fetch_add_explicit(&s_blocks_since_commit, 1,
                                                  memory_order_relaxed) + 1;
    int should_flush =
        force_flush || commits_since >= anchor_freshness_window();

    if (backend == PQC_ANCHOR_BACKEND_HARDWARE || should_flush) {
        int rc = anchor_commit_current_prefix_sync(
            backend, should_flush, external_sync, external_sync_opaque);
        if (rc != 0)
            return rc;
    } else if (external_sync) {
        int rc = external_sync(external_sync_opaque);
        if (rc != 0)
            return rc;
    }

    if (should_flush)
        atomic_store_explicit(&s_blocks_since_commit, 0,
                              memory_order_relaxed);

    return 0;
}

/*
 * pqc_anchor_store — legacy single-file compatibility shim.
 * The normal path keeps freshness-window batching for file backends and stages
 * hardware anchors for later TPM/NV flush.
 */
int pqc_anchor_store(const pqc_anchor_state_t *state)
{
    return pqc_anchor_store_common_sync(state, 0, NULL, NULL);
}

int pqc_anchor_store_force(const pqc_anchor_state_t *state)
{
    return pqc_anchor_store_common_sync(state, 1, NULL, NULL);
}

int pqc_anchor_store_force_external_sync(
    const pqc_anchor_state_t *state,
    pqc_anchor_external_sync_fn sync_fn,
    void *opaque)
{
    if (!sync_fn)
        return pqc_anchor_store_force(state);
    return pqc_anchor_store_common_sync(state, 1, sync_fn, opaque);
}

int pqc_anchor_store_windowed_external_sync(
    const pqc_anchor_state_t *state,
    pqc_anchor_external_sync_fn sync_fn,
    void *opaque)
{
    if (!sync_fn)
        return pqc_anchor_store(state);
    return pqc_anchor_store_common_sync(state, 0, sync_fn, opaque);
}

int pqc_anchor_flush(void)
{
    if (pqc_anchor_backend() != PQC_ANCHOR_BACKEND_HARDWARE) return 0;
    uint64_t start_ns = anchor_now_ns();
    pqc_lock_profile_scope_t scope;
    (void)pqc_profiled_mutex_lock(&g_pending_anchor_lock,
                                  "anchor_pending_lock", __func__, &scope);
    if (!g_pending_anchor_valid) {
        (void)pqc_profiled_mutex_unlock(&g_pending_anchor_lock,
                                        "anchor_pending_lock", __func__,
                                        &scope);
        anchor_trace_event("anchor_flush", "hardware", 0, NULL, start_ns,
                           anchor_now_ns());
        return 0;
    }
    pqc_prefix_anchor_t anchor = g_pending_anchor;
    (void)pqc_profiled_mutex_unlock(&g_pending_anchor_lock,
                                    "anchor_pending_lock", __func__, &scope);
    int rc = write_anchor_tpm(&anchor);
    if (rc == 0)
        clear_pending_hardware_anchor_if_current(&anchor);
    anchor_epoch_record_update(
        "flush", PQC_ANCHOR_BACKEND_HARDWARE,
        rc == 0 ? PQC_ANCHOR_EPOCH_STATUS_COMMITTED
                : PQC_ANCHOR_EPOCH_STATUS_FAILED,
        PQC_ANCHOR_EPOCH_FLUSH_HARDWARE_FORCE, rc, &anchor, start_ns,
        anchor_now_ns());
    anchor_trace_event("anchor_flush", "hardware", rc, &anchor, start_ns,
                       anchor_now_ns());
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
    int record_rc = pqc_anchor_record_file(0, expected_state->epoch,
                                           expected_state->sequence,
                                           expected_state->logical_size);
    if (record_rc != 0)
        return record_rc;

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
    pqc_lock_profile_scope_t scope;
    committed_map_lock(&scope, __func__);
    local_seq = g_global_sequence;
    committed_map_unlock(&scope, __func__);
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
        int rc = pqc_anchor_record_file(1, 5, 10, 4096);
        if (rc == 0)
            rc = pqc_anchor_record_file(2, 7, 11, 8192);
        if (rc == 0)
            rc = pqc_anchor_store_force(&state);
        if (rc == 0) rc = pqc_anchor_load(&state);
        return rc;
    }
    int rc = pqc_anchor_store_force(&state);
    if (rc == 0) rc = pqc_anchor_load(&state);
    return rc;
}

int pqc_anchor_finalize(void)
{
    (void)pqc_anchor_flush();
    return 0;
}
