#include "pqc_file_io.h"

#include "pqc_admission.h"
#include "pqc_anchor_worker.h"
#include "pqc_checkpoint.h"
#include "pqc_crypto.h"
#include "pqc_durability.h"
#include "pqc_fd_context.h"
#include "pqc_format.h"
#include "pqc_keyring.h"
#include "pqc_lock_profile.h"
#include "pqc_metrics.h"
#include "pqc_posix.h"
#include "pqc_publish.h"
#include "pqc_qos.h"
#include "pqc_recovery.h"
#include "pqc_rekey.h"
#include "pqc_storage_path.h"
#include "pqc_test_hooks.h"
#include "pqc_writeback.h"

#include <errno.h>
#include <fcntl.h>
#include <linux/falloc.h>
#include <oqs/oqs.h>
#include <openssl/rand.h>
#include <pthread.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/xattr.h>
#include <time.h>
#include <unistd.h>

typedef enum {
    DRAIN_WAIT_NONE = 0,
    DRAIN_WAIT_WRITEBACK,
    DRAIN_WAIT_ALL,
} drain_wait_mode_t;

static int marker_path_is_hidden_unlinked(const char *path)
{
    if (!path)
        return 0;
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    return strncmp(base, ".fuse_hidden", strlen(".fuse_hidden")) == 0;
}

static void cleanup_hidden_marker_and_sidecars(const char *marker_path)
{
    if (!marker_path_is_hidden_unlinked(marker_path))
        return;
    char sidecar_path[4096 + 16];
    if (pqc_sidecar_path(sidecar_path, sizeof(sidecar_path),
                         marker_path, ".pqcdata") == 0)
        (void)unlink(sidecar_path);
    if (pqc_sidecar_path(sidecar_path, sizeof(sidecar_path),
                         marker_path, ".pqcmeta") == 0)
        (void)unlink(sidecar_path);
    if (pqc_sidecar_path(sidecar_path, sizeof(sidecar_path),
                         marker_path, ".pqcepoch") == 0)
        (void)unlink(sidecar_path);
    (void)unlink(marker_path);
}

static int rekey_enqueue_status_should_log(unsigned long long count)
{
    return count <= 4 || (count & (count - 1)) == 0;
}

static void log_rekey_enqueue_drop(pqc_rekey_enqueue_status_t status, int fd)
{
    static atomic_ullong invalid_fd_count = ATOMIC_VAR_INIT(0);
    static atomic_ullong queue_full_count = ATOMIC_VAR_INIT(0);
    const char *reason = NULL;
    unsigned long long count = 0;

    if (status == PQC_REKEY_ENQUEUE_INVALID_FD)
        reason = "invalid_fd";
    else if (status == PQC_REKEY_ENQUEUE_FULL)
        reason = "queue_full";
    else
        return;

    if (status == PQC_REKEY_ENQUEUE_INVALID_FD)
        count = atomic_fetch_add_explicit(&invalid_fd_count, 1,
                                          memory_order_relaxed) + 1;
    else
        count = atomic_fetch_add_explicit(&queue_full_count, 1,
                                          memory_order_relaxed) + 1;

    if (rekey_enqueue_status_should_log(count))
        pqc_log("REKEY QUEUE: enqueue dropped reason=%s fd=%d count=%llu",
                reason, fd, count);
}

static int rekey_due_locked(const pqc_fd_ctx_t *ctx)
{
    int interval = pqc_rekey_rotation_interval_s();
    if (pqc_rekey_force_on_write_enabled())
        return 1;
    return interval > 0 && (time(NULL) - ctx->last_rekey) >= interval;
}

static int write_buffered_locked(int fd, pqc_fd_ctx_t *ctx,
                                 pqc_lock_profile_scope_t *scope,
                                 const char *buf, size_t size,
                                 off_t offset, const char *site)
{
    size_t bytes_written = 0;
    while (bytes_written < size) {
        size_t chunk = size - bytes_written;
        if (chunk > COALESCE_SIZE - ctx->wbuf_used)
            chunk = COALESCE_SIZE - ctx->wbuf_used;
        if (ctx->wbuf_used == 0)
            ctx->wbuf_base_off = offset + bytes_written;
        memcpy(ctx->wbuf + ctx->wbuf_used, buf + bytes_written, chunk);
        ctx->wbuf_used += chunk;
        bytes_written += chunk;
        if (ctx->wbuf_used >= COALESCE_SIZE) {
            int fr = pqc_writeback_flush_locked(fd, ctx, scope, site, 0);
            if (fr < 0)
                return fr;
        }
    }
    return 0;
}

static int drain_writeback_locked(int fd, pqc_fd_ctx_t *ctx,
                                  pqc_lock_profile_scope_t *scope,
                                  const char *site, int require_durable,
                                  drain_wait_mode_t wait_mode)
{
    int res = 0;
    if (ctx->valid && ctx->wbuf && ctx->wbuf_used > 0)
        res = pqc_writeback_flush_locked(fd, ctx, scope, site,
                                         require_durable);
    if (res == 0 && wait_mode == DRAIN_WAIT_WRITEBACK)
        pqc_fd_context_wait_writeback_locked(ctx, scope, site);
    else if (res == 0 && wait_mode == DRAIN_WAIT_ALL)
        pqc_fd_context_wait_pending_locked(ctx, scope, site);
    return res;
}

static int load_tier_for_path(const char *phys_path, int *out)
{
    if (!phys_path || !out)
        return -EINVAL;
    char value[8] = {0};
    ssize_t n = getxattr(phys_path, PQC_XATTR_TIER, value,
                         sizeof(value) - 1);
    if (n == -1)
        return -errno;
    if (n <= 0)
        return -EINVAL;
    int tier = atoi(value);
    if (tier != PQC_TIER_FULL && tier != PQC_TIER_NONE)
        return -EINVAL;
    *out = tier;
    return 0;
}

static void restore_open_policy_for_fd(int fd, const char *phys_path)
{
    int qos_class = PQC_QOS_CLASS_ELASTIC;
    int rc = pqc_qos_class_load_for_path(phys_path, &qos_class);
    if (rc != 0)
        qos_class = PQC_QOS_CLASS_ELASTIC;
    int tier = PQC_TIER_FULL;
    rc = load_tier_for_path(phys_path, &tier);
    if (rc != 0)
        tier = PQC_TIER_FULL;

    pqc_fd_ctx_t *ctx = pqc_fd_context_for_fd(fd);
    if (!ctx)
        return;
    pqc_lock_profile_scope_t fd_scope;
    (void)pqc_profiled_mutex_lock(&ctx->fd_lock, "fd_lock", __func__,
                                  &fd_scope);
    if (ctx->valid) {
        ctx->tier = tier;
        ctx->qos_class = qos_class;
    }
    (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", __func__,
                                    &fd_scope);
}

static void metadata_publish_turn_finish_locked(file_state_t *state,
                                                uint64_t ticket)
{
    if (!state)
        return;
    if (state->publish_ticket == ticket) {
        ++state->publish_ticket;
        pthread_cond_broadcast(&state->publish_cv);
    }
}

static int metadata_publish_turn_begin(file_state_t *state,
                                       const char *site,
                                       uint64_t *ticket_out,
                                       uint64_t *next_generation_out,
                                       uint64_t *logical_size_out)
{
    if (!state || !ticket_out)
        return -EINVAL;

    pqc_lock_profile_scope_t scope;
    (void)pqc_profiled_mutex_lock(&state->commit_lock, "commit_lock",
                                  site, &scope);
    uint64_t ticket = state->next_publish_ticket++;
    while (state->publish_ticket != ticket)
        (void)pqc_profiled_cond_wait(&state->publish_cv,
                                     &state->commit_lock,
                                     "commit_lock", site, &scope);
    *ticket_out = ticket;
    if (next_generation_out)
        *next_generation_out = state->next_generation;
    if (logical_size_out)
        *logical_size_out = state->logical_size_valid
            ? state->logical_size
            : 0;
    (void)pqc_profiled_mutex_unlock(&state->commit_lock, "commit_lock",
                                    site, &scope);
    return 0;
}

static void metadata_publish_turn_finish(file_state_t *state,
                                         const char *site,
                                         uint64_t ticket)
{
    if (!state)
        return;
    pqc_lock_profile_scope_t scope;
    (void)pqc_profiled_mutex_lock(&state->commit_lock, "commit_lock",
                                  site, &scope);
    metadata_publish_turn_finish_locked(state, ticket);
    (void)pqc_profiled_mutex_unlock(&state->commit_lock, "commit_lock",
                                    site, &scope);
}

static int sync_dirty_sidecars_without_fd_lock(pqc_fd_ctx_t *ctx,
                                               pqc_lock_profile_scope_t *scope,
                                               const char *site)
{
    pqc_fd_sidecar_sync_t sync;
    /*
     * Callers hold fd_lock and have waited for pending writeback jobs.  Dirty
     * sidecar fd/epoch snapshots are fd-context state, not file-state commit
     * state, so commit_lock is intentionally not part of this path.
     */
    int rc = pqc_fd_context_prepare_dirty_sidecar_sync_locked(ctx, &sync);
    if (rc != 0)
        return rc;
    if (!sync.sync_data && !sync.sync_journal && !sync.sync_epoch_log)
        return 0;

    pqc_fd_context_pending_job_begin(ctx);
    (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", site, scope);
    rc = pqc_fd_context_run_dirty_sidecar_sync(&sync);
    (void)pqc_profiled_mutex_lock(&ctx->fd_lock, "fd_lock", site, scope);
    pqc_fd_context_finish_dirty_sidecar_sync_locked(ctx, &sync, rc);
    pqc_fd_context_pending_job_end(ctx);
    return rc;
}

static int read_batch_scratch_acquire_locked(
    pqc_fd_ctx_t *ctx,
    pqc_lock_profile_scope_t *scope,
    const char *site,
    uint8_t **plain_batch,
    uint8_t **cipher_batch,
    size_t *capacity)
{
    if (plain_batch)
        *plain_batch = NULL;
    if (cipher_batch)
        *cipher_batch = NULL;
    if (capacity)
        *capacity = 0;
    if (!ctx || !plain_batch || !cipher_batch || !capacity)
        return -EINVAL;

    while (ctx->read_scratch_in_use)
        (void)pqc_profiled_cond_wait(&ctx->pending_cv, &ctx->fd_lock,
                                     "fd_lock", site, scope);
    if (!ctx->read_plain_batch || !ctx->read_cipher_batch ||
        ctx->read_batch_capacity < PQC_READ_BATCH_SCRATCH_SIZE)
        return -ENOMEM;

    ctx->read_scratch_in_use = 1;
    *plain_batch = ctx->read_plain_batch;
    *cipher_batch = ctx->read_cipher_batch;
    *capacity = ctx->read_batch_capacity;
    return 0;
}

static void read_batch_scratch_release_locked(pqc_fd_ctx_t *ctx)
{
    if (!ctx || !ctx->read_scratch_in_use)
        return;
    ctx->read_scratch_in_use = 0;
    pthread_cond_broadcast(&ctx->pending_cv);
}

static void read_batch_scratch_finish_job_locked(pqc_fd_ctx_t *ctx)
{
    read_batch_scratch_release_locked(ctx);
    pqc_fd_context_pending_job_end(ctx);
}

static int marker_metadata_fsync(void *opaque)
{
    if (!opaque)
        return -EINVAL;
    int fd = *(int *)opaque;
    if (fd < 0)
        return -EBADF;
    return pqc_durability_fsync(fd, PQC_DURABILITY_SITE_MARKER_METADATA);
}

static int read_authenticated_block_batch(
    file_state_t *state,
    int journal_fd,
    int data_fd,
    const char *marker_path,
    pqc_journal_lookup_view_t *journal_view,
    pqc_recovery_epoch_fallback_view_t *epoch_view,
    const uint8_t *key,
    size_t key_len,
    uint64_t file_id,
    uint64_t max_generation,
    uint64_t base,
    uint64_t end,
    uint64_t first_block,
    size_t block_count,
    uint8_t *cipher_batch,
    uint8_t *plain_batch,
    size_t scratch_capacity,
    char *buf)
{
    if (!key || !buf || !cipher_batch || !plain_batch || block_count == 0 ||
        block_count > PQC_READ_BATCH_MAX_BLOCKS)
        return -EINVAL;

    pqc_crypto_block_desc_t descs[PQC_READ_BATCH_MAX_BLOCKS];
    size_t scratch_bytes = block_count * PQC_LOGICAL_BLOCK_SIZE;
    if (scratch_bytes > scratch_capacity)
        return -E2BIG;

    size_t decrypt_count = 0;
    int rc = 0;
    for (size_t bi = 0; bi < block_count; ++bi) {
        uint64_t block = first_block + bi;
        if (block < first_block ||
            block > UINT64_MAX / PQC_LOGICAL_BLOCK_SIZE) {
            rc = -EFBIG;
            goto out;
        }
        block_mapping_t map;
        rc = pqc_file_state_mapping_cache_lookup(state, block,
                                                 max_generation, &map);
        if (rc == -EAGAIN) {
            rc = pqc_recovery_lookup_mapping_committed_epoch_view(
                journal_fd, marker_path, journal_view, epoch_view, file_id,
                block, max_generation, &map);
        }
        if (rc == -ENOENT) {
            memset(plain_batch + bi * PQC_LOGICAL_BLOCK_SIZE, 0,
                   PQC_LOGICAL_BLOCK_SIZE);
            rc = 0;
            continue;
        }
        if (rc != 0)
            goto out;
        if (map.plaintext_length > PQC_LOGICAL_BLOCK_SIZE) {
            rc = -EINVAL;
            goto out;
        }

        size_t packed_offset = bi * PQC_LOGICAL_BLOCK_SIZE;
        if (map.plaintext_length < PQC_LOGICAL_BLOCK_SIZE)
            memset(plain_batch + packed_offset, 0,
                   PQC_LOGICAL_BLOCK_SIZE);
        if (map.plaintext_length > 0) {
            ssize_t n = pread(data_fd, cipher_batch + packed_offset,
                              map.plaintext_length,
                              (off_t)map.ciphertext_offset);
            if (n != (ssize_t)map.plaintext_length) {
                rc = n < 0 ? -errno : -EIO;
                goto out;
            }
        }

        pqc_crypto_block_desc_t *desc = &descs[decrypt_count++];
        memset(desc, 0, sizeof(*desc));
        desc->block = block;
        desc->generation = map.generation;
        desc->length = map.plaintext_length;
        desc->input_offset = packed_offset;
        desc->output_offset = packed_offset;
        desc->ciphertext_offset = map.ciphertext_offset;
        memcpy(desc->tag, map.tag, sizeof(desc->tag));
        rc = pqc_crypto_derive_block_nonce(file_id, block, map.generation,
                                           desc->nonce);
        if (rc != 0)
            goto out;
        pqc_crypto_build_block_aad(desc->aad, file_id, block,
                                   map.generation, map.plaintext_length);
    }

    if (decrypt_count > 0) {
        rc = pqc_crypto_decrypt_block_batch_gcm(
            key, key_len, file_id, descs, decrypt_count, cipher_batch,
            plain_batch, 0);
        if (rc != 0)
            goto out;
    }

    for (size_t bi = 0; bi < block_count; ++bi) {
        uint64_t block = first_block + bi;
        uint64_t block_start = block * PQC_LOGICAL_BLOCK_SIZE;
        uint64_t from = base > block_start ? base - block_start : 0;
        uint64_t block_end = block_start + PQC_LOGICAL_BLOCK_SIZE;
        uint64_t to = end < block_end ? end - block_start
                                      : PQC_LOGICAL_BLOCK_SIZE;
        if (to < from) {
            rc = -EINVAL;
            goto out;
        }
        uint64_t dest = block_start + from - base;
        uint64_t copy_len = to - from;
        if (dest > SIZE_MAX || copy_len > SIZE_MAX) {
            rc = -EOVERFLOW;
            goto out;
        }
        pqc_fault_cutpoint("read_after_auth");
        memcpy(buf + (size_t)dest,
               plain_batch + bi * PQC_LOGICAL_BLOCK_SIZE + from,
               (size_t)copy_len);
    }

out:
    OQS_MEM_cleanse(descs, decrypt_count * sizeof(descs[0]));
    OQS_MEM_cleanse(cipher_batch, scratch_bytes);
    OQS_MEM_cleanse(plain_batch, scratch_bytes);
    return rc;
}

int pqc_open(const char *path, struct fuse_file_info *fi)
{
    if (pqc_is_hidden_sidecar_path(path))
        return -ENOENT;
    char phys_path[4096];
    pqc_storage_path_resolve(phys_path, sizeof(phys_path), path);

    char compat_path[4096];
    const char *open_path = phys_path;
    if (pqc_sqlite_sidecar_redirect_path(compat_path, sizeof(compat_path), path) == 0)
        open_path = compat_path;

    int fd = open(open_path, fi->flags, 0600);
    if (fd == -1)
        return -errno;

    fi->fh = (uint64_t)fd;
    /* SQLite WAL uses a shared-memory sidecar (`-shm`) and expects mmap-capable
     * file descriptors.  The default direct-IO path breaks that contract on
     * this FUSE stack, so we allow a compatibility mode that keeps buffered I/O
     * only for redirected WAL-visible files while preserving encrypted data
     * files as non-mmap direct-I/O handles. */
    int sqlite_mmap_sidecar = (open_path == compat_path);
    fi->direct_io = sqlite_mmap_sidecar ? 0 : 1;
    if (sqlite_mmap_sidecar)
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
    int metadata_rc = pqc_keyring_metadata_load(phys_path, ss, &ss_len, &fid);
    if (metadata_rc != 0 ||
        pqc_fd_context_set(fd, path, phys_path, ss, ss_len, fid,
                            fi->flags) != 0) {
        OQS_MEM_cleanse(ss, sizeof(ss));
        close(fd);
        return metadata_rc == -EKEYREJECTED ? -EKEYREJECTED : -EIO;
    }
    OQS_MEM_cleanse(ss, sizeof(ss));
    pqc_log("OPEN %s: restored authenticated file-key envelope (fid=%llu)",
            path, (unsigned long long)fid);

    restore_open_policy_for_fd(fd, phys_path);
    return 0;
}

int pqc_read(const char *path, char *buf, size_t size, off_t offset,
             struct fuse_file_info *fi)
{
    (void)path;
    int fd  = (int)fi->fh;
    int idx = pqc_fd_context_index_for_fd(fd);
    pqc_fd_ctx_t *ctx = pqc_fd_context_at_index((size_t)idx);
    if (!ctx)
        return -EBADF;
    if (size == 0)
        return 0;

    pqc_lock_profile_scope_t fd_scope;
    (void)pqc_profiled_mutex_lock(&ctx->fd_lock, "fd_lock", __func__,
                                  &fd_scope);
    if (!ctx->valid || ctx->tier == PQC_TIER_NONE) {
        (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", __func__,
                                        &fd_scope);
        int res = (int)pread(fd, buf, size, offset);
        return res <= 0 ? (res == -1 ? -errno : 0) : res;
    }
    {
        pqc_lock_profile_scope_t commit_scope;
        (void)pqc_profiled_mutex_lock(&ctx->state->commit_lock, "commit_lock",
                                      __func__, &commit_scope);
        uint64_t visible_size = ctx->state->logical_size_valid
            ? ctx->state->logical_size
            : ctx->logical_size;
        ctx->logical_size = visible_size;
        if ((uint64_t)offset >= visible_size) {
            (void)pqc_profiled_mutex_unlock(&ctx->state->commit_lock,
                                            "commit_lock", __func__,
                                            &commit_scope);
            (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock",
                                            __func__, &fd_scope);
            return 0;
        }
        (void)pqc_profiled_mutex_unlock(&ctx->state->commit_lock,
                                        "commit_lock", __func__,
                                        &commit_scope);
    }
    uint8_t *read_plain_batch = NULL;
    uint8_t *read_cipher_batch = NULL;
    size_t read_batch_capacity = 0;
    pqc_fd_context_pending_job_begin(ctx);
    int scratch_rc = read_batch_scratch_acquire_locked(
        ctx, &fd_scope, __func__, &read_plain_batch, &read_cipher_batch,
        &read_batch_capacity);
    if (scratch_rc != 0) {
        pqc_fd_context_pending_job_end(ctx);
        (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", __func__,
                                        &fd_scope);
        return scratch_rc;
    }
    if (!ctx->valid) {
        read_batch_scratch_finish_job_locked(ctx);
        (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", __func__,
                                        &fd_scope);
        return -EBADF;
    }
    if (ctx->tier == PQC_TIER_NONE) {
        read_batch_scratch_finish_job_locked(ctx);
        (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", __func__,
                                        &fd_scope);
        int res = (int)pread(fd, buf, size, offset);
        return res <= 0 ? (res == -1 ? -errno : 0) : res;
    }
    pqc_lock_profile_scope_t commit_scope;
    (void)pqc_profiled_mutex_lock(&ctx->state->commit_lock, "commit_lock",
                                  __func__, &commit_scope);
    uint64_t visible_size = ctx->state->logical_size_valid
        ? ctx->state->logical_size
        : ctx->logical_size;
    uint64_t visible_generation = ctx->state->committed_generation;
    ctx->logical_size = visible_size;
    if ((uint64_t)offset >= visible_size) {
        (void)pqc_profiled_mutex_unlock(&ctx->state->commit_lock,
                                        "commit_lock", __func__,
                                        &commit_scope);
        read_batch_scratch_finish_job_locked(ctx);
        (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", __func__,
                                        &fd_scope);
        return 0;
    }
    size_t want = size;
    if (want > visible_size - (uint64_t)offset)
        want = visible_size - (uint64_t)offset;
    (void)pqc_profiled_mutex_unlock(&ctx->state->commit_lock, "commit_lock",
                                    __func__, &commit_scope);
    if (want == 0) {
        read_batch_scratch_finish_job_locked(ctx);
        (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", __func__,
                                        &fd_scope);
        return 0;
    }
    int journal_fd = ctx->journal_fd;
    int data_fd = ctx->data_fd;
    char marker_path[sizeof(ctx->marker_path)];
    memcpy(marker_path, ctx->marker_path, sizeof(marker_path));
    uint8_t ss[64] = {0};
    if (ctx->ss_len > sizeof(ss)) {
        read_batch_scratch_finish_job_locked(ctx);
        (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", __func__,
                                        &fd_scope);
        return -EINVAL;
    }
    uint64_t base = (uint64_t)offset;
    if ((uint64_t)want > UINT64_MAX - base) {
        read_batch_scratch_finish_job_locked(ctx);
        (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", __func__,
                                        &fd_scope);
        return -EFBIG;
    }
    uint64_t end = base + want;
    uint64_t first = base / PQC_LOGICAL_BLOCK_SIZE;
    uint64_t last = (end - 1) / PQC_LOGICAL_BLOCK_SIZE;
    uint64_t file_id = ctx->file_id;
    pqc_journal_lookup_view_t journal_view;
    uint64_t read_journal_cache_epoch = 0;
    int journal_cache_hit =
        pqc_fd_context_read_journal_cache_snapshot_locked(
            ctx, first, last, visible_generation, &journal_view,
            &read_journal_cache_epoch);
    pqc_epoch_log_lookup_view_t epoch_lookup_view;
    uint64_t read_epoch_cache_epoch = 0;
    int epoch_cache_hit =
        pqc_fd_context_read_epoch_cache_snapshot_locked(
            ctx, first, last, visible_generation, file_id,
            &epoch_lookup_view, &read_epoch_cache_epoch);
    size_t ss_len = ctx->ss_len;
    memcpy(ss, ctx->ss, ss_len);
    (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", __func__,
                                    &fd_scope);
    if (!journal_cache_hit)
        pqc_journal_lookup_view_init(&journal_view, first, last,
                                     visible_generation);
    if (!epoch_cache_hit)
        pqc_epoch_log_lookup_view_init(&epoch_lookup_view, first, last,
                                       visible_generation, file_id);

    int rc = 0;
    pqc_recovery_epoch_fallback_view_t epoch_view;
    pqc_recovery_epoch_fallback_view_init(&epoch_view);
    pqc_recovery_epoch_fallback_view_set_lookup(&epoch_view,
                                                &epoch_lookup_view);
    for (uint64_t block = first; block <= last; ) {
        uint64_t remaining_blocks = last - block + 1U;
        size_t batch_blocks = remaining_blocks > PQC_READ_BATCH_MAX_BLOCKS
            ? PQC_READ_BATCH_MAX_BLOCKS
            : (size_t)remaining_blocks;
        rc = read_authenticated_block_batch(
            ctx->state, journal_fd, data_fd, marker_path, &journal_view,
            &epoch_view,
            ss, ss_len, file_id, visible_generation, base, end, block,
            batch_blocks, read_cipher_batch, read_plain_batch,
            read_batch_capacity, buf);
        if (rc != 0)
            break;
        block += batch_blocks;
    }
    pqc_recovery_epoch_fallback_view_close(&epoch_view);
    OQS_MEM_cleanse(ss, sizeof(ss));
    (void)pqc_profiled_mutex_lock(&ctx->fd_lock, "fd_lock", __func__,
                                  &fd_scope);
    if (rc == 0 && !journal_cache_hit && ctx->valid &&
        ctx->journal_fd == journal_fd && ctx->file_id == file_id)
        pqc_fd_context_read_journal_cache_store_locked(
            ctx, &journal_view, read_journal_cache_epoch);
    if (rc == 0 && !epoch_cache_hit && ctx->valid &&
        ctx->file_id == file_id)
        pqc_fd_context_read_epoch_cache_store_locked(
            ctx, &epoch_lookup_view, read_epoch_cache_epoch);
    read_batch_scratch_finish_job_locked(ctx);
    (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", __func__,
                                    &fd_scope);
    pqc_journal_lookup_view_clear(&journal_view);
    pqc_epoch_log_lookup_view_clear(&epoch_lookup_view);
    return rc ? rc : (int)want;
}

int pqc_create(const char *path, mode_t mode,
               struct fuse_file_info *fi)
{
    if (pqc_is_hidden_sidecar_path(path))
        return -ENOENT;
    char phys_path[4096];
    pqc_storage_path_resolve(phys_path, sizeof(phys_path), path);

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
    if (pqc_keyring_metadata_store(phys_path, ss, sizeof(ss), fid) != 0 ||
        pqc_fd_context_set(fd, path, phys_path, ss, sizeof(ss), fid,
                            fi->flags) != 0) {
        OQS_MEM_cleanse(ss, sizeof(ss));
        close(fd);
        unlink(phys_path);
        return -EIO;
    }
    OQS_MEM_cleanse(ss, sizeof(ss));
    pqc_log("CREATE %s: authenticated file-key envelope initialized (fid=%llu)",
            path, (unsigned long long)fid);
    return 0;
}

int pqc_write(const char *path, const char *buf, size_t size,
              off_t offset, struct fuse_file_info *fi)
{
    (void)path;
    int fd  = (int)fi->fh;
    int idx = pqc_fd_context_index_for_fd(fd);
    pqc_fd_ctx_t *ctx = pqc_fd_context_at_index((size_t)idx);
    if (!ctx)
        return -EBADF;
    if (size == 0)
        return 0;

    pqc_lock_profile_scope_t fd_scope;
    (void)pqc_profiled_mutex_lock(&ctx->fd_lock, "fd_lock", __func__,
                                  &fd_scope);

    if (!ctx->valid) {
        (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", __func__,
                                        &fd_scope);
        int res = (int)pwrite(fd, buf, size, offset);
        return res == -1 ? -errno : (int)size;
    }
    if (!ctx->wbuf) {
        (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", __func__,
                                        &fd_scope);
        return -EBADF;
    }

    if (ctx->wbuf_used > 0 &&
        offset != ctx->wbuf_base_off + (off_t)ctx->wbuf_used) {
        int fr = drain_writeback_locked(fd, ctx, &fd_scope, __func__, 0,
                                        DRAIN_WAIT_NONE);
        if (fr < 0) {
            (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock",
                                            __func__, &fd_scope);
            return fr;
        }
    }

    int wr = write_buffered_locked(fd, ctx, &fd_scope, buf, size, offset,
                                   __func__);
    if (wr < 0) {
        (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", __func__,
                                        &fd_scope);
        return wr;
    }
    int rekey_due = pqc_rekey_write_trigger_enabled()
        ? rekey_due_locked(ctx)
        : 0;
    (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", __func__,
                                    &fd_scope);
    if (rekey_due) {
        pqc_rekey_enqueue_status_t rekey_status = pqc_rekey_queue_push(fd);
        log_rekey_enqueue_drop(rekey_status, fd);
    }
    return (int)size;
}

int pqc_fsync(const char *path, int datasync,
              struct fuse_file_info *fi)
{
    (void)path;
    (void)datasync;
    int fd  = (int)fi->fh;
    int idx = pqc_fd_context_index_for_fd(fd);
    pqc_fd_ctx_t *ctx = pqc_fd_context_at_index((size_t)idx);
    if (!ctx)
        return -EBADF;
    int res = 0;
    int need_final_sync = 1;
    int need_anchor_flush = 0;
    uint64_t fsync_epoch = 0;
    pqc_durability_site_t final_sync_site = PQC_DURABILITY_SITE_USER_FILE;

    pqc_lock_profile_scope_t fd_scope;
    (void)pqc_profiled_mutex_lock(&ctx->fd_lock, "fd_lock", __func__,
                                  &fd_scope);
    res = drain_writeback_locked(fd, ctx, &fd_scope, __func__, 1,
                                 DRAIN_WAIT_ALL);
    if (res == 0) {
        res = sync_dirty_sidecars_without_fd_lock(ctx, &fd_scope, __func__);
        if (res == 0) {
            int metadata_dirty =
                pqc_fd_context_fsync_dirty_epoch(ctx, &fsync_epoch);
            int metadata_repairable =
                pqc_fd_context_fsync_epoch_repairable(ctx, fsync_epoch);
            int dirty_sidecars_remaining = ctx->valid &&
                (ctx->data_sidecar_dirty || ctx->journal_sidecar_dirty ||
                 ctx->epoch_log_dirty);
            need_final_sync =
                !ctx->valid || dirty_sidecars_remaining ||
                (metadata_dirty && !metadata_repairable);
            final_sync_site =
                (ctx->valid && ctx->tier != PQC_TIER_NONE)
                    ? PQC_DURABILITY_SITE_MARKER_METADATA
                    : PQC_DURABILITY_SITE_USER_FILE;
            need_anchor_flush =
                metadata_dirty && ctx->valid && ctx->tier != PQC_TIER_NONE;
        }
    }
    (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", __func__,
                                    &fd_scope);
    if (res != 0)
        return res;
    if (!need_final_sync && !need_anchor_flush)
        return 0;

    if (need_anchor_flush && need_final_sync &&
        final_sync_site == PQC_DURABILITY_SITE_MARKER_METADATA) {
        if (pqc_anchor_worker_windowed_file_anchor_enabled())
            res = pqc_anchor_worker_flush_windowed_external_sync(
                marker_metadata_fsync, &fd);
        else
            res = pqc_anchor_worker_flush_now_external_sync(
                marker_metadata_fsync, &fd);
        if (res != 0)
            return -EIO;
        need_anchor_flush = 0;
        need_final_sync = 0;
    } else if (need_anchor_flush && pqc_anchor_worker_flush_now() != 0) {
        return -EIO;
    }

    pqc_fault_cutpoint("fsync_before_return");

    res = need_final_sync
        ? pqc_durability_fdatasync(fd, final_sync_site)
        : 0;
    if (res == 0) {
        (void)pqc_profiled_mutex_lock(&ctx->fd_lock, "fd_lock", __func__,
                                      &fd_scope);
        pqc_fd_context_mark_fsync_synced(ctx, fsync_epoch);
        (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", __func__,
                                        &fd_scope);
    }
    return res;
}

int pqc_flush(const char *path, struct fuse_file_info *fi)
{
    (void)path;
    if (!fi)
        return 0;
    int fd  = (int)fi->fh;
    int idx = pqc_fd_context_index_for_fd(fd);
    pqc_fd_ctx_t *ctx = pqc_fd_context_at_index((size_t)idx);
    if (!ctx)
        return -EBADF;
    int res = 0;

    pqc_lock_profile_scope_t fd_scope;
    (void)pqc_profiled_mutex_lock(&ctx->fd_lock, "fd_lock", __func__,
                                  &fd_scope);
    res = drain_writeback_locked(fd, ctx, &fd_scope, __func__, 0,
                                 DRAIN_WAIT_WRITEBACK);
    (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", __func__,
                                    &fd_scope);
    return res;
}

int pqc_truncate(const char *path, off_t size,
                 struct fuse_file_info *fi)
{
    if (pqc_is_hidden_sidecar_path(path))
        return -ENOENT;
    char phys_path[4096];
    pqc_storage_path_resolve(phys_path, sizeof(phys_path), path);
    if (size < 0)
        return -EINVAL;

    int fd = -1;
    if (fi)
        fd = (int)fi->fh;
    else
        return -ENOTSUP;
    if (fd == -1)
        return -errno;

    int idx = pqc_fd_context_index_for_fd(fd);
    pqc_fd_ctx_t *ctx = pqc_fd_context_at_index((size_t)idx);
    if (!ctx) {
        if (!fi)
            close(fd);
        return -EBADF;
    }
    int res = 0;
    pqc_lock_profile_scope_t fd_scope;
    (void)pqc_profiled_mutex_lock(&ctx->fd_lock, "fd_lock", __func__,
                                  &fd_scope);
    res = drain_writeback_locked(fd, ctx, &fd_scope, __func__, 0,
                                 DRAIN_WAIT_NONE);
    if (res == 0 && ctx->valid) {
        file_state_t *state = ctx->state;
        int journal_fd = ctx->journal_fd;
        char marker_path[sizeof(ctx->marker_path)];
        memcpy(marker_path, ctx->marker_path, sizeof(marker_path));
        pqc_fd_context_pending_job_begin(ctx);
        (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", __func__,
                                        &fd_scope);

        uint64_t publish_ticket = 0;
        int publish_turn_active = 0;
        res = metadata_publish_turn_begin(state, __func__, &publish_ticket,
                                          NULL, NULL);
        if (res == 0)
            publish_turn_active = 1;
        if (res == 0)
            res = pqc_publish_logical_size_store(marker_path, (uint64_t)size);
        if (res == 0 && ftruncate(fd, size) != 0)
            res = -errno;

        if (res == 0 && size == 0) {
            pqc_lock_profile_scope_t commit_scope;
            (void)pqc_profiled_mutex_lock(&state->commit_lock,
                                          "commit_lock", __func__,
                                          &commit_scope);
            state->logical_size = 0;
            state->logical_size_valid = 1;
            (void)pqc_profiled_mutex_unlock(&state->commit_lock,
                                            "commit_lock", __func__,
                                            &commit_scope);
            if (journal_fd >= 0 && ftruncate(journal_fd, 0) != 0)
                res = -errno;
        }

        if (res == 0) {
            pqc_lock_profile_scope_t commit_scope;
            (void)pqc_profiled_mutex_lock(&state->commit_lock,
                                          "commit_lock", __func__,
                                          &commit_scope);
            uint64_t previous_size = state->logical_size_valid
                ? state->logical_size
                : ctx->logical_size;
            state->logical_size = (uint64_t)size;
            state->logical_size_valid = 1;
            if (size == 0) {
                pqc_file_state_mapping_cache_mark_complete_empty_locked(
                    state);
            } else if ((uint64_t)size < previous_size) {
                pqc_file_state_mapping_cache_mark_unknown_locked(state);
            }
            if (publish_turn_active)
                metadata_publish_turn_finish_locked(state, publish_ticket);
            publish_turn_active = 0;
            (void)pqc_profiled_mutex_unlock(&state->commit_lock,
                                            "commit_lock", __func__,
                                            &commit_scope);
        }
        if (publish_turn_active) {
            metadata_publish_turn_finish(state, __func__, publish_ticket);
            publish_turn_active = 0;
        }
        (void)pqc_profiled_mutex_lock(&ctx->fd_lock, "fd_lock", __func__,
                                      &fd_scope);
        if (res == 0 && ctx->valid && ctx->state == state) {
            ctx->logical_size = (uint64_t)size;
            pqc_fd_context_mark_fsync_dirty(ctx);
        }
        pqc_fd_context_pending_job_end(ctx);
    } else if (res == 0) {
        if (ftruncate(fd, size) != 0)
            res = -errno;
    }
    (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", __func__,
                                    &fd_scope);
    if (!fi)
        close(fd);
    return res;
}

int pqc_fallocate(const char *path, int mode, off_t offset,
                  off_t length, struct fuse_file_info *fi)
{
    if (pqc_is_hidden_sidecar_path(path))
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
    int idx = pqc_fd_context_index_for_fd(fd);
    pqc_fd_ctx_t *ctx = pqc_fd_context_at_index((size_t)idx);
    if (!ctx)
        return -EBADF;
    int res = 0;

    pqc_lock_profile_scope_t fd_scope;
    (void)pqc_profiled_mutex_lock(&ctx->fd_lock, "fd_lock", __func__,
                                  &fd_scope);
    res = drain_writeback_locked(fd, ctx, &fd_scope, __func__, 0,
                                 DRAIN_WAIT_NONE);

    if (res == 0 && ctx->valid) {
        uint64_t end = (uint64_t)offset + (uint64_t)length;
        file_state_t *state = ctx->state;
        uint64_t file_id = ctx->file_id;
        char marker_path[sizeof(ctx->marker_path)];
        memcpy(marker_path, ctx->marker_path, sizeof(marker_path));
        pqc_fd_context_pending_job_begin(ctx);
        (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", __func__,
                                        &fd_scope);

        uint64_t publish_ticket = 0;
        uint64_t next_generation = 0;
        uint64_t logical_size = 0;
        int publish_turn_active = 0;
        res = metadata_publish_turn_begin(state, __func__, &publish_ticket,
                                          &next_generation, &logical_size);
        if (res == 0)
            publish_turn_active = 1;
#ifdef FALLOC_FL_KEEP_SIZE
        if (res == 0 && (mode & FALLOC_FL_KEEP_SIZE) == 0 &&
            end > logical_size)
            logical_size = end;
#else
        if (res == 0 && end > logical_size)
            logical_size = end;
#endif
        if (res == 0)
            res = pqc_publish_logical_size_store(marker_path, logical_size);
        if (res == 0 && ftruncate(fd, (off_t)logical_size) != 0)
            res = -errno;
        if (res == 0)
            res = pqc_checkpoint_store_and_stage_anchor(
                marker_path, file_id, next_generation, logical_size,
                next_generation);

        if (res == 0) {
            pqc_lock_profile_scope_t commit_scope;
            (void)pqc_profiled_mutex_lock(&state->commit_lock, "commit_lock",
                                          __func__, &commit_scope);
            state->logical_size = logical_size;
            state->logical_size_valid = 1;
            if (publish_turn_active)
                metadata_publish_turn_finish_locked(state, publish_ticket);
            publish_turn_active = 0;
            (void)pqc_profiled_mutex_unlock(&state->commit_lock,
                                            "commit_lock", __func__,
                                            &commit_scope);
        }
        if (publish_turn_active) {
            metadata_publish_turn_finish(state, __func__, publish_ticket);
            publish_turn_active = 0;
        }
        (void)pqc_profiled_mutex_lock(&ctx->fd_lock, "fd_lock", __func__,
                                      &fd_scope);
        if (res == 0 && ctx->valid && ctx->state == state) {
            ctx->logical_size = logical_size;
            pqc_fd_context_mark_fsync_dirty(ctx);
        }
        pqc_fd_context_pending_job_end(ctx);
    } else if (res == 0) {
        off_t end = offset + length;
#ifdef FALLOC_FL_KEEP_SIZE
        if ((mode & FALLOC_FL_KEEP_SIZE) != 0)
            end = 0;
#endif
        if (end > 0 && ftruncate(fd, end) != 0)
            res = -errno;
    }
    (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", __func__,
                                    &fd_scope);
    return res;
}

int pqc_release(const char *path, struct fuse_file_info *fi)
{
    (void)path;
    int fd  = (int)fi->fh;
    int idx = pqc_fd_context_index_for_fd(fd);
    pqc_fd_ctx_t *ctx = pqc_fd_context_at_index((size_t)idx);
    if (!ctx) {
        close(fd);
        return -EBADF;
    }
    int res = 0;
    char marker_path[4096];
    marker_path[0] = '\0';

    pqc_lock_profile_scope_t fd_scope;
    (void)pqc_profiled_mutex_lock(&ctx->fd_lock, "fd_lock", __func__,
                                  &fd_scope);
    memcpy(marker_path, ctx->marker_path, sizeof(marker_path));
    res = drain_writeback_locked(fd, ctx, &fd_scope, __func__, 0,
                                 DRAIN_WAIT_ALL);
    (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", __func__,
                                    &fd_scope);

    pqc_fd_context_clear(fd);
    close(fd);
    cleanup_hidden_marker_and_sidecars(marker_path);
    return res;
}
