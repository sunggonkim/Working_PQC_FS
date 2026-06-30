#include "pqc_writeback.h"

#include "pqc_block_job.h"
#include "pqc_checkpoint.h"
#include "pqc_flush_batch.h"
#include "pqc_flush_crypto.h"
#include "pqc_format.h"
#include "pqc_lock_profile.h"
#include "pqc_metrics.h"
#include "pqc_epoch_publish.h"
#include "pqc_parallel_commit.h"
#include "pqc_publish.h"
#include "pqc_qos.h"
#include "pqc_scheduler.h"
#include "pqc_strict_publish.h"

#include <errno.h>
#include <openssl/crypto.h>
#include <pthread.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    int storage_fd;
    int data_fd;
    int journal_fd;
    int epoch_log_fd;
    int epoch_log_syncfs_domain_known;
    int epoch_log_same_syncfs_domain;
    int tier;
    int qos_class;
    file_state_t *state;
    uint64_t file_id;
    uint8_t ss[64];
    size_t ss_len;
    char marker_path[4096];
    char epoch_log_path[4096 + 16];
    uint64_t logical_size_cache;
    uint64_t base;
    size_t write_size;
    uint8_t *buf;
    size_t buf_capacity;
    int snapshot_buf_acquired;
    uint8_t *flush_plain_batch;
    uint8_t *flush_cipher_batch;
    size_t flush_batch_capacity;
    int flush_scratch_acquired;
} pqc_writeback_snapshot_t;

static size_t writeback_block_count(uint64_t base, size_t write_size)
{
    uint64_t end = base + write_size;
    uint64_t first = base / PQC_LOGICAL_BLOCK_SIZE;
    uint64_t last = (end - 1) / PQC_LOGICAL_BLOCK_SIZE;
    return (size_t)(last - first + 1);
}

static void publish_turn_finish_locked(file_state_t *state, uint64_t ticket)
{
    if (!state)
        return;
    if (state->publish_ticket == ticket) {
        ++state->publish_ticket;
        pthread_cond_broadcast(&state->publish_cv);
    }
}

static int writeback_use_outer_parallel_commit(void)
{
    pqc_publication_mode_t mode = PQC_PUBLICATION_MODE_STRICT;
    if (pqc_publication_mode_from_config(&mode) != 0)
        return 1;
    return mode != PQC_PUBLICATION_MODE_EPOCH_REDO_LOG;
}

static void writeback_snapshot_cleanup(pqc_writeback_snapshot_t *snapshot)
{
    if (!snapshot)
        return;
    if (snapshot->buf) {
        OPENSSL_cleanse(snapshot->buf, snapshot->write_size);
        snapshot->buf = NULL;
    }
    snapshot->buf_capacity = 0;
    OPENSSL_cleanse(snapshot->ss, sizeof(snapshot->ss));
}

static int writeback_snapshot_acquire_buffer_locked(
    pqc_fd_ctx_t *ctx,
    pqc_writeback_snapshot_t *snapshot,
    pqc_lock_profile_scope_t *fd_scope)
{
    if (!ctx || !snapshot)
        return -EINVAL;
    while (ctx->snapshot_buf_in_use)
        (void)pqc_profiled_cond_wait(&ctx->pending_cv, &ctx->fd_lock,
                                     "fd_lock", __func__, fd_scope);
    if (!ctx->snapshot_buf || ctx->snapshot_buf_capacity < snapshot->write_size)
        return -ENOMEM;
    ctx->snapshot_buf_in_use = 1;
    snapshot->buf = ctx->snapshot_buf;
    snapshot->buf_capacity = ctx->snapshot_buf_capacity;
    snapshot->snapshot_buf_acquired = 1;
    return 0;
}

static void writeback_snapshot_release_buffer_locked(
    pqc_fd_ctx_t *ctx,
    pqc_writeback_snapshot_t *snapshot)
{
    if (!ctx || !snapshot || !snapshot->snapshot_buf_acquired)
        return;
    ctx->snapshot_buf_in_use = 0;
    snapshot->snapshot_buf_acquired = 0;
    pthread_cond_broadcast(&ctx->pending_cv);
}

static int writeback_snapshot_acquire_flush_scratch_locked(
    pqc_fd_ctx_t *ctx,
    pqc_writeback_snapshot_t *snapshot,
    pqc_lock_profile_scope_t *fd_scope)
{
    if (!ctx || !snapshot)
        return -EINVAL;
    while (ctx->flush_scratch_in_use)
        (void)pqc_profiled_cond_wait(&ctx->pending_cv, &ctx->fd_lock,
                                     "fd_lock", __func__, fd_scope);
    if (!ctx->flush_plain_batch || !ctx->flush_cipher_batch ||
        ctx->flush_batch_capacity < PQC_WRITEBACK_BATCH_SCRATCH_SIZE)
        return -ENOMEM;
    ctx->flush_scratch_in_use = 1;
    snapshot->flush_plain_batch = ctx->flush_plain_batch;
    snapshot->flush_cipher_batch = ctx->flush_cipher_batch;
    snapshot->flush_batch_capacity = ctx->flush_batch_capacity;
    snapshot->flush_scratch_acquired = 1;
    return 0;
}

static void writeback_snapshot_release_flush_scratch_locked(
    pqc_fd_ctx_t *ctx,
    const pqc_writeback_snapshot_t *snapshot)
{
    if (!ctx || !snapshot || !snapshot->flush_scratch_acquired)
        return;
    if (ctx->flush_plain_batch == snapshot->flush_plain_batch &&
        ctx->flush_cipher_batch == snapshot->flush_cipher_batch)
        ctx->flush_scratch_in_use = 0;
    pthread_cond_broadcast(&ctx->pending_cv);
}

static int writeback_snapshot_take_locked(int storage_fd,
                                          pqc_fd_ctx_t *ctx,
                                          pqc_writeback_snapshot_t *snapshot,
                                          pqc_lock_profile_scope_t *fd_scope)
{
    if (!ctx || !snapshot)
        return -EINVAL;
    if (ctx->wbuf_used == 0)
        return 0;
    if (!ctx->state)
        return -EIO;

    memset(snapshot, 0, sizeof(*snapshot));
    snapshot->storage_fd = storage_fd;
    snapshot->data_fd = ctx->data_fd;
    snapshot->journal_fd = ctx->journal_fd;
    snapshot->epoch_log_fd = ctx->epoch_log_fd;
    snapshot->epoch_log_syncfs_domain_known =
        ctx->epoch_log_syncfs_domain_known;
    snapshot->epoch_log_same_syncfs_domain =
        ctx->epoch_log_same_syncfs_domain;
    snapshot->tier = ctx->tier;
    snapshot->qos_class = ctx->qos_class;
    snapshot->state = ctx->state;
    snapshot->file_id = ctx->file_id;
    if (ctx->ss_len > sizeof(snapshot->ss))
        return -EINVAL;
    snapshot->ss_len = ctx->ss_len;
    memcpy(snapshot->ss, ctx->ss, ctx->ss_len);
    memcpy(snapshot->marker_path, ctx->marker_path,
           sizeof(snapshot->marker_path) - 1);
    memcpy(snapshot->epoch_log_path, ctx->epoch_log_path,
           sizeof(snapshot->epoch_log_path) - 1);
    snapshot->logical_size_cache = ctx->logical_size;
    snapshot->base = (uint64_t)ctx->wbuf_base_off;
    snapshot->write_size = ctx->wbuf_used;
    int buf_rc = writeback_snapshot_acquire_buffer_locked(ctx, snapshot,
                                                          fd_scope);
    if (buf_rc != 0)
        return buf_rc;
    memcpy(snapshot->buf, ctx->wbuf, snapshot->write_size);
    if (ctx->tier != PQC_TIER_NONE) {
        int scratch_rc = writeback_snapshot_acquire_flush_scratch_locked(
            ctx, snapshot, fd_scope);
        if (scratch_rc != 0) {
            writeback_snapshot_cleanup(snapshot);
            writeback_snapshot_release_buffer_locked(ctx, snapshot);
            return scratch_rc;
        }
    }
    ctx->wbuf_used = 0;
    pqc_fd_context_writeback_job_begin(ctx);
    return 1;
}

static int writeback_flush_plain_snapshot(pqc_writeback_snapshot_t *snapshot)
{
    uint64_t bytes = (uint64_t)snapshot->write_size;
    if (snapshot->base > UINT64_MAX - bytes)
        return -EFBIG;

    uint64_t final_size = snapshot->base + bytes;
    uint64_t next_generation = 0;
    uint64_t publish_ticket = 0;
    int res = 0;

    pqc_lock_profile_scope_t commit_scope;
    (void)pqc_profiled_mutex_lock(&snapshot->state->commit_lock,
                                  "commit_lock", __func__, &commit_scope);
    publish_ticket = snapshot->state->next_publish_ticket++;
    while (snapshot->state->publish_ticket != publish_ticket)
        (void)pqc_profiled_cond_wait(&snapshot->state->publish_cv,
                                     &snapshot->state->commit_lock,
                                     "commit_lock", __func__,
                                     &commit_scope);
    next_generation = snapshot->state->next_generation;
    uint64_t current_size = snapshot->state->logical_size_valid
        ? snapshot->state->logical_size
        : snapshot->logical_size_cache;
    if (final_size < current_size)
        final_size = current_size;
    (void)pqc_profiled_mutex_unlock(&snapshot->state->commit_lock,
                                    "commit_lock", __func__, &commit_scope);

    pqc_block_job_t job;
    pqc_scheduler_data_job_input_t sched_input = {
        .file_id = snapshot->file_id,
        .next_generation = next_generation,
        .logical_offset = snapshot->base,
        .length = (uint32_t)bytes,
        .gpu_load_ewma = pqc_qos_gpu_load_ewma_read(),
    };
    pqc_scheduler_schedule_data_job(&job, &sched_input);

    ssize_t written = pwrite(snapshot->storage_fd, snapshot->buf,
                             snapshot->write_size, (off_t)snapshot->base);
    if (written < 0) {
        res = -errno;
        goto finish_publish_turn;
    }
    if ((size_t)written != snapshot->write_size) {
        res = -EIO;
        goto finish_publish_turn;
    }

    if (final_size != current_size)
        res = pqc_publish_logical_size_store(snapshot->marker_path,
                                             final_size);
    if (res == 0 && final_size != current_size &&
        ftruncate(snapshot->storage_fd, (off_t)final_size) != 0)
        res = -errno;

finish_publish_turn:
    (void)pqc_profiled_mutex_lock(&snapshot->state->commit_lock,
                                  "commit_lock", __func__, &commit_scope);
    if (res == 0) {
        snapshot->state->logical_size = final_size;
        snapshot->state->logical_size_valid = 1;
    }
    publish_turn_finish_locked(snapshot->state, publish_ticket);
    (void)pqc_profiled_mutex_unlock(&snapshot->state->commit_lock,
                                    "commit_lock", __func__, &commit_scope);
    if (res == 0)
        pqc_scheduler_record_data_bytes(bytes, 0, 0);
    return res;
}

static int writeback_flush_authenticated_snapshot(
    pqc_fd_ctx_t *ctx,
    pqc_writeback_snapshot_t *snapshot,
    int require_durable)
{
    uint64_t base = snapshot->base;
    size_t write_size = snapshot->write_size;
    if (base > UINT64_MAX - (uint64_t)write_size)
        return -EFBIG;
    size_t block_count = writeback_block_count(base, write_size);
    uint64_t final_size = 0;
    uint64_t current_logical_size = 0;
    uint64_t first_generation = 0;
    uint64_t reserved_generation = 0;
    uint64_t publish_ticket = 0;
    int publish_turn_active = 0;
    pqc_flush_batch_t batch;
    pqc_flush_batch_init(&batch);
    pqc_parallel_commit_ticket_t parallel_ticket;
    memset(&parallel_ticket, 0, sizeof(parallel_ticket));
    int parallel_commit_active = 0;
    int parallel_commit_finished = 0;
    int res = 0;
    int gpu_admitted = 0;

    pqc_lock_profile_scope_t commit_scope;
    (void)pqc_profiled_mutex_lock(&snapshot->state->commit_lock, "commit_lock",
                                  __func__, &commit_scope);
    publish_ticket = snapshot->state->next_publish_ticket++;
    while (snapshot->state->publish_ticket != publish_ticket)
        (void)pqc_profiled_cond_wait(&snapshot->state->publish_cv,
                                     &snapshot->state->commit_lock,
                                     "commit_lock", __func__,
                                     &commit_scope);
    publish_turn_active = 1;
    current_logical_size = snapshot->state->logical_size_valid
        ? snapshot->state->logical_size
        : snapshot->logical_size_cache;
    final_size = base + (uint64_t)write_size;
    if (final_size < current_logical_size)
        final_size = current_logical_size;
    if (snapshot->state->next_generation >
        UINT64_MAX - (uint64_t)block_count) {
        publish_turn_finish_locked(snapshot->state, publish_ticket);
        (void)pqc_profiled_mutex_unlock(&snapshot->state->commit_lock,
                                        "commit_lock", __func__,
                                        &commit_scope);
        return -EFBIG;
    }
    first_generation = snapshot->state->next_generation + 1;
    reserved_generation = snapshot->state->next_generation + (uint64_t)block_count;
    (void)pqc_profiled_mutex_unlock(&snapshot->state->commit_lock, "commit_lock",
                                    __func__, &commit_scope);

    res = pqc_checkpoint_reserve_generation(snapshot->marker_path,
                                            snapshot->file_id,
                                            current_logical_size,
                                            reserved_generation);
    (void)pqc_profiled_mutex_lock(&snapshot->state->commit_lock, "commit_lock",
                                  __func__, &commit_scope);
    if (res == 0) {
        if (snapshot->state->next_generation < reserved_generation)
            snapshot->state->next_generation = reserved_generation;
        snapshot->state->logical_size = current_logical_size;
        snapshot->state->logical_size_valid = 1;
    } else {
        publish_turn_finish_locked(snapshot->state, publish_ticket);
        publish_turn_active = 0;
    }
    (void)pqc_profiled_mutex_unlock(&snapshot->state->commit_lock, "commit_lock",
                                    __func__, &commit_scope);
    if (res != 0)
        goto out_batch;

    const char *epoch_fallback_marker_path =
        (snapshot->epoch_log_fd >= 0 || snapshot->epoch_log_path[0] != '\0')
            ? snapshot->marker_path
            : NULL;
    res = pqc_flush_batch_prepare(&batch, snapshot->journal_fd,
                                  snapshot->data_fd,
                                  epoch_fallback_marker_path,
                                  snapshot->ss, snapshot->ss_len,
                                  snapshot->file_id,
                                  first_generation,
                                  current_logical_size, base, write_size,
                                  snapshot->buf,
                                  snapshot->flush_plain_batch,
                                  snapshot->flush_cipher_batch,
                                  snapshot->flush_batch_capacity);
    final_size = batch.final_size;
    if (res != 0)
        goto out_batch;

    pqc_block_job_t job;
    pqc_scheduler_data_job_input_t sched_input = {
        .file_id = snapshot->file_id,
        .next_generation = reserved_generation,
        .logical_offset = base,
        .length = (uint32_t)batch.write_size,
        .gpu_load_ewma = pqc_qos_gpu_load_ewma_read(),
    };
    pqc_scheduler_schedule_data_job(&job, &sched_input);
    int use_gpu_batch = (job.target == PQC_JOB_GPU && batch.block_count > 1);
    uint32_t algorithm_id = PQC_ALGO_AES_256_GCM;
    if (use_gpu_batch)
        pqc_scheduler_gpu_admit((uint32_t)(final_size - base));
    gpu_admitted = use_gpu_batch;
    res = pqc_flush_crypto_encrypt(snapshot->ss, snapshot->ss_len,
                                   snapshot->file_id, &batch, use_gpu_batch,
                                   NULL);
    if (res != 0)
        goto out_batch;

    uint64_t published_logical_size = current_logical_size;
    uint64_t data_sidecar_append_offset = 0;
    int data_sidecar_append_offset_valid = 0;
    uint64_t data_sidecar_end_after_append = 0;
    int data_sidecar_end_after_append_valid = 0;
    uint64_t journal_sidecar_append_offset = 0;
    int journal_sidecar_append_offset_valid = 0;
    uint64_t journal_sidecar_end_after_append = 0;
    int journal_sidecar_end_after_append_valid = 0;
    (void)pqc_profiled_mutex_lock(&snapshot->state->commit_lock,
                                  "commit_lock", __func__, &commit_scope);
    if (snapshot->state->data_sidecar_end_valid) {
        data_sidecar_append_offset = snapshot->state->data_sidecar_end;
        data_sidecar_append_offset_valid = 1;
    }
    if (snapshot->state->journal_sidecar_end_valid) {
        journal_sidecar_append_offset =
            snapshot->state->journal_sidecar_end;
        journal_sidecar_append_offset_valid = 1;
    }
    (void)pqc_profiled_mutex_unlock(&snapshot->state->commit_lock,
                                    "commit_lock", __func__, &commit_scope);
    pqc_strict_publish_request_t publish_req = {
        .data_fd = snapshot->data_fd,
        .journal_fd = snapshot->journal_fd,
        .epoch_log_fd = snapshot->epoch_log_fd,
        .epoch_log_syncfs_domain_known =
            snapshot->epoch_log_syncfs_domain_known,
        .epoch_log_same_syncfs_domain =
            snapshot->epoch_log_same_syncfs_domain,
        .storage_fd = snapshot->storage_fd,
        .marker_path = snapshot->marker_path,
        .epoch_log_path = snapshot->epoch_log_path,
        .file_id = snapshot->file_id,
        .final_size = final_size,
        .reserved_generation = reserved_generation,
        .data_sidecar_append_offset = data_sidecar_append_offset,
        .data_sidecar_append_offset_valid =
            data_sidecar_append_offset_valid,
        .data_sidecar_end_after_append = &data_sidecar_end_after_append,
        .data_sidecar_end_after_append_valid =
            &data_sidecar_end_after_append_valid,
        .journal_sidecar_append_offset = journal_sidecar_append_offset,
        .journal_sidecar_append_offset_valid =
            journal_sidecar_append_offset_valid,
        .journal_sidecar_end_after_append =
            &journal_sidecar_end_after_append,
        .journal_sidecar_end_after_append_valid =
            &journal_sidecar_end_after_append_valid,
        .algorithm_id = (uint32_t)algorithm_id,
        .blocks = batch.blocks,
        .block_count = batch.block_count,
        .cipher_batch = batch.cipher_batch,
        .data_sidecar_dirty = &ctx->data_sidecar_dirty,
        .journal_sidecar_dirty = &ctx->journal_sidecar_dirty,
        .epoch_log_dirty = &ctx->epoch_log_dirty,
        .data_sidecar_dirty_epoch = &ctx->data_sidecar_dirty_epoch,
        .journal_sidecar_dirty_epoch = &ctx->journal_sidecar_dirty_epoch,
        .epoch_log_dirty_epoch = &ctx->epoch_log_dirty_epoch,
        .journal_sidecar_epoch_repairable =
            &ctx->journal_sidecar_epoch_repairable,
        .journal_sidecar_epoch_repairable_epoch =
            &ctx->journal_sidecar_epoch_repairable_epoch,
        .logical_size = &published_logical_size,
        .skip_data_fsync = !require_durable,
        .skip_journal_fsync = !require_durable,
        .defer_data_fsync = !require_durable,
        .defer_journal_fsync = !require_durable,
    };

    if (pqc_parallel_commit_runtime_enabled() &&
        writeback_use_outer_parallel_commit()) {
        int parallel_rc = pqc_parallel_commit_runtime_begin(
            snapshot->file_id, (uint64_t)(final_size - base),
            &parallel_ticket);
        if (parallel_ticket.role != PQC_PARALLEL_COMMIT_ROLE_INVALID)
            parallel_commit_active = 1;
        if (parallel_rc != 0 &&
            parallel_ticket.role != PQC_PARALLEL_COMMIT_ROLE_FOLLOWER) {
            res = parallel_rc;
            goto out_batch;
        }
    }

    res = pqc_publication_dispatch_commit(&publish_req);
    (void)pqc_profiled_mutex_lock(&snapshot->state->commit_lock, "commit_lock",
                                  __func__, &commit_scope);
    if (data_sidecar_end_after_append_valid) {
        snapshot->state->data_sidecar_end = data_sidecar_end_after_append;
        snapshot->state->data_sidecar_end_valid = 1;
    } else if (res != 0) {
        snapshot->state->data_sidecar_end_valid = 0;
    }
    if (journal_sidecar_end_after_append_valid) {
        snapshot->state->journal_sidecar_end =
            journal_sidecar_end_after_append;
        snapshot->state->journal_sidecar_end_valid = 1;
    } else if (res != 0) {
        snapshot->state->journal_sidecar_end_valid = 0;
    }
    if (res == 0) {
        block_mapping_t live_mappings[PQC_WRITEBACK_MAX_BLOCKS];
        for (size_t bi = 0; bi < batch.block_count; ++bi) {
            const pqc_crypto_block_desc_t *block = &batch.blocks[bi];
            live_mappings[bi] = (block_mapping_t) {
                .logical_block = block->block,
                .generation = block->generation,
                .ciphertext_offset = block->ciphertext_offset,
                .plaintext_length = block->length,
                .algorithm_id = algorithm_id,
            };
            memcpy(live_mappings[bi].tag, block->tag,
                   sizeof(live_mappings[bi].tag));
        }
        (void)pqc_file_state_mapping_cache_store_locked(
            snapshot->state, live_mappings, batch.block_count);
        snapshot->state->logical_size = published_logical_size;
        snapshot->state->logical_size_valid = 1;
        if (snapshot->state->committed_generation < reserved_generation)
            snapshot->state->committed_generation = reserved_generation;
    }
    publish_turn_finish_locked(snapshot->state, publish_ticket);
    publish_turn_active = 0;
    (void)pqc_profiled_mutex_unlock(&snapshot->state->commit_lock, "commit_lock",
                                    __func__, &commit_scope);
    if (parallel_commit_active &&
        parallel_ticket.role == PQC_PARALLEL_COMMIT_ROLE_LEADER) {
        int finish_rc = pqc_parallel_commit_runtime_finish(&parallel_ticket,
                                                           res);
        parallel_commit_finished = 1;
        if (res == 0 && finish_rc != 0)
            res = finish_rc;
    }
out_batch:
    if (parallel_commit_active && !parallel_commit_finished &&
        parallel_ticket.role == PQC_PARALLEL_COMMIT_ROLE_LEADER)
        (void)pqc_parallel_commit_runtime_finish(&parallel_ticket, res);
    if (publish_turn_active) {
        (void)pqc_profiled_mutex_lock(&snapshot->state->commit_lock, "commit_lock",
                                      __func__, &commit_scope);
        publish_turn_finish_locked(snapshot->state, publish_ticket);
        (void)pqc_profiled_mutex_unlock(&snapshot->state->commit_lock,
                                        "commit_lock", __func__,
                                        &commit_scope);
    }
    if (gpu_admitted)
        pqc_scheduler_gpu_release((uint32_t)(final_size - base));
    pqc_flush_batch_cleanup(&batch);
    if (res != 0)
        pqc_log("authenticated flush failed: %s", strerror(-res));
    if (res == 0) {
        pqc_scheduler_record_data_bytes(0,
                                        (uint64_t)(final_size - base),
                                        (uint64_t)(final_size - base) * 64ULL);
    }
    return res;
}

int pqc_writeback_flush_locked(int storage_fd,
                               pqc_fd_ctx_t *ctx,
                               pqc_lock_profile_scope_t *fd_scope,
                               const char *fd_site,
                               int require_durable)
{
    pqc_writeback_snapshot_t snapshot;
    int snap_rc = writeback_snapshot_take_locked(storage_fd, ctx, &snapshot,
                                                 fd_scope);
    if (snap_rc <= 0)
        return snap_rc;

    (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock",
                                    fd_site, fd_scope);
    pqc_qos_apply_runtime_throttle(snapshot.write_size, snapshot.qos_class);
    int res = snapshot.tier == PQC_TIER_NONE
        ? writeback_flush_plain_snapshot(&snapshot)
        : writeback_flush_authenticated_snapshot(ctx, &snapshot,
                                                 require_durable);
    (void)pqc_profiled_mutex_lock(&ctx->fd_lock, "fd_lock",
                                  fd_site, fd_scope);
    writeback_snapshot_cleanup(&snapshot);
    writeback_snapshot_release_flush_scratch_locked(ctx, &snapshot);
    writeback_snapshot_release_buffer_locked(ctx, &snapshot);

    if (res == 0 && ctx->valid && ctx->state == snapshot.state) {
        pqc_fd_context_mark_fsync_dirty(ctx);
        if (ctx->journal_sidecar_epoch_repairable)
            pqc_fd_context_mark_fsync_epoch_repairable(ctx);
        pqc_lock_profile_scope_t commit_scope;
        (void)pqc_profiled_mutex_lock(&snapshot.state->commit_lock,
                                      "commit_lock", __func__,
                                      &commit_scope);
        if (snapshot.state->logical_size_valid)
            ctx->logical_size = snapshot.state->logical_size;
        (void)pqc_profiled_mutex_unlock(&snapshot.state->commit_lock,
                                        "commit_lock", __func__,
                                        &commit_scope);
    }
    pqc_fd_context_writeback_job_end(ctx);
    return res;
}
