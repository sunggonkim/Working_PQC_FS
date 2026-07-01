#include "pqc_fd_context.h"

#include "pqc_checkpoint.h"
#include "pqc_durability.h"
#include "pqc_epoch_log.h"
#include "pqc_epoch_publish.h"
#include "pqc_format.h"
#include "pqc_journal.h"
#include "pqc_keyring.h"
#include "pqc_lock_profile.h"
#include "pqc_posix.h"
#include "pqc_publish.h"
#include "pqc_rekey.h"
#include "pqc_test_hooks.h"

#include <errno.h>
#include <fcntl.h>
#include <oqs/oqs.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static pqc_fd_ctx_t g_fd_ctx[PQC_MAX_FD];

static int fd_context_logical_path_equal(const char *a, const char *b)
{
    if (!a || !b)
        return 0;
    if (strncmp(a, b, 4096) == 0)
        return 1;
    if (a[0] == '/' && strncmp(a + 1, b, 4095) == 0)
        return 1;
    if (b[0] == '/' && strncmp(a, b + 1, 4095) == 0)
        return 1;
    return 0;
}

static void fd_context_copy_cstr(char *dst, size_t dst_size, const char *src)
{
    if (!dst || dst_size == 0)
        return;
    if (!src) {
        dst[0] = '\0';
        return;
    }
    size_t len = strnlen(src, dst_size - 1);
    memcpy(dst, src, len);
    dst[len] = '\0';
}

typedef struct {
    int           data_fd;
    int           journal_fd;
    int           epoch_log_fd;
    uint8_t      *wbuf;
    uint8_t      *snapshot_buf;
    size_t        snapshot_buf_capacity;
    uint8_t      *flush_plain_batch;
    uint8_t      *flush_cipher_batch;
    size_t        flush_batch_capacity;
    uint8_t      *read_plain_batch;
    uint8_t      *read_cipher_batch;
    size_t        read_batch_capacity;
    pqc_journal_lookup_view_t *read_journal_cache;
    pqc_epoch_log_lookup_view_t *read_epoch_cache;
    file_state_t *state;
} pqc_fd_retired_resources_t;

/*
 * Mounted write/read scratch is CPU-owned foreground memory.  The CUDA
 * executor owns its own managed buffers; keeping FUSE scratch ordinary memory
 * avoids UVM placement work on the AES-GCM data path.
 */
static uint8_t *fd_lifecycle_scratch_alloc(size_t capacity)
{
    if (capacity == 0)
        return NULL;
    return (uint8_t *)malloc(capacity);
}

static void fd_lifecycle_scratch_free(uint8_t **buf, size_t capacity)
{
    if (!buf || !*buf)
        return;
    OQS_MEM_cleanse(*buf, capacity);
    free(*buf);
    *buf = NULL;
}

static pqc_journal_lookup_view_t *fd_lifecycle_read_cache_alloc(void)
{
    pqc_journal_lookup_view_t *cache =
        (pqc_journal_lookup_view_t *)calloc(1, sizeof(*cache));
    return cache;
}

static pqc_epoch_log_lookup_view_t *fd_lifecycle_epoch_cache_alloc(void)
{
    pqc_epoch_log_lookup_view_t *cache =
        (pqc_epoch_log_lookup_view_t *)calloc(1, sizeof(*cache));
    return cache;
}

static void fd_lifecycle_read_cache_free(pqc_journal_lookup_view_t **cache)
{
    if (!cache || !*cache)
        return;
    pqc_journal_lookup_view_clear(*cache);
    free(*cache);
    *cache = NULL;
}

static void fd_lifecycle_epoch_cache_free(pqc_epoch_log_lookup_view_t **cache)
{
    if (!cache || !*cache)
        return;
    pqc_epoch_log_lookup_view_clear(*cache);
    free(*cache);
    *cache = NULL;
}

static void retired_resources_init(pqc_fd_retired_resources_t *retired)
{
    if (!retired)
        return;
    retired->data_fd = -1;
    retired->journal_fd = -1;
    retired->epoch_log_fd = -1;
    retired->wbuf = NULL;
    retired->snapshot_buf = NULL;
    retired->snapshot_buf_capacity = 0;
    retired->flush_plain_batch = NULL;
    retired->flush_cipher_batch = NULL;
    retired->flush_batch_capacity = 0;
    retired->read_plain_batch = NULL;
    retired->read_cipher_batch = NULL;
    retired->read_batch_capacity = 0;
    retired->read_journal_cache = NULL;
    retired->read_epoch_cache = NULL;
    retired->state = NULL;
}

static void retired_resources_release(pqc_fd_retired_resources_t *retired)
{
    if (!retired)
        return;
    fd_lifecycle_scratch_free(&retired->wbuf, COALESCE_SIZE);
    fd_lifecycle_scratch_free(&retired->snapshot_buf,
                              retired->snapshot_buf_capacity);
    retired->snapshot_buf_capacity = 0;
    fd_lifecycle_scratch_free(&retired->flush_cipher_batch,
                              retired->flush_batch_capacity);
    fd_lifecycle_scratch_free(&retired->flush_plain_batch,
                              retired->flush_batch_capacity);
    retired->flush_batch_capacity = 0;
    fd_lifecycle_scratch_free(&retired->read_cipher_batch,
                              retired->read_batch_capacity);
    fd_lifecycle_scratch_free(&retired->read_plain_batch,
                              retired->read_batch_capacity);
    retired->read_batch_capacity = 0;
    fd_lifecycle_read_cache_free(&retired->read_journal_cache);
    fd_lifecycle_epoch_cache_free(&retired->read_epoch_cache);
    if (retired->data_fd >= 0) {
        close(retired->data_fd);
        retired->data_fd = -1;
    }
    if (retired->journal_fd >= 0) {
        close(retired->journal_fd);
        retired->journal_fd = -1;
    }
    if (retired->epoch_log_fd >= 0) {
        close(retired->epoch_log_fd);
        retired->epoch_log_fd = -1;
    }
    pqc_file_state_release(retired->state);
    retired->state = NULL;
}

static int fd_context_should_open_epoch_log(int open_flags)
{
    if ((open_flags & O_ACCMODE) == O_RDONLY)
        return 0;

    pqc_publication_mode_t mode = PQC_PUBLICATION_MODE_STRICT;
    if (pqc_publication_mode_from_config(&mode) != 0)
        return 0;
    return mode == PQC_PUBLICATION_MODE_EPOCH_REDO_LOG;
}

static int fd_context_open_is_writeable(int open_flags)
{
    return (open_flags & O_ACCMODE) != O_RDONLY;
}

static int fd_context_open_is_readable(int open_flags)
{
    return (open_flags & O_ACCMODE) != O_WRONLY;
}

typedef struct {
    uint64_t next_generation;
    uint64_t logical_size;
    int      logical_size_valid;
    int      data_sidecar_end_valid;
    int      journal_sidecar_end_valid;
} fd_context_state_snapshot_t;

static fd_context_state_snapshot_t
fd_context_visible_state_snapshot(file_state_t *state)
{
    fd_context_state_snapshot_t snapshot;
    memset(&snapshot, 0, sizeof(snapshot));
    if (!state)
        return snapshot;

    pqc_lock_profile_scope_t commit_scope;
    (void)pqc_profiled_mutex_lock(&state->commit_lock, "commit_lock",
                                  __func__, &commit_scope);
    snapshot.logical_size_valid = state->logical_size_valid;
    snapshot.data_sidecar_end_valid = state->data_sidecar_end_valid;
    snapshot.journal_sidecar_end_valid = state->journal_sidecar_end_valid;
    if (snapshot.logical_size_valid) {
        snapshot.next_generation = state->next_generation;
        snapshot.logical_size = state->logical_size;
    }
    (void)pqc_profiled_mutex_unlock(&state->commit_lock, "commit_lock",
                                    __func__, &commit_scope);
    return snapshot;
}

static int fd_context_same_syncfs_domain(int first_fd, int second_fd,
                                         int *same_domain)
{
    if (!same_domain)
        return -EINVAL;
    *same_domain = 0;
    if (first_fd < 0 || second_fd < 0)
        return -EINVAL;

    struct stat first_st;
    struct stat second_st;
    if (fstat(first_fd, &first_st) != 0)
        return -errno;
    if (fstat(second_fd, &second_st) != 0)
        return -errno;
    *same_domain = first_st.st_dev == second_st.st_dev;
    return 0;
}

void pqc_fd_context_table_init(void)
{
    for (int i = 0; i < PQC_MAX_FD; i++) {
        pthread_mutex_init(&g_fd_ctx[i].fd_lock, NULL);
        pthread_cond_init(&g_fd_ctx[i].pending_cv, NULL);
        g_fd_ctx[i].valid = 0;
        g_fd_ctx[i].wbuf = NULL;
        g_fd_ctx[i].snapshot_buf = NULL;
        g_fd_ctx[i].snapshot_buf_capacity = 0;
        g_fd_ctx[i].snapshot_buf_in_use = 0;
        g_fd_ctx[i].flush_plain_batch = NULL;
        g_fd_ctx[i].flush_cipher_batch = NULL;
        g_fd_ctx[i].flush_batch_capacity = 0;
        g_fd_ctx[i].flush_scratch_in_use = 0;
        g_fd_ctx[i].read_plain_batch = NULL;
        g_fd_ctx[i].read_cipher_batch = NULL;
        g_fd_ctx[i].read_batch_capacity = 0;
        g_fd_ctx[i].read_scratch_in_use = 0;
        g_fd_ctx[i].read_journal_cache = NULL;
        g_fd_ctx[i].read_journal_cache_valid = 0;
        g_fd_ctx[i].read_journal_cache_epoch = 0;
        g_fd_ctx[i].read_epoch_cache = NULL;
        g_fd_ctx[i].read_epoch_cache_valid = 0;
        g_fd_ctx[i].read_epoch_cache_epoch = 0;
        g_fd_ctx[i].pending_writeback_jobs = 0;
        g_fd_ctx[i].data_fd = -1;
        g_fd_ctx[i].journal_fd = -1;
        g_fd_ctx[i].epoch_log_fd = -1;
        g_fd_ctx[i].epoch_fallback_enabled = 0;
        g_fd_ctx[i].epoch_log_syncfs_domain_known = 0;
        g_fd_ctx[i].epoch_log_same_syncfs_domain = 0;
        g_fd_ctx[i].logical_path[0] = '\0';
        g_fd_ctx[i].epoch_log_path[0] = '\0';
        g_fd_ctx[i].data_sidecar_dirty = 0;
        g_fd_ctx[i].journal_sidecar_dirty = 0;
        g_fd_ctx[i].epoch_log_dirty = 0;
        g_fd_ctx[i].data_sidecar_dirty_epoch = 0;
        g_fd_ctx[i].journal_sidecar_dirty_epoch = 0;
        g_fd_ctx[i].epoch_log_dirty_epoch = 0;
        g_fd_ctx[i].journal_sidecar_epoch_repairable = 0;
        g_fd_ctx[i].journal_sidecar_epoch_repairable_epoch = 0;
        g_fd_ctx[i].fsync_metadata_epoch = 0;
        g_fd_ctx[i].fsync_metadata_synced_epoch = 0;
        g_fd_ctx[i].fsync_metadata_epoch_repairable = 0;
        g_fd_ctx[i].fsync_metadata_epoch_repairable_epoch = 0;
    }
}

int pqc_fd_context_index_for_fd(int fd)
{
    return fd % PQC_MAX_FD;
}

size_t pqc_fd_context_capacity(void)
{
    return PQC_MAX_FD;
}

pqc_fd_ctx_t *pqc_fd_context_at_index(size_t idx)
{
    return idx < PQC_MAX_FD ? &g_fd_ctx[idx] : NULL;
}

pqc_fd_ctx_t *pqc_fd_context_for_fd(int fd)
{
    int idx = pqc_fd_context_index_for_fd(fd);
    return idx >= 0 ? &g_fd_ctx[idx] : NULL;
}

int pqc_fd_context_path_is_open(const char *marker_path)
{
    if (!marker_path)
        return 0;
    uint8_t ss[64] = {0};
    size_t ss_len = 0;
    uint64_t marker_fid = 0;
    int have_marker_fid =
        pqc_keyring_metadata_load(marker_path, ss, &ss_len, &marker_fid) == 0;
    OQS_MEM_cleanse(ss, sizeof(ss));
    for (size_t i = 0; i < PQC_MAX_FD; ++i) {
        pqc_fd_ctx_t *ctx = &g_fd_ctx[i];
        pqc_lock_profile_scope_t scope;
        (void)pqc_profiled_mutex_lock(&ctx->fd_lock, "fd_lock", __func__,
                                      &scope);
        int match = ctx->valid &&
            (strncmp(ctx->marker_path, marker_path,
                     sizeof(ctx->marker_path)) == 0 ||
             (have_marker_fid && ctx->file_id == marker_fid));
        (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", __func__,
                                        &scope);
        if (match)
            return 1;
    }
    return 0;
}

int pqc_fd_context_logical_path_is_open(const char *logical_path)
{
    if (!logical_path)
        return 0;
    for (size_t i = 0; i < PQC_MAX_FD; ++i) {
        pqc_fd_ctx_t *ctx = &g_fd_ctx[i];
        pqc_lock_profile_scope_t scope;
        (void)pqc_profiled_mutex_lock(&ctx->fd_lock, "fd_lock", __func__,
                                      &scope);
        int match = ctx->valid &&
            fd_context_logical_path_equal(ctx->logical_path, logical_path);
        (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", __func__,
                                        &scope);
        if (match)
            return 1;
    }
    return 0;
}

int pqc_fd_context_any_open(void)
{
    for (size_t i = 0; i < PQC_MAX_FD; ++i) {
        pqc_fd_ctx_t *ctx = &g_fd_ctx[i];
        pqc_lock_profile_scope_t scope;
        (void)pqc_profiled_mutex_lock(&ctx->fd_lock, "fd_lock", __func__,
                                      &scope);
        int valid = ctx->valid;
        (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", __func__,
                                        &scope);
        if (valid)
            return 1;
    }
    return 0;
}

int pqc_fd_context_all_open_markers_hidden(void)
{
    int saw_open = 0;
    for (size_t i = 0; i < PQC_MAX_FD; ++i) {
        pqc_fd_ctx_t *ctx = &g_fd_ctx[i];
        pqc_lock_profile_scope_t scope;
        (void)pqc_profiled_mutex_lock(&ctx->fd_lock, "fd_lock", __func__,
                                      &scope);
        if (ctx->valid) {
            saw_open = 1;
            const char *base = strrchr(ctx->marker_path, '/');
            base = base ? base + 1 : ctx->marker_path;
            int hidden = strncmp(base, ".fuse_hidden",
                                 strlen(".fuse_hidden")) == 0;
            (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock",
                                            __func__, &scope);
            if (!hidden)
                return 0;
            continue;
        }
        (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", __func__,
                                        &scope);
    }
    return saw_open;
}

int pqc_fd_context_rename_path(const char *from_marker_path,
                               const char *to_marker_path)
{
    if (!from_marker_path || !to_marker_path)
        return -EINVAL;
    if (strlen(to_marker_path) >= sizeof(g_fd_ctx[0].marker_path))
        return -ENAMETOOLONG;

    char to_epoch_log_path[sizeof(g_fd_ctx[0].epoch_log_path)];
    int epoch_path_rc = pqc_sidecar_path(to_epoch_log_path,
                                         sizeof(to_epoch_log_path),
                                         to_marker_path, ".pqcepoch");
    if (epoch_path_rc != 0)
        return epoch_path_rc;

    uint8_t ss[64] = {0};
    size_t ss_len = 0;
    uint64_t from_fid = 0;
    int have_from_fid =
        pqc_keyring_metadata_load(from_marker_path, ss, &ss_len, &from_fid) == 0;
    OQS_MEM_cleanse(ss, sizeof(ss));

    for (size_t i = 0; i < PQC_MAX_FD; ++i) {
        pqc_fd_ctx_t *ctx = &g_fd_ctx[i];
        pqc_lock_profile_scope_t scope;
        (void)pqc_profiled_mutex_lock(&ctx->fd_lock, "fd_lock", __func__,
                                      &scope);
        int match = ctx->valid &&
            (strncmp(ctx->marker_path, from_marker_path,
                     sizeof(ctx->marker_path)) == 0 ||
             (have_from_fid && ctx->file_id == from_fid));
        if (match) {
            pqc_fd_context_wait_pending_locked(ctx, &scope, __func__);
            pqc_fd_context_wait_writeback_locked(ctx, &scope, __func__);
            fd_context_copy_cstr(ctx->marker_path,
                                 sizeof(ctx->marker_path),
                                 to_marker_path);
            if (ctx->epoch_log_path[0] != '\0')
                fd_context_copy_cstr(ctx->epoch_log_path,
                                     sizeof(ctx->epoch_log_path),
                                     to_epoch_log_path);
        }
        (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", __func__,
                                        &scope);
    }
    return 0;
}

int pqc_fd_context_rename_logical_path(const char *from_logical_path,
                                       const char *to_logical_path)
{
    if (!from_logical_path || !to_logical_path)
        return -EINVAL;
    if (strlen(to_logical_path) >= sizeof(g_fd_ctx[0].logical_path))
        return -ENAMETOOLONG;

    for (size_t i = 0; i < PQC_MAX_FD; ++i) {
        pqc_fd_ctx_t *ctx = &g_fd_ctx[i];
        pqc_lock_profile_scope_t scope;
        (void)pqc_profiled_mutex_lock(&ctx->fd_lock, "fd_lock", __func__,
                                      &scope);
        int match = ctx->valid &&
            fd_context_logical_path_equal(ctx->logical_path,
                                          from_logical_path);
        if (match) {
            pqc_fd_context_wait_pending_locked(ctx, &scope, __func__);
            pqc_fd_context_wait_writeback_locked(ctx, &scope, __func__);
            fd_context_copy_cstr(ctx->logical_path,
                                 sizeof(ctx->logical_path),
                                 to_logical_path);
        }
        (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", __func__,
                                        &scope);
    }
    return 0;
}

int pqc_fd_context_set(int fd,
                       const char *logical_path,
                       const char *marker_path,
                       const uint8_t *ss,
                       size_t ss_len,
                       uint64_t fid,
                       int open_flags,
                       int qos_class,
                       int tier)
{
    pqc_fd_ctx_t *ctx = pqc_fd_context_for_fd(fd);
    if (!ctx)
        return -EBADF;
    if (!logical_path || strlen(logical_path) >= sizeof(ctx->logical_path))
        return -ENAMETOOLONG;
    if (ss_len > sizeof(ctx->ss))
        return -EINVAL;
    if (qos_class != PQC_QOS_CLASS_ELASTIC &&
        qos_class != PQC_QOS_CLASS_LATENCY)
        return -EINVAL;
    if (tier != PQC_TIER_FULL && tier != PQC_TIER_NONE)
        return -EINVAL;

    pqc_lock_profile_scope_t fd_scope;
    (void)pqc_profiled_mutex_lock(&ctx->fd_lock, "fd_lock", __func__,
                                  &fd_scope);
    if (ctx->valid || ctx->pending_jobs > 0 ||
        ctx->pending_writeback_jobs > 0 || ctx->wbuf ||
        ctx->snapshot_buf || ctx->snapshot_buf_in_use ||
        ctx->flush_plain_batch || ctx->flush_cipher_batch ||
        ctx->flush_scratch_in_use || ctx->read_scratch_in_use ||
        ctx->read_plain_batch || ctx->read_cipher_batch ||
        ctx->read_journal_cache || ctx->read_epoch_cache ||
        ctx->data_fd >= 0 || ctx->journal_fd >= 0 ||
        ctx->epoch_log_fd >= 0 || ctx->state) {
        (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", __func__,
                                        &fd_scope);
        return -EBUSY;
    }
    (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", __func__,
                                    &fd_scope);

    pqc_fd_retired_resources_t setup;
    retired_resources_init(&setup);
    setup.state = pqc_file_state_acquire(fd);
    if (!setup.state) {
        int err = errno ? errno : EIO;
        return -err;
    }

    int writeable_open = fd_context_open_is_writeable(open_flags);
    int readable_open = fd_context_open_is_readable(open_flags);
    char data_path[4096 + 16], journal_path[4096 + 16];
    char epoch_log_path[4096 + 16];
    epoch_log_path[0] = '\0';
    if (pqc_sidecar_path(data_path, sizeof(data_path), marker_path, ".pqcdata") ||
        pqc_sidecar_path(journal_path, sizeof(journal_path), marker_path, ".pqcmeta")) {
        retired_resources_release(&setup);
        return -ENAMETOOLONG;
    }
    setup.data_fd = open(data_path,
                         writeable_open ? (O_RDWR | O_CREAT) : O_RDONLY,
                         0600);
    setup.journal_fd = open(journal_path,
                            writeable_open
                                ? (O_RDWR | O_CREAT | O_APPEND)
                                : O_RDONLY,
                            0600);
    if (setup.data_fd < 0 || setup.journal_fd < 0) {
        int err = errno ? errno : EIO;
        retired_resources_release(&setup);
        return -err;
    }
    if (fd_context_should_open_epoch_log(open_flags)) {
        if (pqc_sidecar_path(epoch_log_path, sizeof(epoch_log_path),
                             marker_path, ".pqcepoch") != 0) {
            retired_resources_release(&setup);
            return -ENAMETOOLONG;
        }
        setup.epoch_log_fd = open(epoch_log_path,
                                  O_RDWR | O_CREAT | O_APPEND | O_CLOEXEC,
                                  0600);
    if (setup.epoch_log_fd < 0) {
        int err = errno ? errno : EIO;
        retired_resources_release(&setup);
        return -err;
    }
    }

    int epoch_log_syncfs_domain_known = 0;
    int epoch_log_same_syncfs_domain = 0;
    if (setup.epoch_log_fd >= 0 &&
        fd_context_same_syncfs_domain(setup.data_fd, setup.epoch_log_fd,
                                      &epoch_log_same_syncfs_domain) == 0) {
        epoch_log_syncfs_domain_known = 1;
    }

    uint64_t logical_size = 0;
    uint64_t journal_max_generation = 0;
    fd_context_state_snapshot_t state_snapshot =
        fd_context_visible_state_snapshot(setup.state);
    int state_generation_ready = state_snapshot.logical_size_valid;
    if (state_generation_ready)
        logical_size = state_snapshot.logical_size;

    struct stat journal_st;
    int journal_sidecar_stat_ready = 0;
    if (!state_generation_ready ||
        !state_snapshot.journal_sidecar_end_valid) {
        if (fstat(setup.journal_fd, &journal_st) != 0 ||
            journal_st.st_size < 0) {
            int err = errno ? errno : EIO;
            retired_resources_release(&setup);
            return -err;
        }
        journal_sidecar_stat_ready = 1;
    }
    pqc_epoch_log_replay_summary_t epoch_summary;
    memset(&epoch_summary, 0, sizeof(epoch_summary));
    pqc_checkpoint_t ckpt = {0};
    int ckpt_rc = -ENOENT;
    int journal_empty =
        journal_sidecar_stat_ready && journal_st.st_size == 0;
    int epoch_has_committed_prefix = 0;
    if (state_generation_ready) {
        journal_max_generation = state_snapshot.next_generation;
    } else {
        if (pqc_publish_logical_size_load(marker_path, &logical_size) != 0)
            logical_size = 0;

        int epoch_replay_rc = setup.epoch_log_fd >= 0
            ? pqc_epoch_log_replay_fd(setup.epoch_log_fd, fid, &epoch_summary)
            : pqc_epoch_log_replay_path(marker_path, fid, &epoch_summary);
        if (epoch_replay_rc != 0 && epoch_replay_rc != -ENOENT) {
            retired_resources_release(&setup);
            return epoch_replay_rc;
        }

        ckpt_rc = pqc_checkpoint_load_and_verify_anchor(marker_path, fid,
                                                        &ckpt);
        if (ckpt_rc == 0) {
            if (ckpt.logical_size > logical_size)
                logical_size = ckpt.logical_size;
            pqc_fault_cutpoint("remount_after_checkpoint_load");
        } else if (ckpt_rc != -ENODATA && ckpt_rc != -ENOENT) {
            retired_resources_release(&setup);
            return ckpt_rc;
        }

        if (!journal_empty) {
            uint64_t tail_highwater = 0;
            int tail_rc = pqc_journal_tail_highwater_generation_at(
                setup.journal_fd, (uint64_t)journal_st.st_size,
                &tail_highwater);
            if (tail_rc == 0 && ckpt_rc == 0 &&
                tail_highwater == ckpt.max_generation) {
                journal_max_generation = tail_highwater;
            } else {
                journal_max_generation =
                    pqc_journal_max_generation(setup.journal_fd);
            }
        }

        epoch_has_committed_prefix =
            epoch_replay_rc == 0 && epoch_summary.committed_records > 0;
        int checkpoint_covers_epoch =
            epoch_has_committed_prefix && ckpt_rc == 0 &&
            ckpt.max_generation >= epoch_summary.max_generation;
        if (epoch_has_committed_prefix && !checkpoint_covers_epoch &&
            writeable_open) {
            int epoch_compact_rc = pqc_epoch_log_compact_checkpoint(
                marker_path, fid, setup.journal_fd, journal_max_generation,
                &epoch_summary);
            if (epoch_compact_rc == 0) {
                ckpt_rc = pqc_checkpoint_load_and_verify_anchor(marker_path,
                                                                fid, &ckpt);
                if (ckpt_rc != 0) {
                    retired_resources_release(&setup);
                    return ckpt_rc;
                }
                if (ckpt.logical_size > logical_size)
                    logical_size = ckpt.logical_size;
                if (journal_max_generation < epoch_summary.max_generation)
                    journal_max_generation = epoch_summary.max_generation;
                pqc_fault_cutpoint("epoch_checkpoint_compaction_after");
            } else if (epoch_compact_rc != -ENOENT &&
                       epoch_compact_rc != -EAGAIN) {
                retired_resources_release(&setup);
                return epoch_compact_rc;
            }
        } else if (epoch_has_committed_prefix) {
            if (journal_max_generation < epoch_summary.max_generation)
                journal_max_generation = epoch_summary.max_generation;
            if (logical_size < epoch_summary.logical_size_after)
                logical_size = epoch_summary.logical_size_after;
        }
    }

    if (writeable_open) {
        setup.wbuf = fd_lifecycle_scratch_alloc(COALESCE_SIZE);
        if (!setup.wbuf) {
            retired_resources_release(&setup);
            return -ENOMEM;
        }
        setup.snapshot_buf_capacity = COALESCE_SIZE;
        setup.snapshot_buf =
            fd_lifecycle_scratch_alloc(setup.snapshot_buf_capacity);
        if (!setup.snapshot_buf) {
            retired_resources_release(&setup);
            return -ENOMEM;
        }
        setup.flush_batch_capacity = PQC_WRITEBACK_BATCH_SCRATCH_SIZE;
        setup.flush_plain_batch =
            fd_lifecycle_scratch_alloc(setup.flush_batch_capacity);
        setup.flush_cipher_batch =
            fd_lifecycle_scratch_alloc(setup.flush_batch_capacity);
        if (!setup.flush_plain_batch || !setup.flush_cipher_batch) {
            retired_resources_release(&setup);
            return -ENOMEM;
        }
    }
    int epoch_fallback_needed =
        epoch_has_committed_prefix || setup.epoch_log_fd >= 0;
    if (readable_open) {
        setup.read_batch_capacity = PQC_READ_BATCH_SCRATCH_SIZE;
        setup.read_plain_batch =
            fd_lifecycle_scratch_alloc(setup.read_batch_capacity);
        setup.read_cipher_batch =
            fd_lifecycle_scratch_alloc(setup.read_batch_capacity);
        if (!setup.read_plain_batch || !setup.read_cipher_batch) {
            retired_resources_release(&setup);
            return -ENOMEM;
        }
        setup.read_journal_cache = fd_lifecycle_read_cache_alloc();
        if (!setup.read_journal_cache) {
            retired_resources_release(&setup);
            return -ENOMEM;
        }
        if (epoch_fallback_needed)
            setup.read_epoch_cache = fd_lifecycle_epoch_cache_alloc();
        if (epoch_fallback_needed && !setup.read_epoch_cache) {
            retired_resources_release(&setup);
            return -ENOMEM;
        }
    }

    struct stat data_st;
    int data_sidecar_stat_ready = 0;
    if (!state_snapshot.data_sidecar_end_valid)
        data_sidecar_stat_ready =
            fstat(setup.data_fd, &data_st) == 0 && data_st.st_size >= 0;
    pqc_lock_profile_scope_t commit_scope;
    (void)pqc_profiled_mutex_lock(&setup.state->commit_lock,
                                  "commit_lock", __func__, &commit_scope);
    if (!setup.state->data_sidecar_end_valid &&
        data_sidecar_stat_ready) {
        setup.state->data_sidecar_end = (uint64_t)data_st.st_size;
        setup.state->data_sidecar_end_valid = 1;
    }
    if (!setup.state->journal_sidecar_end_valid &&
        journal_sidecar_stat_ready) {
        setup.state->journal_sidecar_end = (uint64_t)journal_st.st_size;
        setup.state->journal_sidecar_end_valid = 1;
    }
    int initializing_visible_state = !setup.state->logical_size_valid;
    if (setup.state->next_generation < journal_max_generation)
        setup.state->next_generation = journal_max_generation;
    if (initializing_visible_state &&
        setup.state->committed_generation < journal_max_generation)
        setup.state->committed_generation = journal_max_generation;
    if (ckpt_rc == 0 && setup.state->next_generation < ckpt.max_generation)
        setup.state->next_generation = ckpt.max_generation;
    if (initializing_visible_state) {
        setup.state->logical_size = logical_size;
        setup.state->logical_size_valid = 1;
        if (journal_empty && !epoch_has_committed_prefix &&
            (ckpt_rc != 0 || ckpt.max_generation == 0)) {
            pqc_file_state_mapping_cache_mark_complete_empty_locked(
                setup.state);
        } else {
            pqc_file_state_mapping_cache_mark_unknown_locked(setup.state);
        }
    } else {
        logical_size = setup.state->logical_size;
    }
    (void)pqc_profiled_mutex_unlock(&setup.state->commit_lock,
                                    "commit_lock", __func__, &commit_scope);

    (void)pqc_profiled_mutex_lock(&ctx->fd_lock, "fd_lock", __func__,
                                  &fd_scope);
    if (ctx->valid || ctx->pending_jobs > 0 ||
        ctx->pending_writeback_jobs > 0 || ctx->wbuf ||
        ctx->snapshot_buf || ctx->snapshot_buf_in_use ||
        ctx->flush_plain_batch || ctx->flush_cipher_batch ||
        ctx->flush_scratch_in_use || ctx->read_scratch_in_use ||
        ctx->read_plain_batch || ctx->read_cipher_batch ||
        ctx->read_journal_cache || ctx->read_epoch_cache ||
        ctx->data_fd >= 0 || ctx->journal_fd >= 0 ||
        ctx->epoch_log_fd >= 0 || ctx->state) {
        (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", __func__,
                                        &fd_scope);
        retired_resources_release(&setup);
        return -EBUSY;
    }
    memcpy(ctx->ss, ss, ss_len);
    ctx->ss_len = ss_len;
    ctx->file_id = fid;
    ctx->state = setup.state;
    setup.state = NULL;
    ctx->data_fd = setup.data_fd;
    setup.data_fd = -1;
    ctx->journal_fd = setup.journal_fd;
    setup.journal_fd = -1;
    ctx->epoch_log_fd = setup.epoch_log_fd;
    setup.epoch_log_fd = -1;
    ctx->epoch_fallback_enabled = epoch_fallback_needed;
    ctx->epoch_log_syncfs_domain_known = epoch_log_syncfs_domain_known;
    ctx->epoch_log_same_syncfs_domain = epoch_log_same_syncfs_domain;
    ctx->data_sidecar_dirty = 0;
    ctx->journal_sidecar_dirty = 0;
    ctx->epoch_log_dirty = 0;
    ctx->data_sidecar_dirty_epoch = 0;
    ctx->journal_sidecar_dirty_epoch = 0;
    ctx->epoch_log_dirty_epoch = 0;
    ctx->journal_sidecar_epoch_repairable = 0;
    ctx->journal_sidecar_epoch_repairable_epoch = 0;
    ctx->fsync_metadata_epoch = 0;
    ctx->fsync_metadata_synced_epoch = 0;
    ctx->fsync_metadata_epoch_repairable = 0;
    ctx->fsync_metadata_epoch_repairable_epoch = 0;
    fd_context_copy_cstr(ctx->logical_path, sizeof(ctx->logical_path),
                         logical_path);
    fd_context_copy_cstr(ctx->marker_path, sizeof(ctx->marker_path),
                         marker_path);
    fd_context_copy_cstr(ctx->epoch_log_path, sizeof(ctx->epoch_log_path),
                         epoch_log_path);
    ctx->logical_size = logical_size;
    ctx->valid = 1;
    ctx->pending_writeback_jobs = 0;
    ctx->tier = tier;
    ctx->qos_class = qos_class;
    ctx->key_epoch = 0;
    ctx->last_rekey = pqc_rekey_write_trigger_enabled() ? time(NULL) : 0;
    ctx->wbuf = setup.wbuf;
    setup.wbuf = NULL;
    ctx->snapshot_buf = setup.snapshot_buf;
    setup.snapshot_buf = NULL;
    ctx->snapshot_buf_capacity = setup.snapshot_buf_capacity;
    setup.snapshot_buf_capacity = 0;
    ctx->snapshot_buf_in_use = 0;
    ctx->flush_plain_batch = setup.flush_plain_batch;
    setup.flush_plain_batch = NULL;
    ctx->flush_cipher_batch = setup.flush_cipher_batch;
    setup.flush_cipher_batch = NULL;
    ctx->flush_batch_capacity = setup.flush_batch_capacity;
    setup.flush_batch_capacity = 0;
    ctx->flush_scratch_in_use = 0;
    ctx->read_plain_batch = setup.read_plain_batch;
    setup.read_plain_batch = NULL;
    ctx->read_cipher_batch = setup.read_cipher_batch;
    setup.read_cipher_batch = NULL;
    ctx->read_batch_capacity = setup.read_batch_capacity;
    setup.read_batch_capacity = 0;
    ctx->read_scratch_in_use = 0;
    ctx->read_journal_cache = setup.read_journal_cache;
    setup.read_journal_cache = NULL;
    ctx->read_journal_cache_valid = 0;
    ctx->read_journal_cache_epoch = 0;
    ctx->read_epoch_cache = setup.read_epoch_cache;
    setup.read_epoch_cache = NULL;
    ctx->read_epoch_cache_valid = 0;
    ctx->read_epoch_cache_epoch = 0;
    ctx->wbuf_used = 0;
    ctx->wbuf_base_off = 0;
    (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", __func__,
                                    &fd_scope);
    return 0;
}

void pqc_fd_context_clear(int fd)
{
    pqc_fd_ctx_t *ctx = pqc_fd_context_for_fd(fd);
    if (!ctx)
        return;
    pqc_fd_retired_resources_t retired;
    retired_resources_init(&retired);

    pqc_lock_profile_scope_t fd_scope;
    (void)pqc_profiled_mutex_lock(&ctx->fd_lock, "fd_lock", __func__,
                                  &fd_scope);
    pqc_fd_context_wait_pending_locked(ctx, &fd_scope, __func__);
    OQS_MEM_cleanse(ctx->ss, sizeof(ctx->ss));
    ctx->ss_len = 0;
    retired.wbuf = ctx->wbuf;
    ctx->wbuf = NULL;
    retired.snapshot_buf = ctx->snapshot_buf;
    ctx->snapshot_buf = NULL;
    retired.snapshot_buf_capacity = ctx->snapshot_buf_capacity;
    ctx->snapshot_buf_capacity = 0;
    ctx->snapshot_buf_in_use = 0;
    retired.flush_plain_batch = ctx->flush_plain_batch;
    ctx->flush_plain_batch = NULL;
    retired.flush_cipher_batch = ctx->flush_cipher_batch;
    ctx->flush_cipher_batch = NULL;
    retired.flush_batch_capacity = ctx->flush_batch_capacity;
    ctx->flush_batch_capacity = 0;
    ctx->flush_scratch_in_use = 0;
    retired.read_plain_batch = ctx->read_plain_batch;
    ctx->read_plain_batch = NULL;
    retired.read_cipher_batch = ctx->read_cipher_batch;
    ctx->read_cipher_batch = NULL;
    retired.read_batch_capacity = ctx->read_batch_capacity;
    ctx->read_batch_capacity = 0;
    ctx->read_scratch_in_use = 0;
    retired.read_journal_cache = ctx->read_journal_cache;
    ctx->read_journal_cache = NULL;
    ctx->read_journal_cache_valid = 0;
    ctx->read_journal_cache_epoch = 0;
    retired.read_epoch_cache = ctx->read_epoch_cache;
    ctx->read_epoch_cache = NULL;
    ctx->read_epoch_cache_valid = 0;
    ctx->read_epoch_cache_epoch = 0;
    retired.data_fd = ctx->data_fd;
    retired.journal_fd = ctx->journal_fd;
    retired.epoch_log_fd = ctx->epoch_log_fd;
    retired.state = ctx->state;
    ctx->pending_jobs = 0;
    ctx->pending_writeback_jobs = 0;
    ctx->wbuf_used = 0;
    ctx->wbuf_base_off = 0;
    ctx->data_fd = -1;
    ctx->journal_fd = -1;
    ctx->epoch_log_fd = -1;
    ctx->epoch_fallback_enabled = 0;
    ctx->epoch_log_syncfs_domain_known = 0;
    ctx->epoch_log_same_syncfs_domain = 0;
    ctx->data_sidecar_dirty = 0;
    ctx->journal_sidecar_dirty = 0;
    ctx->epoch_log_dirty = 0;
    ctx->data_sidecar_dirty_epoch = 0;
    ctx->journal_sidecar_dirty_epoch = 0;
    ctx->epoch_log_dirty_epoch = 0;
    ctx->journal_sidecar_epoch_repairable = 0;
    ctx->journal_sidecar_epoch_repairable_epoch = 0;
    ctx->fsync_metadata_epoch = 0;
    ctx->fsync_metadata_synced_epoch = 0;
    ctx->fsync_metadata_epoch_repairable = 0;
    ctx->fsync_metadata_epoch_repairable_epoch = 0;
    ctx->logical_size = 0;
    ctx->logical_path[0] = '\0';
    ctx->marker_path[0] = '\0';
    ctx->epoch_log_path[0] = '\0';
    ctx->file_id = 0;
    ctx->tier = 0;
    ctx->qos_class = 0;
    ctx->key_epoch = 0;
    ctx->last_rekey = 0;
    ctx->valid = 0;
    ctx->state = NULL;
    (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock", __func__,
                                    &fd_scope);

    retired_resources_release(&retired);
}

void pqc_fd_context_wait_pending_locked(pqc_fd_ctx_t *ctx,
                                        pqc_lock_profile_scope_t *scope,
                                        const char *site)
{
    while (ctx && (ctx->pending_jobs > 0 || ctx->snapshot_buf_in_use ||
                   ctx->flush_scratch_in_use ||
                   ctx->read_scratch_in_use))
        (void)pqc_profiled_cond_wait(&ctx->pending_cv, &ctx->fd_lock,
                                     "fd_lock", site, scope);
}

void pqc_fd_context_wait_writeback_locked(pqc_fd_ctx_t *ctx,
                                          pqc_lock_profile_scope_t *scope,
                                          const char *site)
{
    while (ctx && (ctx->pending_writeback_jobs > 0 ||
                   ctx->snapshot_buf_in_use ||
                   ctx->flush_scratch_in_use))
        (void)pqc_profiled_cond_wait(&ctx->pending_cv, &ctx->fd_lock,
                                     "fd_lock", site, scope);
}

void pqc_fd_context_pending_job_begin(void *arg)
{
    pqc_fd_ctx_t *ctx = (pqc_fd_ctx_t *)arg;
    if (ctx)
        ++ctx->pending_jobs;
}

void pqc_fd_context_pending_job_end(void *arg)
{
    pqc_fd_ctx_t *ctx = (pqc_fd_ctx_t *)arg;
    if (!ctx)
        return;
    if (ctx->pending_jobs > 0)
        --ctx->pending_jobs;
    if (ctx->pending_jobs == 0)
        pthread_cond_broadcast(&ctx->pending_cv);
}

void pqc_fd_context_writeback_job_begin(void *arg)
{
    pqc_fd_ctx_t *ctx = (pqc_fd_ctx_t *)arg;
    if (!ctx)
        return;
    pqc_fd_context_pending_job_begin(ctx);
    ++ctx->pending_writeback_jobs;
}

void pqc_fd_context_writeback_job_end(void *arg)
{
    pqc_fd_ctx_t *ctx = (pqc_fd_ctx_t *)arg;
    if (!ctx)
        return;
    if (ctx->pending_writeback_jobs > 0)
        --ctx->pending_writeback_jobs;
    pqc_fd_context_pending_job_end(ctx);
    if (ctx->pending_writeback_jobs == 0)
        pthread_cond_broadcast(&ctx->pending_cv);
}

static void sidecar_sync_init(pqc_fd_sidecar_sync_t *sync)
{
    if (!sync)
        return;
    sync->data_fd = -1;
    sync->journal_fd = -1;
    sync->epoch_log_fd = -1;
    sync->sync_data = 0;
    sync->sync_journal = 0;
    sync->sync_epoch_log = 0;
    sync->sync_data_epoch_domain_known = 0;
    sync->sync_data_epoch_same_domain = 0;
    sync->data_epoch = 0;
    sync->journal_epoch = 0;
    sync->epoch_log_epoch = 0;
}

static void sidecar_sync_reset_fds(pqc_fd_sidecar_sync_t *sync)
{
    if (!sync)
        return;
    sync->data_fd = -1;
    sync->journal_fd = -1;
    sync->epoch_log_fd = -1;
}

int pqc_fd_context_prepare_dirty_sidecar_sync_locked(
    pqc_fd_ctx_t *ctx,
    pqc_fd_sidecar_sync_t *sync)
{
    sidecar_sync_init(sync);
    if (!sync || !ctx || !ctx->valid)
        return 0;

    if (ctx->data_sidecar_dirty && ctx->data_fd >= 0) {
        sync->data_fd = ctx->data_fd;
        sync->sync_data = 1;
        sync->data_epoch = ctx->data_sidecar_dirty_epoch;
    }
    if (ctx->journal_sidecar_dirty &&
        ctx->journal_sidecar_epoch_repairable &&
        ctx->journal_sidecar_epoch_repairable_epoch ==
            ctx->journal_sidecar_dirty_epoch) {
        ctx->journal_sidecar_dirty = 0;
        ctx->journal_sidecar_epoch_repairable = 0;
        ctx->journal_sidecar_epoch_repairable_epoch = 0;
    } else if (ctx->journal_sidecar_dirty && ctx->journal_fd >= 0) {
        sync->journal_fd = ctx->journal_fd;
        sync->sync_journal = 1;
        sync->journal_epoch = ctx->journal_sidecar_dirty_epoch;
    }
    if (ctx->epoch_log_dirty && ctx->epoch_log_fd >= 0) {
        sync->epoch_log_fd = ctx->epoch_log_fd;
        sync->sync_epoch_log = 1;
        sync->epoch_log_epoch = ctx->epoch_log_dirty_epoch;
        sync->sync_data_epoch_domain_known =
            ctx->epoch_log_syncfs_domain_known;
        sync->sync_data_epoch_same_domain =
            ctx->epoch_log_same_syncfs_domain;
    }
    return 0;
}

int pqc_fd_context_run_dirty_sidecar_sync(pqc_fd_sidecar_sync_t *sync)
{
    if (!sync)
        return 0;
    int rc = 0;
    int data_epoch_coalesced = 0;
    if (sync->sync_data && sync->data_fd >= 0 &&
        sync->sync_epoch_log && sync->epoch_log_fd >= 0) {
        int same_domain = 0;
        int domain_known = sync->sync_data_epoch_domain_known;
        if (domain_known) {
            same_domain = sync->sync_data_epoch_same_domain;
        } else if (fd_context_same_syncfs_domain(sync->data_fd,
                                                 sync->epoch_log_fd,
                                                 &same_domain) == 0) {
            domain_known = 1;
        }
        if (domain_known && same_domain) {
            int sync_rc = pqc_durability_syncfs(
                sync->epoch_log_fd, PQC_DURABILITY_SITE_EPOCH_LOG);
            if (sync_rc != 0 && rc == 0)
                rc = sync_rc;
            data_epoch_coalesced = sync_rc == 0;
        }
    }
    if (sync->sync_data && sync->data_fd >= 0) {
        if (!data_epoch_coalesced) {
            int sync_rc = pqc_durability_fdatasync(
                sync->data_fd, PQC_DURABILITY_SITE_DATA_SIDECAR);
            if (sync_rc != 0 && rc == 0)
                rc = sync_rc;
        }
    }
    if (sync->sync_journal && sync->journal_fd >= 0) {
        int sync_rc = pqc_durability_fdatasync(
            sync->journal_fd, PQC_DURABILITY_SITE_JOURNAL_SIDECAR);
        if (sync_rc != 0 && rc == 0)
            rc = sync_rc;
    }
    if (sync->sync_epoch_log && sync->epoch_log_fd >= 0) {
        if (!data_epoch_coalesced) {
            int sync_rc = pqc_durability_fdatasync(
                sync->epoch_log_fd, PQC_DURABILITY_SITE_EPOCH_LOG);
            if (sync_rc != 0 && rc == 0)
                rc = sync_rc;
        }
    }
    sidecar_sync_reset_fds(sync);
    return rc;
}

void pqc_fd_context_finish_dirty_sidecar_sync_locked(
    pqc_fd_ctx_t *ctx,
    const pqc_fd_sidecar_sync_t *sync,
    int sync_rc)
{
    if (!ctx || !sync || sync_rc != 0 || !ctx->valid)
        return;
    if (sync->sync_data &&
        ctx->data_sidecar_dirty &&
        ctx->data_sidecar_dirty_epoch == sync->data_epoch)
        ctx->data_sidecar_dirty = 0;
    if (sync->sync_journal &&
        ctx->journal_sidecar_dirty &&
        ctx->journal_sidecar_dirty_epoch == sync->journal_epoch) {
        ctx->journal_sidecar_dirty = 0;
        ctx->journal_sidecar_epoch_repairable = 0;
        ctx->journal_sidecar_epoch_repairable_epoch = 0;
    }
    if (sync->sync_epoch_log &&
        ctx->epoch_log_dirty &&
        ctx->epoch_log_dirty_epoch == sync->epoch_log_epoch)
        ctx->epoch_log_dirty = 0;
}

void pqc_fd_context_mark_fsync_dirty(pqc_fd_ctx_t *ctx)
{
    if (!ctx || !ctx->valid)
        return;
    pqc_fd_context_invalidate_read_journal_cache(ctx);
    pqc_fd_context_invalidate_read_epoch_cache(ctx);
    if (ctx->fsync_metadata_epoch == UINT64_MAX) {
        ctx->fsync_metadata_epoch = 1;
        ctx->fsync_metadata_synced_epoch = 0;
    } else {
        ++ctx->fsync_metadata_epoch;
    }
    ctx->fsync_metadata_epoch_repairable = 0;
    ctx->fsync_metadata_epoch_repairable_epoch = 0;
}

static uint64_t fd_context_epoch_advance(uint64_t epoch)
{
    return epoch == UINT64_MAX ? 1 : epoch + 1U;
}

void pqc_fd_context_invalidate_read_journal_cache(pqc_fd_ctx_t *ctx)
{
    if (!ctx)
        return;
    ctx->read_journal_cache_valid = 0;
    if (ctx->read_journal_cache)
        pqc_journal_lookup_view_clear(ctx->read_journal_cache);
    ctx->read_journal_cache_epoch =
        fd_context_epoch_advance(ctx->read_journal_cache_epoch);
}

int pqc_fd_context_read_journal_cache_snapshot_locked(
    pqc_fd_ctx_t *ctx,
    uint64_t first_block,
    uint64_t last_block,
    uint64_t max_generation,
    pqc_journal_lookup_view_t *out,
    uint64_t *epoch_out)
{
    if (epoch_out)
        *epoch_out = ctx ? ctx->read_journal_cache_epoch : 0;
    if (!ctx || !out || !ctx->valid || !ctx->read_journal_cache ||
        !ctx->read_journal_cache_valid)
        return 0;
    const pqc_journal_lookup_view_t *cache = ctx->read_journal_cache;
    if (!cache->initialized || cache->overflow ||
        cache->max_generation != max_generation ||
        first_block < cache->first_block ||
        last_block > cache->last_block)
        return 0;
    *out = *cache;
    return 1;
}

void pqc_fd_context_read_journal_cache_store_locked(
    pqc_fd_ctx_t *ctx,
    const pqc_journal_lookup_view_t *view,
    uint64_t epoch_snapshot)
{
    if (!ctx || !view || !ctx->valid || !ctx->read_journal_cache ||
        view->overflow || !view->initialized ||
        ctx->read_journal_cache_epoch != epoch_snapshot)
        return;
    *ctx->read_journal_cache = *view;
    ctx->read_journal_cache_valid = 1;
}

void pqc_fd_context_invalidate_read_epoch_cache(pqc_fd_ctx_t *ctx)
{
    if (!ctx)
        return;
    ctx->read_epoch_cache_valid = 0;
    if (ctx->read_epoch_cache)
        pqc_epoch_log_lookup_view_clear(ctx->read_epoch_cache);
    ctx->read_epoch_cache_epoch =
        fd_context_epoch_advance(ctx->read_epoch_cache_epoch);
}

int pqc_fd_context_read_epoch_cache_snapshot_locked(
    pqc_fd_ctx_t *ctx,
    uint64_t first_block,
    uint64_t last_block,
    uint64_t max_generation,
    uint64_t file_id,
    pqc_epoch_log_lookup_view_t *out,
    uint64_t *epoch_out)
{
    if (epoch_out)
        *epoch_out = ctx ? ctx->read_epoch_cache_epoch : 0;
    if (!ctx || !out || !ctx->valid || !ctx->read_epoch_cache ||
        !ctx->read_epoch_cache_valid)
        return 0;
    const pqc_epoch_log_lookup_view_t *cache = ctx->read_epoch_cache;
    if (!cache->initialized || cache->overflow ||
        cache->max_generation != max_generation ||
        cache->file_id != file_id ||
        first_block < cache->first_block ||
        last_block > cache->last_block)
        return 0;
    *out = *cache;
    return 1;
}

void pqc_fd_context_read_epoch_cache_store_locked(
    pqc_fd_ctx_t *ctx,
    const pqc_epoch_log_lookup_view_t *view,
    uint64_t epoch_snapshot)
{
    if (!ctx || !view || !ctx->valid || !ctx->read_epoch_cache ||
        view->overflow || !view->initialized ||
        ctx->read_epoch_cache_epoch != epoch_snapshot)
        return;
    *ctx->read_epoch_cache = *view;
    ctx->read_epoch_cache_valid = 1;
}

void pqc_fd_context_mark_fsync_epoch_repairable(pqc_fd_ctx_t *ctx)
{
    if (!ctx || !ctx->valid)
        return;
    ctx->fsync_metadata_epoch_repairable = 1;
    ctx->fsync_metadata_epoch_repairable_epoch = ctx->fsync_metadata_epoch;
}

int pqc_fd_context_fsync_dirty_epoch(const pqc_fd_ctx_t *ctx,
                                     uint64_t *epoch_out)
{
    if (epoch_out)
        *epoch_out = 0;
    if (!ctx || !ctx->valid)
        return 0;
    if (epoch_out)
        *epoch_out = ctx->fsync_metadata_epoch;
    return ctx->fsync_metadata_epoch != ctx->fsync_metadata_synced_epoch;
}

int pqc_fd_context_fsync_epoch_repairable(const pqc_fd_ctx_t *ctx,
                                          uint64_t epoch)
{
    return ctx && ctx->valid &&
        ctx->fsync_metadata_epoch_repairable &&
        ctx->fsync_metadata_epoch_repairable_epoch == epoch;
}

void pqc_fd_context_mark_fsync_synced(pqc_fd_ctx_t *ctx, uint64_t epoch)
{
    if (!ctx || !ctx->valid)
        return;
    if (ctx->fsync_metadata_epoch == epoch) {
        ctx->fsync_metadata_synced_epoch = epoch;
        ctx->fsync_metadata_epoch_repairable = 0;
        ctx->fsync_metadata_epoch_repairable_epoch = 0;
    }
}

int pqc_fd_context_fsync_clean_locked(const pqc_fd_ctx_t *ctx)
{
    if (!ctx || !ctx->valid)
        return 0;
    if (ctx->data_sidecar_dirty || ctx->journal_sidecar_dirty ||
        ctx->epoch_log_dirty)
        return 0;
    return ctx->fsync_metadata_epoch == ctx->fsync_metadata_synced_epoch;
}
