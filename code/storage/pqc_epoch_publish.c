#include "pqc_epoch_publish.h"

#include "pqc_config.h"
#include "pqc_anchor_worker.h"
#include "pqc_durability.h"
#include "pqc_epoch_log.h"
#include "pqc_lock_profile.h"
#include "pqc_posix.h"
#include "pqc_trace_sink.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    char log_path[4096 + 16];
    size_t record_count;
    int barrier_rc;
    uint32_t epoch_log_sync_count;
    uint32_t group_size;
    uint64_t group_epoch;
    uint64_t group_wait_ns;
    uint64_t group_open_wait_ns;
    uint64_t group_sync_ns;
    uint64_t group_completion_wait_ns;
    const char *group_role;
    const char *sync_primitive;
    uint64_t file_id;
    uint64_t payload_bytes;
    int log_fd;
    int owns_log_fd;
    int force_syncfs;
    int defer_durability;
    int windowed_file_anchor;
    int data_fsync_fallback_count;
    uint64_t epoch_dirty_epoch;
    int trace_emitted;
} pqc_epoch_publish_context_t;

typedef struct {
    pthread_mutex_t lock;
    pthread_cond_t cv;
    uint64_t next_epoch;
    uint64_t completed_epoch;
    uint64_t open_epoch;
    uint64_t active_epoch;
    uint32_t open_group_size;
    uint64_t open_group_bytes;
    int open_force_syncfs;
    uint32_t completed_group_size;
    uint64_t completed_group_bytes;
    int completed_force_syncfs;
    int open_rc;
    int completed_rc;
    int open;
    int active;
} pqc_epoch_group_barrier_t;

static pqc_epoch_group_barrier_t g_epoch_group_barrier = {
    .lock = PTHREAD_MUTEX_INITIALIZER,
    .cv = PTHREAD_COND_INITIALIZER,
    .next_epoch = 1,
};

static pqc_trace_sink_t g_publication_trace_sink =
    PQC_TRACE_SINK_INITIALIZER;
static atomic_int       g_publication_mode_cached = ATOMIC_VAR_INIT(-1);
static atomic_uint       g_epoch_group_max_cached = ATOMIC_VAR_INIT(0);
static atomic_ullong    g_epoch_group_wait_ns_cached = ATOMIC_VAR_INIT(0);
static atomic_int       g_epoch_group_wait_ns_ready = ATOMIC_VAR_INIT(0);

static int epoch_barrier_lock(pqc_epoch_group_barrier_t *barrier,
                              pqc_lock_profile_scope_t *scope,
                              const char *site)
{
    return pqc_profiled_mutex_lock(&barrier->lock, "epoch_barrier_lock",
                                   site, scope);
}

static int epoch_barrier_unlock(pqc_epoch_group_barrier_t *barrier,
                                pqc_lock_profile_scope_t *scope,
                                const char *site)
{
    return pqc_profiled_mutex_unlock(&barrier->lock, "epoch_barrier_lock",
                                     site, scope);
}

static int epoch_barrier_wait(pqc_epoch_group_barrier_t *barrier,
                              pqc_lock_profile_scope_t *scope,
                              const char *site)
{
    return pqc_profiled_cond_wait(&barrier->cv, &barrier->lock,
                                  "epoch_barrier_lock", site, scope);
}

static int epoch_barrier_timedwait(pqc_epoch_group_barrier_t *barrier,
                                   pqc_lock_profile_scope_t *scope,
                                   const char *site,
                                   const struct timespec *deadline)
{
    return pqc_profiled_cond_timedwait(&barrier->cv, &barrier->lock,
                                       "epoch_barrier_lock", site, scope,
                                       deadline);
}

static int pqc_streq(const char *a, const char *b)
{
    return a && b && strcmp(a, b) == 0;
}

int pqc_publication_mode_parse(const char *raw,
                               pqc_publication_mode_t *out)
{
    if (!out)
        return -EINVAL;
    if (!raw || !*raw || pqc_streq(raw, "strict")) {
        *out = PQC_PUBLICATION_MODE_STRICT;
        return 0;
    }
    if (pqc_streq(raw, "epoch") || pqc_streq(raw, "epoch-redo-log")) {
        *out = PQC_PUBLICATION_MODE_EPOCH_REDO_LOG;
        return 0;
    }
    if (pqc_streq(raw, "epoch-skeleton")) {
        *out = PQC_PUBLICATION_MODE_EPOCH_UNAVAILABLE;
        return 0;
    }
    return -EINVAL;
}

const char *pqc_publication_mode_name(pqc_publication_mode_t mode)
{
    switch (mode) {
    case PQC_PUBLICATION_MODE_STRICT:
        return "strict";
    case PQC_PUBLICATION_MODE_EPOCH_UNAVAILABLE:
        return "epoch-unavailable";
    case PQC_PUBLICATION_MODE_EPOCH_REDO_LOG:
        return "epoch-redo-log";
    }
    return "unknown";
}

int pqc_publication_mode_from_config(pqc_publication_mode_t *out)
{
    if (!out)
        return -EINVAL;

    int cached = atomic_load_explicit(&g_publication_mode_cached,
                                      memory_order_acquire);
    if (cached >= 0) {
        *out = (pqc_publication_mode_t)cached;
        return 0;
    }

    const char *raw =
        pqc_config_nonempty_or_default("PQC_PUBLICATION_MODE", "strict");
    pqc_publication_mode_t parsed = PQC_PUBLICATION_MODE_STRICT;
    int rc = pqc_publication_mode_parse(raw, &parsed);
    if (rc != 0)
        return rc;

    int expected = -1;
    (void)atomic_compare_exchange_strong_explicit(
        &g_publication_mode_cached, &expected, (int)parsed,
        memory_order_release, memory_order_relaxed);
    cached = atomic_load_explicit(&g_publication_mode_cached,
                                  memory_order_acquire);
    *out = (pqc_publication_mode_t)cached;
    return 0;
}

static uint64_t pqc_publication_now_ns(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        return 0;
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static int pqc_epoch_deadline_from_now(uint64_t wait_ns,
                                       struct timespec *deadline)
{
    if (!deadline)
        return -EINVAL;
    if (clock_gettime(CLOCK_REALTIME, deadline) != 0)
        return -errno;
    uint64_t nsec = (uint64_t)deadline->tv_nsec + wait_ns;
    deadline->tv_sec += (time_t)(nsec / UINT64_C(1000000000));
    deadline->tv_nsec = (long)(nsec % UINT64_C(1000000000));
    return 0;
}

static uint32_t pqc_epoch_group_max_from_config(void)
{
    unsigned int cached = atomic_load_explicit(&g_epoch_group_max_cached,
                                               memory_order_acquire);
    if (cached != 0)
        return cached;

    uint64_t fallback =
        pqc_config_u64_or_default("PQC_PARALLEL_COMMIT_GROUP_MAX", 1);
    uint64_t value = pqc_config_u64_or_default("PQC_EPOCH_GROUP_MAX",
                                               fallback);
    if (value == 0)
        value = 1;
    if (value > UINT32_MAX)
        value = UINT32_MAX;
    unsigned int computed = (unsigned int)value;
    unsigned int expected = 0;
    (void)atomic_compare_exchange_strong_explicit(
        &g_epoch_group_max_cached, &expected, computed,
        memory_order_release, memory_order_relaxed);
    return atomic_load_explicit(&g_epoch_group_max_cached,
                                memory_order_acquire);
}

static uint64_t pqc_epoch_group_wait_ns_from_config(void)
{
    if (atomic_load_explicit(&g_epoch_group_wait_ns_ready,
                             memory_order_acquire))
        return (uint64_t)atomic_load_explicit(&g_epoch_group_wait_ns_cached,
                                              memory_order_relaxed);

    uint64_t fallback =
        pqc_config_u64_or_default("PQC_PARALLEL_COMMIT_WAIT_NS", 0);
    uint64_t value = pqc_config_u64_or_default("PQC_EPOCH_GROUP_WAIT_NS",
                                               fallback);
    atomic_store_explicit(&g_epoch_group_wait_ns_cached,
                          (unsigned long long)value,
                          memory_order_relaxed);
    atomic_store_explicit(&g_epoch_group_wait_ns_ready, 1,
                          memory_order_release);
    return value;
}

static int pqc_epoch_sync_fd(int fd, uint32_t group_size, int force_syncfs,
                             const char **primitive)
{
    if (primitive)
        *primitive = (force_syncfs || group_size > 1)
            ? "syncfs" : "fdatasync";
    if (!force_syncfs && group_size <= 1)
        return pqc_durability_fdatasync(fd,
                                        PQC_DURABILITY_SITE_EPOCH_LOG);
    return pqc_durability_syncfs(fd, PQC_DURABILITY_SITE_EPOCH_LOG);
}

static int pqc_epoch_same_syncfs_domain(int data_fd, int epoch_fd)
{
    struct stat data_st;
    struct stat epoch_st;

    if (fstat(data_fd, &data_st) != 0 || fstat(epoch_fd, &epoch_st) != 0)
        return 0;
    return data_st.st_dev == epoch_st.st_dev;
}

static int pqc_epoch_request_same_syncfs_domain(
    const pqc_strict_publish_request_t *req,
    int epoch_fd)
{
    if (!req)
        return 0;
    if (req->epoch_log_syncfs_domain_known)
        return req->epoch_log_same_syncfs_domain;
    return pqc_epoch_same_syncfs_domain(req->data_fd, epoch_fd);
}

static int pqc_epoch_group_barrier_wait(
    int fd, uint64_t file_id, uint64_t bytes, int append_rc,
    pqc_epoch_publish_context_t *ctx)
{
    (void)file_id;
    uint32_t max_group = pqc_epoch_group_max_from_config();
    uint64_t max_wait_ns = pqc_epoch_group_wait_ns_from_config();
    uint64_t start_ns = pqc_publication_now_ns();

    if (max_group <= 1 || max_wait_ns == 0) {
        int rc = append_rc;
        const char *primitive = "fdatasync";
        uint64_t sync_start_ns = pqc_publication_now_ns();
        if (rc == 0)
            rc = pqc_epoch_sync_fd(fd, 1, ctx && ctx->force_syncfs,
                                   &primitive);
        uint64_t sync_end_ns = pqc_publication_now_ns();
        if (ctx) {
            ctx->barrier_rc = rc;
            ctx->epoch_log_sync_count = rc == 0 ? 1U : 0U;
            ctx->group_size = 1;
            ctx->group_epoch = 0;
            ctx->group_role = max_group <= 1 ? "single" : "no-wait-single";
            ctx->sync_primitive = primitive;
            uint64_t end_ns = sync_end_ns;
            ctx->group_wait_ns = end_ns >= start_ns ? end_ns - start_ns : 0;
            ctx->group_open_wait_ns = 0;
            ctx->group_sync_ns = sync_end_ns >= sync_start_ns
                ? sync_end_ns - sync_start_ns : 0;
            ctx->group_completion_wait_ns = 0;
        }
        return rc;
    }

    pqc_epoch_group_barrier_t *barrier = &g_epoch_group_barrier;
    pqc_lock_profile_scope_t barrier_scope;
    int rc = epoch_barrier_lock(barrier, &barrier_scope, __func__);
    if (rc != 0)
        return -rc;

    while (barrier->active && !barrier->open)
        (void)epoch_barrier_wait(barrier, &barrier_scope, __func__);

    if (barrier->open && barrier->open_group_size < max_group) {
        uint64_t epoch = barrier->open_epoch;
        barrier->open_group_size++;
        barrier->open_group_bytes += bytes;
        if (ctx && ctx->force_syncfs)
            barrier->open_force_syncfs = 1;
        if (append_rc != 0 && barrier->open_rc == 0)
            barrier->open_rc = append_rc;
        if (barrier->open_group_size >= max_group)
            pthread_cond_broadcast(&barrier->cv);
        while (barrier->completed_epoch < epoch)
            (void)epoch_barrier_wait(barrier, &barrier_scope, __func__);
        int result = barrier->completed_rc;
        if (ctx) {
            ctx->barrier_rc = result;
            ctx->epoch_log_sync_count = 0;
            ctx->group_size = barrier->completed_group_size;
            ctx->group_epoch = epoch;
            ctx->group_role = "follower";
            ctx->sync_primitive = "none";
            uint64_t end_ns = pqc_publication_now_ns();
            ctx->group_wait_ns = end_ns >= start_ns ? end_ns - start_ns : 0;
            ctx->group_open_wait_ns = 0;
            ctx->group_sync_ns = 0;
            ctx->group_completion_wait_ns =
                end_ns >= start_ns ? end_ns - start_ns : 0;
        }
        (void)epoch_barrier_unlock(barrier, &barrier_scope, __func__);
        return result;
    }

    while (barrier->active)
        (void)epoch_barrier_wait(barrier, &barrier_scope, __func__);

    uint64_t epoch = barrier->next_epoch++;
    barrier->open = 1;
    barrier->open_epoch = epoch;
    barrier->open_group_size = 1;
    barrier->open_group_bytes = bytes;
    barrier->open_force_syncfs = ctx && ctx->force_syncfs;
    barrier->open_rc = append_rc;

    if (max_wait_ns > 0) {
        struct timespec deadline;
        rc = pqc_epoch_deadline_from_now(max_wait_ns, &deadline);
        if (rc != 0) {
            barrier->open = 0;
            pthread_cond_broadcast(&barrier->cv);
            (void)epoch_barrier_unlock(barrier, &barrier_scope, __func__);
            return rc;
        }
        while (barrier->open_group_size < max_group) {
            rc = epoch_barrier_timedwait(barrier, &barrier_scope, __func__,
                                         &deadline);
            if (rc == ETIMEDOUT)
                break;
            if (rc != 0) {
                barrier->open = 0;
                pthread_cond_broadcast(&barrier->cv);
                (void)epoch_barrier_unlock(barrier, &barrier_scope,
                                           __func__);
                return -rc;
            }
        }
    }

    barrier->open = 0;
    barrier->active = 1;
    barrier->active_epoch = epoch;
    barrier->completed_group_size = barrier->open_group_size;
    barrier->completed_group_bytes = barrier->open_group_bytes;
    barrier->completed_force_syncfs = barrier->open_force_syncfs;
    barrier->open_force_syncfs = 0;
    uint32_t group_size = barrier->completed_group_size;
    int force_syncfs = barrier->completed_force_syncfs;
    int group_rc = barrier->open_rc;
    pthread_cond_broadcast(&barrier->cv);
    (void)epoch_barrier_unlock(barrier, &barrier_scope, __func__);

    const char *primitive = "none";
    int sync_rc = group_rc;
    uint64_t sync_start_ns = pqc_publication_now_ns();
    if (sync_rc == 0)
        sync_rc = pqc_epoch_sync_fd(fd, group_size, force_syncfs,
                                    &primitive);
    uint64_t sync_end_ns = pqc_publication_now_ns();

    rc = epoch_barrier_lock(barrier, &barrier_scope, __func__);
    if (rc != 0)
        return -rc;
    barrier->completed_rc = sync_rc;
    barrier->completed_epoch = epoch;
    barrier->active = 0;
    pthread_cond_broadcast(&barrier->cv);
    if (ctx) {
        ctx->barrier_rc = sync_rc;
        ctx->epoch_log_sync_count = sync_rc == 0 ? 1U : 0U;
        ctx->group_size = group_size;
        ctx->group_epoch = epoch;
        ctx->group_role = "leader";
        ctx->sync_primitive = primitive;
        uint64_t end_ns = sync_end_ns;
        ctx->group_wait_ns = end_ns >= start_ns ? end_ns - start_ns : 0;
        ctx->group_open_wait_ns =
            sync_start_ns >= start_ns ? sync_start_ns - start_ns : 0;
        ctx->group_sync_ns =
            sync_end_ns >= sync_start_ns ? sync_end_ns - sync_start_ns : 0;
        ctx->group_completion_wait_ns = 0;
    }
    (void)epoch_barrier_unlock(barrier, &barrier_scope, __func__);
    return sync_rc;
}

static uint64_t pqc_publication_request_bytes(
    const pqc_strict_publish_request_t *req)
{
    uint64_t total = 0;
    if (!req || !req->blocks)
        return 0;
    for (size_t i = 0; i < req->block_count; ++i)
        total += req->blocks[i].length;
    return total;
}

static void pqc_publication_trace_write_line(const char *line, size_t len)
{
    (void)pqc_trace_sink_write_env(&g_publication_trace_sink,
                                   "PQC_PUBLICATION_TRACE_PATH",
                                   line, len);
}

void pqc_publication_trace_shutdown(void)
{
    pqc_trace_sink_close(&g_publication_trace_sink);
}

static void pqc_publication_trace_decision(
    const char *mode, int rc, const pqc_strict_publish_request_t *req,
    uint64_t elapsed_ns, uint32_t data_fsync_count,
    uint32_t journal_fsync_count, uint32_t epoch_log_fsync_count)
{
    uint32_t sync_count = data_fsync_count + journal_fsync_count +
                          epoch_log_fsync_count;
    char line[1024];
    int n = snprintf(line, sizeof(line),
                     "{\"event\":\"publication_dispatch\",\"mode\":\"%s\","
                     "\"rc\":%d,\"elapsed_ns\":%llu,\"block_count\":%zu,"
                     "\"payload_bytes\":%llu,\"sync_count\":%u,"
                     "\"data_fsync_count\":%u,\"journal_fsync_count\":%u,"
                     "\"epoch_log_fsync_count\":%u}\n",
                     mode ? mode : "invalid", rc,
                     (unsigned long long)elapsed_ns,
                     req ? req->block_count : 0,
                     (unsigned long long)pqc_publication_request_bytes(req),
                     sync_count, data_fsync_count, journal_fsync_count,
                     epoch_log_fsync_count);
    if (n > 0 && (size_t)n < sizeof(line))
        pqc_publication_trace_write_line(line, (size_t)n);
}

static void pqc_publication_trace_epoch_append(
    const pqc_epoch_publish_context_t *ctx, int rc)
{
    char line[8192];
    int n = snprintf(line, sizeof(line),
                     "{\"event\":\"epoch_redo_log_append\","
                     "\"mode\":\"epoch-redo-log\","
                     "\"rc\":%d,\"record_count\":%zu,\"barrier_rc\":%d,"
                     "\"epoch_log_sync_count\":%u,\"group_size\":%u,"
                     "\"group_epoch\":%llu,\"group_role\":\"%s\","
                     "\"group_wait_ns\":%llu,\"sync_primitive\":\"%s\","
                     "\"group_open_wait_ns\":%llu,"
                     "\"group_sync_ns\":%llu,"
                     "\"group_completion_wait_ns\":%llu,"
                     "\"force_syncfs\":%d,"
                     "\"data_fsync_fallback_count\":%d,"
                     "\"log_path\":\"%s\"}\n",
                     rc, ctx ? ctx->record_count : 0,
                     ctx ? ctx->barrier_rc : 0,
                     ctx ? ctx->epoch_log_sync_count : 0,
                     ctx ? ctx->group_size : 0,
                     (unsigned long long)(ctx ? ctx->group_epoch : 0),
                     ctx && ctx->group_role ? ctx->group_role : "unknown",
                     (unsigned long long)(ctx ? ctx->group_wait_ns : 0),
                     ctx && ctx->sync_primitive ? ctx->sync_primitive : "unknown",
                     (unsigned long long)(ctx ? ctx->group_open_wait_ns : 0),
                     (unsigned long long)(ctx ? ctx->group_sync_ns : 0),
                     (unsigned long long)(ctx ? ctx->group_completion_wait_ns : 0),
                     ctx ? ctx->force_syncfs : 0,
                     ctx ? ctx->data_fsync_fallback_count : 0,
                     ctx ? ctx->log_path : "");
    if (n > 0 && (size_t)n < sizeof(line))
        pqc_publication_trace_write_line(line, (size_t)n);
}

static int pqc_epoch_publish_after_data_fsync(
    const pqc_strict_publish_request_t *req, void *opaque)
{
    pqc_epoch_publish_context_t *ctx =
        (pqc_epoch_publish_context_t *)opaque;
    if (!req || !ctx || !req->marker_path || !req->blocks)
        return -EINVAL;
    int rc = 0;
    if (req->epoch_log_path && req->epoch_log_path[0] != '\0') {
        int n = snprintf(ctx->log_path, sizeof(ctx->log_path), "%s",
                         req->epoch_log_path);
        if (n < 0 || (size_t)n >= sizeof(ctx->log_path))
            return -ENAMETOOLONG;
    } else {
        rc = pqc_sidecar_path(ctx->log_path, sizeof(ctx->log_path),
                              req->marker_path, ".pqcepoch");
        if (rc != 0)
            return rc;
    }

    int fd = req->epoch_log_fd;
    if (fd >= 0) {
        ctx->owns_log_fd = 0;
    } else {
        fd = open(ctx->log_path,
                  O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
        if (fd < 0)
            return -errno;
        ctx->owns_log_fd = 1;
    }
    ctx->log_fd = fd;
    ctx->file_id = req->file_id;
    ctx->payload_bytes = pqc_publication_request_bytes(req);

    ctx->force_syncfs =
        pqc_epoch_request_same_syncfs_domain(req, fd) &&
        pqc_epoch_group_max_from_config() > 1 &&
        pqc_epoch_group_wait_ns_from_config() > 0;
    ctx->data_fsync_fallback_count = 0;
    if (!ctx->defer_durability && !ctx->force_syncfs) {
        rc = pqc_durability_fdatasync(
            req->data_fd, PQC_DURABILITY_SITE_DATA_SIDECAR);
        if (rc == 0)
            ctx->data_fsync_fallback_count = 1;
    }

    if (req->block_count > PQC_WRITEBACK_MAX_BLOCKS)
        rc = -E2BIG;

    pqc_epoch_log_record_t records[PQC_WRITEBACK_MAX_BLOCKS + 1U];
    size_t record_count = 0;
    for (size_t bi = 0; rc == 0 && bi < req->block_count; ++bi) {
        const pqc_crypto_block_desc_t *block = &req->blocks[bi];
        pqc_epoch_log_record_t *record = &records[record_count++];
        *record = (pqc_epoch_log_record_t) {
            .record_type = PQC_EPOCH_LOG_RECORD_BLOCK,
            .flags = 0,
            .algorithm_id = req->algorithm_id,
            .epoch = req->reserved_generation,
            .sequence = block->generation,
            .file_id = req->file_id,
            .logical_block = block->block,
            .generation = block->generation,
            .ciphertext_offset = block->ciphertext_offset,
            .logical_size_after = req->final_size,
            .plaintext_length = block->length,
        };
        memcpy(record->tag, block->tag, sizeof(record->tag));
    }

    if (rc == 0) {
        records[record_count++] = (pqc_epoch_log_record_t) {
            .record_type = PQC_EPOCH_LOG_RECORD_COMMIT,
            .flags = 0,
            .algorithm_id = req->algorithm_id,
            .epoch = req->reserved_generation,
            .sequence = req->reserved_generation,
            .file_id = req->file_id,
            .logical_block = 0,
            .generation = req->reserved_generation,
            .ciphertext_offset = 0,
            .logical_size_after = req->final_size,
            .plaintext_length = 0,
        };
        rc = pqc_epoch_log_append_records_fd(fd, records, record_count);
        if (rc == 0)
            ctx->record_count = record_count;
    }
    if (rc == 0 && req->epoch_log_dirty && req->epoch_log_dirty_epoch) {
        ++*req->epoch_log_dirty_epoch;
        ctx->epoch_dirty_epoch = *req->epoch_log_dirty_epoch;
        *req->epoch_log_dirty = 1;
    }

    if (rc != 0) {
        int close_rc = 0;
        if (ctx->owns_log_fd)
            close_rc = close(ctx->log_fd);
        ctx->log_fd = -1;
        if (close_rc != 0 && rc == 0)
            rc = -errno;
        pqc_publication_trace_epoch_append(ctx, rc);
        ctx->trace_emitted = 1;
    }
    return rc;
}

static int pqc_epoch_publish_sync_barrier(void *opaque)
{
    pqc_epoch_publish_context_t *ctx =
        (pqc_epoch_publish_context_t *)opaque;
    if (!ctx || ctx->log_fd < 0)
        return -EINVAL;
    return pqc_epoch_group_barrier_wait(ctx->log_fd, ctx->file_id,
                                        ctx->payload_bytes, 0, ctx);
}

static int pqc_epoch_publish_after_metadata(
    const pqc_strict_publish_request_t *req, void *opaque)
{
    (void)req;
    pqc_epoch_publish_context_t *ctx =
        (pqc_epoch_publish_context_t *)opaque;
    if (!ctx || ctx->log_fd < 0)
        return -EINVAL;

    int rc = 0;
    if (ctx->defer_durability) {
        ctx->barrier_rc = 0;
        ctx->epoch_log_sync_count = 0;
        ctx->group_size = 0;
        ctx->sync_primitive = "deferred";
    } else if (ctx->force_syncfs) {
        if (ctx->windowed_file_anchor)
            rc = pqc_anchor_worker_flush_windowed_external_sync(
                pqc_epoch_publish_sync_barrier, ctx);
        else
            rc = pqc_anchor_worker_flush_now_external_sync(
                pqc_epoch_publish_sync_barrier, ctx);
    } else {
        if (ctx->windowed_file_anchor)
            rc = pqc_anchor_worker_flush_windowed_external_sync(
                pqc_epoch_publish_sync_barrier, ctx);
        else
            rc = pqc_anchor_worker_flush_now();
        if (rc == 0)
            rc = pqc_epoch_publish_sync_barrier(ctx);
    }
    int close_rc = 0;
    if (ctx->owns_log_fd)
        close_rc = close(ctx->log_fd);
    ctx->log_fd = -1;
    if (rc == 0 && close_rc != 0)
        rc = -errno;
    if (rc == 0 && !ctx->defer_durability &&
        req->epoch_log_dirty && req->epoch_log_dirty_epoch &&
        *req->epoch_log_dirty_epoch == ctx->epoch_dirty_epoch)
        *req->epoch_log_dirty = 0;
    pqc_publication_trace_epoch_append(ctx, rc);
    ctx->trace_emitted = 1;
    return rc;
}

int pqc_publication_dispatch_commit(const pqc_strict_publish_request_t *req)
{
    uint64_t start_ns = pqc_publication_now_ns();
    pqc_publication_mode_t mode = PQC_PUBLICATION_MODE_STRICT;
    int rc = pqc_publication_mode_from_config(&mode);
    if (rc != 0) {
        uint64_t end_ns = pqc_publication_now_ns();
        pqc_publication_trace_decision("invalid", rc, req,
                                       end_ns >= start_ns ? end_ns - start_ns : 0,
                                       0, 0, 0);
        return rc;
    }
    if (mode == PQC_PUBLICATION_MODE_STRICT) {
        rc = pqc_strict_publish_commit(req);
        uint64_t end_ns = pqc_publication_now_ns();
        pqc_publication_trace_decision(
            pqc_publication_mode_name(mode), rc, req,
            end_ns >= start_ns ? end_ns - start_ns : 0,
            rc == 0 ? 1U : 0U, rc == 0 ? 1U : 0U, 0);
        return rc;
    }
    if (mode == PQC_PUBLICATION_MODE_EPOCH_REDO_LOG) {
        pqc_epoch_publish_context_t ctx;
        memset(&ctx, 0, sizeof(ctx));
        ctx.log_fd = -1;
        ctx.owns_log_fd = 0;
        ctx.defer_durability =
            (req->defer_data_fsync || req->defer_journal_fsync);
        ctx.windowed_file_anchor =
            pqc_anchor_worker_windowed_file_anchor_enabled();
        pqc_strict_publish_request_t epoch_req = *req;
        epoch_req.after_data_fsync = pqc_epoch_publish_after_data_fsync;
        epoch_req.after_data_fsync_opaque = &ctx;
        epoch_req.after_metadata_publish = pqc_epoch_publish_after_metadata;
        epoch_req.after_metadata_publish_opaque = &ctx;
        epoch_req.skip_data_fsync = 1;
        epoch_req.skip_journal_append = 1;
        epoch_req.skip_journal_fsync = 1;
        rc = pqc_strict_publish_commit(&epoch_req);
        if (ctx.log_fd >= 0) {
            int close_rc = 0;
            if (ctx.owns_log_fd)
                close_rc = close(ctx.log_fd);
            ctx.log_fd = -1;
            if (rc == 0 && close_rc != 0)
                rc = -errno;
        }
        if (!ctx.trace_emitted) {
            pqc_publication_trace_epoch_append(&ctx, rc);
            ctx.trace_emitted = 1;
        }
        uint64_t end_ns = pqc_publication_now_ns();
        pqc_publication_trace_decision(
            pqc_publication_mode_name(mode), rc, req,
            end_ns >= start_ns ? end_ns - start_ns : 0,
            (rc == 0 && ctx.data_fsync_fallback_count > 0) ? 1U : 0U, 0,
            (rc == 0 && ctx.barrier_rc == 0)
                ? ctx.epoch_log_sync_count : 0U);
        return rc;
    }

    rc = -ENOTSUP;
    uint64_t end_ns = pqc_publication_now_ns();
    pqc_publication_trace_decision(pqc_publication_mode_name(mode), rc, req,
                                   end_ns >= start_ns ? end_ns - start_ns : 0,
                                   0, 0, 0);
    return rc;
}
