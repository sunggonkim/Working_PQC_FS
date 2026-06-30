#include "pqc_rekey.h"

#include "cuda_pqc.h"
#include "pqc_admission.h"
#include "pqc_config.h"
#include "pqc_fd_context.h"
#include "pqc_keyring.h"
#include "pqc_lock_profile.h"
#include "pqc_metrics.h"
#include "pqc_plane_trace.h"
#include "pqc_qos.h"
#include "pqc_scheduler.h"

#include <errno.h>
#include <limits.h>
#include <openssl/crypto.h>
#include <openssl/rand.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define PQC_KEY_ROTATION_INTERVAL_S 0
#define PQC_REKEY_QUEUE_MAX 4096

typedef struct {
    int      fd_list[PQC_REKEY_QUEUE_MAX];
    unsigned char queued[PQC_MAX_FD];
    size_t   head;
    size_t   tail;
    size_t   count;
    pqc_rekey_queue_stats_t stats;
    pthread_mutex_t lock;
    pthread_cond_t  cond;
} pqc_rekey_queue_t;

static pqc_rekey_queue_t g_rekey_queue = {
    .queued = {0},
    .head = 0,
    .tail = 0,
    .count = 0,
    .stats = {0},
    .lock = PTHREAD_MUTEX_INITIALIZER,
    .cond = PTHREAD_COND_INITIALIZER
};

static atomic_int g_rekey_stop = ATOMIC_VAR_INIT(1);
static atomic_int g_force_rekey_on_write = ATOMIC_VAR_INIT(-1);
static atomic_int g_write_trigger_enabled = ATOMIC_VAR_INIT(-1);
static pthread_mutex_t g_rekey_lifecycle_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_t g_rekey_thread;
static int g_rekey_thread_started = 0;
static int g_rekey_thread_joining = 0;

typedef struct {
    OQS_KEM *kem;
    uint8_t *public_key;
    size_t   public_key_len;
    uint8_t *ct_batch;
    size_t   ct_batch_capacity;
    uint8_t *ss_batch;
    size_t   ss_batch_capacity;
    uint8_t *seed_batch;
    size_t   seed_batch_capacity;
} pqc_rekey_worker_ctx_t;

static pqc_rekey_worker_ctx_t *g_rekey_worker_ctx = NULL;

static void rekey_lifecycle_lock(pqc_lock_profile_scope_t *scope,
                                 const char *site)
{
    (void)pqc_profiled_mutex_lock(&g_rekey_lifecycle_lock,
                                  "rekey_lifecycle_lock", site, scope);
}

static void rekey_lifecycle_unlock(pqc_lock_profile_scope_t *scope,
                                   const char *site)
{
    (void)pqc_profiled_mutex_unlock(&g_rekey_lifecycle_lock,
                                    "rekey_lifecycle_lock", site, scope);
}

static void rekey_queue_lock(pqc_lock_profile_scope_t *scope,
                             const char *site)
{
    (void)pqc_profiled_mutex_lock(&g_rekey_queue.lock,
                                  "rekey_queue_lock", site, scope);
}

static void rekey_queue_unlock(pqc_lock_profile_scope_t *scope,
                               const char *site)
{
    (void)pqc_profiled_mutex_unlock(&g_rekey_queue.lock,
                                    "rekey_queue_lock", site, scope);
}

static int rekey_queue_timedwait(pqc_lock_profile_scope_t *scope,
                                 const char *site,
                                 const struct timespec *deadline)
{
    return pqc_profiled_cond_timedwait(&g_rekey_queue.cond,
                                       &g_rekey_queue.lock,
                                       "rekey_queue_lock", site, scope,
                                       deadline);
}

typedef struct {
    char     marker_path[4096];
    uint8_t  ss[64];
    size_t   ss_len;
    uint64_t file_id;
    uint64_t key_epoch;
} pqc_rekey_snapshot_t;

static void rekey_snapshot_clear(pqc_rekey_snapshot_t *snapshot)
{
    if (!snapshot)
        return;
    OPENSSL_cleanse(snapshot->ss, sizeof(snapshot->ss));
    memset(snapshot, 0, sizeof(*snapshot));
}

static void rekey_worker_ctx_free(pqc_rekey_worker_ctx_t *ctx)
{
    if (!ctx)
        return;
    if (ctx->public_key) {
        OPENSSL_cleanse(ctx->public_key, ctx->public_key_len);
        free(ctx->public_key);
    }
    if (ctx->ct_batch) {
        OPENSSL_cleanse(ctx->ct_batch, ctx->ct_batch_capacity);
        free(ctx->ct_batch);
    }
    if (ctx->ss_batch) {
        OPENSSL_cleanse(ctx->ss_batch, ctx->ss_batch_capacity);
        free(ctx->ss_batch);
    }
    if (ctx->seed_batch) {
        OPENSSL_cleanse(ctx->seed_batch, ctx->seed_batch_capacity);
        free(ctx->seed_batch);
    }
    OPENSSL_cleanse(ctx, sizeof(*ctx));
    free(ctx);
}

static uint8_t *rekey_scratch_alloc(size_t capacity)
{
    if (capacity == 0)
        return NULL;
    return malloc(capacity);
}

static pqc_rekey_worker_ctx_t *rekey_worker_ctx_new(
    OQS_KEM *kem,
    const uint8_t *public_key)
{
    if (!kem || !public_key || kem->length_public_key == 0 ||
        kem->length_ciphertext == 0 || kem->length_shared_secret == 0)
        return NULL;
    if (kem->length_ciphertext > SIZE_MAX / PQC_REKEY_QUEUE_MAX ||
        kem->length_shared_secret > SIZE_MAX / PQC_REKEY_QUEUE_MAX ||
        MLKEM768_ENCAPS_SEED_BYTES > SIZE_MAX / PQC_REKEY_QUEUE_MAX)
        return NULL;

    pqc_rekey_worker_ctx_t *ctx = calloc(1, sizeof(*ctx));
    if (!ctx)
        return NULL;
    ctx->kem = kem;
    ctx->public_key = malloc(kem->length_public_key);
    if (!ctx->public_key) {
        rekey_worker_ctx_free(ctx);
        return NULL;
    }
    memcpy(ctx->public_key, public_key, kem->length_public_key);
    ctx->public_key_len = kem->length_public_key;
    ctx->ct_batch_capacity = kem->length_ciphertext * PQC_REKEY_QUEUE_MAX;
    ctx->ss_batch_capacity = kem->length_shared_secret * PQC_REKEY_QUEUE_MAX;
    ctx->seed_batch_capacity =
        MLKEM768_ENCAPS_SEED_BYTES * PQC_REKEY_QUEUE_MAX;
    ctx->ct_batch = rekey_scratch_alloc(ctx->ct_batch_capacity);
    ctx->ss_batch = rekey_scratch_alloc(ctx->ss_batch_capacity);
    ctx->seed_batch = rekey_scratch_alloc(ctx->seed_batch_capacity);
    if (!ctx->ct_batch || !ctx->ss_batch || !ctx->seed_batch) {
        rekey_worker_ctx_free(ctx);
        return NULL;
    }
    return ctx;
}

static int rekey_snapshot_take_locked(const pqc_fd_ctx_t *ctx,
                                      pqc_rekey_snapshot_t *snapshot)
{
    if (!ctx || !snapshot || !ctx->valid)
        return 0;
    if (ctx->ss_len == 0 || ctx->ss_len > sizeof(snapshot->ss))
        return -EINVAL;

    memset(snapshot, 0, sizeof(*snapshot));
    memcpy(snapshot->marker_path, ctx->marker_path,
           sizeof(snapshot->marker_path) - 1);
    memcpy(snapshot->ss, ctx->ss, ctx->ss_len);
    snapshot->ss_len = ctx->ss_len;
    snapshot->file_id = ctx->file_id;
    snapshot->key_epoch = ctx->key_epoch;
    return 1;
}

static int rekey_snapshot_still_current_locked(const pqc_fd_ctx_t *ctx,
                                               const pqc_rekey_snapshot_t *snapshot)
{
    if (!ctx || !snapshot || !ctx->valid)
        return 0;
    return ctx->file_id == snapshot->file_id &&
           ctx->key_epoch == snapshot->key_epoch &&
           ctx->ss_len == snapshot->ss_len &&
           memcmp(ctx->ss, snapshot->ss, snapshot->ss_len) == 0 &&
           strncmp(ctx->marker_path, snapshot->marker_path,
                   sizeof(snapshot->marker_path)) == 0;
}

static int rekey_stop_requested(void)
{
    return atomic_load_explicit(&g_rekey_stop, memory_order_acquire) != 0;
}

static void rekey_stop_set(int stop)
{
    atomic_store_explicit(&g_rekey_stop, stop ? 1 : 0, memory_order_release);
}

static void rekey_queue_reset_locked(void)
{
    memset(g_rekey_queue.queued, 0, sizeof(g_rekey_queue.queued));
    g_rekey_queue.head = 0;
    g_rekey_queue.tail = 0;
    g_rekey_queue.count = 0;
}

static void rekey_queue_stats_reset_locked(void)
{
    memset(&g_rekey_queue.stats, 0, sizeof(g_rekey_queue.stats));
}

static pqc_rekey_enqueue_status_t rekey_queue_push_locked(int fd)
{
    int idx = pqc_fd_context_index_for_fd(fd);
    if (idx < 0 || (size_t)idx >= sizeof(g_rekey_queue.queued)) {
        g_rekey_queue.stats.invalid_fd++;
        return PQC_REKEY_ENQUEUE_INVALID_FD;
    }
    if (g_rekey_queue.queued[idx]) {
        g_rekey_queue.stats.duplicate++;
        return PQC_REKEY_ENQUEUE_DUPLICATE;
    }
    if (g_rekey_queue.count >= PQC_REKEY_QUEUE_MAX) {
        g_rekey_queue.stats.queue_full++;
        return PQC_REKEY_ENQUEUE_FULL;
    }

    g_rekey_queue.fd_list[g_rekey_queue.tail] = fd;
    g_rekey_queue.tail = (g_rekey_queue.tail + 1) % PQC_REKEY_QUEUE_MAX;
    g_rekey_queue.count++;
    g_rekey_queue.queued[idx] = 1;
    g_rekey_queue.stats.accepted++;
    pthread_cond_signal(&g_rekey_queue.cond);
    return PQC_REKEY_ENQUEUE_ACCEPTED;
}

static int rekey_queue_pop_locked(int *fd)
{
    if (!fd || g_rekey_queue.count == 0)
        return 0;
    int popped = g_rekey_queue.fd_list[g_rekey_queue.head];
    g_rekey_queue.head = (g_rekey_queue.head + 1) % PQC_REKEY_QUEUE_MAX;
    g_rekey_queue.count--;
    *fd = popped;
    return 1;
}

static void rekey_queue_batch_done(const int *fds, size_t count)
{
    if (!fds || count == 0)
        return;
    pqc_lock_profile_scope_t queue_scope;
    rekey_queue_lock(&queue_scope, __func__);
    for (size_t i = 0; i < count; i++) {
        int idx = pqc_fd_context_index_for_fd(fds[i]);
        if (idx >= 0 && (size_t)idx < sizeof(g_rekey_queue.queued))
            g_rekey_queue.queued[idx] = 0;
    }
    rekey_queue_unlock(&queue_scope, __func__);
}

static void rekey_queue_collect_locked(int *fds,
                                       size_t *batch_size,
                                       size_t configured_max_batch)
{
    while (g_rekey_queue.count > 0 &&
           *batch_size < configured_max_batch) {
        if (!rekey_queue_pop_locked(&fds[*batch_size]))
            break;
        (*batch_size)++;
    }
}

static void rekey_collect_wait_deadline(struct timespec *deadline,
                                        long collect_ms)
{
    clock_gettime(CLOCK_REALTIME, deadline);
    deadline->tv_sec += collect_ms / 1000;
    deadline->tv_nsec += (collect_ms % 1000) * 1000000L;
    if (deadline->tv_nsec >= 1000000000L) {
        deadline->tv_sec++;
        deadline->tv_nsec -= 1000000000L;
    }
}

int pqc_rekey_rotation_interval_s(void)
{
    static int cached = -2;
    if (cached != -2)
        return cached;

    long v = pqc_config_long_or_default("PQC_KEY_ROTATION_INTERVAL_S",
                                        PQC_KEY_ROTATION_INTERVAL_S);
    if (v < 0)
        v = 0;
    cached = (int)v;
    return cached;
}

int pqc_rekey_force_on_write_enabled(void)
{
    int cached = atomic_load_explicit(&g_force_rekey_on_write,
                                      memory_order_acquire);
    if (cached >= 0)
        return cached;

    int enabled = pqc_config_present("PQC_FORCE_REKEY_ON_WRITE") ? 1 : 0;
    int expected = -1;
    if (atomic_compare_exchange_strong_explicit(
            &g_force_rekey_on_write, &expected, enabled,
            memory_order_release, memory_order_acquire))
        return enabled;
    return atomic_load_explicit(&g_force_rekey_on_write,
                                memory_order_acquire);
}

int pqc_rekey_write_trigger_enabled(void)
{
    int cached = atomic_load_explicit(&g_write_trigger_enabled,
                                      memory_order_acquire);
    if (cached >= 0)
        return cached;

    int enabled = pqc_rekey_force_on_write_enabled() ||
                  pqc_rekey_rotation_interval_s() > 0;
    int expected = -1;
    if (atomic_compare_exchange_strong_explicit(
            &g_write_trigger_enabled, &expected, enabled ? 1 : 0,
            memory_order_release, memory_order_acquire))
        return enabled ? 1 : 0;
    return atomic_load_explicit(&g_write_trigger_enabled,
                                memory_order_acquire);
}

pqc_rekey_enqueue_status_t pqc_rekey_queue_push(int fd)
{
    pqc_rekey_enqueue_status_t status;
    pqc_lock_profile_scope_t queue_scope;
    rekey_queue_lock(&queue_scope, __func__);
    status = rekey_queue_push_locked(fd);
    rekey_queue_unlock(&queue_scope, __func__);
    return status;
}

void pqc_rekey_queue_stats_snapshot(pqc_rekey_queue_stats_t *out)
{
    if (!out)
        return;
    pqc_lock_profile_scope_t queue_scope;
    rekey_queue_lock(&queue_scope, __func__);
    *out = g_rekey_queue.stats;
    rekey_queue_unlock(&queue_scope, __func__);
}

int pqc_rekey_queue_accounting_selftest(void)
{
    (void)pqc_rekey_worker_stop();

    pqc_lock_profile_scope_t queue_scope;
    rekey_queue_lock(&queue_scope, __func__);
    rekey_queue_reset_locked();
    rekey_queue_stats_reset_locked();

    pqc_rekey_enqueue_status_t invalid = rekey_queue_push_locked(-1);
    pqc_rekey_enqueue_status_t accepted = rekey_queue_push_locked(7);
    pqc_rekey_enqueue_status_t duplicate = rekey_queue_push_locked(7);

    rekey_queue_reset_locked();
    g_rekey_queue.count = PQC_REKEY_QUEUE_MAX;
    g_rekey_queue.head = 0;
    g_rekey_queue.tail = 0;
    memset(g_rekey_queue.queued, 0, sizeof(g_rekey_queue.queued));
    pqc_rekey_enqueue_status_t full = rekey_queue_push_locked(8);

    pqc_rekey_queue_stats_t stats = g_rekey_queue.stats;
    int ok = invalid == PQC_REKEY_ENQUEUE_INVALID_FD &&
             accepted == PQC_REKEY_ENQUEUE_ACCEPTED &&
             duplicate == PQC_REKEY_ENQUEUE_DUPLICATE &&
             full == PQC_REKEY_ENQUEUE_FULL &&
             stats.accepted == 1 &&
             stats.duplicate == 1 &&
             stats.invalid_fd == 1 &&
             stats.queue_full == 1;

    rekey_queue_reset_locked();
    rekey_queue_stats_reset_locked();
    rekey_queue_unlock(&queue_scope, __func__);
    return ok ? 0 : -1;
}

static void *rekey_worker_main(void *arg)
{
    pqc_rekey_worker_ctx_t *worker_ctx = (pqc_rekey_worker_ctx_t *)arg;
    OQS_KEM *kem = worker_ctx ? worker_ctx->kem : NULL;
    const uint8_t *public_key = worker_ctx ? worker_ctx->public_key : NULL;
    int fds[PQC_REKEY_QUEUE_MAX];
    while (kem && public_key && !rekey_stop_requested()) {
        long configured_max_batch =
            pqc_config_positive_long_or_default("PQC_REKEY_BATCH_MAX", 64);
        if (configured_max_batch > PQC_REKEY_QUEUE_MAX)
            configured_max_batch = PQC_REKEY_QUEUE_MAX;
        pqc_lock_profile_scope_t queue_scope;
        rekey_queue_lock(&queue_scope, __func__);
        while (g_rekey_queue.count == 0 && !rekey_stop_requested()) {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_sec += 1;
            (void)rekey_queue_timedwait(&queue_scope, __func__, &ts);
        }
        if (rekey_stop_requested()) {
            rekey_queue_unlock(&queue_scope, __func__);
            break;
        }
        size_t batch_size = 0;
        rekey_queue_collect_locked(fds, &batch_size,
                                   (size_t)configured_max_batch);
        rekey_queue_unlock(&queue_scope, __func__);

        long collect_ms =
            pqc_config_positive_long_or_default("PQC_REKEY_BATCH_COLLECT_MS", 0);
        if (collect_ms > 0 && batch_size < (size_t)configured_max_batch) {
            rekey_queue_lock(&queue_scope, __func__);
            if (g_rekey_queue.count == 0 && !rekey_stop_requested()) {
                struct timespec deadline;
                rekey_collect_wait_deadline(&deadline, collect_ms);
                while (g_rekey_queue.count == 0 &&
                       !rekey_stop_requested()) {
                    int wait_rc = rekey_queue_timedwait(
                        &queue_scope, __func__, &deadline);
                    if (wait_rc == ETIMEDOUT)
                        break;
                }
            }
            rekey_queue_collect_locked(fds, &batch_size,
                                       (size_t)configured_max_batch);
            rekey_queue_unlock(&queue_scope, __func__);
        }

        if (rekey_stop_requested()) {
            rekey_queue_batch_done(fds, batch_size);
            break;
        }
        if (batch_size == 0)
            continue;

        double t0 = pqc_metrics_time_us();
        double load = 0.0;
        if (getloadavg(&load, 1) < 0)
            load = 0.0;
        double gpu_ewma = pqc_qos_gpu_load_ewma_read();
        const size_t key_work_bytes =
            batch_size * (kem->length_ciphertext +
                          kem->length_shared_secret);
        int min_batch =
            (int)pqc_config_long_legacy_or_default("PQC_GPU_MIN_BATCH", 16);
        pqc_job_target_t target = PQC_JOB_CPU;
        pqc_admission_context_t admission_ctx;
        memset(&admission_ctx, 0, sizeof(admission_ctx));
        admission_ctx.batch_count = batch_size;
        admission_ctx.bytes_total = key_work_bytes;
        admission_ctx.batch_age_ns = (uint64_t)collect_ms * 1000000ULL;
        admission_ctx.gpu_kernel_est_ns =
            (uint64_t)pqc_config_positive_long_or_default(
                "PQC_REKEY_GPU_KERNEL_EST_NS", 250000L);
        admission_ctx.gpu_h2d_staging_ns = key_work_bytes;
        admission_ctx.gpu_d2h_staging_ns =
            batch_size * kem->length_shared_secret;
        admission_ctx.cpu_queue_depth = batch_size;
        admission_ctx.gpu_queue_depth = pqc_scheduler_gpu_inflight_jobs();
        admission_ctx.cpu_load_avg = load;
        admission_ctx.gpu_load_avg = gpu_ewma;
        admission_ctx.ai_inference_deadline_ns =
            (uint64_t)pqc_config_positive_long_or_default(
                "PQC_REKEY_DEADLINE_NS", 10000000L);
        admission_ctx.uma_migration_cost_ns = 0;
        admission_ctx.uma_migration_bytes_est = 0;
        if ((int)batch_size >= min_batch && pqc_admit(&admission_ctx) == 0)
            target = admission_ctx.chosen_target;

        size_t ct_required = 0;
        size_t ss_required = 0;
        size_t seed_required = 0;
        uint8_t *ct_batch = worker_ctx->ct_batch;
        uint8_t *ss_batch = worker_ctx->ss_batch;
        uint8_t *seed_batch = worker_ctx->seed_batch;
        int success = 0;
        int gpu_used = 0;
        int scratch_ready =
            batch_size <= PQC_REKEY_QUEUE_MAX &&
            batch_size <= SIZE_MAX / kem->length_ciphertext &&
            batch_size <= SIZE_MAX / kem->length_shared_secret;
        if (scratch_ready) {
            ct_required = kem->length_ciphertext * batch_size;
            ss_required = kem->length_shared_secret * batch_size;
            scratch_ready = ct_batch && ss_batch &&
                            ct_required <= worker_ctx->ct_batch_capacity &&
                            ss_required <= worker_ctx->ss_batch_capacity;
        }
        int seed_ready =
            batch_size <= PQC_REKEY_QUEUE_MAX &&
            batch_size <= SIZE_MAX / MLKEM768_ENCAPS_SEED_BYTES;
        if (seed_ready) {
            seed_required = MLKEM768_ENCAPS_SEED_BYTES * batch_size;
            seed_ready = seed_batch &&
                         seed_required <= worker_ctx->seed_batch_capacity &&
                         seed_required <= (size_t)INT_MAX;
        }

        if (scratch_ready) {
            if (target == PQC_JOB_GPU && seed_ready &&
                skim_cuda_pqc_available()) {
                pqc_scheduler_gpu_admit((uint32_t)(key_work_bytes > UINT32_MAX ?
                                                   UINT32_MAX : key_work_bytes));
                int rc = RAND_bytes(seed_batch, (int)seed_required) == 1 ?
                    0 : -1;
                int burn_iters =
                    (int)pqc_config_long_legacy_or_default("PQC_GPU_BURN_ITERS", 1);
                if (rc == 0) {
                    for (int iter = 0; iter < burn_iters; iter++) {
                        rc = skim_cuda_mlkem768_encaps_batch(
                            public_key, seed_batch, ct_batch, ss_batch,
                            batch_size);
                        if (rc != 0)
                            break;
                    }
                }
                if (rc == 0) {
                    success = 1;
                    gpu_used = 1;
                }
                pqc_scheduler_gpu_release((uint32_t)(key_work_bytes > UINT32_MAX ?
                                                     UINT32_MAX : key_work_bytes));
            }
            if (!success) {
                success = 1;
                for (size_t i = 0; i < batch_size; i++) {
                    if (OQS_KEM_encaps(kem,
                                       ct_batch +
                                           i * kem->length_ciphertext,
                                       ss_batch +
                                           i * kem->length_shared_secret,
                                       public_key) != OQS_SUCCESS) {
                        success = 0;
                        break;
                    }
                }
            }
        }

        if (success) {
            size_t refreshed = 0;
            size_t metadata_failures = 0;
            for (size_t i = 0; i < batch_size; i++) {
                int fd = fds[i];
                pqc_fd_ctx_t *ctx = pqc_fd_context_for_fd(fd);
                if (!ctx)
                    continue;
                pqc_rekey_snapshot_t snapshot;
                memset(&snapshot, 0, sizeof(snapshot));
                pqc_lock_profile_scope_t fd_scope;
                (void)pqc_profiled_mutex_lock(&ctx->fd_lock, "fd_lock",
                                              __func__, &fd_scope);
                int snap_rc = rekey_snapshot_take_locked(ctx, &snapshot);
                (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock",
                                                __func__, &fd_scope);
                if (snap_rc <= 0) {
                    if (snap_rc < 0)
                        metadata_failures++;
                    rekey_snapshot_clear(&snapshot);
                    continue;
                }

                /*
                 * The current format uses one file key for all authenticated
                 * generations.  Changing ctx->ss here without re-encrypting old
                 * blocks would make committed data unreadable, so the mounted
                 * rekey worker refreshes the metadata envelope and leaves real
                 * transactional data-key rotation to a future format change.
                 */
                int store_rc = pqc_keyring_metadata_store(snapshot.marker_path,
                                                          snapshot.ss,
                                                          snapshot.ss_len,
                                                          snapshot.file_id);
                if (store_rc != 0) {
                    metadata_failures++;
                    rekey_snapshot_clear(&snapshot);
                    continue;
                }

                (void)pqc_profiled_mutex_lock(&ctx->fd_lock, "fd_lock",
                                              __func__, &fd_scope);
                if (rekey_snapshot_still_current_locked(ctx, &snapshot)) {
                    ctx->key_epoch++;
                    ctx->last_rekey = time(NULL);
                    refreshed++;
                }
                (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock",
                                                __func__, &fd_scope);
                rekey_snapshot_clear(&snapshot);
            }
            pqc_log("REKEY WORKER: batched %zu files %.1fµs (target=%s, run=%s)",
                    batch_size, pqc_metrics_time_us() - t0,
                    target == PQC_JOB_GPU ? "GPU" : "CPU",
                    gpu_used ? "GPU" : "CPU");
            if (metadata_failures > 0)
                pqc_log("REKEY WORKER DETAIL: metadata_refresh_failed=%zu refreshed=%zu",
                        metadata_failures, refreshed);
            pqc_log("REKEY WORKER DETAIL: work_bytes=%zu budget_ns=%llu decision_reason=%u deferral_reason=%u",
                    key_work_bytes,
                    (unsigned long long)admission_ctx.ai_qos_budget_remaining_ns,
                    (unsigned int)admission_ctx.decision_reason,
                    (unsigned int)admission_ctx.deferral_reason);

            pqc_scheduler_record_key_plane_batch(batch_size, gpu_used);
            pqc_plane_trace_record_keyplane_batch(
                (uint64_t)batch_size, (uint64_t)refreshed,
                (uint64_t)key_work_bytes, target == PQC_JOB_GPU,
                gpu_used, metadata_failures == 0);
        } else {
            pqc_log("REKEY WORKER: batched rekey FAILED for %zu files", batch_size);
            pqc_plane_trace_record_keyplane_batch(
                (uint64_t)batch_size, 0, (uint64_t)key_work_bytes,
                target == PQC_JOB_GPU, gpu_used, 0);
        }

        rekey_queue_batch_done(fds, batch_size);
        if (scratch_ready) {
            OPENSSL_cleanse(ct_batch, ct_required);
            OPENSSL_cleanse(ss_batch, ss_required);
        }
        if (seed_ready)
            OPENSSL_cleanse(seed_batch, seed_required);
    }
    return NULL;
}

int pqc_rekey_worker_start(OQS_KEM *kem, const uint8_t *public_key)
{
    if (!kem || !public_key)
        return -EINVAL;
    pqc_lock_profile_scope_t lifecycle_scope;
    rekey_lifecycle_lock(&lifecycle_scope, __func__);
    if (g_rekey_thread_started || g_rekey_thread_joining) {
        rekey_lifecycle_unlock(&lifecycle_scope, __func__);
        return 0;
    }
    pqc_rekey_worker_ctx_t *worker_ctx = rekey_worker_ctx_new(kem, public_key);
    if (!worker_ctx) {
        rekey_lifecycle_unlock(&lifecycle_scope, __func__);
        return -ENOMEM;
    }
    pqc_lock_profile_scope_t queue_scope;
    rekey_queue_lock(&queue_scope, __func__);
    rekey_queue_reset_locked();
    rekey_stop_set(0);
    rekey_queue_unlock(&queue_scope, __func__);
    if (pthread_create(&g_rekey_thread, NULL, rekey_worker_main,
                       worker_ctx) != 0) {
        int err = errno ? errno : EIO;
        rekey_stop_set(1);
        rekey_worker_ctx_free(worker_ctx);
        rekey_lifecycle_unlock(&lifecycle_scope, __func__);
        return -err;
    }
    g_rekey_worker_ctx = worker_ctx;
    g_rekey_thread_started = 1;
    rekey_lifecycle_unlock(&lifecycle_scope, __func__);
    return 0;
}

int pqc_rekey_worker_stop(void)
{
    pqc_lock_profile_scope_t lifecycle_scope;
    rekey_lifecycle_lock(&lifecycle_scope, __func__);
    if (!g_rekey_thread_started || g_rekey_thread_joining) {
        rekey_lifecycle_unlock(&lifecycle_scope, __func__);
        return 0;
    }
    g_rekey_thread_joining = 1;
    pthread_t thread_to_join = g_rekey_thread;
    pqc_rekey_worker_ctx_t *ctx_to_free = g_rekey_worker_ctx;
    rekey_lifecycle_unlock(&lifecycle_scope, __func__);

    pqc_lock_profile_scope_t queue_scope;
    rekey_queue_lock(&queue_scope, __func__);
    rekey_stop_set(1);
    pthread_cond_broadcast(&g_rekey_queue.cond);
    rekey_queue_unlock(&queue_scope, __func__);
    (void)pthread_join(thread_to_join, NULL);

    rekey_queue_lock(&queue_scope, __func__);
    rekey_queue_reset_locked();
    rekey_queue_unlock(&queue_scope, __func__);

    rekey_lifecycle_lock(&lifecycle_scope, __func__);
    g_rekey_thread_started = 0;
    g_rekey_thread_joining = 0;
    if (g_rekey_worker_ctx == ctx_to_free)
        g_rekey_worker_ctx = NULL;
    rekey_lifecycle_unlock(&lifecycle_scope, __func__);

    rekey_worker_ctx_free(ctx_to_free);
    return 1;
}
