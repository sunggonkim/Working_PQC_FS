#include "pqc_parallel_commit.h"

#include "pqc_config.h"
#include "pqc_lock_profile.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define PQC_PARALLEL_TRACE_PATH_MAX 4096U

typedef struct {
    pthread_mutex_t lock;
    pthread_cond_t cv;
    uint64_t next_epoch;
    uint64_t completed_epoch;
    uint64_t open_epoch;
    uint64_t active_epoch;
    uint32_t open_group_size;
    uint64_t open_group_bytes;
    uint32_t completed_group_size;
    uint64_t completed_group_bytes;
    int completed_rc;
    int open;
    int active;
    uint64_t total_epochs;
    uint64_t total_requests;
    uint64_t total_leaders;
    uint64_t total_followers;
    uint64_t total_group_size;
    uint32_t max_group_size;
    uint64_t max_group_bytes;
    uint32_t max_queue_depth;
    uint64_t total_wait_ns;
    uint64_t max_wait_ns;
    uint64_t wait_timeout_epochs;
    uint64_t full_group_epochs;
} pqc_parallel_commit_shard_t;

struct pqc_parallel_commit_coordinator {
    pqc_parallel_commit_config_t config;
    pqc_parallel_commit_shard_t shards[PQC_PARALLEL_COMMIT_MAX_SHARDS];
};

static pthread_mutex_t g_runtime_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_runtime_cv = PTHREAD_COND_INITIALIZER;
static pqc_parallel_commit_coordinator_t *g_runtime_coordinator = NULL;
static pqc_parallel_commit_config_t g_runtime_config;
static char *g_runtime_trace_path = NULL;
static uint32_t g_runtime_active_calls = 0;
static atomic_int g_runtime_advisory_enabled = ATOMIC_VAR_INIT(0);
static int g_runtime_enabled = 0;

typedef struct {
    char path[PQC_PARALLEL_TRACE_PATH_MAX];
    int has_path;
    const char *event;
    uint64_t file_id;
    uint64_t bytes;
    pqc_parallel_commit_ticket_t ticket;
    pqc_parallel_commit_config_t config;
    int rc;
} pqc_parallel_runtime_trace_record_t;

static int parallel_shard_lock(pqc_parallel_commit_shard_t *shard,
                               pqc_lock_profile_scope_t *scope,
                               const char *site)
{
    return pqc_profiled_mutex_lock(&shard->lock, "parallel_shard_lock",
                                   site, scope);
}

static int parallel_shard_unlock(pqc_parallel_commit_shard_t *shard,
                                 pqc_lock_profile_scope_t *scope,
                                 const char *site)
{
    return pqc_profiled_mutex_unlock(&shard->lock, "parallel_shard_lock",
                                     site, scope);
}

static int parallel_shard_wait(pqc_parallel_commit_shard_t *shard,
                               pqc_lock_profile_scope_t *scope,
                               const char *site)
{
    return pqc_profiled_cond_wait(&shard->cv, &shard->lock,
                                  "parallel_shard_lock", site, scope);
}

static int parallel_shard_timedwait(pqc_parallel_commit_shard_t *shard,
                                    pqc_lock_profile_scope_t *scope,
                                    const char *site,
                                    const struct timespec *deadline)
{
    return pqc_profiled_cond_timedwait(&shard->cv, &shard->lock,
                                       "parallel_shard_lock", site, scope,
                                       deadline);
}

static int parallel_runtime_lock(pqc_lock_profile_scope_t *scope,
                                 const char *site)
{
    return pqc_profiled_mutex_lock(&g_runtime_lock,
                                   "parallel_runtime_lock", site, scope);
}

static int parallel_runtime_unlock(pqc_lock_profile_scope_t *scope,
                                   const char *site)
{
    return pqc_profiled_mutex_unlock(&g_runtime_lock,
                                     "parallel_runtime_lock", site, scope);
}

static int parallel_runtime_wait(pqc_lock_profile_scope_t *scope,
                                 const char *site)
{
    return pqc_profiled_cond_wait(&g_runtime_cv, &g_runtime_lock,
                                  "parallel_runtime_lock", site, scope);
}

static uint64_t splitmix64(uint64_t value)
{
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

uint32_t pqc_parallel_commit_shard_for_file(uint64_t file_id,
                                            uint32_t shard_count)
{
    if (shard_count == 0)
        return 0;
    if (shard_count > PQC_PARALLEL_COMMIT_MAX_SHARDS)
        shard_count = PQC_PARALLEL_COMMIT_MAX_SHARDS;
    return (uint32_t)(splitmix64(file_id) % shard_count);
}

static int deadline_from_now(uint64_t wait_ns, struct timespec *deadline)
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

static uint64_t monotonic_now_ns(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
        return 0;
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) +
           (uint64_t)now.tv_nsec;
}

static uint32_t normalized_shard_count(const pqc_parallel_commit_config_t *config)
{
    uint32_t count = config ? config->shard_count : 0;
    if (count == 0)
        count = 1;
    if (count > PQC_PARALLEL_COMMIT_MAX_SHARDS)
        count = PQC_PARALLEL_COMMIT_MAX_SHARDS;
    return count;
}

int pqc_parallel_commit_init(pqc_parallel_commit_coordinator_t **out,
                             const pqc_parallel_commit_config_t *config)
{
    if (!out)
        return -EINVAL;
    *out = NULL;

    pqc_parallel_commit_coordinator_t *coordinator =
        calloc(1, sizeof(*coordinator));
    if (!coordinator)
        return -ENOMEM;

    coordinator->config.shard_count = normalized_shard_count(config);
    coordinator->config.max_group_size =
        config && config->max_group_size ? config->max_group_size : 1;
    coordinator->config.max_wait_ns = config ? config->max_wait_ns : 0;

    if (coordinator->config.max_group_size == 0)
        coordinator->config.max_group_size = 1;

    for (uint32_t i = 0; i < coordinator->config.shard_count; ++i) {
        pqc_parallel_commit_shard_t *shard = &coordinator->shards[i];
        int rc = pthread_mutex_init(&shard->lock, NULL);
        if (rc != 0) {
            pqc_parallel_commit_destroy(coordinator);
            return -rc;
        }
        rc = pthread_cond_init(&shard->cv, NULL);
        if (rc != 0) {
            pqc_parallel_commit_destroy(coordinator);
            return -rc;
        }
        shard->next_epoch = 1;
    }

    *out = coordinator;
    return 0;
}

void pqc_parallel_commit_destroy(pqc_parallel_commit_coordinator_t *coordinator)
{
    if (!coordinator)
        return;
    for (uint32_t i = 0; i < coordinator->config.shard_count; ++i) {
        pthread_cond_destroy(&coordinator->shards[i].cv);
        pthread_mutex_destroy(&coordinator->shards[i].lock);
    }
    free(coordinator);
}

static void ticket_clear(pqc_parallel_commit_ticket_t *ticket)
{
    if (ticket)
        memset(ticket, 0, sizeof(*ticket));
}

static void fill_ticket(pqc_parallel_commit_ticket_t *ticket,
                        pqc_parallel_commit_role_t role,
                        uint32_t shard_id,
                        uint32_t shard_count,
                        uint64_t epoch,
                        uint32_t group_size,
                        uint64_t group_bytes,
                        uint32_t observed_queue_depth,
                        uint64_t wait_ns,
                        int result_rc)
{
    if (!ticket)
        return;
    ticket->role = role;
    ticket->shard = shard_id;
    ticket->shard_count = shard_count;
    ticket->epoch = epoch;
    ticket->group_size = group_size;
    ticket->group_bytes = group_bytes;
    ticket->observed_queue_depth = observed_queue_depth;
    ticket->wait_ns = wait_ns;
    ticket->result_rc = result_rc;
}

static void record_wait_locked(pqc_parallel_commit_shard_t *shard,
                               uint32_t queue_depth,
                               uint64_t wait_ns)
{
    if (queue_depth > shard->max_queue_depth)
        shard->max_queue_depth = queue_depth;
    shard->total_wait_ns += wait_ns;
    if (wait_ns > shard->max_wait_ns)
        shard->max_wait_ns = wait_ns;
}

static void close_open_epoch_locked(pqc_parallel_commit_shard_t *shard,
                                    int timeout)
{
    shard->open = 0;
    shard->active = 1;
    shard->active_epoch = shard->open_epoch;
    shard->total_epochs++;
    shard->total_group_size += shard->open_group_size;
    if (shard->open_group_size > shard->max_group_size)
        shard->max_group_size = shard->open_group_size;
    if (shard->open_group_bytes > shard->max_group_bytes)
        shard->max_group_bytes = shard->open_group_bytes;
    if (timeout)
        shard->wait_timeout_epochs++;
    else
        shard->full_group_epochs++;
    pthread_cond_broadcast(&shard->cv);
}

int pqc_parallel_commit_begin(pqc_parallel_commit_coordinator_t *coordinator,
                              uint64_t file_id,
                              uint64_t bytes,
                              pqc_parallel_commit_ticket_t *ticket)
{
    if (!coordinator || !ticket)
        return -EINVAL;
    ticket_clear(ticket);
    uint64_t start_ns = monotonic_now_ns();

    uint32_t shard_id = pqc_parallel_commit_shard_for_file(
        file_id, coordinator->config.shard_count);
    pqc_parallel_commit_shard_t *shard = &coordinator->shards[shard_id];
    uint32_t max_group = coordinator->config.max_group_size;
    uint64_t max_wait_ns = coordinator->config.max_wait_ns;
    if (max_group == 0)
        max_group = 1;

    pqc_lock_profile_scope_t shard_scope;
    int rc = parallel_shard_lock(shard, &shard_scope, __func__);
    if (rc != 0)
        return -rc;

    while (shard->active && !shard->open)
        (void)parallel_shard_wait(shard, &shard_scope, __func__);

    if (shard->open && shard->open_group_size < max_group) {
        uint64_t epoch = shard->open_epoch;
        shard->open_group_size++;
        shard->open_group_bytes += bytes;
        uint32_t queue_depth = shard->open_group_size;
        shard->total_requests++;
        shard->total_followers++;
        if (shard->open_group_size >= max_group)
            pthread_cond_broadcast(&shard->cv);
        while (shard->completed_epoch < epoch)
            (void)parallel_shard_wait(shard, &shard_scope, __func__);
        uint64_t now_ns = monotonic_now_ns();
        uint64_t wait_ns = now_ns >= start_ns ? now_ns - start_ns : 0;
        record_wait_locked(shard, queue_depth, wait_ns);
        fill_ticket(ticket, PQC_PARALLEL_COMMIT_ROLE_FOLLOWER, shard_id,
                    coordinator->config.shard_count, epoch,
                    shard->completed_group_size, shard->completed_group_bytes,
                    queue_depth, wait_ns,
                    shard->completed_rc);
        int result = shard->completed_rc;
        (void)parallel_shard_unlock(shard, &shard_scope, __func__);
        return result;
    }

    while (shard->active)
        (void)parallel_shard_wait(shard, &shard_scope, __func__);

    uint64_t epoch = shard->next_epoch++;
    shard->open = 1;
    shard->open_epoch = epoch;
    shard->open_group_size = 1;
    shard->open_group_bytes = bytes;
    shard->total_requests++;
    shard->total_leaders++;

    int timeout = 0;
    if (max_group > 1 && max_wait_ns > 0) {
        struct timespec deadline;
        rc = deadline_from_now(max_wait_ns, &deadline);
        if (rc != 0) {
            shard->open = 0;
            (void)parallel_shard_unlock(shard, &shard_scope, __func__);
            return rc;
        }
        while (shard->open_group_size < max_group) {
            rc = parallel_shard_timedwait(shard, &shard_scope, __func__,
                                          &deadline);
            if (rc == ETIMEDOUT) {
                timeout = 1;
                break;
            }
            if (rc != 0) {
                shard->open = 0;
                (void)parallel_shard_unlock(shard, &shard_scope, __func__);
                return -rc;
            }
        }
    } else if (max_group > 1) {
        timeout = 1;
    }

    close_open_epoch_locked(shard, timeout);
    uint64_t now_ns = monotonic_now_ns();
    uint64_t wait_ns = now_ns >= start_ns ? now_ns - start_ns : 0;
    uint32_t queue_depth = shard->open_group_size;
    record_wait_locked(shard, queue_depth, wait_ns);
    fill_ticket(ticket, PQC_PARALLEL_COMMIT_ROLE_LEADER, shard_id,
                coordinator->config.shard_count, epoch, shard->open_group_size,
                shard->open_group_bytes, queue_depth, wait_ns, 0);
    (void)parallel_shard_unlock(shard, &shard_scope, __func__);
    return 0;
}

int pqc_parallel_commit_finish(pqc_parallel_commit_coordinator_t *coordinator,
                               const pqc_parallel_commit_ticket_t *ticket,
                               int result_rc)
{
    if (!coordinator || !ticket ||
        ticket->role != PQC_PARALLEL_COMMIT_ROLE_LEADER ||
        ticket->shard >= coordinator->config.shard_count)
        return -EINVAL;

    pqc_parallel_commit_shard_t *shard = &coordinator->shards[ticket->shard];
    pqc_lock_profile_scope_t shard_scope;
    int rc = parallel_shard_lock(shard, &shard_scope, __func__);
    if (rc != 0)
        return -rc;
    if (!shard->active || shard->active_epoch != ticket->epoch) {
        (void)parallel_shard_unlock(shard, &shard_scope, __func__);
        return -EINVAL;
    }
    shard->completed_group_size = ticket->group_size;
    shard->completed_group_bytes = ticket->group_bytes;
    shard->completed_rc = result_rc;
    shard->completed_epoch = ticket->epoch;
    shard->active = 0;
    pthread_cond_broadcast(&shard->cv);
    (void)parallel_shard_unlock(shard, &shard_scope, __func__);
    return 0;
}

void pqc_parallel_commit_stats_snapshot(
    pqc_parallel_commit_coordinator_t *coordinator,
    pqc_parallel_commit_stats_t *out)
{
    if (!out)
        return;
    memset(out, 0, sizeof(*out));
    if (!coordinator)
        return;
    out->shard_count = coordinator->config.shard_count;
    for (uint32_t i = 0; i < coordinator->config.shard_count; ++i) {
        pqc_parallel_commit_shard_t *shard = &coordinator->shards[i];
        pqc_lock_profile_scope_t shard_scope;
        if (parallel_shard_lock(shard, &shard_scope, __func__) != 0)
            continue;
        out->total_epochs += shard->total_epochs;
        out->total_requests += shard->total_requests;
        out->total_leaders += shard->total_leaders;
        out->total_followers += shard->total_followers;
        out->total_group_size += shard->total_group_size;
        if (shard->max_group_size > out->max_observed_group_size)
            out->max_observed_group_size = shard->max_group_size;
        if (shard->max_group_bytes > out->max_observed_group_bytes)
            out->max_observed_group_bytes = shard->max_group_bytes;
        if (shard->max_queue_depth > out->max_observed_queue_depth)
            out->max_observed_queue_depth = shard->max_queue_depth;
        out->total_wait_ns += shard->total_wait_ns;
        if (shard->max_wait_ns > out->max_wait_ns)
            out->max_wait_ns = shard->max_wait_ns;
        out->wait_timeout_epochs += shard->wait_timeout_epochs;
        out->full_group_epochs += shard->full_group_epochs;
        (void)parallel_shard_unlock(shard, &shard_scope, __func__);
    }
}

static int mode_enabled(const char *mode)
{
    if (!mode || !*mode)
        return 0;
    return strcmp(mode, "epoch-gated-strict") == 0 ||
           strcmp(mode, "epoch") == 0 ||
           strcmp(mode, "parallel") == 0 ||
           strcmp(mode, "on") == 0 ||
           strcmp(mode, "1") == 0;
}

static const char *role_name(pqc_parallel_commit_role_t role)
{
    switch (role) {
    case PQC_PARALLEL_COMMIT_ROLE_LEADER:
        return "leader";
    case PQC_PARALLEL_COMMIT_ROLE_FOLLOWER:
        return "follower";
    default:
        return "invalid";
    }
}

static char *runtime_trace_path_dup(const char *path)
{
    if (!path)
        return NULL;
    size_t len = strlen(path) + 1;
    char *copy = malloc(len);
    if (!copy)
        return NULL;
    memcpy(copy, path, len);
    return copy;
}

static int runtime_trace_path_copy_bounded(char *dest, size_t dest_size,
                                           const char *src)
{
    if (!dest || dest_size == 0 || !src || !*src)
        return 0;

    for (size_t i = 0; i < dest_size; ++i) {
        dest[i] = src[i];
        if (src[i] == '\0')
            return 1;
    }

    dest[dest_size - 1] = '\0';
    return 0;
}

static int runtime_trace_path_probe(const char *path)
{
    if (!path)
        return 0;
    int fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    if (fd < 0)
        return -errno;
    if (close(fd) != 0)
        return -errno;
    return 0;
}

static void runtime_trace_record_clear(
    pqc_parallel_runtime_trace_record_t *record)
{
    if (!record)
        return;
    memset(record, 0, sizeof(*record));
}

static void runtime_trace_snapshot_locked(
    pqc_parallel_runtime_trace_record_t *record,
    const char *event,
    uint64_t file_id,
    uint64_t bytes,
    const pqc_parallel_commit_ticket_t *ticket,
    int rc)
{
    if (!record)
        return;
    memset(record, 0, sizeof(*record));
    if (!g_runtime_trace_path || !event)
        return;
    if (!runtime_trace_path_copy_bounded(record->path,
                                         sizeof(record->path),
                                         g_runtime_trace_path))
        return;
    record->has_path = 1;
    record->event = event;
    record->file_id = file_id;
    record->bytes = bytes;
    if (ticket)
        record->ticket = *ticket;
    record->config = g_runtime_config;
    record->rc = rc;
}

static void runtime_trace_emit(pqc_parallel_runtime_trace_record_t *record)
{
    if (!record || !record->has_path || !record->event)
        return;

    uint32_t shard_count = record->ticket.shard_count ?
        record->ticket.shard_count : record->config.shard_count;
    char line[1024];
    int len = snprintf(
        line, sizeof(line),
        "{\"event\":\"%s\","
        "\"file_id\":%llu,"
        "\"bytes\":%llu,"
        "\"role\":\"%s\","
        "\"shard\":%u,"
        "\"shard_count\":%u,"
        "\"config_shard_count\":%u,"
        "\"config_max_group_size\":%u,"
        "\"config_max_wait_ns\":%llu,"
        "\"epoch\":%llu,"
        "\"group_size\":%u,"
        "\"group_bytes\":%llu,"
        "\"queue_depth\":%u,"
        "\"wait_ns\":%llu,"
        "\"rc\":%d}\n",
        record->event,
        (unsigned long long)record->file_id,
        (unsigned long long)record->bytes,
        role_name(record->ticket.role),
        record->ticket.shard,
        shard_count,
        record->config.shard_count,
        record->config.max_group_size,
        (unsigned long long)record->config.max_wait_ns,
        (unsigned long long)record->ticket.epoch,
        record->ticket.group_size,
        (unsigned long long)record->ticket.group_bytes,
        record->ticket.observed_queue_depth,
        (unsigned long long)record->ticket.wait_ns,
        record->rc);
    if (len <= 0 || (size_t)len >= sizeof(line)) {
        runtime_trace_record_clear(record);
        return;
    }

    int fd = open(record->path,
                  O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    if (fd >= 0) {
        size_t off = 0;
        while (off < (size_t)len) {
            ssize_t written = write(fd, line + off, (size_t)len - off);
            if (written < 0) {
                if (errno == EINTR)
                    continue;
                break;
            }
            if (written == 0)
                break;
            off += (size_t)written;
        }
        (void)close(fd);
    }
    runtime_trace_record_clear(record);
}

static pqc_parallel_commit_coordinator_t *runtime_coordinator_acquire(void)
{
    pqc_parallel_commit_coordinator_t *coordinator = NULL;
    pqc_lock_profile_scope_t runtime_scope;
    if (parallel_runtime_lock(&runtime_scope, __func__) != 0)
        return NULL;
    if (g_runtime_enabled && g_runtime_coordinator) {
        coordinator = g_runtime_coordinator;
        g_runtime_active_calls++;
    }
    (void)parallel_runtime_unlock(&runtime_scope, __func__);
    return coordinator;
}

static void runtime_coordinator_release(void)
{
    pqc_lock_profile_scope_t runtime_scope;
    if (parallel_runtime_lock(&runtime_scope, __func__) != 0)
        return;
    if (g_runtime_active_calls > 0)
        g_runtime_active_calls--;
    if (g_runtime_active_calls == 0)
        pthread_cond_broadcast(&g_runtime_cv);
    (void)parallel_runtime_unlock(&runtime_scope, __func__);
}

static void runtime_wait_for_idle_locked(pqc_lock_profile_scope_t *scope,
                                         const char *site)
{
    while (g_runtime_active_calls > 0)
        (void)parallel_runtime_wait(scope, site);
}

int pqc_parallel_commit_runtime_init_from_config(void)
{
    const char *mode =
        pqc_config_nonempty_or_default("PQC_PARALLEL_COMMIT_MODE", "off");
    if (!mode_enabled(mode))
        return 0;

    pqc_parallel_commit_config_t config = {
        .shard_count = (uint32_t)pqc_config_u64_or_default(
            "PQC_PARALLEL_COMMIT_SHARDS", 4),
        .max_group_size = (uint32_t)pqc_config_u64_or_default(
            "PQC_PARALLEL_COMMIT_GROUP_MAX", 1),
        .max_wait_ns = pqc_config_u64_or_default(
            "PQC_PARALLEL_COMMIT_WAIT_NS", 0),
    };
    if (config.shard_count == 0)
        config.shard_count = 1;
    if (config.shard_count > PQC_PARALLEL_COMMIT_MAX_SHARDS)
        config.shard_count = PQC_PARALLEL_COMMIT_MAX_SHARDS;
    if (config.max_group_size == 0)
        config.max_group_size = 1;

    pqc_parallel_commit_coordinator_t *coordinator = NULL;
    int rc = pqc_parallel_commit_init(&coordinator, &config);
    if (rc != 0)
        return rc;

    char *trace_path_copy = NULL;
    const char *trace_path =
        pqc_config_get_nonempty("PQC_PARALLEL_COMMIT_TRACE_PATH");
    if (trace_path) {
        rc = runtime_trace_path_probe(trace_path);
        if (rc != 0) {
            pqc_parallel_commit_destroy(coordinator);
            return rc;
        }
        trace_path_copy = runtime_trace_path_dup(trace_path);
        if (!trace_path_copy) {
            pqc_parallel_commit_destroy(coordinator);
            return -ENOMEM;
        }
    }

    pqc_parallel_runtime_trace_record_t trace_record = {0};
    pqc_parallel_commit_coordinator_t *old_coordinator = NULL;
    char *old_trace_path = NULL;
    pqc_lock_profile_scope_t runtime_scope;
    rc = parallel_runtime_lock(&runtime_scope, __func__);
    if (rc != 0) {
        free(trace_path_copy);
        pqc_parallel_commit_destroy(coordinator);
        return -rc;
    }
    atomic_store_explicit(&g_runtime_advisory_enabled, 0,
                          memory_order_release);
    g_runtime_enabled = 0;
    old_coordinator = g_runtime_coordinator;
    g_runtime_coordinator = NULL;
    runtime_wait_for_idle_locked(&runtime_scope, __func__);
    old_trace_path = g_runtime_trace_path;
    g_runtime_coordinator = coordinator;
    g_runtime_config = config;
    g_runtime_trace_path = trace_path_copy;
    trace_path_copy = NULL;
    g_runtime_enabled = 1;
    atomic_store_explicit(&g_runtime_advisory_enabled, 1,
                          memory_order_release);
    runtime_trace_snapshot_locked(&trace_record, "runtime_init", 0, 0,
                                  NULL, 0);
    (void)parallel_runtime_unlock(&runtime_scope, __func__);
    free(old_trace_path);
    pqc_parallel_commit_destroy(old_coordinator);
    runtime_trace_emit(&trace_record);
    return 0;
}

void pqc_parallel_commit_runtime_shutdown(void)
{
    pqc_parallel_runtime_trace_record_t trace_record = {0};
    pqc_lock_profile_scope_t runtime_scope;
    if (parallel_runtime_lock(&runtime_scope, __func__) != 0)
        return;
    atomic_store_explicit(&g_runtime_advisory_enabled, 0,
                          memory_order_release);
    g_runtime_enabled = 0;
    pqc_parallel_commit_coordinator_t *coordinator = g_runtime_coordinator;
    g_runtime_coordinator = NULL;
    runtime_wait_for_idle_locked(&runtime_scope, __func__);
    runtime_trace_snapshot_locked(&trace_record, "runtime_shutdown", 0, 0,
                                  NULL, 0);
    char *trace_path = g_runtime_trace_path;
    g_runtime_trace_path = NULL;
    memset(&g_runtime_config, 0, sizeof(g_runtime_config));
    (void)parallel_runtime_unlock(&runtime_scope, __func__);

    runtime_trace_emit(&trace_record);
    free(trace_path);
    pqc_parallel_commit_destroy(coordinator);
}

int pqc_parallel_commit_runtime_enabled(void)
{
    return atomic_load_explicit(&g_runtime_advisory_enabled,
                                memory_order_acquire) != 0;
}

int pqc_parallel_commit_runtime_begin(uint64_t file_id,
                                      uint64_t bytes,
                                      pqc_parallel_commit_ticket_t *ticket)
{
    if (!ticket)
        return -EINVAL;
    memset(ticket, 0, sizeof(*ticket));
    pqc_parallel_commit_coordinator_t *coordinator =
        runtime_coordinator_acquire();
    if (!coordinator)
        return 0;
    int rc = pqc_parallel_commit_begin(coordinator, file_id, bytes, ticket);
    int leader_ref =
        rc == 0 && ticket->role == PQC_PARALLEL_COMMIT_ROLE_LEADER;
    if (leader_ref) {
        ticket->runtime_coordinator = coordinator;
        ticket->runtime_ref_held = 1;
    }
    pqc_parallel_runtime_trace_record_t trace_record = {0};
    pqc_lock_profile_scope_t runtime_scope;
    if (parallel_runtime_lock(&runtime_scope, __func__) == 0) {
        runtime_trace_snapshot_locked(&trace_record, "begin", file_id,
                                      bytes, ticket, rc);
        (void)parallel_runtime_unlock(&runtime_scope, __func__);
    }
    if (!leader_ref)
        runtime_coordinator_release();
    runtime_trace_emit(&trace_record);
    return rc;
}

int pqc_parallel_commit_runtime_finish(pqc_parallel_commit_ticket_t *ticket,
                                       int result_rc)
{
    if (!ticket || ticket->role != PQC_PARALLEL_COMMIT_ROLE_LEADER)
        return 0;

    int held_ref = ticket->runtime_ref_held && ticket->runtime_coordinator;
    pqc_parallel_commit_coordinator_t *coordinator = held_ref ?
        ticket->runtime_coordinator : runtime_coordinator_acquire();
    if (!coordinator)
        return 0;

    int rc = pqc_parallel_commit_finish(coordinator, ticket, result_rc);
    pqc_parallel_runtime_trace_record_t trace_record = {0};
    pqc_lock_profile_scope_t runtime_scope;
    if (parallel_runtime_lock(&runtime_scope, __func__) == 0) {
        runtime_trace_snapshot_locked(&trace_record, "finish", 0,
                                      ticket->group_bytes, ticket,
                                      rc == 0 ? result_rc : rc);
        (void)parallel_runtime_unlock(&runtime_scope, __func__);
    }
    if (held_ref) {
        ticket->runtime_ref_held = 0;
        ticket->runtime_coordinator = NULL;
    }
    runtime_coordinator_release();
    runtime_trace_emit(&trace_record);
    return rc;
}

void pqc_parallel_commit_runtime_stats_snapshot(
    pqc_parallel_commit_stats_t *out)
{
    pqc_parallel_commit_coordinator_t *coordinator =
        runtime_coordinator_acquire();
    pqc_parallel_commit_stats_snapshot(coordinator, out);
    if (coordinator)
        runtime_coordinator_release();
}
