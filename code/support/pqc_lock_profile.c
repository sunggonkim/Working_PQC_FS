#include "pqc_lock_profile.h"

#include "pqc_config.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdatomic.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define PQC_LOCK_PROFILE_BUCKETS 24
#define PQC_LOCK_ORDER_STACK_MAX 32
#define PQC_LOCK_PROFILE_TRACE_LINE_CAP 2048

typedef struct {
    const char *name;
    uint64_t samples;
    uint64_t wait_total_ns;
    uint64_t hold_total_ns;
    uint64_t wait_max_ns;
    uint64_t hold_max_ns;
    uint64_t wait_buckets[PQC_LOCK_PROFILE_BUCKETS];
    uint64_t hold_buckets[PQC_LOCK_PROFILE_BUCKETS];
    uint64_t cond_wait_samples;
    uint64_t cond_wait_total_ns;
    uint64_t cond_wait_max_ns;
    uint64_t cond_wait_buckets[PQC_LOCK_PROFILE_BUCKETS];
    uint64_t order_violations;
} pqc_lock_profile_stat_t;

typedef struct {
    pthread_mutex_t *mutex;
    pthread_cond_t *cond;
    int *ready;
} pqc_lock_profile_cond_test_arg_t;

typedef struct {
    pthread_mutex_t *mutex;
    const char *name;
    int rank;
} pqc_lock_order_held_t;

static pthread_mutex_t g_profile_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_profile_fd = -1;
static atomic_int g_profile_initialized = ATOMIC_VAR_INIT(0);
static atomic_int g_profile_enabled = ATOMIC_VAR_INIT(0);
static atomic_int g_profile_dump_on_shutdown = ATOMIC_VAR_INIT(0);
static uint64_t g_profile_sequence = 0;
static __thread pqc_lock_order_held_t
    g_lock_order_stack[PQC_LOCK_ORDER_STACK_MAX];
static __thread size_t g_lock_order_depth = 0;

static int profile_internal_lock(void)
{
    return pthread_mutex_lock(&g_profile_lock);
}

static int profile_internal_unlock(void)
{
    return pthread_mutex_unlock(&g_profile_lock);
}

static pqc_lock_profile_stat_t g_profile_stats[] = {
    { .name = "file_state_table_lock" },
    { .name = "fd_lock" },
    { .name = "commit_lock" },
    { .name = "committed_map_lock" },
    { .name = "anchor_pending_lock" },
    { .name = "file_anchor_commit_lock" },
    { .name = "anchor_epoch_record_lock" },
    { .name = "anchor_lifecycle_lock" },
    { .name = "anchor_worker_lock" },
    { .name = "rekey_lifecycle_lock" },
    { .name = "rekey_queue_lock" },
    { .name = "epoch_barrier_lock" },
    { .name = "parallel_shard_lock" },
    { .name = "parallel_runtime_lock" },
    { .name = "admission_state_lock" },
    { .name = "admission_trace_lock" },
    { .name = "qos_gpu_load_lock" },
    { .name = "qos_throttle_lock" },
    { .name = "scheduler_lock" },
    { .name = "fault_cutpoint_lock" },
    { .name = "trace_sink_lock" },
};

static uint64_t monotonic_ns(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        return 0;
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static uint64_t elapsed_ns(uint64_t end_ns, uint64_t start_ns)
{
    return end_ns >= start_ns ? end_ns - start_ns : 0;
}

static size_t profile_stat_count(void)
{
    return sizeof(g_profile_stats) / sizeof(g_profile_stats[0]);
}

static pqc_lock_profile_stat_t *profile_stat_for_lock(const char *lock_name)
{
    if (!lock_name)
        return NULL;
    for (size_t i = 0; i < profile_stat_count(); i++) {
        if (strcmp(g_profile_stats[i].name, lock_name) == 0)
            return &g_profile_stats[i];
    }
    return NULL;
}

static void profile_trace_write_locked(const char *line)
{
    if (!line || g_profile_fd < 0)
        return;
    size_t len = strlen(line);
    size_t off = 0;
    while (off < len) {
        ssize_t written = write(g_profile_fd, line + off, len - off);
        if (written < 0) {
            if (errno == EINTR)
                continue;
            break;
        }
        if (written == 0)
            break;
        off += (size_t)written;
    }
}

static void profile_trace_appendf_locked(const char *fmt, ...)
{
    if (!fmt || g_profile_fd < 0)
        return;
    char line[PQC_LOCK_PROFILE_TRACE_LINE_CAP];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    if (n < 0)
        return;
    line[sizeof(line) - 1] = '\0';
    profile_trace_write_locked(line);
}

static int lock_rank_for_name(const char *lock_name)
{
    if (!lock_name)
        return 0;
    if (strcmp(lock_name, "file_state_table_lock") == 0)
        return 5;
    if (strcmp(lock_name, "fd_lock") == 0)
        return 10;
    if (strcmp(lock_name, "commit_lock") == 0)
        return 20;
    if (strcmp(lock_name, "committed_map_lock") == 0)
        return 30;
    if (strcmp(lock_name, "anchor_lifecycle_lock") == 0)
        return 30;
    if (strcmp(lock_name, "rekey_lifecycle_lock") == 0)
        return 30;
    if (strcmp(lock_name, "anchor_worker_lock") == 0)
        return 40;
    if (strcmp(lock_name, "rekey_queue_lock") == 0)
        return 40;
    if (strcmp(lock_name, "epoch_barrier_lock") == 0)
        return 40;
    if (strcmp(lock_name, "parallel_runtime_lock") == 0)
        return 40;
    if (strcmp(lock_name, "parallel_shard_lock") == 0)
        return 40;
    if (strcmp(lock_name, "anchor_pending_lock") == 0)
        return 50;
    if (strcmp(lock_name, "file_anchor_commit_lock") == 0)
        return 50;
    if (strcmp(lock_name, "anchor_epoch_record_lock") == 0)
        return 55;
    if (strcmp(lock_name, "admission_state_lock") == 0)
        return 60;
    if (strcmp(lock_name, "admission_trace_lock") == 0)
        return 60;
    if (strcmp(lock_name, "qos_gpu_load_lock") == 0)
        return 60;
    if (strcmp(lock_name, "qos_throttle_lock") == 0)
        return 60;
    if (strcmp(lock_name, "scheduler_lock") == 0)
        return 60;
    if (strcmp(lock_name, "fault_cutpoint_lock") == 0)
        return 70;
    if (strcmp(lock_name, "trace_sink_lock") == 0)
        return 80;
    return 0;
}

static void record_order_violation_locked(const char *lock_name)
{
    pqc_lock_profile_stat_t *stat = profile_stat_for_lock(lock_name);
    if (stat)
        stat->order_violations++;
}

static void log_order_violation(const char *lock_name,
                                const char *site,
                                pthread_mutex_t *mutex,
                                int rank,
                                const char *held_name,
                                int held_rank,
                                const char *reason)
{
    if (profile_internal_lock() != 0)
        return;
    record_order_violation_locked(lock_name);
    profile_trace_appendf_locked(
        "{\"event\":\"lock_order_violation\","
        "\"seq\":%" PRIu64 ","
        "\"lock\":\"%s\",\"site\":\"%s\","
        "\"thread\":\"%llu\",\"addr\":\"%p\","
        "\"rank\":%d,\"held_lock\":\"%s\","
        "\"held_rank\":%d,\"reason\":\"%s\"}\n",
        ++g_profile_sequence,
        lock_name ? lock_name : "unknown",
        site ? site : "unknown",
        (unsigned long long)(uintptr_t)pthread_self(),
        (void *)mutex,
        rank,
        held_name ? held_name : "none",
        held_rank,
        reason ? reason : "unknown");
    (void)profile_internal_unlock();
}

static int lock_order_check_before_lock(pthread_mutex_t *mutex,
                                        const char *lock_name,
                                        const char *site,
                                        int rank)
{
    if (rank <= 0)
        return 0;
    for (size_t i = 0; i < g_lock_order_depth; i++) {
        if (g_lock_order_stack[i].mutex == mutex) {
            log_order_violation(lock_name, site, mutex, rank,
                                g_lock_order_stack[i].name,
                                g_lock_order_stack[i].rank,
                                "recursive_lock");
            return EDEADLK;
        }
    }
    if (g_lock_order_depth > 0) {
        const pqc_lock_order_held_t *held =
            &g_lock_order_stack[g_lock_order_depth - 1];
        if (held->rank > rank) {
            log_order_violation(lock_name, site, mutex, rank, held->name,
                                held->rank, "rank_inversion");
            return EDEADLK;
        }
    }
    return 0;
}

static void lock_order_push(pthread_mutex_t *mutex,
                            const char *lock_name,
                            const char *site,
                            int rank)
{
    if (rank <= 0)
        return;
    if (g_lock_order_depth >= PQC_LOCK_ORDER_STACK_MAX) {
        log_order_violation(lock_name, site, mutex, rank, "lock_stack", rank,
                            "stack_overflow");
        return;
    }
    g_lock_order_stack[g_lock_order_depth++] = (pqc_lock_order_held_t) {
        .mutex = mutex,
        .name = lock_name,
        .rank = rank,
    };
}

static void lock_order_pop(pthread_mutex_t *mutex,
                           const char *lock_name,
                           const char *site,
                           int rank)
{
    if (rank <= 0)
        return;
    if (g_lock_order_depth == 0) {
        log_order_violation(lock_name, site, mutex, rank, "none", 0,
                            "unlock_empty_stack");
        return;
    }
    const pqc_lock_order_held_t *held =
        &g_lock_order_stack[g_lock_order_depth - 1];
    if (held->mutex == mutex) {
        g_lock_order_depth--;
        return;
    }
    log_order_violation(lock_name, site, mutex, rank, held->name,
                        held->rank, "unlock_not_top");
    for (size_t i = g_lock_order_depth; i > 0; i--) {
        if (g_lock_order_stack[i - 1].mutex == mutex) {
            memmove(&g_lock_order_stack[i - 1],
                    &g_lock_order_stack[i],
                    (g_lock_order_depth - i) *
                        sizeof(g_lock_order_stack[0]));
            g_lock_order_depth--;
            return;
        }
    }
}

static int latency_bucket(uint64_t ns)
{
    uint64_t limit = 128;
    for (int i = 0; i < PQC_LOCK_PROFILE_BUCKETS - 1; i++) {
        if (ns <= limit)
            return i;
        if (limit > UINT64_MAX / 2)
            break;
        limit <<= 1;
    }
    return PQC_LOCK_PROFILE_BUCKETS - 1;
}

static uint64_t bucket_upper_bound_ns(int bucket)
{
    if (bucket >= PQC_LOCK_PROFILE_BUCKETS - 1)
        return UINT64_MAX;
    return UINT64_C(128) << bucket;
}

static uint64_t percentile_from_buckets(const uint64_t *buckets,
                                        uint64_t samples,
                                        unsigned percentile)
{
    if (!buckets || samples == 0)
        return 0;
    uint64_t rank = (samples * percentile + 99) / 100;
    if (rank == 0)
        rank = 1;
    uint64_t seen = 0;
    for (int i = 0; i < PQC_LOCK_PROFILE_BUCKETS; i++) {
        seen += buckets[i];
        if (seen >= rank)
            return bucket_upper_bound_ns(i);
    }
    return UINT64_MAX;
}

static void profile_stats_reset_locked(void)
{
    for (size_t i = 0; i < profile_stat_count(); i++) {
        const char *name = g_profile_stats[i].name;
        memset(&g_profile_stats[i], 0, sizeof(g_profile_stats[i]));
        g_profile_stats[i].name = name;
    }
}

static void record_lock_sample_locked(const char *lock_name,
                                      uint64_t wait_ns,
                                      uint64_t hold_ns)
{
    pqc_lock_profile_stat_t *stat = profile_stat_for_lock(lock_name);
    if (!stat)
        return;
    stat->samples++;
    stat->wait_total_ns += wait_ns;
    stat->hold_total_ns += hold_ns;
    if (wait_ns > stat->wait_max_ns)
        stat->wait_max_ns = wait_ns;
    if (hold_ns > stat->hold_max_ns)
        stat->hold_max_ns = hold_ns;
    stat->wait_buckets[latency_bucket(wait_ns)]++;
    stat->hold_buckets[latency_bucket(hold_ns)]++;
}

static void record_cond_wait_sample_locked(const char *lock_name,
                                           uint64_t cond_wait_ns)
{
    pqc_lock_profile_stat_t *stat = profile_stat_for_lock(lock_name);
    if (!stat)
        return;
    stat->cond_wait_samples++;
    stat->cond_wait_total_ns += cond_wait_ns;
    if (cond_wait_ns > stat->cond_wait_max_ns)
        stat->cond_wait_max_ns = cond_wait_ns;
    stat->cond_wait_buckets[latency_bucket(cond_wait_ns)]++;
}

static void record_scope_segment(pthread_mutex_t *mutex,
                                 const char *lock_name,
                                 const char *site,
                                 pqc_lock_profile_scope_t *scope,
                                 uint64_t release_ns,
                                 int unlock_rc,
                                 const char *event,
                                 const char *boundary)
{
    if (!scope || !scope->enabled)
        return;

    uint64_t wait_ns = elapsed_ns(scope->acquired_ns, scope->wait_start_ns);
    uint64_t hold_ns = elapsed_ns(release_ns, scope->acquired_ns);
    if (profile_internal_lock() != 0) {
        scope->enabled = 0;
        return;
    }
    profile_trace_appendf_locked(
        "{\"event\":\"%s\","
        "\"seq\":%" PRIu64 ","
        "\"lock\":\"%s\",\"site\":\"%s\","
        "%s%s%s"
        "\"thread\":\"%llu\",\"addr\":\"%p\","
        "\"wait_ns\":%" PRIu64 ","
        "\"hold_ns\":%" PRIu64 ",\"unlock_rc\":%d}\n",
        event ? event : "lock_hold",
        ++g_profile_sequence,
        lock_name ? lock_name : "unknown",
        site ? site : "unknown",
        boundary ? "\"boundary\":\"" : "",
        boundary ? boundary : "",
        boundary ? "\"," : "",
        (unsigned long long)(uintptr_t)pthread_self(),
        (void *)mutex,
        wait_ns,
        hold_ns,
        unlock_rc);
    if (unlock_rc == 0)
        record_lock_sample_locked(lock_name, wait_ns, hold_ns);
    (void)profile_internal_unlock();
    scope->enabled = 0;
}

static void dump_stats_locked(FILE *out)
{
    if (!out)
        return;
    fprintf(out,
            "{\"event\":\"lock_profile_summary_begin\","
            "\"clock\":\"CLOCK_MONOTONIC\","
            "\"percentiles_are_histogram_upper_bounds\":true}\n");
    for (size_t i = 0; i < profile_stat_count(); i++) {
        const pqc_lock_profile_stat_t *s = &g_profile_stats[i];
        uint64_t wait_avg = s->samples ? s->wait_total_ns / s->samples : 0;
        uint64_t hold_avg = s->samples ? s->hold_total_ns / s->samples : 0;
        uint64_t cond_wait_avg = s->cond_wait_samples
            ? s->cond_wait_total_ns / s->cond_wait_samples
            : 0;
        fprintf(out,
                "{\"event\":\"lock_profile_summary\","
                "\"lock\":\"%s\","
                "\"samples\":%" PRIu64 ","
                "\"wait_avg_ns\":%" PRIu64 ","
                "\"wait_max_ns\":%" PRIu64 ","
                "\"wait_p50_le_ns\":%" PRIu64 ","
                "\"wait_p95_le_ns\":%" PRIu64 ","
                "\"wait_p99_le_ns\":%" PRIu64 ","
                "\"hold_avg_ns\":%" PRIu64 ","
                "\"hold_max_ns\":%" PRIu64 ","
                "\"hold_p50_le_ns\":%" PRIu64 ","
                "\"hold_p95_le_ns\":%" PRIu64 ","
                "\"hold_p99_le_ns\":%" PRIu64 ","
                "\"cond_wait_samples\":%" PRIu64 ","
                "\"cond_wait_avg_ns\":%" PRIu64 ","
                "\"cond_wait_max_ns\":%" PRIu64 ","
                "\"cond_wait_p50_le_ns\":%" PRIu64 ","
                "\"cond_wait_p95_le_ns\":%" PRIu64 ","
                "\"cond_wait_p99_le_ns\":%" PRIu64 ","
                "\"order_violations\":%" PRIu64 "}\n",
                s->name,
                s->samples,
                wait_avg,
                s->wait_max_ns,
                percentile_from_buckets(s->wait_buckets, s->samples, 50),
                percentile_from_buckets(s->wait_buckets, s->samples, 95),
                percentile_from_buckets(s->wait_buckets, s->samples, 99),
                hold_avg,
                s->hold_max_ns,
                percentile_from_buckets(s->hold_buckets, s->samples, 50),
                percentile_from_buckets(s->hold_buckets, s->samples, 95),
                percentile_from_buckets(s->hold_buckets, s->samples, 99),
                s->cond_wait_samples,
                cond_wait_avg,
                s->cond_wait_max_ns,
                percentile_from_buckets(s->cond_wait_buckets,
                                        s->cond_wait_samples, 50),
                percentile_from_buckets(s->cond_wait_buckets,
                                        s->cond_wait_samples, 95),
                percentile_from_buckets(s->cond_wait_buckets,
                                        s->cond_wait_samples, 99),
                s->order_violations);
    }
    fprintf(out, "{\"event\":\"lock_profile_summary_end\"}\n");
}

static void dump_stats_trace_locked(void)
{
    profile_trace_appendf_locked(
        "{\"event\":\"lock_profile_summary_begin\","
        "\"clock\":\"CLOCK_MONOTONIC\","
        "\"percentiles_are_histogram_upper_bounds\":true}\n");
    for (size_t i = 0; i < profile_stat_count(); i++) {
        const pqc_lock_profile_stat_t *s = &g_profile_stats[i];
        uint64_t wait_avg = s->samples ? s->wait_total_ns / s->samples : 0;
        uint64_t hold_avg = s->samples ? s->hold_total_ns / s->samples : 0;
        uint64_t cond_wait_avg = s->cond_wait_samples
            ? s->cond_wait_total_ns / s->cond_wait_samples
            : 0;
        profile_trace_appendf_locked(
            "{\"event\":\"lock_profile_summary\","
            "\"lock\":\"%s\","
            "\"samples\":%" PRIu64 ","
            "\"wait_avg_ns\":%" PRIu64 ","
            "\"wait_max_ns\":%" PRIu64 ","
            "\"wait_p50_le_ns\":%" PRIu64 ","
            "\"wait_p95_le_ns\":%" PRIu64 ","
            "\"wait_p99_le_ns\":%" PRIu64 ","
            "\"hold_avg_ns\":%" PRIu64 ","
            "\"hold_max_ns\":%" PRIu64 ","
            "\"hold_p50_le_ns\":%" PRIu64 ","
            "\"hold_p95_le_ns\":%" PRIu64 ","
            "\"hold_p99_le_ns\":%" PRIu64 ","
            "\"cond_wait_samples\":%" PRIu64 ","
            "\"cond_wait_avg_ns\":%" PRIu64 ","
            "\"cond_wait_max_ns\":%" PRIu64 ","
            "\"cond_wait_p50_le_ns\":%" PRIu64 ","
            "\"cond_wait_p95_le_ns\":%" PRIu64 ","
            "\"cond_wait_p99_le_ns\":%" PRIu64 ","
            "\"order_violations\":%" PRIu64 "}\n",
            s->name,
            s->samples,
            wait_avg,
            s->wait_max_ns,
            percentile_from_buckets(s->wait_buckets, s->samples, 50),
            percentile_from_buckets(s->wait_buckets, s->samples, 95),
            percentile_from_buckets(s->wait_buckets, s->samples, 99),
            hold_avg,
            s->hold_max_ns,
            percentile_from_buckets(s->hold_buckets, s->samples, 50),
            percentile_from_buckets(s->hold_buckets, s->samples, 95),
            percentile_from_buckets(s->hold_buckets, s->samples, 99),
            s->cond_wait_samples,
            cond_wait_avg,
            s->cond_wait_max_ns,
            percentile_from_buckets(s->cond_wait_buckets,
                                    s->cond_wait_samples, 50),
            percentile_from_buckets(s->cond_wait_buckets,
                                    s->cond_wait_samples, 95),
            percentile_from_buckets(s->cond_wait_buckets,
                                    s->cond_wait_samples, 99),
            s->order_violations);
    }
    profile_trace_appendf_locked("{\"event\":\"lock_profile_summary_end\"}\n");
}

int pqc_lock_profile_init_from_config(void)
{
    if (atomic_load_explicit(&g_profile_initialized, memory_order_acquire))
        return 0;

    int lock_rc = profile_internal_lock();
    if (lock_rc != 0)
        return -lock_rc;
    if (atomic_load_explicit(&g_profile_initialized, memory_order_relaxed)) {
        int unlock_rc = profile_internal_unlock();
        return unlock_rc != 0 ? -unlock_rc : 0;
    }

    const char *path = pqc_config_get_nonempty("PQC_LOCK_PROFILE_PATH");
    int enabled = path != NULL || pqc_config_enabled("PQC_LOCK_PROFILE") ||
                  pqc_config_enabled("PQC_LOCK_PROFILE_SUMMARY");
    int dump_on_shutdown =
        enabled && (path == NULL || pqc_config_enabled("PQC_LOCK_PROFILE_SUMMARY"));

    if (path) {
        g_profile_fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
        if (g_profile_fd < 0) {
            int err = errno ? errno : EIO;
            atomic_store_explicit(&g_profile_initialized, 1,
                                  memory_order_release);
            atomic_store_explicit(&g_profile_enabled, 0,
                                  memory_order_release);
            int unlock_rc = profile_internal_unlock();
            if (unlock_rc != 0)
                return -unlock_rc;
            return -err;
        }
        profile_trace_appendf_locked(
            "{\"event\":\"lock_profile_begin\","
            "\"clock\":\"CLOCK_MONOTONIC\","
            "\"path_recorded_by_harness\":true,"
            "\"histogram_buckets\":%d}\n",
            PQC_LOCK_PROFILE_BUCKETS);
    }

    profile_stats_reset_locked();
    g_profile_sequence = 0;
    atomic_store_explicit(&g_profile_dump_on_shutdown, dump_on_shutdown,
                          memory_order_release);
    atomic_store_explicit(&g_profile_enabled, enabled, memory_order_release);
    atomic_store_explicit(&g_profile_initialized, 1, memory_order_release);
    int unlock_rc = profile_internal_unlock();
    return unlock_rc != 0 ? -unlock_rc : 0;
}

void pqc_lock_profile_shutdown(void)
{
    if (!atomic_load_explicit(&g_profile_initialized, memory_order_acquire))
        return;

    if (profile_internal_lock() != 0)
        return;
    if (atomic_load_explicit(&g_profile_enabled, memory_order_relaxed)) {
        if (atomic_load_explicit(&g_profile_dump_on_shutdown,
                                 memory_order_relaxed))
            dump_stats_locked(stderr);
        if (g_profile_fd >= 0) {
            dump_stats_trace_locked();
            profile_trace_appendf_locked(
                "{\"event\":\"lock_profile_end\","
                "\"events\":%" PRIu64 "}\n",
                g_profile_sequence);
            (void)close(g_profile_fd);
            g_profile_fd = -1;
        }
    }
    atomic_store_explicit(&g_profile_enabled, 0, memory_order_release);
    (void)profile_internal_unlock();
}

int pqc_lock_profile_enabled(void)
{
    if (!atomic_load_explicit(&g_profile_initialized, memory_order_acquire))
        (void)pqc_lock_profile_init_from_config();
    return atomic_load_explicit(&g_profile_enabled, memory_order_acquire);
}

static int profile_enabled_fast(void)
{
    if (atomic_load_explicit(&g_profile_initialized, memory_order_acquire))
        return atomic_load_explicit(&g_profile_enabled, memory_order_relaxed);
    return pqc_lock_profile_enabled();
}

void pqc_lock_profile_dump(FILE *out)
{
    if (profile_internal_lock() != 0)
        return;
    dump_stats_locked(out);
    (void)profile_internal_unlock();
}

int pqc_profiled_mutex_lock(pthread_mutex_t *mutex, const char *lock_name,
                            const char *site,
                            pqc_lock_profile_scope_t *scope)
{
    if (scope)
        scope->enabled = 0;
    if (!profile_enabled_fast())
        return pthread_mutex_lock(mutex);

    int rank = lock_rank_for_name(lock_name);
    int order_rc = lock_order_check_before_lock(mutex, lock_name, site, rank);
    if (order_rc != 0)
        return order_rc;

    uint64_t wait_start_ns = monotonic_ns();
    int rc = pthread_mutex_lock(mutex);
    uint64_t acquired_ns = monotonic_ns();
    if (rc == 0) {
        if (scope) {
            scope->wait_start_ns = wait_start_ns;
            scope->acquired_ns = acquired_ns;
            scope->enabled = 1;
        }
        lock_order_push(mutex, lock_name, site, rank);
    } else if (rc != 0) {
        if (profile_internal_lock() == 0) {
            profile_trace_appendf_locked(
                "{\"event\":\"lock_acquire_error\","
                "\"seq\":%" PRIu64 ","
                "\"lock\":\"%s\",\"site\":\"%s\",\"rc\":%d}\n",
                ++g_profile_sequence,
                lock_name ? lock_name : "unknown",
                site ? site : "unknown",
                rc);
            (void)profile_internal_unlock();
        }
    }
    return rc;
}

int pqc_profiled_mutex_unlock(pthread_mutex_t *mutex, const char *lock_name,
                              const char *site,
                              pqc_lock_profile_scope_t *scope)
{
    if (!scope || !scope->enabled)
        return pthread_mutex_unlock(mutex);

    int rank = lock_rank_for_name(lock_name);
    uint64_t release_ns = monotonic_ns();
    int rc = pthread_mutex_unlock(mutex);
    if (rc == 0)
        lock_order_pop(mutex, lock_name, site, rank);
    record_scope_segment(mutex, lock_name, site, scope, release_ns, rc,
                         "lock_hold", NULL);
    return rc;
}

int pqc_profiled_cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex,
                           const char *lock_name, const char *site,
                           pqc_lock_profile_scope_t *scope)
{
    if (!scope || !scope->enabled)
        return pthread_cond_wait(cond, mutex);

    uint64_t release_ns = monotonic_ns();
    record_scope_segment(mutex, lock_name, site, scope, release_ns, 0,
                         "lock_hold_segment", "cond_wait");

    uint64_t cond_start_ns = monotonic_ns();
    int rc = pthread_cond_wait(cond, mutex);
    uint64_t acquired_ns = monotonic_ns();
    if (profile_enabled_fast()) {
        uint64_t cond_wait_ns = elapsed_ns(acquired_ns, cond_start_ns);
        if (profile_internal_lock() == 0) {
            profile_trace_appendf_locked(
                "{\"event\":\"lock_cond_wait\","
                "\"seq\":%" PRIu64 ","
                "\"lock\":\"%s\",\"site\":\"%s\","
                "\"thread\":\"%llu\",\"addr\":\"%p\","
                "\"cond_wait_ns\":%" PRIu64 ",\"wait_rc\":%d}\n",
                ++g_profile_sequence,
                lock_name ? lock_name : "unknown",
                site ? site : "unknown",
                (unsigned long long)(uintptr_t)pthread_self(),
                (void *)mutex,
                cond_wait_ns,
                rc);
            record_cond_wait_sample_locked(lock_name, cond_wait_ns);
            (void)profile_internal_unlock();
        }
        scope->wait_start_ns = acquired_ns;
        scope->acquired_ns = acquired_ns;
        scope->enabled = 1;
    }
    return rc;
}

int pqc_profiled_cond_timedwait(pthread_cond_t *cond,
                                pthread_mutex_t *mutex,
                                const char *lock_name,
                                const char *site,
                                pqc_lock_profile_scope_t *scope,
                                const struct timespec *abstime)
{
    if (!scope || !scope->enabled)
        return pthread_cond_timedwait(cond, mutex, abstime);

    uint64_t release_ns = monotonic_ns();
    record_scope_segment(mutex, lock_name, site, scope, release_ns, 0,
                         "lock_hold_segment", "cond_timedwait");

    uint64_t cond_start_ns = monotonic_ns();
    int rc = pthread_cond_timedwait(cond, mutex, abstime);
    uint64_t acquired_ns = monotonic_ns();
    if (profile_enabled_fast()) {
        uint64_t cond_wait_ns = elapsed_ns(acquired_ns, cond_start_ns);
        if (profile_internal_lock() == 0) {
            profile_trace_appendf_locked(
                "{\"event\":\"lock_cond_timedwait\","
                "\"seq\":%" PRIu64 ","
                "\"lock\":\"%s\",\"site\":\"%s\","
                "\"thread\":\"%llu\",\"addr\":\"%p\","
                "\"cond_wait_ns\":%" PRIu64 ",\"wait_rc\":%d}\n",
                ++g_profile_sequence,
                lock_name ? lock_name : "unknown",
                site ? site : "unknown",
                (unsigned long long)(uintptr_t)pthread_self(),
                (void *)mutex,
                cond_wait_ns,
                rc);
            record_cond_wait_sample_locked(lock_name, cond_wait_ns);
            (void)profile_internal_unlock();
        }
        scope->wait_start_ns = acquired_ns;
        scope->acquired_ns = acquired_ns;
        scope->enabled = 1;
    }
    return rc;
}

static void *lock_profile_cond_test_signal(void *arg)
{
    pqc_lock_profile_cond_test_arg_t *test =
        (pqc_lock_profile_cond_test_arg_t *)arg;
    if (!test)
        return NULL;
    pthread_mutex_lock(test->mutex);
    *test->ready = 1;
    pthread_cond_signal(test->cond);
    pthread_mutex_unlock(test->mutex);
    return NULL;
}

int pqc_lock_profile_self_test(FILE *out)
{
    const char *names[] = {
        "file_state_table_lock",
        "fd_lock",
        "commit_lock",
        "committed_map_lock",
        "anchor_pending_lock",
        "file_anchor_commit_lock",
        "anchor_epoch_record_lock",
        "anchor_lifecycle_lock",
        "anchor_worker_lock",
        "rekey_lifecycle_lock",
        "rekey_queue_lock",
        "epoch_barrier_lock",
        "parallel_shard_lock",
        "parallel_runtime_lock",
        "admission_state_lock",
        "admission_trace_lock",
        "qos_gpu_load_lock",
        "qos_throttle_lock",
        "scheduler_lock",
        "fault_cutpoint_lock",
        "trace_sink_lock",
    };
    const size_t count = sizeof(names) / sizeof(names[0]);
    pthread_mutex_t locks[sizeof(names) / sizeof(names[0])];

    if (profile_internal_lock() != 0)
        return -1;
    profile_stats_reset_locked();
    g_profile_sequence = 0;
    atomic_store_explicit(&g_profile_initialized, 1, memory_order_release);
    atomic_store_explicit(&g_profile_enabled, 1, memory_order_release);
    atomic_store_explicit(&g_profile_dump_on_shutdown, 0,
                          memory_order_release);
    if (profile_internal_unlock() != 0)
        return -1;
    g_lock_order_depth = 0;

    int ok = 1;
    for (size_t i = 0; i < count; i++) {
        if (pthread_mutex_init(&locks[i], NULL) != 0) {
            ok = 0;
            continue;
        }
        for (int iter = 0; iter < 3; iter++) {
            pqc_lock_profile_scope_t scope;
            if (pqc_profiled_mutex_lock(&locks[i], names[i],
                                        "lock_profile_self_test",
                                        &scope) != 0) {
                ok = 0;
                continue;
            }
            if (pqc_profiled_mutex_unlock(&locks[i], names[i],
                                          "lock_profile_self_test",
                                          &scope) != 0)
                ok = 0;
        }
        if (pthread_mutex_destroy(&locks[i]) != 0)
            ok = 0;
    }

    if (profile_internal_lock() != 0) {
        ok = 0;
    } else {
        for (size_t i = 0; i < count; i++) {
            pqc_lock_profile_stat_t *stat = profile_stat_for_lock(names[i]);
            if (!stat || stat->samples != 3)
                ok = 0;
        }
        if (profile_internal_unlock() != 0)
            ok = 0;
    }

    pthread_mutex_t cond_mutex;
    pthread_cond_t cond;
    pthread_t signal_thread;
    int ready = 0;
    if (pthread_mutex_init(&cond_mutex, NULL) != 0) {
        ok = 0;
    } else if (pthread_cond_init(&cond, NULL) != 0) {
        (void)pthread_mutex_destroy(&cond_mutex);
        ok = 0;
    } else {
        pqc_lock_profile_cond_test_arg_t arg = {
            .mutex = &cond_mutex,
            .cond = &cond,
            .ready = &ready,
        };
        pqc_lock_profile_scope_t scope;
        if (pqc_profiled_mutex_lock(&cond_mutex, "fd_lock",
                                    "lock_profile_cond_self_test",
                                    &scope) != 0) {
            ok = 0;
        } else if (pthread_create(&signal_thread, NULL,
                                  lock_profile_cond_test_signal,
                                  &arg) != 0) {
            (void)pqc_profiled_mutex_unlock(&cond_mutex, "fd_lock",
                                            "lock_profile_cond_self_test",
                                            &scope);
            ok = 0;
        } else {
            while (!ready) {
                if (pqc_profiled_cond_wait(&cond, &cond_mutex, "fd_lock",
                                           "lock_profile_cond_self_test",
                                           &scope) != 0) {
                    ok = 0;
                    break;
                }
            }
            if (pqc_profiled_mutex_unlock(&cond_mutex, "fd_lock",
                                          "lock_profile_cond_self_test",
                                          &scope) != 0)
                ok = 0;
            if (pthread_join(signal_thread, NULL) != 0)
                ok = 0;
        }
        if (pthread_cond_destroy(&cond) != 0)
            ok = 0;
        if (pthread_mutex_destroy(&cond_mutex) != 0)
            ok = 0;
    }

    pthread_mutex_t low_lock;
    pthread_mutex_t high_lock;
    int low_lock_initialized = pthread_mutex_init(&low_lock, NULL) == 0;
    int high_lock_initialized = pthread_mutex_init(&high_lock, NULL) == 0;
    if (!low_lock_initialized || !high_lock_initialized) {
        ok = 0;
    } else {
        pqc_lock_profile_scope_t low_scope;
        pqc_lock_profile_scope_t high_scope;
        if (pqc_profiled_mutex_lock(&low_lock, "fd_lock",
                                    "lock_order_allowed_self_test",
                                    &low_scope) != 0) {
            ok = 0;
        } else if (pqc_profiled_mutex_lock(&high_lock, "commit_lock",
                                           "lock_order_allowed_self_test",
                                           &high_scope) != 0) {
            (void)pqc_profiled_mutex_unlock(
                &low_lock, "fd_lock", "lock_order_allowed_self_test",
                &low_scope);
            ok = 0;
        } else {
            if (pqc_profiled_mutex_unlock(
                    &high_lock, "commit_lock",
                    "lock_order_allowed_self_test", &high_scope) != 0)
                ok = 0;
            if (pqc_profiled_mutex_unlock(
                    &low_lock, "fd_lock", "lock_order_allowed_self_test",
                    &low_scope) != 0)
                ok = 0;
        }

        if (pqc_profiled_mutex_lock(&high_lock, "commit_lock",
                                    "lock_order_inversion_self_test",
                                    &high_scope) != 0) {
            ok = 0;
        } else {
            int inversion_rc = pqc_profiled_mutex_lock(
                &low_lock, "fd_lock", "lock_order_inversion_self_test",
                &low_scope);
            if (inversion_rc != EDEADLK)
                ok = 0;
            if (pqc_profiled_mutex_unlock(
                    &high_lock, "commit_lock",
                    "lock_order_inversion_self_test", &high_scope) != 0)
                ok = 0;
        }
    }
    if (low_lock_initialized && pthread_mutex_destroy(&low_lock) != 0)
        ok = 0;
    if (high_lock_initialized && pthread_mutex_destroy(&high_lock) != 0)
        ok = 0;

    if (profile_internal_lock() != 0) {
        ok = 0;
    } else {
        pqc_lock_profile_stat_t *fd_stat = profile_stat_for_lock("fd_lock");
        pqc_lock_profile_stat_t *commit_stat =
            profile_stat_for_lock("commit_lock");
        if (!fd_stat || fd_stat->cond_wait_samples == 0 ||
            fd_stat->samples < 5)
            ok = 0;
        if (!commit_stat || commit_stat->samples < 5 || !fd_stat ||
            fd_stat->order_violations == 0)
            ok = 0;
        if (g_lock_order_depth != 0)
            ok = 0;
        dump_stats_locked(out);
        atomic_store_explicit(&g_profile_enabled, 0, memory_order_release);
        if (profile_internal_unlock() != 0)
            ok = 0;
    }
    return ok ? 0 : -1;
}
