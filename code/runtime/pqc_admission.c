/**
 * pqc_admission.c — QoS-Aware Admission Control Framework (M5)
 *
 * Implementation status: M5 PHASE 1 SKELETON
 *
 * This file contains the explicit admission controller and its trace-based
 * policy hooks. The current implementation is telemetry-driven and keeps the
 * coherence path as a rolling software proxy until hardware counters are wired.
 *
 * Blocking items (M5 prerequisite):
 *   1. M4 completion: file-key lifecycle must be operational.
 *   2. GPU occupancy telemetry: CUPTI or similar (M6, but need mock for M5).
 *   3. Scheduler trace output format and file rotation.
 *   4. Microbenchmark cost model for GPU staging (H2D, D2H latency).
 *
 * Copyright 2025 AEGIS-Q Authors. Apache-2.0.
 */

#include "pqc_admission.h"
#include "pqc_block_job.h"
#include "pqc_config.h"
#include "pqc_format.h"
#include "pqc_lock_profile.h"
#include "pqc_posix.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <sys/xattr.h>
#include <unistd.h>

#define PQC_ADMISSION_DEFAULT_GPU_MIN_BATCH_BYTES 4096ULL
#define PQC_ADMISSION_DEFAULT_AI_QOS_MIN_BUDGET_NS 1730000ULL
#define PQC_ADMISSION_DEFAULT_DEADLINE_MARGIN_NS 100000ULL
#define PQC_ADMISSION_DEFAULT_PRODUCER_SLACK_STALE_NS 250000000ULL
#define PQC_ADMISSION_DEFAULT_QUEUE_PRESSURE_THRESHOLD 0.8

/* Global state (protected by state_lock, trace_fd protected by trace_lock) */
static struct {
    int trace_fd;
    pthread_mutex_t trace_lock;
    pthread_mutex_t state_lock;
    uint64_t ai_budget_ns;
    uint64_t ai_queue_depth;
    uint64_t ai_budget_update_mono_ns;

    /* Cumulative statistics */
    pqc_admission_stats_t stats;

    /* Config parameters (tunable) */
    uint64_t gpu_min_batch_bytes;
    uint64_t ai_qos_min_budget_ns;
    uint64_t deadline_margin_ns;
    uint64_t producer_slack_stale_ns;
    double queue_pressure_threshold;  /* ratio: cpu_depth / gpu_depth */
    uint64_t uma_window_bytes;
    uint64_t uma_window_latency_ns;
    uint64_t uma_sample_count;
    uint64_t uma_latency_ema_ns;
    uint64_t uma_bytes_ema;
} g_admission __attribute__((unused)) = {
    .trace_fd = -1,
    .trace_lock = PTHREAD_MUTEX_INITIALIZER,
    .state_lock = PTHREAD_MUTEX_INITIALIZER,
    .ai_budget_ns = 0,
    .ai_queue_depth = 0,
    .ai_budget_update_mono_ns = 0,
    .stats = {0},
    .gpu_min_batch_bytes = PQC_ADMISSION_DEFAULT_GPU_MIN_BATCH_BYTES,
    .ai_qos_min_budget_ns = PQC_ADMISSION_DEFAULT_AI_QOS_MIN_BUDGET_NS,
    .deadline_margin_ns = PQC_ADMISSION_DEFAULT_DEADLINE_MARGIN_NS,
    .producer_slack_stale_ns = PQC_ADMISSION_DEFAULT_PRODUCER_SLACK_STALE_NS,
    .queue_pressure_threshold = PQC_ADMISSION_DEFAULT_QUEUE_PRESSURE_THRESHOLD,
    .uma_window_bytes = 0,
    .uma_window_latency_ns = 0,
    .uma_sample_count = 0,
    .uma_latency_ema_ns = 0,
    .uma_bytes_ema = 0,
};

static struct {
    double mem_bandwidth_util;
    double tensor_core_util;
} g_telemetry = {0.0, 0.0};

typedef struct {
    uint64_t ai_budget_ns;
    uint64_t ai_queue_depth;
    uint64_t ai_budget_update_mono_ns;
    pqc_admission_stats_t stats;
    uint64_t gpu_min_batch_bytes;
    uint64_t ai_qos_min_budget_ns;
    uint64_t deadline_margin_ns;
    uint64_t producer_slack_stale_ns;
    double queue_pressure_threshold;
    uint64_t uma_window_bytes;
    uint64_t uma_window_latency_ns;
    uint64_t uma_sample_count;
    uint64_t uma_latency_ema_ns;
    uint64_t uma_bytes_ema;
    double telemetry_mem_bandwidth_util;
    double telemetry_tensor_core_util;
} admission_state_snapshot_t;

static uint64_t monotonic_now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static int admission_trace_lock(pqc_lock_profile_scope_t *scope,
                                const char *site)
{
    return pqc_profiled_mutex_lock(&g_admission.trace_lock,
                                   "admission_trace_lock", site, scope);
}

static int admission_trace_unlock(pqc_lock_profile_scope_t *scope,
                                  const char *site)
{
    return pqc_profiled_mutex_unlock(&g_admission.trace_lock,
                                     "admission_trace_lock", site, scope);
}

static int admission_state_lock(pqc_lock_profile_scope_t *scope,
                                const char *site)
{
    return pqc_profiled_mutex_lock(&g_admission.state_lock,
                                   "admission_state_lock", site, scope);
}

static int admission_state_unlock(pqc_lock_profile_scope_t *scope,
                                  const char *site)
{
    return pqc_profiled_mutex_unlock(&g_admission.state_lock,
                                     "admission_state_lock", site, scope);
}

static void admission_state_snapshot_locked(admission_state_snapshot_t *snapshot)
{
    if (!snapshot)
        return;
    snapshot->ai_budget_ns = g_admission.ai_budget_ns;
    snapshot->ai_queue_depth = g_admission.ai_queue_depth;
    snapshot->ai_budget_update_mono_ns = g_admission.ai_budget_update_mono_ns;
    memcpy(&snapshot->stats, &g_admission.stats, sizeof(snapshot->stats));
    snapshot->gpu_min_batch_bytes = g_admission.gpu_min_batch_bytes;
    snapshot->ai_qos_min_budget_ns = g_admission.ai_qos_min_budget_ns;
    snapshot->deadline_margin_ns = g_admission.deadline_margin_ns;
    snapshot->producer_slack_stale_ns = g_admission.producer_slack_stale_ns;
    snapshot->queue_pressure_threshold = g_admission.queue_pressure_threshold;
    snapshot->uma_window_bytes = g_admission.uma_window_bytes;
    snapshot->uma_window_latency_ns = g_admission.uma_window_latency_ns;
    snapshot->uma_sample_count = g_admission.uma_sample_count;
    snapshot->uma_latency_ema_ns = g_admission.uma_latency_ema_ns;
    snapshot->uma_bytes_ema = g_admission.uma_bytes_ema;
    snapshot->telemetry_mem_bandwidth_util = g_telemetry.mem_bandwidth_util;
    snapshot->telemetry_tensor_core_util = g_telemetry.tensor_core_util;
}

static void admission_state_restore_locked(const admission_state_snapshot_t *snapshot)
{
    if (!snapshot)
        return;
    g_admission.ai_budget_ns = snapshot->ai_budget_ns;
    g_admission.ai_queue_depth = snapshot->ai_queue_depth;
    g_admission.ai_budget_update_mono_ns = snapshot->ai_budget_update_mono_ns;
    memcpy(&g_admission.stats, &snapshot->stats, sizeof(g_admission.stats));
    g_admission.gpu_min_batch_bytes = snapshot->gpu_min_batch_bytes;
    g_admission.ai_qos_min_budget_ns = snapshot->ai_qos_min_budget_ns;
    g_admission.deadline_margin_ns = snapshot->deadline_margin_ns;
    g_admission.producer_slack_stale_ns = snapshot->producer_slack_stale_ns;
    g_admission.queue_pressure_threshold = snapshot->queue_pressure_threshold;
    g_admission.uma_window_bytes = snapshot->uma_window_bytes;
    g_admission.uma_window_latency_ns = snapshot->uma_window_latency_ns;
    g_admission.uma_sample_count = snapshot->uma_sample_count;
    g_admission.uma_latency_ema_ns = snapshot->uma_latency_ema_ns;
    g_admission.uma_bytes_ema = snapshot->uma_bytes_ema;
    g_telemetry.mem_bandwidth_util = snapshot->telemetry_mem_bandwidth_util;
    g_telemetry.tensor_core_util = snapshot->telemetry_tensor_core_util;
}

static void admission_init_rollback_published_state(
    const admission_state_snapshot_t *snapshot)
{
    pqc_lock_profile_scope_t rollback_scope;
    if (admission_state_lock(&rollback_scope, "admission_init_rollback") != 0) {
        fprintf(stderr,
                "[PQC] Admission init rollback failed: state lock unavailable\n");
        return;
    }
    admission_state_restore_locked(snapshot);
    (void)admission_state_unlock(&rollback_scope, "admission_init_rollback");
}

static void admission_trace_write_locked(const char *line)
{
    if (!line || g_admission.trace_fd < 0)
        return;
    size_t len = strlen(line);
    size_t off = 0;
    while (off < len) {
        ssize_t written = write(g_admission.trace_fd, line + off, len - off);
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

static void admission_trace_append(const char *line)
{
    if (!line)
        return;
    pqc_lock_profile_scope_t trace_scope;
    if (admission_trace_lock(&trace_scope, "admission_trace_append") != 0)
        return;
    admission_trace_write_locked(line);
    (void)admission_trace_unlock(&trace_scope, "admission_trace_append");
}

static void scheduler_trace_log_snapshot(const pqc_admission_context_t *ctx,
                                         double mem_bandwidth_util,
                                         double tensor_core_util);

enum admission_stats_delta {
    ADMISSION_STATS_GPU_ADMITTED = 1u << 0,
    ADMISSION_STATS_CPU_ROUTED = 1u << 1,
    ADMISSION_STATS_AI_BUDGET_EXHAUSTED = 1u << 2,
    ADMISSION_STATS_COHERENCE_RISK = 1u << 3,
    ADMISSION_STATS_QUEUE_PRESSURE = 1u << 4,
    ADMISSION_STATS_DEADLINE_EXCEEDED = 1u << 5,
    ADMISSION_STATS_SIZE_TOO_SMALL = 1u << 6,
    ADMISSION_STATS_STALE_TELEMETRY = 1u << 7,
};

static void admission_stats_record(unsigned flags)
{
    pqc_lock_profile_scope_t stats_scope;
    if (admission_state_lock(&stats_scope, __func__) != 0)
        return;

    g_admission.stats.total_requests++;
    if (flags & ADMISSION_STATS_GPU_ADMITTED)
        g_admission.stats.gpu_admitted_count++;
    if (flags & ADMISSION_STATS_CPU_ROUTED)
        g_admission.stats.cpu_routed_count++;
    if (flags & ADMISSION_STATS_AI_BUDGET_EXHAUSTED)
        g_admission.stats.ai_budget_exhausted_count++;
    if (flags & ADMISSION_STATS_COHERENCE_RISK)
        g_admission.stats.coherence_risk_count++;
    if (flags & ADMISSION_STATS_QUEUE_PRESSURE)
        g_admission.stats.queue_pressure_count++;
    if (flags & ADMISSION_STATS_DEADLINE_EXCEEDED)
        g_admission.stats.deadline_exceeded_count++;
    if (flags & ADMISSION_STATS_SIZE_TOO_SMALL)
        g_admission.stats.size_too_small_count++;
    if (flags & ADMISSION_STATS_STALE_TELEMETRY)
        g_admission.stats.stale_telemetry_count++;

    (void)admission_state_unlock(&stats_scope, __func__);
}

const char *pqc_qos_class_name(int qos_class)
{
    return qos_class == PQC_QOS_CLASS_LATENCY ? "latency" : "elastic";
}

int pqc_parse_qos_class_value(const char *value, size_t size, int *out)
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
    return pqc_parse_qos_class_value(value, (size_t)n, out);
}

int pqc_qos_class_load_for_path(const char *phys_path, int *out)
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
        if (!pqc_path_has_suffix(phys_path, suffix))
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


/* ──────────────────────────────────────────────────────────────────────────
 *  Initialization & Shutdown
 * ────────────────────────────────────────────────────────────────────────── */

int pqc_admission_init(const char *trace_file_path)
{
    /* Read policy parameters from environment.  Legacy helpers preserve the
     * previous strto* prefix/invalid handling while recording parse status. */
    uint64_t gpu_min_batch_bytes =
        pqc_config_u64_legacy_or_default("PQC_GPU_MIN_BATCH_BYTES",
                                         PQC_ADMISSION_DEFAULT_GPU_MIN_BATCH_BYTES);
    uint64_t ai_qos_min_budget_ns =
        pqc_config_u64_legacy_or_default("PQC_AI_QOS_MIN_BUDGET_NS",
                                         PQC_ADMISSION_DEFAULT_AI_QOS_MIN_BUDGET_NS);
    uint64_t deadline_margin_ns =
        pqc_config_u64_legacy_or_default("PQC_DEADLINE_MARGIN_NS",
                                         PQC_ADMISSION_DEFAULT_DEADLINE_MARGIN_NS);
    uint64_t producer_slack_stale_ns =
        pqc_config_u64_legacy_or_default("PQC_PRODUCER_SLACK_STALE_NS",
                                         PQC_ADMISSION_DEFAULT_PRODUCER_SLACK_STALE_NS);
    double queue_pressure_threshold =
        pqc_config_double_legacy_or_default(
            "PQC_QUEUE_PRESSURE_THRESHOLD",
            PQC_ADMISSION_DEFAULT_QUEUE_PRESSURE_THRESHOLD);
    double telemetry_mem_bandwidth =
        pqc_config_double_legacy_or_default("PQC_TELEMETRY_MEM_BANDWIDTH", 0.0);
    double telemetry_tensor_core =
        pqc_config_double_legacy_or_default("PQC_TELEMETRY_TENSOR_CORE", 0.0);

    uint64_t ai_budget_ns = 0;
    uint64_t ai_budget_update_mono_ns = 0;
    if (pqc_config_present("PQC_ADMISSION_INITIAL_BUDGET_NS")) {
        ai_budget_ns =
            pqc_config_u64_legacy_or_default("PQC_ADMISSION_INITIAL_BUDGET_NS",
                                             0);
        ai_budget_update_mono_ns = monotonic_now_ns();
    }
    uint64_t ai_queue_depth =
        pqc_config_u64_legacy_or_default("PQC_ADMISSION_INITIAL_QUEUE_DEPTH",
                                         0);

    int trace_fd = -1;
    if (trace_file_path && trace_file_path[0]) {
        trace_fd = open(trace_file_path,
                        O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
        if (trace_fd < 0) {
            fprintf(stderr, "[PQC] Failed to open trace file: %s\n",
                    trace_file_path);
            return -1;
        }
    }

    pqc_lock_profile_scope_t state_scope;
    if (admission_state_lock(&state_scope, "admission_init_policy") != 0) {
        if (trace_fd >= 0)
            (void)close(trace_fd);
        return -1;
    }

    admission_state_snapshot_t previous_state;
    admission_state_snapshot_locked(&previous_state);

    g_admission.gpu_min_batch_bytes = gpu_min_batch_bytes;
    g_admission.ai_qos_min_budget_ns = ai_qos_min_budget_ns;
    g_admission.deadline_margin_ns = deadline_margin_ns;
    g_admission.producer_slack_stale_ns = producer_slack_stale_ns;
    g_admission.queue_pressure_threshold = queue_pressure_threshold;
    g_admission.ai_budget_ns = ai_budget_ns;
    g_admission.ai_budget_update_mono_ns = ai_budget_update_mono_ns;
    g_admission.ai_queue_depth = ai_queue_depth;
    g_admission.uma_window_bytes = 0;
    g_admission.uma_window_latency_ns = 0;
    g_admission.uma_sample_count = 0;
    g_admission.uma_latency_ema_ns = 0;
    g_admission.uma_bytes_ema = 0;
    memset(&g_admission.stats, 0, sizeof(g_admission.stats));
    g_telemetry.mem_bandwidth_util = telemetry_mem_bandwidth;
    g_telemetry.tensor_core_util = telemetry_tensor_core;

    (void)admission_state_unlock(&state_scope, "admission_init_policy");

    pqc_lock_profile_scope_t trace_scope;
    if (admission_trace_lock(&trace_scope, "admission_init_trace_fd") != 0) {
        if (trace_fd >= 0)
            (void)close(trace_fd);
        admission_init_rollback_published_state(&previous_state);
        return -1;
    }
    if (g_admission.trace_fd >= 0)
        (void)close(g_admission.trace_fd);
    g_admission.trace_fd = trace_fd;
    (void)admission_trace_unlock(&trace_scope, "admission_init_trace_fd");

    char header[512];
    snprintf(header, sizeof(header),
             "# pqc_admission_init: "
             "gpu_min_batch=%zu bytes, "
             "ai_qos_min_budget=%llu ns, "
             "deadline_margin=%llu ns, "
             "producer_slack_stale=%llu ns, "
             "queue_threshold=%.2f\n",
             (size_t)gpu_min_batch_bytes,
             (unsigned long long)ai_qos_min_budget_ns,
             (unsigned long long)deadline_margin_ns,
             (unsigned long long)producer_slack_stale_ns,
             queue_pressure_threshold);
    admission_trace_append(header);

    fprintf(stderr, "[PQC] Admission controller initialized: trace=%s\n",
            trace_fd >= 0 ? trace_file_path : "disabled");
    return 0;
}

void pqc_admission_shutdown(void)
{
    pqc_admission_stats_t stats_snapshot;
    memset(&stats_snapshot, 0, sizeof(stats_snapshot));
    int stats_available = 0;
    pqc_lock_profile_scope_t state_scope;
    if (admission_state_lock(&state_scope, __func__) == 0) {
        memcpy(&stats_snapshot, &g_admission.stats, sizeof(stats_snapshot));
        (void)admission_state_unlock(&state_scope, __func__);
        stats_available = 1;
    }

    char final_stats[512];
    if (stats_available) {
        snprintf(final_stats, sizeof(final_stats),
                 "# FINAL STATS: "
                 "total_requests=%llu, "
                 "gpu_admitted=%llu, "
                 "cpu_routed=%llu, "
                 "ai_budget_exhausted=%llu, "
                 "deadline_exceeded=%llu, "
                 "size_too_small=%llu, "
                 "stale_telemetry=%llu\n",
                 (unsigned long long)stats_snapshot.total_requests,
                 (unsigned long long)stats_snapshot.gpu_admitted_count,
                 (unsigned long long)stats_snapshot.cpu_routed_count,
                 (unsigned long long)stats_snapshot.ai_budget_exhausted_count,
                 (unsigned long long)stats_snapshot.deadline_exceeded_count,
                 (unsigned long long)stats_snapshot.size_too_small_count,
                 (unsigned long long)stats_snapshot.stale_telemetry_count);
    } else {
        snprintf(final_stats, sizeof(final_stats),
                 "# FINAL STATS: unavailable=1\n");
    }
    pqc_lock_profile_scope_t trace_scope;
    if (admission_trace_lock(&trace_scope, "admission_shutdown_trace") == 0) {
        if (g_admission.trace_fd >= 0) {
            admission_trace_write_locked(final_stats);
            (void)close(g_admission.trace_fd);
            g_admission.trace_fd = -1;
        }
        (void)admission_trace_unlock(&trace_scope,
                                     "admission_shutdown_trace");
    }

    fprintf(stderr,
            "[PQC] Admission stats: total=%llu, gpu=%llu, cpu=%llu, "
            "ai_exhausted=%llu, deadline=%llu, size_small=%llu, stale=%llu%s\n",
            (unsigned long long)stats_snapshot.total_requests,
            (unsigned long long)stats_snapshot.gpu_admitted_count,
            (unsigned long long)stats_snapshot.cpu_routed_count,
            (unsigned long long)stats_snapshot.ai_budget_exhausted_count,
            (unsigned long long)stats_snapshot.deadline_exceeded_count,
            (unsigned long long)stats_snapshot.size_too_small_count,
            (unsigned long long)stats_snapshot.stale_telemetry_count,
            stats_available ? "" : ", unavailable=true");
}

/* ──────────────────────────────────────────────────────────────────────────
 *  Route Decision & Tracing
 * ────────────────────────────────────────────────────────────────────────── */

int pqc_admit(pqc_admission_context_t *ctx)
{
    if (!ctx) return -1;

    ctx->decision_reason = 0;
    ctx->deferral_reason = 0;
    ctx->chosen_target = PQC_JOB_CPU;  /* Default: conservative */
    ctx->service_time_ns = ctx->gpu_kernel_est_ns +
                           ctx->gpu_h2d_staging_ns +
                           ctx->gpu_d2h_staging_ns;
    ctx->queue_delay_ns = ctx->cpu_queue_depth * 1000ULL;

    /* ─────────────────────────────────────────────────────────────────────
     * Step 0: Read dynamic telemetry under lock
     * ───────────────────────────────────────────────────────────────────── */
    uint64_t now_mono_ns = monotonic_now_ns();
    uint64_t budget_update_mono_ns;
    uint64_t stale_after_ns;
    uint64_t gpu_min_batch_bytes;
    uint64_t deadline_margin_ns;
    uint64_t uma_sample_count;
    uint64_t uma_latency_ema_ns;
    uint64_t uma_bytes_ema;
    double queue_pressure_threshold;
    pqc_lock_profile_scope_t state_scope;
    if (admission_state_lock(&state_scope, __func__) != 0) {
        ctx->producer_slack_stale_after_ns = 0;
        ctx->producer_slack_age_ns = 0;
        ctx->producer_slack_stale = 1;
        ctx->ai_qos_budget_remaining_ns = 0;
        ctx->chosen_target = PQC_JOB_CPU;
        ctx->decision_reason |= PQC_ROUTE_REASON_STALE_TELEMETRY;
        ctx->deferral_reason = PQC_ROUTE_REASON_STALE_TELEMETRY;
        admission_stats_record(ADMISSION_STATS_STALE_TELEMETRY);
        scheduler_trace_log_snapshot(ctx, 0.0, 0.0);
        return 0;
    }
    ctx->ai_qos_budget_remaining_ns = g_admission.ai_budget_ns;
    budget_update_mono_ns = g_admission.ai_budget_update_mono_ns;
    stale_after_ns = g_admission.producer_slack_stale_ns;
    gpu_min_batch_bytes = g_admission.gpu_min_batch_bytes;
    deadline_margin_ns = g_admission.deadline_margin_ns;
    uma_sample_count = g_admission.uma_sample_count;
    uma_latency_ema_ns = g_admission.uma_latency_ema_ns;
    uma_bytes_ema = g_admission.uma_bytes_ema;
    queue_pressure_threshold = g_admission.queue_pressure_threshold;
    double mem_bw = g_telemetry.mem_bandwidth_util;
    double tc_util = g_telemetry.tensor_core_util;
    (void)admission_state_unlock(&state_scope, __func__);

    ctx->producer_slack_stale_after_ns = stale_after_ns;
    if (budget_update_mono_ns == 0) {
        ctx->producer_slack_age_ns = 0;
        ctx->producer_slack_stale = 0;
    } else {
        ctx->producer_slack_age_ns =
            now_mono_ns >= budget_update_mono_ns ? now_mono_ns - budget_update_mono_ns : 0;
        ctx->producer_slack_stale =
            (stale_after_ns > 0 && ctx->producer_slack_age_ns > stale_after_ns) ? 1u : 0u;
    }

    /* ─────────────────────────────────────────────────────────────────────
     * Step 1: Hard constraints (size and deadline)
     * ───────────────────────────────────────────────────────────────────── */

    if (ctx->bytes_total < gpu_min_batch_bytes) {
        ctx->chosen_target = PQC_JOB_CPU;
        ctx->decision_reason |= PQC_ROUTE_REASON_SIZE_TOO_SMALL;
        ctx->deferral_reason = PQC_ROUTE_REASON_SIZE_TOO_SMALL;
        admission_stats_record(ADMISSION_STATS_SIZE_TOO_SMALL);
        scheduler_trace_log_snapshot(ctx, mem_bw, tc_util);
        return 0;
    }

    if (ctx->batch_age_ns > ctx->ai_inference_deadline_ns) {
        ctx->chosen_target = PQC_JOB_CPU;
        ctx->decision_reason |= PQC_ROUTE_REASON_DEADLINE_ELAPSED;
        ctx->deferral_reason = PQC_ROUTE_REASON_DEADLINE_ELAPSED;
        admission_stats_record(ADMISSION_STATS_DEADLINE_EXCEEDED);
        scheduler_trace_log_snapshot(ctx, mem_bw, tc_util);
        return 0;
    }

    if (ctx->producer_slack_stale) {
        ctx->chosen_target = PQC_JOB_CPU;
        ctx->ai_qos_budget_remaining_ns = 0;
        ctx->decision_reason |= PQC_ROUTE_REASON_STALE_TELEMETRY;
        ctx->deferral_reason = PQC_ROUTE_REASON_STALE_TELEMETRY;
        admission_stats_record(ADMISSION_STATS_STALE_TELEMETRY);
        scheduler_trace_log_snapshot(ctx, mem_bw, tc_util);
        return 0;
    }

    /* ─────────────────────────────────────────────────────────────────────
     * Step 2: AI QoS budget check (priority for inference)
     * ───────────────────────────────────────────────────────────────────── */

    uint64_t expected_service = ctx->gpu_kernel_est_ns +
                                ctx->gpu_h2d_staging_ns +
                                ctx->gpu_d2h_staging_ns;

    if (ctx->ai_qos_budget_remaining_ns < expected_service) {
        ctx->chosen_target = PQC_JOB_CPU;
        ctx->decision_reason |= PQC_ROUTE_REASON_AI_QOS_EXHAUSTED;
        ctx->deferral_reason = PQC_ROUTE_REASON_AI_QOS_EXHAUSTED;
        admission_stats_record(ADMISSION_STATS_AI_BUDGET_EXHAUSTED);
        scheduler_trace_log_snapshot(ctx, mem_bw, tc_util);
        return 0;
    }

    /* ─────────────────────────────────────────────────────────────────────
     * Step 2.5: Phase-Aware Memory Interleaving (Milestone 3)
     * Detect LLM execution phase using memory bandwidth and Tensor Core telemetry.
     * ───────────────────────────────────────────────────────────────────── */
    int is_llm_decoding = (mem_bw > 0.70 && tc_util < 0.30);
    int is_llm_prefill  = (tc_util >= 0.30 && mem_bw <= 0.60);

    if (is_llm_decoding) {
        /* Force PQC encryption to CPU fast lane to avoid GPU memory staging contention. */
        ctx->chosen_target = PQC_JOB_CPU;
        ctx->decision_reason |= PQC_ROUTE_REASON_AI_QOS_EXHAUSTED;
        ctx->deferral_reason = PQC_ROUTE_REASON_AI_QOS_EXHAUSTED;
        admission_stats_record(ADMISSION_STATS_AI_BUDGET_EXHAUSTED);
        scheduler_trace_log_snapshot(ctx, mem_bw, tc_util);
        return 0;
    }

    /* ─────────────────────────────────────────────────────────────────────
     * Step 3: Coherence proxy and queue pressure heuristic
     * ───────────────────────────────────────────────────────────────────── */

    uint64_t coherence_cost_ns = ctx->uma_migration_cost_ns;
    if (coherence_cost_ns == 0 && uma_sample_count > 0) {
        coherence_cost_ns = uma_latency_ema_ns;
        ctx->uma_migration_cost_ns = coherence_cost_ns;
        ctx->uma_migration_bytes_est = uma_bytes_ema;
    }

    if (coherence_cost_ns > deadline_margin_ns) {
        ctx->chosen_target = PQC_JOB_CPU;
        ctx->decision_reason |= PQC_ROUTE_REASON_COHERENCE_RISK;
        ctx->deferral_reason = PQC_ROUTE_REASON_COHERENCE_RISK;
        admission_stats_record(ADMISSION_STATS_COHERENCE_RISK);
        scheduler_trace_log_snapshot(ctx, mem_bw, tc_util);
        return 0;
    }

    double cpu_queue_pressure = (double)ctx->cpu_queue_depth +
                                ctx->cpu_load_avg * 10.0;
    double gpu_queue_pressure = (double)ctx->gpu_queue_depth +
                                ctx->gpu_load_avg * 5.0;

    double threshold = queue_pressure_threshold;
    if (is_llm_prefill) {
        threshold = threshold * 1.5; /* Soften threshold to encourage GPU placement */
    }

    if (gpu_queue_pressure < cpu_queue_pressure * threshold) {
        /* GPU is less congested; route there */
        ctx->chosen_target = PQC_JOB_GPU;
        ctx->decision_reason |= PQC_ROUTE_REASON_QUEUE_PRESSURE;
        admission_stats_record(ADMISSION_STATS_GPU_ADMITTED |
                               ADMISSION_STATS_QUEUE_PRESSURE);
    } else {
        /* CPU is better utilized */
        ctx->chosen_target = PQC_JOB_CPU;
        ctx->decision_reason |= PQC_ROUTE_REASON_GPU_ELIGIBLE;
        ctx->deferral_reason = PQC_ROUTE_REASON_QUEUE_PRESSURE;
        admission_stats_record(ADMISSION_STATS_CPU_ROUTED |
                               ADMISSION_STATS_QUEUE_PRESSURE);
    }

    /* ─────────────────────────────────────────────────────────────────────
     * Step 4: Log decision and update stats
     * ───────────────────────────────────────────────────────────────────── */

    scheduler_trace_log_snapshot(ctx, mem_bw, tc_util);

    return 0;
}

void pqc_admission_update_ai_budget(uint64_t ai_budget_ns,
                                   uint64_t ai_inference_queue_depth)
{
    pqc_lock_profile_scope_t state_scope;
    if (admission_state_lock(&state_scope, __func__) != 0)
        return;

    uint64_t old_budget = g_admission.ai_budget_ns;
    g_admission.ai_budget_ns = ai_budget_ns;
    g_admission.ai_queue_depth = ai_inference_queue_depth;
    g_admission.ai_budget_update_mono_ns = monotonic_now_ns();
    (void)admission_state_unlock(&state_scope, __func__);

    /* Log budget transition */
    if (old_budget != ai_budget_ns) {
        char line[256];
        snprintf(line, sizeof(line),
                 "# AI_BUDGET_UPDATE: old=%llu ns, new=%llu ns, queue_depth=%llu\n",
                 (unsigned long long)old_budget,
                 (unsigned long long)ai_budget_ns,
                 (unsigned long long)ai_inference_queue_depth);
        admission_trace_append(line);
    }
}

/* ──────────────────────────────────────────────────────────────────────────
 *  Scheduler Trace Logging
 * ────────────────────────────────────────────────────────────────────────── */

static void scheduler_trace_log_snapshot(const pqc_admission_context_t *ctx,
                                         double mem_bandwidth_util,
                                         double tensor_core_util)
{
    if (!ctx) return;

    /* Get current time */
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t now_ns = ts.tv_sec * 1000000000ULL + ts.tv_nsec;

    /* Format JSON record (JSONL format) */
    char json_buf[4096];
    snprintf(json_buf, sizeof(json_buf),
        "{"
        "\"timestamp_ns\": %llu, "
        "\"trace_timestamp_clock\": \"CLOCK_REALTIME\", "
        "\"age_clock\": \"CLOCK_MONOTONIC\", "
        "\"batch_age_ns\": %llu, "
        "\"ai_inference_deadline_ns\": %llu, "
        "\"batch_count\": %zu, "
        "\"bytes_total\": %zu, "
        "\"queue_delay_ns\": %llu, "
        "\"service_time_ns\": %llu, "
        "\"gpu_kernel_est_ns\": %llu, "
        "\"gpu_h2d_staging_ns\": %llu, "
        "\"gpu_d2h_staging_ns\": %llu, "
        "\"cpu_queue_depth\": %llu, "
        "\"gpu_queue_depth\": %llu, "
        "\"cpu_load_avg\": %.2f, "
        "\"gpu_load_avg\": %.2f, "
        "\"telemetry_mem_bandwidth_util\": %.4f, "
        "\"telemetry_tensor_core_util\": %.4f, "
        "\"ai_qos_budget_remaining_ns\": %llu, "
        "\"producer_slack_age_ns\": %llu, "
        "\"producer_slack_stale_after_ns\": %llu, "
        "\"producer_slack_stale\": %s, "
        "\"uma_migration_bytes_est\": %llu, "
        "\"uma_migration_cost_ns\": %llu, "
        "\"chosen_target\": \"%s\", "
        "\"deferral_reason\": %u, "
        "\"decision_reason\": %u"
        "}\n",
        (unsigned long long)now_ns,
        (unsigned long long)ctx->batch_age_ns,
        (unsigned long long)ctx->ai_inference_deadline_ns,
        ctx->batch_count,
        ctx->bytes_total,
        (unsigned long long)ctx->queue_delay_ns,
        (unsigned long long)ctx->service_time_ns,
        (unsigned long long)ctx->gpu_kernel_est_ns,
        (unsigned long long)ctx->gpu_h2d_staging_ns,
        (unsigned long long)ctx->gpu_d2h_staging_ns,
        (unsigned long long)ctx->cpu_queue_depth,
        (unsigned long long)ctx->gpu_queue_depth,
        ctx->cpu_load_avg,
        ctx->gpu_load_avg,
        mem_bandwidth_util,
        tensor_core_util,
        (unsigned long long)ctx->ai_qos_budget_remaining_ns,
        (unsigned long long)ctx->producer_slack_age_ns,
        (unsigned long long)ctx->producer_slack_stale_after_ns,
        ctx->producer_slack_stale ? "true" : "false",
        (unsigned long long)ctx->uma_migration_bytes_est,
        (unsigned long long)ctx->uma_migration_cost_ns,
        (ctx->chosen_target == PQC_JOB_GPU ? "GPU" : "CPU"),
        (unsigned int)ctx->deferral_reason,
        (unsigned int)ctx->decision_reason);

    admission_trace_append(json_buf);
}

void pqc_scheduler_trace_log(const pqc_admission_context_t *ctx)
{
    if (!ctx) return;

    double mem_bandwidth_util = 0.0;
    double tensor_core_util = 0.0;
    pqc_lock_profile_scope_t state_scope;
    if (admission_state_lock(&state_scope, __func__) == 0) {
        mem_bandwidth_util = g_telemetry.mem_bandwidth_util;
        tensor_core_util = g_telemetry.tensor_core_util;
        (void)admission_state_unlock(&state_scope, __func__);
    }

    scheduler_trace_log_snapshot(ctx, mem_bandwidth_util, tensor_core_util);
}

void pqc_scheduler_trace_stats(pqc_admission_stats_t *out_stats)
{
    if (!out_stats) return;

    memset(out_stats, 0, sizeof(*out_stats));

    pqc_lock_profile_scope_t state_scope;
    if (admission_state_lock(&state_scope, __func__) != 0)
        return;
    memcpy(out_stats, &g_admission.stats, sizeof(*out_stats));
    (void)admission_state_unlock(&state_scope, __func__);
}

/* ──────────────────────────────────────────────────────────────────────────
 *  Telemetry Integration (software rolling proxy + CUPTI hooks)
 * ────────────────────────────────────────────────────────────────────────── */

void pqc_admission_record_uma_event(size_t uma_bytes, uint64_t uma_latency_ns)
{
    pqc_lock_profile_scope_t state_scope;
    if (admission_state_lock(&state_scope, __func__) != 0)
        return;
    g_admission.uma_sample_count++;
    g_admission.uma_window_bytes += uma_bytes;
    g_admission.uma_window_latency_ns += uma_latency_ns;
    if (g_admission.uma_sample_count == 1) {
        g_admission.uma_bytes_ema = uma_bytes;
        g_admission.uma_latency_ema_ns = uma_latency_ns;
    } else {
        g_admission.uma_bytes_ema = ((g_admission.uma_bytes_ema * 7u) + uma_bytes) / 8u;
        g_admission.uma_latency_ema_ns =
            ((g_admission.uma_latency_ema_ns * 7u) + uma_latency_ns) / 8u;
    }
    (void)admission_state_unlock(&state_scope, __func__);
}

void pqc_admission_update_telemetry(double mem_bandwidth_util, double tensor_core_util)
{
    pqc_lock_profile_scope_t state_scope;
    if (admission_state_lock(&state_scope, __func__) != 0)
        return;
    g_telemetry.mem_bandwidth_util = mem_bandwidth_util;
    g_telemetry.tensor_core_util = tensor_core_util;
    (void)admission_state_unlock(&state_scope, __func__);
}
