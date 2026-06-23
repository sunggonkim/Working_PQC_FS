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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>

/* Global state (protected by mutex) */
static struct {
    FILE *trace_file;
    pthread_mutex_t trace_lock;
    uint64_t ai_budget_ns;
    uint64_t ai_queue_depth;

    /* Cumulative statistics */
    pqc_admission_stats_t stats;

    /* Config parameters (tunable) */
    uint64_t gpu_min_batch_bytes;
    uint64_t ai_qos_min_budget_ns;
    uint64_t deadline_margin_ns;
    double queue_pressure_threshold;  /* ratio: cpu_depth / gpu_depth */
    uint64_t uma_window_bytes;
    uint64_t uma_window_latency_ns;
    uint64_t uma_sample_count;
    uint64_t uma_latency_ema_ns;
    uint64_t uma_bytes_ema;
} g_admission __attribute__((unused)) = {
    .trace_file = NULL,
    .trace_lock = PTHREAD_MUTEX_INITIALIZER,
    .ai_budget_ns = 0,
    .ai_queue_depth = 0,
    .stats = {0},
    .gpu_min_batch_bytes = 4096,
    .ai_qos_min_budget_ns = 1730000,  /* TensorRT YOLOv8n p99 + margin */
    .deadline_margin_ns = 100000,      /* 100 µs */
    .queue_pressure_threshold = 0.8,
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


/* ──────────────────────────────────────────────────────────────────────────
 *  Initialization & Shutdown
 * ────────────────────────────────────────────────────────────────────────── */

int pqc_admission_init(const char *trace_file_path)
{
    if (!trace_file_path) return -1;

    /* Open trace file for append (create if missing) */
    g_admission.trace_file = fopen(trace_file_path, "a");
    if (!g_admission.trace_file) {
        fprintf(stderr, "[PQC] Failed to open trace file: %s\n", trace_file_path);
        return -1;
    }

    /* Read policy parameters from environment */
    const char *env;

    env = getenv("PQC_GPU_MIN_BATCH_BYTES");
    if (env) g_admission.gpu_min_batch_bytes = strtoul(env, NULL, 10);

    env = getenv("PQC_AI_QOS_MIN_BUDGET_NS");
    if (env) g_admission.ai_qos_min_budget_ns = strtoull(env, NULL, 10);

    env = getenv("PQC_DEADLINE_MARGIN_NS");
    if (env) g_admission.deadline_margin_ns = strtoull(env, NULL, 10);

    env = getenv("PQC_QUEUE_PRESSURE_THRESHOLD");
    if (env) g_admission.queue_pressure_threshold = strtod(env, NULL);

    env = getenv("PQC_TELEMETRY_MEM_BANDWIDTH");
    if (env) g_telemetry.mem_bandwidth_util = strtod(env, NULL);

    env = getenv("PQC_TELEMETRY_TENSOR_CORE");
    if (env) g_telemetry.tensor_core_util = strtod(env, NULL);

    /* Write trace header */
    fprintf(g_admission.trace_file,
            "# pqc_admission_init: "
            "gpu_min_batch=%zu bytes, "
            "ai_qos_min_budget=%llu ns, "
            "deadline_margin=%llu ns, "
            "queue_threshold=%.2f\n",
            g_admission.gpu_min_batch_bytes,
            (unsigned long long)g_admission.ai_qos_min_budget_ns,
            (unsigned long long)g_admission.deadline_margin_ns,
            g_admission.queue_pressure_threshold);
    fflush(g_admission.trace_file);

    memset(&g_admission.stats, 0, sizeof(g_admission.stats));

    fprintf(stderr, "[PQC] Admission controller initialized: trace=%s\n", trace_file_path);
    return 0;
}

void pqc_admission_shutdown(void)
{
    if (g_admission.trace_file) {
        pqc_admission_stats_t stats_snapshot;
        pthread_mutex_lock(&g_admission.trace_lock);
        memcpy(&stats_snapshot, &g_admission.stats, sizeof(stats_snapshot));
        /* Write final statistics */
        fprintf(g_admission.trace_file,
                "# FINAL STATS: "
                "total_requests=%llu, "
                "gpu_admitted=%llu, "
                "cpu_routed=%llu, "
                "ai_budget_exhausted=%llu, "
                "deadline_exceeded=%llu, "
                "size_too_small=%llu\n",
                (unsigned long long)stats_snapshot.total_requests,
                (unsigned long long)stats_snapshot.gpu_admitted_count,
                (unsigned long long)stats_snapshot.cpu_routed_count,
                (unsigned long long)stats_snapshot.ai_budget_exhausted_count,
                (unsigned long long)stats_snapshot.deadline_exceeded_count,
                (unsigned long long)stats_snapshot.size_too_small_count);
        fflush(g_admission.trace_file);
        pthread_mutex_unlock(&g_admission.trace_lock);
        fclose(g_admission.trace_file);
        g_admission.trace_file = NULL;
    }

    fprintf(stderr,
            "[PQC] Admission stats: total=%llu, gpu=%llu, cpu=%llu, "
            "ai_exhausted=%llu, deadline=%llu, size_small=%llu\n",
            (unsigned long long)g_admission.stats.total_requests,
            (unsigned long long)g_admission.stats.gpu_admitted_count,
            (unsigned long long)g_admission.stats.cpu_routed_count,
            (unsigned long long)g_admission.stats.ai_budget_exhausted_count,
            (unsigned long long)g_admission.stats.deadline_exceeded_count,
            (unsigned long long)g_admission.stats.size_too_small_count);
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
     * Step 1: Hard constraints (size and deadline)
     * ───────────────────────────────────────────────────────────────────── */

    if (ctx->bytes_total < g_admission.gpu_min_batch_bytes) {
        ctx->chosen_target = PQC_JOB_CPU;
        ctx->decision_reason |= PQC_ROUTE_REASON_SIZE_TOO_SMALL;
        ctx->deferral_reason = PQC_ROUTE_REASON_SIZE_TOO_SMALL;
        pthread_mutex_lock(&g_admission.trace_lock);
        g_admission.stats.total_requests++;
        g_admission.stats.size_too_small_count++;
        pthread_mutex_unlock(&g_admission.trace_lock);
        pqc_scheduler_trace_log(ctx);
        return 0;
    }

    if (ctx->batch_age_ns > ctx->ai_inference_deadline_ns) {
        ctx->chosen_target = PQC_JOB_CPU;
        ctx->decision_reason |= PQC_ROUTE_REASON_DEADLINE_ELAPSED;
        ctx->deferral_reason = PQC_ROUTE_REASON_DEADLINE_ELAPSED;
        pthread_mutex_lock(&g_admission.trace_lock);
        g_admission.stats.total_requests++;
        g_admission.stats.deadline_exceeded_count++;
        pthread_mutex_unlock(&g_admission.trace_lock);
        pqc_scheduler_trace_log(ctx);
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
        pthread_mutex_lock(&g_admission.trace_lock);
        g_admission.stats.total_requests++;
        g_admission.stats.ai_budget_exhausted_count++;
        pthread_mutex_unlock(&g_admission.trace_lock);
        pqc_scheduler_trace_log(ctx);
        return 0;
    }

    /* ─────────────────────────────────────────────────────────────────────
     * Step 2.5: Phase-Aware Memory Interleaving (Milestone 3)
     * Detect LLM execution phase using memory bandwidth and Tensor Core telemetry.
     * ───────────────────────────────────────────────────────────────────── */
    int is_llm_decoding = (g_telemetry.mem_bandwidth_util > 0.70 && g_telemetry.tensor_core_util < 0.30);
    int is_llm_prefill  = (g_telemetry.tensor_core_util >= 0.30 && g_telemetry.mem_bandwidth_util <= 0.60);

    if (is_llm_decoding) {
        /* Force PQC encryption to CPU fast lane to avoid GPU memory staging contention. */
        ctx->chosen_target = PQC_JOB_CPU;
        ctx->decision_reason |= PQC_ROUTE_REASON_AI_QOS_EXHAUSTED;
        ctx->deferral_reason = PQC_ROUTE_REASON_AI_QOS_EXHAUSTED;
        pthread_mutex_lock(&g_admission.trace_lock);
        g_admission.stats.total_requests++;
        g_admission.stats.ai_budget_exhausted_count++;
        pthread_mutex_unlock(&g_admission.trace_lock);
        pqc_scheduler_trace_log(ctx);
        return 0;
    }

    /* ─────────────────────────────────────────────────────────────────────
     * Step 3: Coherence proxy and queue pressure heuristic
     * ───────────────────────────────────────────────────────────────────── */

    uint64_t coherence_cost_ns = ctx->uma_migration_cost_ns;
    if (coherence_cost_ns == 0 && g_admission.uma_sample_count > 0) {
        coherence_cost_ns = g_admission.uma_latency_ema_ns;
        ctx->uma_migration_cost_ns = coherence_cost_ns;
        ctx->uma_migration_bytes_est = g_admission.uma_bytes_ema;
    }

    if (coherence_cost_ns > g_admission.deadline_margin_ns) {
        ctx->chosen_target = PQC_JOB_CPU;
        ctx->decision_reason |= PQC_ROUTE_REASON_COHERENCE_RISK;
        ctx->deferral_reason = PQC_ROUTE_REASON_COHERENCE_RISK;
        pthread_mutex_lock(&g_admission.trace_lock);
        g_admission.stats.total_requests++;
        g_admission.stats.coherence_risk_count++;
        pthread_mutex_unlock(&g_admission.trace_lock);
        pqc_scheduler_trace_log(ctx);
        return 0;
    }

    double cpu_queue_pressure = (double)ctx->cpu_queue_depth +
                                ctx->cpu_load_avg * 10.0;
    double gpu_queue_pressure = (double)ctx->gpu_queue_depth +
                                ctx->gpu_load_avg * 5.0;

    double threshold = g_admission.queue_pressure_threshold;
    if (is_llm_prefill) {
        threshold = threshold * 1.5; /* Soften threshold to encourage GPU placement */
    }

    if (gpu_queue_pressure < cpu_queue_pressure * threshold) {
        /* GPU is less congested; route there */
        ctx->chosen_target = PQC_JOB_GPU;
        ctx->decision_reason |= PQC_ROUTE_REASON_QUEUE_PRESSURE;
        pthread_mutex_lock(&g_admission.trace_lock);
        g_admission.stats.total_requests++;
        g_admission.stats.gpu_admitted_count++;
        g_admission.stats.queue_pressure_count++;
        pthread_mutex_unlock(&g_admission.trace_lock);
    } else {
        /* CPU is better utilized */
        ctx->chosen_target = PQC_JOB_CPU;
        ctx->decision_reason |= PQC_ROUTE_REASON_GPU_ELIGIBLE;
        ctx->deferral_reason = PQC_ROUTE_REASON_QUEUE_PRESSURE;
        pthread_mutex_lock(&g_admission.trace_lock);
        g_admission.stats.total_requests++;
        g_admission.stats.cpu_routed_count++;
        g_admission.stats.queue_pressure_count++;
        pthread_mutex_unlock(&g_admission.trace_lock);
    }

    /* ─────────────────────────────────────────────────────────────────────
     * Step 4: Log decision and update stats
     * ───────────────────────────────────────────────────────────────────── */

    pqc_scheduler_trace_log(ctx);

    return 0;
}

void pqc_admission_update_ai_budget(uint64_t ai_budget_ns,
                                   uint64_t ai_inference_queue_depth)
{
    pthread_mutex_lock(&g_admission.trace_lock);

    uint64_t old_budget = g_admission.ai_budget_ns;
    g_admission.ai_budget_ns = ai_budget_ns;
    g_admission.ai_queue_depth = ai_inference_queue_depth;

    /* Log budget transition */
    if (old_budget != ai_budget_ns && g_admission.trace_file) {
        fprintf(g_admission.trace_file,
                "# AI_BUDGET_UPDATE: old=%llu ns, new=%llu ns, queue_depth=%llu\n",
                (unsigned long long)old_budget,
                (unsigned long long)ai_budget_ns,
                (unsigned long long)ai_inference_queue_depth);
        fflush(g_admission.trace_file);
    }

    pthread_mutex_unlock(&g_admission.trace_lock);
}

/* ──────────────────────────────────────────────────────────────────────────
 *  Scheduler Trace Logging
 * ────────────────────────────────────────────────────────────────────────── */

void pqc_scheduler_trace_log(const pqc_admission_context_t *ctx)
{
    if (!ctx || !g_admission.trace_file) return;

    /* Get current time */
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t now_ns = ts.tv_sec * 1000000000ULL + ts.tv_nsec;

    /* Format JSON record (JSONL format) */
    char json_buf[2048];
    snprintf(json_buf, sizeof(json_buf),
        "{"
        "\"timestamp_ns\": %llu, "
        "\"batch_age_ns\": %llu, "
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
        "\"ai_qos_budget_remaining_ns\": %llu, "
        "\"uma_migration_bytes_est\": %llu, "
        "\"uma_migration_cost_ns\": %llu, "
        "\"chosen_target\": \"%s\", "
        "\"deferral_reason\": 0x%02x, "
        "\"decision_reason\": 0x%02x"
        "}\n",
        (unsigned long long)now_ns,
        (unsigned long long)ctx->batch_age_ns,
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
        (unsigned long long)ctx->ai_qos_budget_remaining_ns,
        (unsigned long long)ctx->uma_migration_bytes_est,
        (unsigned long long)ctx->uma_migration_cost_ns,
        (ctx->chosen_target == PQC_JOB_GPU ? "GPU" : "CPU"),
        (unsigned int)ctx->deferral_reason,
        (unsigned int)ctx->decision_reason);

    uint64_t total_requests;

    /* Write with lock to ensure atomicity and a consistent counter snapshot. */
    pthread_mutex_lock(&g_admission.trace_lock);
    total_requests = g_admission.stats.total_requests;

    fprintf(g_admission.trace_file, "%s", json_buf);

    /* Periodic flush (every 10 records) */
    if ((total_requests % 10) == 0) {
        fflush(g_admission.trace_file);
    }

    pthread_mutex_unlock(&g_admission.trace_lock);
}

void pqc_scheduler_trace_stats(pqc_admission_stats_t *out_stats)
{
    if (!out_stats) return;

    pthread_mutex_lock(&g_admission.trace_lock);
    memcpy(out_stats, &g_admission.stats, sizeof(*out_stats));
    pthread_mutex_unlock(&g_admission.trace_lock);
}

/* ──────────────────────────────────────────────────────────────────────────
 *  Telemetry Integration (software rolling proxy + CUPTI hooks)
 * ────────────────────────────────────────────────────────────────────────── */

void pqc_admission_record_uma_event(size_t uma_bytes, uint64_t uma_latency_ns)
{
    pthread_mutex_lock(&g_admission.trace_lock);
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
    pthread_mutex_unlock(&g_admission.trace_lock);
}

void pqc_admission_update_telemetry(double mem_bandwidth_util, double tensor_core_util)
{
    pthread_mutex_lock(&g_admission.trace_lock);
    g_telemetry.mem_bandwidth_util = mem_bandwidth_util;
    g_telemetry.tensor_core_util = tensor_core_util;
    pthread_mutex_unlock(&g_admission.trace_lock);
}

