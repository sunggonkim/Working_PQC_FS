/**
 * pqc_admission.h — QoS-Aware Admission Control Framework (M5)
 *
 * Purpose:
 *   Replace threshold-only routing with explicit admission control:
 *   - Track batch age, expected service time, queue depth, staging cost.
 *   - Make route decision based on AI SLO budget, coherence cost, queue pressure.
 *   - Log route reason for scheduler trace analysis (E3/E4 evaluation).
 *
 * Design:
 *   The admission controller is the central policy arbiter:
 *   - Input:  pqc_admission_context_t (batch characteristics + system state)
 *   - Output: route decision (CPU or GPU) + reason code
 *   - Side effect: log decision to scheduler trace for reproducibility
 *
 * Evaluation Contract (M5 Gate):
 *   "Scheduler traces show a causal route change when the SLO budget changes.
 *    CPU fast-lane p99 remains bounded under a saturated elastic queue."
 *
 *   This requires:
 *   - Route decision is deterministic given input state.
 *   - Trace records exact state at decision time.
 *   - E4 can replay traces and verify SLO protection causality.
 *
 * Copyright 2025 AEGIS-Q Authors. Apache-2.0.
 */

#ifndef PQC_ADMISSION_H
#define PQC_ADMISSION_H

#include <stddef.h>
#include <stdint.h>
#include "pqc_block_job.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ──────────────────────────────────────────────────────────────────────────
 *  Admission Controller Initialization & Config
 * ────────────────────────────────────────────────────────────────────────── */

/**
 * pqc_admission_init()
 *
 * Initialize the admission controller.
 *   - Open scheduler trace file when explicitly configured.
 *   - Load policy parameters (AI SLO budget, queue thresholds, etc.).
 *   - Initialize telemetry buffers.
 *
 * Called once during pqc_fuse startup.
 * Returns: 0 on success, -1 on explicit trace-file or config loading error.
 */
int pqc_admission_init(const char *trace_file_path);

/**
 * pqc_admission_shutdown()
 *
 * Close scheduler trace and clean up resources.
 * Called during pqc_fuse unmount.
 */
void pqc_admission_shutdown(void);

/* ──────────────────────────────────────────────────────────────────────────
 *  Route Decision & Tracing
 * ────────────────────────────────────────────────────────────────────────── */

/**
 * pqc_admit()
 *
 * Core admission control decision function.
 *
 * Input:
 *   - ctx->batch_count, bytes_total: job characteristics
 *   - ctx->batch_age_ns: monotonic relative time since job submission
 *   - ctx->gpu_kernel_est_ns, gpu_h2d_staging_ns, gpu_d2h_staging_ns: cost estimates
 *   - ctx->cpu_queue_depth, gpu_queue_depth: current queue lengths
 *   - ctx->cpu_load_avg, gpu_load_avg: rolling load estimates
 *   - ctx->ai_inference_deadline_ns: producer-supplied relative deadline/slack
 *   - ctx->ai_qos_budget_remaining_ns: GPU slack after AI inference reservation
 *   - ctx->producer_slack_age_ns/stale_after_ns/stale: filled by controller
 *   - ctx->uma_migration_cost_ns: UVM stall cost derived from a rolling proxy
 *     when CUPTI counters are unavailable
 *
 * Output:
 *   - ctx->chosen_target: PQC_JOB_CPU or PQC_JOB_GPU
 *   - ctx->decision_reason: bitmask of pqc_admission_reason_t
 *
 * Decision Logic:
 *
 *   1. Hard constraints (force CPU):
 *      - plane not GPU-eligible (done in pqc_block_job_gpu_eligible)
 *      - request too small for GPU overhead amortization
 *      - batch_age > ai_inference_deadline_ns
 *
 *   2. AI QoS protection (priority for inference):
 *      - if producer slack is stale:
 *          → CPU (reason = STALE_TELEMETRY)
 *      - if ai_qos_budget_remaining_ns < gpu_kernel_est_ns + staging:
 *          → CPU (reason = AI_QOS_EXHAUSTED)
 *      - else: proceed to queue pressure check
 *
 *   3. Coherence cost (software proxy or CUPTI-backed counter):
 *      - if uma_migration_cost_ns > policy threshold:
 *          → CPU (reason = COHERENCE_RISK)
 *
 *   4. Queue pressure heuristic:
 *      - if gpu_queue_depth << cpu_queue_depth:
 *          → GPU (reason = QUEUE_PRESSURE)
 *      - else:
 *          → CPU (reason = GPU_ELIGIBLE, but CPU better utilized)
 *
 * Returns: 0 on success, -1 on internal error (do not use for routing).
 *
 * M5 Gate Criterion:
 *   "Scheduler traces show a causal route change when the SLO budget changes.
 *    CPU fast-lane p99 remains bounded under a saturated elastic queue."
 */
int pqc_admit(pqc_admission_context_t *ctx);

/**
 * pqc_admission_update_ai_budget()
 *
 * Called periodically by the AI load monitor to update GPU SLO budget.
 *
 * Input:
 *   - ai_budget_ns: remaining GPU slack for elastic jobs (nanoseconds)
 *   - ai_inference_queue_depth: number of pending AI jobs
 *
 * The admission controller uses this to make QoS-aware decisions.
 * Called by a foreground producer or telemetry bridge.  The sample is treated
 * as stale after PQC_PRODUCER_SLACK_STALE_NS (default 250 ms), and missing or
 * stale slack fails closed to CPU.
 */
void pqc_admission_update_ai_budget(uint64_t ai_budget_ns,
                                   uint64_t ai_inference_queue_depth);

const char *pqc_qos_class_name(int qos_class);
int pqc_parse_qos_class_value(const char *value, size_t size, int *out);
int pqc_qos_class_load_for_path(const char *phys_path, int *out);

/* ──────────────────────────────────────────────────────────────────────────
 *  Scheduler Trace Logging
 * ────────────────────────────────────────────────────────────────────────── */

/**
 * pqc_scheduler_trace_log()
 *
 * Record a route decision to the scheduler trace.
 * Format: JSON Lines (one JSON object per line).
 *
 * Trace record:
 * {
 *   "timestamp_ns": <unix ns from CLOCK_REALTIME>,
 *   "trace_timestamp_clock": "CLOCK_REALTIME",
 *   "age_clock": "CLOCK_MONOTONIC",
 *   "batch_age_ns": <uint64>,
 *   "ai_inference_deadline_ns": <uint64>,
 *   "batch_count": <size_t>,
 *   "bytes_total": <size_t>,
 *   "queue_delay_ns": <uint64>,
 *   "service_time_ns": <uint64>,
 *   "gpu_kernel_est_ns": <uint64>,
 *   "gpu_h2d_staging_ns": <uint64>,
 *   "gpu_d2h_staging_ns": <uint64>,
 *   "cpu_queue_depth": <uint64>,
 *   "gpu_queue_depth": <uint64>,
 *   "ai_qos_budget_remaining_ns": <uint64>,
 *   "producer_slack_age_ns": <uint64>,
 *   "producer_slack_stale_after_ns": <uint64>,
 *   "producer_slack_stale": <bool>,
 *   "chosen_target": "CPU" or "GPU",
 *   "deferral_reason": <hex bitmask>,
 *   "decision_reason": <hex bitmask>,
 *   "expected_service_ns": <uint64>
 * }
 *
 * Used for E3/E4 reproducibility and causal analysis.
 * Thread-safe (uses mutex lock on trace file).
 */
void pqc_scheduler_trace_log(const pqc_admission_context_t *ctx);

/**
 * pqc_scheduler_trace_stats()
 *
 * Aggregate statistics from trace (for summary reports).
 * Called after evaluation to produce results tables.
 *
 * Output:
 *   - gpu_admitted_count: jobs routed to GPU
 *   - cpu_routed_count: jobs routed to CPU
 *   - ai_budget_exhausted_count: jobs spilled due to AI QoS
 *   - coherence_risk_count: jobs spilled due to UVM stall risk
 *   - ...
 *
 * All counts cumulative since pqc_admission_init().
 */
typedef struct {
    uint64_t gpu_admitted_count;
    uint64_t cpu_routed_count;
    uint64_t ai_budget_exhausted_count;
    uint64_t coherence_risk_count;
    uint64_t queue_pressure_count;
    uint64_t deadline_exceeded_count;
    uint64_t size_too_small_count;
    uint64_t stale_telemetry_count;
    uint64_t total_requests;
} pqc_admission_stats_t;

void pqc_scheduler_trace_stats(pqc_admission_stats_t *out_stats);

/* ──────────────────────────────────────────────────────────────────────────
 *  Telemetry Integration (software rolling proxy + CUPTI-ready hooks)
 * ────────────────────────────────────────────────────────────────────────── */

/**
 * pqc_admission_record_uma_event()
 *
 * Record a UMA migration/stall sample into the rolling admission proxy.
 * When hardware counters are unavailable, the admission controller uses this
 * software proxy to approximate coherence pressure.
 *
 * Input:
 *   - uma_bytes: number of bytes migrated
 *   - uma_latency_ns: measured migration stall time
 *
 * Used to feed coherence_cost_ns into admission decisions.
 */
extern void pqc_admission_record_uma_event(size_t uma_bytes,
                                          uint64_t uma_latency_ns);

/**
 * pqc_admission_update_telemetry()
 *
 * Update global telemetry values for memory bandwidth and Tensor Core utilization.
 * Used by the co-running stress test and profilers to drive phase-aware decisions.
 */
extern void pqc_admission_update_telemetry(double mem_bandwidth_util,
                                           double tensor_core_util);


#ifdef __cplusplus
}
#endif

#endif /* PQC_ADMISSION_H */
