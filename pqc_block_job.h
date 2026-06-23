#ifndef PQC_BLOCK_JOB_H
#define PQC_BLOCK_JOB_H

#include <stdint.h>
#include <stddef.h>

/*
 * AEGIS-Q Placement Model
 * =======================
 *
 * The scheduler partitions secure-storage work into four planes.
 * Each plane has a distinct latency sensitivity and batch-parallelism profile,
 * which determines its natural execution target:
 *
 *   DATA_PLANE      – Real-time authenticated I/O (AES-GCM per block).
 *                     Latency-critical; CPU AES-NI/ARM-Crypto dominates.
 *                     Always routed to the CPU fast lane.
 *
 *   KEY_PLANE       – PQC key lifecycle (ML-KEM keygen/encaps/decaps).
 *                     Throughput-oriented; NTT polynomial arithmetic is
 *                     structurally suited for GPU warp parallelism.
 *                     Routed to the GPU elastic lane when QoS budget allows.
 *
 *   INTEGRITY_PLANE – Merkle / hash-tree maintenance (SHA-256 / BLAKE3).
 *                     Data-parallel leaf hashing scales with batch size.
 *                     Routed to the GPU elastic lane when QoS budget allows.
 *
 *   FRESHNESS_PLANE – Journal commit and TPM anchor updates.
 *                     Must be serialized; always CPU, always background.
 *
 * The separation is the implementation of the research question:
 * "When should each plane use CPU, GPU, or MIG to jointly preserve
 *  application QoS and storage freshness?"
 */

typedef enum {
    PQC_PLANE_DATA       = 0,  /* AES-GCM block I/O  → CPU fast lane      */
    PQC_PLANE_KEY        = 1,  /* ML-KEM lifecycle   → GPU elastic lane    */
    PQC_PLANE_INTEGRITY  = 2,  /* Hash / Merkle tree → GPU elastic lane    */
    PQC_PLANE_FRESHNESS  = 3,  /* Journal / anchor   → CPU background      */
} pqc_plane_t;

#define PQC_JOB_FLAG_ENCRYPT      0x01u
#define PQC_JOB_FLAG_READMOD      0x02u
#define PQC_JOB_FLAG_GPU_ELIGIBLE 0x04u
/* Set when the AI inference engine holds the GPU; admission must check budget. */
#define PQC_JOB_FLAG_AI_ACTIVE    0x08u
/* Set when the job deadline has elapsed; must spill to CPU immediately. */
#define PQC_JOB_FLAG_DEADLINE     0x10u

typedef enum {
    PQC_JOB_CPU = 0,
    PQC_JOB_GPU = 1,
} pqc_job_target_t;

typedef struct {
    /* Identity */
    uint64_t file_id;
    uint64_t logical_block;
    uint64_t generation;
    uint64_t logical_offset;
    uint32_t plaintext_length;
    uint32_t flags;

    /* Plane classification — determines GPU eligibility. */
    pqc_plane_t plane;

    /* Scheduler signals */
    uint64_t submit_ns;
    uint64_t cpu_queue_depth;
    uint64_t gpu_queue_depth;
    uint64_t gpu_wait_ns;
    uint64_t coherence_cost_ns;

    /*
     * AI QoS budget remaining (nanoseconds).
     * Filled by the admission controller from GPU occupancy telemetry.
     * Zero means the inference engine currently owns the GPU fully.
     * The elastic lane is only available when this value exceeds the
     * estimated GPU kernel duration for the job.
     */
    uint64_t ai_qos_budget_ns;

    pqc_job_target_t target;
} pqc_block_job_t;

typedef struct {
    uint64_t submitted;
    uint64_t cpu_executed;
    uint64_t gpu_executed;
    uint64_t bytes_cpu;
    uint64_t bytes_gpu;
    uint64_t gpu_migration_ns;

    /* Per-plane counters */
    uint64_t data_plane_cpu;
    uint64_t key_plane_cpu;
    uint64_t key_plane_gpu;
    uint64_t integrity_plane_cpu;
    uint64_t integrity_plane_gpu;
    uint64_t freshness_plane_cpu;

    /* QoS protection counters */
    uint64_t spilled_ai_budget;   /* jobs spilled because ai_qos_budget_ns == 0  */
    uint64_t spilled_deadline;    /* jobs spilled because deadline elapsed        */
    uint64_t spilled_queue;       /* jobs spilled because GPU queue too deep      */
} pqc_scheduler_stats_t;

typedef struct {
    size_t   gpu_min_bytes;
    uint64_t gpu_queue_penalty_ns;
    uint64_t coherence_penalty_ns;
    uint64_t gpu_max_inflight_jobs;
    uint64_t gpu_max_inflight_bytes;
    uint64_t gpu_max_wait_ns;
    double   cpu_load_bias;
    double   gpu_queue_bias;
    uint64_t gpu_contention_penalty_ns;
    /*
     * Traceable contention score threshold.  The scheduler computes a
     * deterministic score from queue depth, wait time, coherence cost, and
     * AI budget slack instead of relying on a learning policy.  That makes
     * admission decisions reproducible enough for paper evidence.
     */
    uint64_t contention_score_ns;
    /*
     * Minimum AI QoS budget (ns) that must remain before a GPU elastic-lane
     * job is admitted.  Set from measured TensorRT YOLOv8 p99 baseline
     * (1.729 ms → 1_730_000 ns) plus a safety margin.
     */
    uint64_t ai_qos_min_budget_ns;
} pqc_scheduler_policy_t;

static inline void pqc_block_job_init(pqc_block_job_t *job,
                                      uint64_t file_id,
                                      uint64_t logical_block,
                                      uint64_t generation,
                                      uint64_t logical_offset,
                                      uint32_t length,
                                      uint32_t flags,
                                      pqc_plane_t plane)
{
    if (!job) return;
    job->file_id         = file_id;
    job->logical_block   = logical_block;
    job->generation      = generation;
    job->logical_offset  = logical_offset;
    job->plaintext_length = length;
    job->flags           = flags;
    job->plane           = plane;
    job->submit_ns       = 0;
    job->cpu_queue_depth = 0;
    job->gpu_queue_depth = 0;
    job->gpu_wait_ns     = 0;
    job->coherence_cost_ns = 0;
    job->ai_qos_budget_ns  = 0;
    job->target          = PQC_JOB_CPU;
}

/*
 * Returns non-zero if the job's plane is structurally eligible for the
 * GPU elastic lane.
 *
 * DATA_PLANE and FRESHNESS_PLANE are never GPU-eligible:
 *   - DATA_PLANE: CPU AES-NI/ARM-Crypto is ≥10× faster; GPU staging overhead
 *                 dominates at every measured request size (workload_map.csv).
 *   - FRESHNESS_PLANE: journal commit must be serialized on the CPU.
 *
 * KEY_PLANE and INTEGRITY_PLANE are eligible when flags confirm it and the
 * GPU flag was not overridden by the caller for correctness reasons.
 */
static inline int pqc_block_job_gpu_eligible(const pqc_block_job_t *job)
{
    if (!job) return 0;
    if (job->plane == PQC_PLANE_DATA || job->plane == PQC_PLANE_FRESHNESS)
        return 0;
    return (job->flags & PQC_JOB_FLAG_GPU_ELIGIBLE) != 0;
}

/*
 * Core placement decision for the Elastic Lane scheduler.
 *
 * The function implements the research question: "when should a KEY_PLANE or
 * INTEGRITY_PLANE job use the GPU elastic lane?"
 *
 * Conditions that force CPU (in priority order):
 *   1. Plane is DATA or FRESHNESS — structural, never GPU.
 *   2. GPU not flagged eligible by the job builder.
 *   3. Request too small for GPU launch overhead to be amortized.
 *   4. Coherence cost exceeds policy limit (UVM migration stall risk).
 *   5. Deadline has elapsed (flag or gpu_wait_ns exceeded limit).
 *   6. AI inference engine holds the GPU (ai_qos_budget_ns < threshold).
 *   7. GPU queue pressure exceeds CPU pressure.
 *
 * This mirrors the evaluation contract for E3/E4: the scheduler must
 * protect inference p99 (Table yolo_baseline) even under elastic PQC load.
 */
static inline pqc_job_target_t pqc_block_job_choose_target(
    const pqc_block_job_t *job,
    const pqc_scheduler_policy_t *policy,
    double cpu_load,
    double gpu_load)
{
    if (!job || !policy) return PQC_JOB_CPU;

    /* Rule 1: Plane-level hard constraint */
    if (!pqc_block_job_gpu_eligible(job)) return PQC_JOB_CPU;

    /* Rule 2: Request too small */
    if (job->plaintext_length > 0 && job->plaintext_length < (uint32_t)policy->gpu_min_bytes) return PQC_JOB_CPU;

    /* Rule 3: UVM coherence cost too high */
    if (job->coherence_cost_ns > policy->coherence_penalty_ns) return PQC_JOB_CPU;

    /* Rule 4: Deadline elapsed */
    if ((job->flags & PQC_JOB_FLAG_DEADLINE) ||
        job->gpu_wait_ns > policy->gpu_max_wait_ns) return PQC_JOB_CPU;

    /* Rule 5: AI QoS protection — do not starve inference engine.
     * The elastic lane is admitted only when sufficient GPU slack exists
     * beyond the AI reservation and the estimated contention cost does not
     * exceed the remaining budget. */
    uint64_t estimated_gpu_ns = job->gpu_wait_ns +
                                 job->coherence_cost_ns +
                                 policy->gpu_contention_penalty_ns;
    uint64_t contention_score = estimated_gpu_ns;
    if (job->gpu_queue_depth > job->cpu_queue_depth) {
        contention_score += (job->gpu_queue_depth - job->cpu_queue_depth) * policy->gpu_queue_penalty_ns;
    } else {
        contention_score += (job->cpu_queue_depth - job->gpu_queue_depth);
    }
    if (policy->ai_qos_min_budget_ns > 0) {
        if (job->ai_qos_budget_ns < policy->ai_qos_min_budget_ns)
            return PQC_JOB_CPU;
        if (job->ai_qos_budget_ns <= estimated_gpu_ns ||
            job->ai_qos_budget_ns <= contention_score)
            return PQC_JOB_CPU;
    }

    /* Rule 6: GPU queue pressure vs CPU pressure */
    double gpu_pressure = (double)job->gpu_queue_depth + gpu_load * policy->gpu_queue_bias;
    double cpu_pressure = (double)job->cpu_queue_depth + cpu_load * policy->cpu_load_bias;
    if (gpu_pressure + 0.25 < cpu_pressure)
        return PQC_JOB_GPU;

    /* Rule 7: contention-aware backoff.
     * If the estimated GPU contention is already large, spill earlier.
     * This is intentionally conservative so the policy can recover tail
     * latency rather than merely observe the violation. */
    if (contention_score >= policy->contention_score_ns)
        return PQC_JOB_CPU;

    if (gpu_pressure + 0.05 < cpu_pressure)
        return PQC_JOB_GPU;
    return PQC_JOB_CPU;
}

/* ═════════════════════════════════════════════════════════════════════════════
 *  M4 PHASE 1: ML-KEM-768 File-Key Lifecycle Metadata
 * ═════════════════════════════════════════════════════════════════════════════
 *
 * The AEGIS-Q submission requires:
 *  - ML-KEM-768 file-key establishment (not just per-write encapsulation)
 *  - Encapsulated ciphertext stored in authenticated metadata
 *  - TPM-sealed device recipient key path
 *  - Key rotation and zeroization tracking
 *  - GPU key residency honestly documented
 */

typedef enum {
    PQC_KEY_STATE_CREATED   = 0,    /* File created; ML-KEM-768 key established. */
    PQC_KEY_STATE_ACTIVE    = 1,    /* Normal operation; in-use shared secrets. */
    PQC_KEY_STATE_ROTATING  = 2,    /* Key rotation in progress. */
    PQC_KEY_STATE_ZEROIZED  = 3,    /* Key material destroyed (file ready for archive/delete). */
} pqc_key_lifecycle_state_t;

/*
 * ML-KEM-768 File Key Metadata
 * ─────────────────────────────
 * Stored in file extended attributes (xattr) or authenticated sidecar.
 * Gates file creation, recovery path, and remount operations.
 * 
 * Layout (implemented in the current metadata path):
 *   magic(4) | version(1) | state(1) | epoch(8) | tpm_handle(4) |
 *   ciphertext_len(2) | ciphertext(1184 for ML-KEM-768) | checksum(4)
 * Total ≈ 1210 bytes.
 */
typedef struct {
    /* Unique identifier for this file's key establishment */
    uint64_t file_id;
    
    /* ML-KEM-768 encapsulated ciphertext (stored in metadata) */
    uint8_t ciphertext[1184];      /* ML-KEM-768 ct = 1184 bytes */
    size_t ciphertext_len;
    
    /* TPM 2.0 / RPMB handle for sealed device key (M7 anchor implementation) */
    uint32_t tpm_handle;
    
    /* Rotation epoch: incremented on each key rotation; gate for stale writes */
    uint64_t current_epoch;
    uint64_t previous_epoch;
    uint64_t previous_epoch_deadline_ns;
    
    /* State machine: CREATED → ACTIVE → ROTATING → ZEROIZED */
    pqc_key_lifecycle_state_t state;
    
    /* Timestamps for rotation policy */
    uint64_t created_ns;
    uint64_t last_rotated_ns;
    uint64_t next_rotation_deadline_ns;
    
    /* GPU key residency tracking: how many GPU operations have touched this key? */
    uint64_t gpu_ops_count;
    
    /* Zeroization audit trail (timestamp when keys were destroyed) */
    uint64_t zeroized_ns;
} pqc_file_key_metadata_t;

/*
 * File Key Lifecycle Operations (M4 skeleton; to be implemented)
 * ──────────────────────────────────────────────────────────────
 * These gate the file creation, recovery, and rotation paths.
 */

/**
 * pqc_file_key_create()
 *   Called during file creation (pqc_fuse open/create handler).
 *   - Generate ML-KEM-768 keypair (CPU).
 *   - Perform ML-KEM-768 encapsulation on a recipient public key.
 *   - Store encapsulated ciphertext in file metadata.
 *   - Provision TPM-sealed device key path (M7 future).
 *   - Return 0 on success, -1 on error.
 *
 * Contract: After this call, the file can be opened and read/written only
 * with the ML-KEM-provisioned key material.
 */
extern int pqc_file_key_create(pqc_file_key_metadata_t *key_md,
                               uint64_t file_id);

/**
 * pqc_file_key_decrypt_shared_secret()
 *   Called during file read (pqc_fuse read handler).
 *   - Retrieve the encapsulated ciphertext from file metadata.
 *   - Invoke ML-KEM-768 decapsulation (CPU or GPU).
 *   - Return the shared secret (32 bytes).
 *   - Contract: Decapsulation result must match recorded ciphertext.
 */
extern int pqc_file_key_decrypt_shared_secret(
    const pqc_file_key_metadata_t *key_md,
    uint8_t out_shared_secret[32]);

/**
 * pqc_file_key_rotate()
 *   Called periodically or on admin request.
 *   - Generate new ML-KEM-768 keypair.
 *   - Encapsulate new shared secret.
 *   - Increment epoch; store new ciphertext as "current", old as "previous".
 *   - Update next_rotation_deadline_ns.
 *   - Zeroize old key material (document GPU residency policy).
 */
extern int pqc_file_key_rotate(pqc_file_key_metadata_t *key_md);

/**
 * pqc_file_key_zeroize()
 *   Called on file deletion or archive.
 *   - Destroy all key material (CPU and GPU if applicable).
 *   - Record timestamp; mark state as ZEROIZED.
 *   - Update GPU ops counter to document key residency.
 */
extern int pqc_file_key_zeroize(pqc_file_key_metadata_t *key_md);

/* ═════════════════════════════════════════════════════════════════════════════
 *  M5 PHASE 1: Explicit Admission Control Framework
 * ═════════════════════════════════════════════════════════════════════════════
 *
 * Replace threshold-only routing with explicit admission control:
 *  - Track batch age, expected service time, queue depth, staging cost.
 *  - Make route decision based on SLO budget, coherence cost, and queue pressure.
 *  - Log route reason for scheduler trace analysis and E3/E4 evaluation.
 */

typedef enum {
    PQC_ROUTE_REASON_GPU_ELIGIBLE           = 0x01,   /* Plane eligible + budget available */
    PQC_ROUTE_REASON_AI_QOS_EXHAUSTED       = 0x02,   /* AI inference owns GPU; CPU fallback */
    PQC_ROUTE_REASON_QUEUE_PRESSURE         = 0x04,   /* CPU queue << GPU queue; dispatch to GPU */
    PQC_ROUTE_REASON_DEADLINE_ELAPSED       = 0x08,   /* Deadline exceeded; must spill to CPU */
    PQC_ROUTE_REASON_SIZE_TOO_SMALL         = 0x10,   /* Request too small; staging cost dominates */
    PQC_ROUTE_REASON_COHERENCE_RISK         = 0x20,   /* UVM migration stall risk (M6 CUPTI future) */
    PQC_ROUTE_REASON_STAGING_COST_HIGH      = 0x40,   /* Expected staging time > execution time */
} pqc_admission_reason_t;

/*
 * Route Decision Context (M5 skeleton)
 * ────────────────────────────────────
 * Passed to admission controller to record detailed routing decision.
 */
typedef struct {
    /* Batch characteristics */
    size_t batch_count;
    size_t bytes_total;
    uint64_t batch_age_ns;              /* Time since job submission */
    
    /* Estimated costs */
    uint64_t gpu_kernel_est_ns;         /* Estimated GPU kernel duration */
    uint64_t gpu_h2d_staging_ns;        /* Host→Device transfer time */
    uint64_t gpu_d2h_staging_ns;        /* Device→Host transfer time */
    uint64_t queue_delay_ns;            /* Estimated queue delay before service */
    uint64_t service_time_ns;           /* Estimated end-to-end service time */
    
    /* Queue & load state */
    uint64_t cpu_queue_depth;
    uint64_t gpu_queue_depth;
    double cpu_load_avg;
    double gpu_load_avg;
    
    /* QoS budget state */
    uint64_t ai_qos_budget_remaining_ns;
    uint64_t ai_inference_deadline_ns;
    
    /* Coherence & UMA state (M6 future: CUPTI/Nsight data) */
    uint64_t uma_migration_bytes_est;
    uint64_t uma_migration_cost_ns;
    
    /* Route decision output */
    pqc_job_target_t chosen_target;
    pqc_admission_reason_t decision_reason;
    pqc_admission_reason_t deferral_reason;
} pqc_admission_context_t;

/**
 * pqc_admit()
 *   Explicit admission control decision function (M5 skeleton).
 *   - Input:  pqc_admission_context_t with batch characteristics and state
 *   - Output: chosen_target (CPU vs GPU) and decision_reason for logging
 *   - Contract: Decision must be logged to scheduler trace for E3/E4 reproducibility.
 *
 * Implementation (M5 task):
 *   1. Evaluate AI QoS budget first (priority for inference protection).
 *   2. Then evaluate queue pressure and coherence cost.
 *   3. Set decision_reason bitmask to explain choice.
 *   4. Log to scheduler trace: batch_id, reason, route, service_time_est.
 */
extern int pqc_admit(pqc_admission_context_t *ctx);

/**
 * pqc_scheduler_trace_log()
 *   Record route decision to scheduler trace for E3/E4 analysis.
 *   - Trace file: experiments/scheduler_trace_<run_id>.jsonl
 *   - Each line: { "batch_id", "submit_ns", "chosen_target", "reason", "gpu_ops_ns", ... }
 */
extern void pqc_scheduler_trace_log(const pqc_admission_context_t *ctx);

#endif
