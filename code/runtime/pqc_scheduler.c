#include "pqc_scheduler.h"

#include "pqc_config.h"
#include "pqc_format.h"
#include "pqc_lock_profile.h"

#include <pthread.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdio.h>

typedef struct {
    atomic_ullong submitted;
    atomic_ullong cpu_executed;
    atomic_ullong gpu_executed;
    atomic_ullong bytes_cpu;
    atomic_ullong bytes_gpu;
    atomic_ullong gpu_migration_ns;
    atomic_ullong data_plane_cpu;
    atomic_ullong key_plane_cpu;
    atomic_ullong key_plane_gpu;
    atomic_ullong integrity_plane_cpu;
    atomic_ullong integrity_plane_gpu;
    atomic_ullong freshness_plane_cpu;
    atomic_ullong spilled_ai_budget;
    atomic_ullong spilled_deadline;
    atomic_ullong spilled_queue;
} pqc_scheduler_atomic_stats_t;

static pthread_mutex_t g_sched_lock = PTHREAD_MUTEX_INITIALIZER;
static pqc_scheduler_atomic_stats_t g_sched_stats;
static atomic_ullong g_gpu_inflight_jobs;
static atomic_ullong g_gpu_inflight_bytes;
static atomic_int g_data_accounting_enabled = ATOMIC_VAR_INIT(0);
static pqc_scheduler_policy_t g_sched_policy = {
    .gpu_min_bytes = 131072,
    .gpu_queue_penalty_ns = 25000,
    .coherence_penalty_ns = 25000,
    .gpu_contention_penalty_ns = 25000,
    .contention_score_ns = 75000,
    .gpu_max_inflight_jobs = 2,
    .gpu_max_inflight_bytes = 256 * 1024,
    .gpu_max_wait_ns = 25000,
    .cpu_load_bias = 1.0,
    .gpu_queue_bias = 1.0,
};

static int scheduler_lock(pqc_lock_profile_scope_t *scope, const char *site)
{
    return pqc_profiled_mutex_lock(&g_sched_lock, "scheduler_lock", site, scope);
}

static int scheduler_unlock(pqc_lock_profile_scope_t *scope, const char *site)
{
    return pqc_profiled_mutex_unlock(&g_sched_lock, "scheduler_lock", site, scope);
}

static void sched_stat_add(atomic_ullong *counter, uint64_t value)
{
    if (value == 0)
        return;
    atomic_fetch_add_explicit(counter, (unsigned long long)value,
                              memory_order_relaxed);
}

static uint64_t sched_stat_load(atomic_ullong *counter)
{
    return (uint64_t)atomic_load_explicit(counter, memory_order_relaxed);
}

static void sched_stat_saturating_sub(atomic_ullong *counter, uint64_t value)
{
    unsigned long long cur =
        atomic_load_explicit(counter, memory_order_relaxed);
    for (;;) {
        unsigned long long next =
            cur > (unsigned long long)value
                ? cur - (unsigned long long)value
                : 0;
        if (atomic_compare_exchange_weak_explicit(
                counter, &cur, next, memory_order_relaxed,
                memory_order_relaxed))
            return;
    }
}

static pqc_scheduler_policy_t default_policy(void)
{
    pqc_scheduler_policy_t policy = {
        .gpu_min_bytes = 131072,
        .gpu_queue_penalty_ns = 25000,
        .coherence_penalty_ns = 25000,
        .gpu_contention_penalty_ns = 25000,
        .contention_score_ns = 75000,
        .gpu_max_inflight_jobs = 2,
        .gpu_max_inflight_bytes = 256 * 1024,
        .gpu_max_wait_ns = 25000,
        .cpu_load_bias = 1.0,
        .gpu_queue_bias = 1.0,
    };
    return policy;
}

void pqc_scheduler_reload_runtime_policy_from_env(void)
{
    pqc_scheduler_policy_t current = default_policy();
    pqc_lock_profile_scope_t scope;
    if (scheduler_lock(&scope, "scheduler_reload_snapshot") == 0) {
        current = g_sched_policy;
        (void)scheduler_unlock(&scope, "scheduler_reload_snapshot");
    }

    pqc_scheduler_policy_t next = current;
    next.gpu_min_bytes =
        (size_t)pqc_config_positive_long_or_default("PQC_GPU_MIN_BYTES",
                                                    (long)next.gpu_min_bytes);
    next.gpu_queue_penalty_ns =
        (uint64_t)pqc_config_positive_long_or_default("PQC_GPU_QUEUE_PENALTY_NS",
                                                      (long)next.gpu_queue_penalty_ns);
    next.coherence_penalty_ns =
        (uint64_t)pqc_config_positive_long_or_default("PQC_COHERENCE_PENALTY_NS",
                                                      (long)next.coherence_penalty_ns);
    next.gpu_contention_penalty_ns =
        (uint64_t)pqc_config_positive_long_or_default("PQC_GPU_CONTENTION_PENALTY_NS",
                                                      (long)next.gpu_contention_penalty_ns);
    next.contention_score_ns =
        (uint64_t)pqc_config_positive_long_or_default("PQC_CONTENTION_SCORE_NS",
                                                      (long)next.contention_score_ns);
    next.gpu_max_inflight_jobs =
        (uint64_t)pqc_config_positive_long_or_default("PQC_GPU_MAX_INFLIGHT_JOBS",
                                                      (long)next.gpu_max_inflight_jobs);
    next.gpu_max_inflight_bytes =
        (uint64_t)pqc_config_positive_long_or_default("PQC_GPU_MAX_INFLIGHT_BYTES",
                                                      (long)next.gpu_max_inflight_bytes);
    next.gpu_max_wait_ns =
        (uint64_t)pqc_config_positive_long_or_default("PQC_GPU_MAX_WAIT_NS",
                                                      (long)next.gpu_max_wait_ns);
    next.ai_qos_min_budget_ns =
        (uint64_t)pqc_config_positive_long_or_default("PQC_AI_QOS_MIN_BUDGET_NS",
                                                      (long)next.ai_qos_min_budget_ns);

    if (scheduler_lock(&scope, "scheduler_reload_publish") == 0) {
        g_sched_policy = next;
        (void)scheduler_unlock(&scope, "scheduler_reload_publish");
    }
}

pqc_scheduler_policy_t pqc_scheduler_policy_from_env(void)
{
    pqc_scheduler_policy_t policy = default_policy();
    policy.gpu_min_bytes =
        (size_t)pqc_config_positive_long_or_default("PQC_GPU_MIN_BYTES",
                                                    (long)policy.gpu_min_bytes);
    policy.gpu_queue_penalty_ns =
        (uint64_t)pqc_config_positive_long_or_default("PQC_GPU_QUEUE_PENALTY_NS",
                                                      (long)policy.gpu_queue_penalty_ns);
    policy.coherence_penalty_ns =
        (uint64_t)pqc_config_positive_long_or_default("PQC_COHERENCE_PENALTY_NS",
                                                      (long)policy.coherence_penalty_ns);
    policy.gpu_contention_penalty_ns =
        (uint64_t)pqc_config_positive_long_or_default("PQC_GPU_CONTENTION_PENALTY_NS",
                                                      (long)policy.gpu_contention_penalty_ns);
    policy.contention_score_ns =
        (uint64_t)pqc_config_positive_long_or_default("PQC_CONTENTION_SCORE_NS",
                                                      (long)policy.contention_score_ns);
    policy.gpu_max_inflight_jobs =
        (uint64_t)pqc_config_positive_long_or_default("PQC_GPU_MAX_INFLIGHT_JOBS",
                                                      (long)policy.gpu_max_inflight_jobs);
    policy.gpu_max_inflight_bytes =
        (uint64_t)pqc_config_positive_long_or_default("PQC_GPU_MAX_INFLIGHT_BYTES",
                                                      (long)policy.gpu_max_inflight_bytes);
    policy.gpu_max_wait_ns =
        (uint64_t)pqc_config_positive_long_or_default("PQC_GPU_MAX_WAIT_NS",
                                                      (long)policy.gpu_max_wait_ns);
    policy.ai_qos_min_budget_ns =
        (uint64_t)pqc_config_positive_long_or_default("PQC_AI_QOS_MIN_BUDGET_NS",
                                                      (long)policy.ai_qos_min_budget_ns);
    return policy;
}

void pqc_scheduler_runtime_policy_snapshot(pqc_scheduler_policy_t *out)
{
    if (!out)
        return;
    pqc_lock_profile_scope_t scope;
    if (scheduler_lock(&scope, "scheduler_policy_snapshot") != 0) {
        *out = default_policy();
        return;
    }
    *out = g_sched_policy;
    (void)scheduler_unlock(&scope, "scheduler_policy_snapshot");
}

void pqc_scheduler_smoke_report(FILE *out)
{
    if (!out)
        out = stderr;

    pqc_scheduler_policy_t policy = pqc_scheduler_policy_from_env();
    uint64_t supplied_budget_ns = 2000000;
    uint64_t supplied_cpu_queue_depth = 2;
    uint64_t supplied_pressure_gpu_queue_depth = 2;
    supplied_budget_ns =
        pqc_config_u64_or_default("PQC_SCHED_SMOKE_AI_BUDGET_NS",
                                  supplied_budget_ns);
    supplied_cpu_queue_depth =
        pqc_config_u64_or_default("PQC_SCHED_SMOKE_CPU_QUEUE_DEPTH",
                                  supplied_cpu_queue_depth);
    supplied_pressure_gpu_queue_depth =
        pqc_config_u64_or_default("PQC_SCHED_SMOKE_GPU_QUEUE_DEPTH",
                                  supplied_pressure_gpu_queue_depth);

    pqc_block_job_t jobs[3];
    pqc_block_job_init(&jobs[0], 1, 0, 1, 0, 4096,
                       PQC_JOB_FLAG_ENCRYPT | PQC_JOB_FLAG_READMOD, PQC_PLANE_KEY);
    pqc_block_job_init(&jobs[1], 1, 1, 2, 4096, 131072,
                       PQC_JOB_FLAG_ENCRYPT | PQC_JOB_FLAG_READMOD | PQC_JOB_FLAG_GPU_ELIGIBLE,
                       PQC_PLANE_KEY);
    pqc_block_job_init(&jobs[2], 1, 2, 3, 20480, 262144,
                       PQC_JOB_FLAG_ENCRYPT | PQC_JOB_FLAG_READMOD | PQC_JOB_FLAG_GPU_ELIGIBLE,
                       PQC_PLANE_KEY);
    jobs[0].cpu_queue_depth = supplied_cpu_queue_depth;
    jobs[1].cpu_queue_depth = supplied_cpu_queue_depth > 0 ? supplied_cpu_queue_depth - 1 : 0;
    jobs[2].cpu_queue_depth = supplied_cpu_queue_depth;
    jobs[0].ai_qos_budget_ns = supplied_budget_ns;
    jobs[1].ai_qos_budget_ns = supplied_budget_ns;
    jobs[2].ai_qos_budget_ns = supplied_budget_ns;
    jobs[0].coherence_cost_ns = 0;
    jobs[1].coherence_cost_ns = 1024;
    jobs[2].coherence_cost_ns = 8192;
    fprintf(out,
            "{\"event\":\"scheduler_smoke_begin\",\"gpu_min_bytes\":%zu,"
            "\"coherence_penalty_ns\":%llu,\"supplied_ai_budget_ns\":%llu}\n",
            policy.gpu_min_bytes,
            (unsigned long long)policy.coherence_penalty_ns,
            (unsigned long long)supplied_budget_ns);
    for (size_t i = 0; i < 3; ++i) {
        jobs[i].target = pqc_block_job_choose_target(&jobs[i], &policy, 0.5,
                                                     (double)jobs[i].gpu_queue_depth);
        fprintf(out,
                "{\"event\":\"scheduler_smoke_job\",\"index\":%zu,"
                "\"bytes\":%u,\"target\":\"%s\",\"coherence_ns\":%llu,"
                "\"cpu_queue_depth\":%llu,\"gpu_queue_depth\":%llu,"
                "\"gpu_wait_ns\":%llu}\n",
                i,
                jobs[i].plaintext_length,
                jobs[i].target == PQC_JOB_GPU ? "GPU" : "CPU",
                (unsigned long long)jobs[i].coherence_cost_ns,
                (unsigned long long)jobs[i].cpu_queue_depth,
                (unsigned long long)jobs[i].gpu_queue_depth,
                (unsigned long long)jobs[i].gpu_wait_ns);
    }

    pqc_block_job_t spill = {0};
    pqc_block_job_init(&spill, 9, 9, 9, 0, 131072,
                       PQC_JOB_FLAG_ENCRYPT | PQC_JOB_FLAG_READMOD | PQC_JOB_FLAG_GPU_ELIGIBLE,
                       PQC_PLANE_KEY);
    spill.cpu_queue_depth = 0;
    spill.gpu_queue_depth = supplied_pressure_gpu_queue_depth;
    spill.gpu_wait_ns = policy.gpu_max_wait_ns + 1;
    spill.ai_qos_budget_ns = supplied_budget_ns;
    spill.coherence_cost_ns = 2048;
    spill.target = pqc_block_job_choose_target(&spill, &policy, 0.5,
                                               (double)spill.gpu_queue_depth);
    fprintf(out,
            "{\"event\":\"scheduler_smoke_pressure_job\",\"bytes\":%u,"
            "\"target\":\"%s\",\"gpu_wait_ns\":%llu,\"gpu_queue_depth\":%llu}\n",
            spill.plaintext_length,
            spill.target == PQC_JOB_GPU ? "GPU" : "CPU",
            (unsigned long long)spill.gpu_wait_ns,
            (unsigned long long)spill.gpu_queue_depth);
    fprintf(out, "{\"event\":\"scheduler_smoke_end\",\"jobs\":3}\n");
}

void pqc_scheduler_set_data_accounting_enabled(int enabled)
{
    atomic_store_explicit(&g_data_accounting_enabled, enabled ? 1 : 0,
                          memory_order_release);
}

int pqc_scheduler_data_accounting_enabled(void)
{
    return atomic_load_explicit(&g_data_accounting_enabled,
                                memory_order_acquire) != 0;
}

void pqc_scheduler_schedule_data_job(pqc_block_job_t *job,
                                     const pqc_scheduler_data_job_input_t *input)
{
    if (!job || !input)
        return;

    pqc_block_job_init(job, input->file_id,
                       input->logical_offset / PQC_LOGICAL_BLOCK_SIZE,
                       input->next_generation + 1,
                       input->logical_offset, input->length,
                       PQC_JOB_FLAG_ENCRYPT | PQC_JOB_FLAG_READMOD,
                       PQC_PLANE_DATA);
    job->target = PQC_JOB_CPU;
}

void pqc_scheduler_gpu_admit(uint32_t bytes)
{
    atomic_fetch_add_explicit(&g_gpu_inflight_jobs, 1ULL,
                              memory_order_relaxed);
    sched_stat_add(&g_gpu_inflight_bytes, bytes);
}

void pqc_scheduler_gpu_release(uint32_t bytes)
{
    sched_stat_saturating_sub(&g_gpu_inflight_jobs, 1);
    sched_stat_saturating_sub(&g_gpu_inflight_bytes, bytes);
}

uint64_t pqc_scheduler_gpu_inflight_jobs(void)
{
    return sched_stat_load(&g_gpu_inflight_jobs);
}

void pqc_scheduler_record_key_plane_batch(size_t batch_size, int gpu_used)
{
    if (batch_size == 0)
        return;
    sched_stat_add(&g_sched_stats.submitted, (uint64_t)batch_size);
    if (gpu_used) {
        sched_stat_add(&g_sched_stats.gpu_executed, (uint64_t)batch_size);
        sched_stat_add(&g_sched_stats.key_plane_gpu, (uint64_t)batch_size);
    } else {
        sched_stat_add(&g_sched_stats.cpu_executed, (uint64_t)batch_size);
        sched_stat_add(&g_sched_stats.key_plane_cpu, (uint64_t)batch_size);
    }
}

void pqc_scheduler_record_data_bytes(uint64_t cpu_bytes,
                                     uint64_t gpu_bytes,
                                     uint64_t migration_ns)
{
    if (!pqc_scheduler_data_accounting_enabled())
        return;
    if (cpu_bytes > 0 || gpu_bytes > 0) {
        sched_stat_add(&g_sched_stats.submitted, 1);
        if (gpu_bytes > 0) {
            sched_stat_add(&g_sched_stats.gpu_executed, 1);
        } else {
            sched_stat_add(&g_sched_stats.cpu_executed, 1);
            sched_stat_add(&g_sched_stats.data_plane_cpu, 1);
        }
    }
    sched_stat_add(&g_sched_stats.bytes_cpu, cpu_bytes);
    sched_stat_add(&g_sched_stats.bytes_gpu, gpu_bytes);
    sched_stat_add(&g_sched_stats.gpu_migration_ns, migration_ns);
}

void pqc_scheduler_stats_snapshot(pqc_scheduler_stats_t *out)
{
    if (!out)
        return;
    out->submitted = sched_stat_load(&g_sched_stats.submitted);
    out->cpu_executed = sched_stat_load(&g_sched_stats.cpu_executed);
    out->gpu_executed = sched_stat_load(&g_sched_stats.gpu_executed);
    out->bytes_cpu = sched_stat_load(&g_sched_stats.bytes_cpu);
    out->bytes_gpu = sched_stat_load(&g_sched_stats.bytes_gpu);
    out->gpu_migration_ns = sched_stat_load(&g_sched_stats.gpu_migration_ns);
    out->data_plane_cpu = sched_stat_load(&g_sched_stats.data_plane_cpu);
    out->key_plane_cpu = sched_stat_load(&g_sched_stats.key_plane_cpu);
    out->key_plane_gpu = sched_stat_load(&g_sched_stats.key_plane_gpu);
    out->integrity_plane_cpu =
        sched_stat_load(&g_sched_stats.integrity_plane_cpu);
    out->integrity_plane_gpu =
        sched_stat_load(&g_sched_stats.integrity_plane_gpu);
    out->freshness_plane_cpu =
        sched_stat_load(&g_sched_stats.freshness_plane_cpu);
    out->spilled_ai_budget =
        sched_stat_load(&g_sched_stats.spilled_ai_budget);
    out->spilled_deadline =
        sched_stat_load(&g_sched_stats.spilled_deadline);
    out->spilled_queue = sched_stat_load(&g_sched_stats.spilled_queue);
}
