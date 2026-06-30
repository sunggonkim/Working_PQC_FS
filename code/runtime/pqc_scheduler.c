#include "pqc_scheduler.h"

#include "cuda_aead.h"
#include "pqc_admission.h"
#include "pqc_config.h"
#include "pqc_format.h"
#include "pqc_lock_profile.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static pthread_mutex_t g_sched_lock = PTHREAD_MUTEX_INITIALIZER;
static pqc_scheduler_stats_t g_sched_stats = {0};
static uint64_t g_gpu_inflight_jobs = 0;
static uint64_t g_gpu_inflight_bytes = 0;
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

static uint64_t monotonic_now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
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
    job->submit_ns = monotonic_now_ns();

    double load = 0.0;
    if (getloadavg(&load, 1) < 0)
        load = 0.0;

    job->cpu_queue_depth = 1 + (load > 1.5 ? 1 : 0);
    job->gpu_queue_depth = 0;
    job->coherence_cost_ns = skim_cuda_aead_is_uma() ? 0 : (uint64_t)input->length * 64ULL;

    pqc_scheduler_policy_t policy;
    uint64_t inflight_jobs = 0;
    uint64_t inflight_bytes = 0;
    pqc_lock_profile_scope_t scope;
    if (scheduler_lock(&scope, "scheduler_data_job_snapshot") == 0) {
        policy = g_sched_policy;
        inflight_jobs = g_gpu_inflight_jobs;
        inflight_bytes = g_gpu_inflight_bytes;
        (void)scheduler_unlock(&scope, "scheduler_data_job_snapshot");
    } else {
        policy = default_policy();
    }

    if (input->length >= policy.gpu_min_bytes)
        job->flags |= PQC_JOB_FLAG_GPU_ELIGIBLE;
    job->gpu_queue_depth = inflight_jobs;
    job->gpu_wait_ns = inflight_jobs * policy.gpu_queue_penalty_ns +
                       inflight_bytes / 4096ULL * (policy.gpu_queue_penalty_ns / 4ULL);

    int gpu_pressure_spill = (policy.gpu_max_inflight_jobs > 0 &&
                              inflight_jobs >= policy.gpu_max_inflight_jobs) ||
                             (policy.gpu_max_inflight_bytes > 0 &&
                              inflight_bytes + input->length > policy.gpu_max_inflight_bytes);
    if (gpu_pressure_spill) {
        job->target = PQC_JOB_CPU;
    } else if (pqc_config_present("PQC_ENABLE_ADMISSION_ON_WRITE") &&
               pqc_block_job_gpu_eligible(job)) {
        pqc_admission_context_t admission_ctx;
        memset(&admission_ctx, 0, sizeof(admission_ctx));
        admission_ctx.batch_count = 1;
        admission_ctx.bytes_total = input->length;
        admission_ctx.batch_age_ns = monotonic_now_ns() - job->submit_ns;
        admission_ctx.gpu_kernel_est_ns = input->length / 4ULL + 100000ULL;
        admission_ctx.gpu_h2d_staging_ns =
            skim_cuda_aead_is_uma() ? 0ULL : (uint64_t)input->length * 8ULL;
        admission_ctx.gpu_d2h_staging_ns =
            skim_cuda_aead_is_uma() ? 0ULL : (uint64_t)input->length * 8ULL;
        admission_ctx.cpu_queue_depth = job->cpu_queue_depth;
        admission_ctx.gpu_queue_depth = job->gpu_queue_depth;
        admission_ctx.cpu_load_avg = load;
        admission_ctx.gpu_load_avg = input->gpu_load_ewma;
        admission_ctx.ai_inference_deadline_ns =
            (uint64_t)pqc_config_positive_long_or_default("PQC_ADMISSION_WRITE_DEADLINE_NS",
                                                          10000000L);
        admission_ctx.uma_migration_cost_ns = job->coherence_cost_ns;
        admission_ctx.uma_migration_bytes_est = skim_cuda_aead_is_uma() ? 0ULL : input->length;
        if (pqc_admit(&admission_ctx) == 0)
            job->target = admission_ctx.chosen_target;
        else
            job->target = PQC_JOB_CPU;
    } else {
        job->target = pqc_block_job_choose_target(job, &policy, load, input->gpu_load_ewma);
    }

    if (scheduler_lock(&scope, "scheduler_data_job_stats") == 0) {
        g_sched_stats.submitted++;
        if (job->target == PQC_JOB_GPU) {
            g_sched_stats.gpu_executed++;
            g_sched_stats.gpu_migration_ns += job->coherence_cost_ns;
        } else {
            g_sched_stats.cpu_executed++;
        }
        (void)scheduler_unlock(&scope, "scheduler_data_job_stats");
    }
}

void pqc_scheduler_gpu_admit(uint32_t bytes)
{
    pqc_lock_profile_scope_t scope;
    if (scheduler_lock(&scope, "scheduler_gpu_admit") != 0)
        return;
    ++g_gpu_inflight_jobs;
    g_gpu_inflight_bytes += bytes;
    (void)scheduler_unlock(&scope, "scheduler_gpu_admit");
}

void pqc_scheduler_gpu_release(uint32_t bytes)
{
    pqc_lock_profile_scope_t scope;
    if (scheduler_lock(&scope, "scheduler_gpu_release") != 0)
        return;
    if (g_gpu_inflight_jobs > 0)
        --g_gpu_inflight_jobs;
    if (g_gpu_inflight_bytes >= bytes)
        g_gpu_inflight_bytes -= bytes;
    else
        g_gpu_inflight_bytes = 0;
    (void)scheduler_unlock(&scope, "scheduler_gpu_release");
}

uint64_t pqc_scheduler_gpu_inflight_jobs(void)
{
    pqc_lock_profile_scope_t scope;
    if (scheduler_lock(&scope, "scheduler_gpu_inflight_jobs") != 0)
        return 0;
    uint64_t jobs = g_gpu_inflight_jobs;
    (void)scheduler_unlock(&scope, "scheduler_gpu_inflight_jobs");
    return jobs;
}

void pqc_scheduler_record_key_plane_batch(size_t batch_size, int gpu_used)
{
    pqc_lock_profile_scope_t scope;
    if (scheduler_lock(&scope, "scheduler_record_key_plane_batch") != 0)
        return;
    g_sched_stats.submitted += batch_size;
    if (gpu_used) {
        g_sched_stats.gpu_executed += batch_size;
        g_sched_stats.key_plane_gpu += batch_size;
    } else {
        g_sched_stats.cpu_executed += batch_size;
        g_sched_stats.key_plane_cpu += batch_size;
    }
    (void)scheduler_unlock(&scope, "scheduler_record_key_plane_batch");
}

void pqc_scheduler_record_data_bytes(uint64_t cpu_bytes,
                                     uint64_t gpu_bytes,
                                     uint64_t migration_ns)
{
    pqc_lock_profile_scope_t scope;
    if (scheduler_lock(&scope, "scheduler_record_data_bytes") != 0)
        return;
    g_sched_stats.bytes_cpu += cpu_bytes;
    g_sched_stats.bytes_gpu += gpu_bytes;
    g_sched_stats.gpu_migration_ns += migration_ns;
    (void)scheduler_unlock(&scope, "scheduler_record_data_bytes");
}

void pqc_scheduler_stats_snapshot(pqc_scheduler_stats_t *out)
{
    if (!out)
        return;
    pqc_lock_profile_scope_t scope;
    if (scheduler_lock(&scope, "scheduler_stats_snapshot") != 0) {
        memset(out, 0, sizeof(*out));
        return;
    }
    *out = g_sched_stats;
    (void)scheduler_unlock(&scope, "scheduler_stats_snapshot");
}
