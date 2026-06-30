#ifndef PQC_SCHEDULER_H
#define PQC_SCHEDULER_H

#include "pqc_block_job.h"

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t file_id;
    uint64_t next_generation;
    uint64_t logical_offset;
    uint32_t length;
    double gpu_load_ewma;
} pqc_scheduler_data_job_input_t;

void pqc_scheduler_reload_runtime_policy_from_env(void);
pqc_scheduler_policy_t pqc_scheduler_policy_from_env(void);
void pqc_scheduler_runtime_policy_snapshot(pqc_scheduler_policy_t *out);
void pqc_scheduler_smoke_report(FILE *out);

void pqc_scheduler_schedule_data_job(pqc_block_job_t *job,
                                     const pqc_scheduler_data_job_input_t *input);
void pqc_scheduler_gpu_admit(uint32_t bytes);
void pqc_scheduler_gpu_release(uint32_t bytes);
uint64_t pqc_scheduler_gpu_inflight_jobs(void);

void pqc_scheduler_record_key_plane_batch(size_t batch_size, int gpu_used);
void pqc_scheduler_record_data_bytes(uint64_t cpu_bytes,
                                     uint64_t gpu_bytes,
                                     uint64_t migration_ns);
void pqc_scheduler_stats_snapshot(pqc_scheduler_stats_t *out);

#ifdef __cplusplus
}
#endif

#endif /* PQC_SCHEDULER_H */
