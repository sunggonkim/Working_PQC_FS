#ifndef PQC_PARALLEL_COMMIT_H
#define PQC_PARALLEL_COMMIT_H

#include <stdint.h>

#define PQC_PARALLEL_COMMIT_MAX_SHARDS 64U

typedef enum {
    PQC_PARALLEL_COMMIT_ROLE_INVALID = 0,
    PQC_PARALLEL_COMMIT_ROLE_LEADER = 1,
    PQC_PARALLEL_COMMIT_ROLE_FOLLOWER = 2,
} pqc_parallel_commit_role_t;

typedef struct {
    uint32_t shard_count;
    uint32_t max_group_size;
    uint64_t max_wait_ns;
} pqc_parallel_commit_config_t;

typedef struct pqc_parallel_commit_coordinator pqc_parallel_commit_coordinator_t;

typedef struct {
    pqc_parallel_commit_role_t role;
    uint32_t shard;
    uint32_t shard_count;
    uint64_t epoch;
    uint32_t group_size;
    uint64_t group_bytes;
    uint32_t observed_queue_depth;
    uint64_t wait_ns;
    int result_rc;
    pqc_parallel_commit_coordinator_t *runtime_coordinator;
    int runtime_ref_held;
} pqc_parallel_commit_ticket_t;

typedef struct {
    uint32_t shard_count;
    uint64_t total_epochs;
    uint64_t total_requests;
    uint64_t total_leaders;
    uint64_t total_followers;
    uint64_t total_group_size;
    uint32_t max_observed_group_size;
    uint64_t max_observed_group_bytes;
    uint32_t max_observed_queue_depth;
    uint64_t total_wait_ns;
    uint64_t max_wait_ns;
    uint64_t wait_timeout_epochs;
    uint64_t full_group_epochs;
} pqc_parallel_commit_stats_t;

uint32_t pqc_parallel_commit_shard_for_file(uint64_t file_id,
                                            uint32_t shard_count);
int pqc_parallel_commit_init(pqc_parallel_commit_coordinator_t **out,
                             const pqc_parallel_commit_config_t *config);
void pqc_parallel_commit_destroy(pqc_parallel_commit_coordinator_t *coordinator);
int pqc_parallel_commit_begin(pqc_parallel_commit_coordinator_t *coordinator,
                              uint64_t file_id,
                              uint64_t bytes,
                              pqc_parallel_commit_ticket_t *ticket);
int pqc_parallel_commit_finish(pqc_parallel_commit_coordinator_t *coordinator,
                               const pqc_parallel_commit_ticket_t *ticket,
                               int result_rc);
void pqc_parallel_commit_stats_snapshot(
    pqc_parallel_commit_coordinator_t *coordinator,
    pqc_parallel_commit_stats_t *out);

int pqc_parallel_commit_runtime_init_from_config(void);
void pqc_parallel_commit_runtime_shutdown(void);
int pqc_parallel_commit_runtime_enabled(void);
int pqc_parallel_commit_runtime_begin(uint64_t file_id,
                                      uint64_t bytes,
                                      pqc_parallel_commit_ticket_t *ticket);
int pqc_parallel_commit_runtime_finish(pqc_parallel_commit_ticket_t *ticket,
                                       int result_rc);
void pqc_parallel_commit_runtime_stats_snapshot(
    pqc_parallel_commit_stats_t *out);

#endif /* PQC_PARALLEL_COMMIT_H */
