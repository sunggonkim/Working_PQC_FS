#ifndef PQC_PLANE_TRACE_H
#define PQC_PLANE_TRACE_H

#include <stdint.h>

typedef struct {
    uint64_t data_aes_gcm_encrypt_blocks;
    uint64_t data_aes_gcm_encrypt_bytes;
    uint64_t data_aes_gcm_decrypt_blocks;
    uint64_t data_aes_gcm_decrypt_bytes;
    uint64_t data_route_cpu_blocks;
    uint64_t data_route_gpu_blocks;
    uint64_t data_gpu_fallback_events;
    uint64_t keyplane_batches;
    uint64_t keyplane_candidate_files;
    uint64_t keyplane_refreshed_files;
    uint64_t keyplane_work_bytes;
    uint64_t keyplane_cpu_batches;
    uint64_t keyplane_gpu_batches;
    uint64_t keyplane_gpu_fallback_events;
    uint64_t keyplane_failed_batches;
    uint64_t freshness_anchor_events;
    uint64_t freshness_anchor_successes;
    uint64_t freshness_anchor_failures;
    uint64_t freshness_anchor_file_backend;
    uint64_t freshness_anchor_hardware_backend;
} pqc_plane_trace_snapshot_t;

void pqc_plane_trace_record_data_encrypt(uint64_t blocks, uint64_t bytes,
                                         int gpu_used);
void pqc_plane_trace_record_data_decrypt(uint64_t blocks, uint64_t bytes,
                                         int gpu_used);
void pqc_plane_trace_record_data_gpu_fallback(void);
void pqc_plane_trace_record_keyplane_batch(uint64_t candidate_files,
                                           uint64_t refreshed_files,
                                           uint64_t work_bytes,
                                           int target_gpu,
                                           int gpu_used,
                                           int success);
void pqc_plane_trace_record_freshness_anchor(uint32_t backend, int rc);
void pqc_plane_trace_snapshot(pqc_plane_trace_snapshot_t *out);
void pqc_plane_trace_reset(void);
int pqc_plane_trace_dump_if_requested(void);

#endif /* PQC_PLANE_TRACE_H */
