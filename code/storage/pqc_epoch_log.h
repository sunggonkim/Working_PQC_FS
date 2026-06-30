#ifndef PQC_EPOCH_LOG_H
#define PQC_EPOCH_LOG_H

#include <stddef.h>
#include <stdint.h>

#include "pqc_format.h"

typedef struct {
    uint32_t record_type;
    uint32_t flags;
    uint32_t algorithm_id;
    uint64_t epoch;
    uint64_t sequence;
    uint64_t file_id;
    uint64_t logical_block;
    uint64_t generation;
    uint64_t ciphertext_offset;
    uint64_t logical_size_after;
    uint32_t plaintext_length;
    uint8_t tag[PQC_AEAD_TAG_SIZE];
} pqc_epoch_log_record_t;

typedef struct {
    size_t decoded_records;
    size_t block_records;
    size_t commit_records;
    size_t committed_records;
    size_t uncommitted_records;
    size_t duplicate_generation_records;
    size_t journal_repair_records;
    size_t torn_tail_bytes;
    uint64_t committed_epoch;
    uint64_t committed_sequence;
    uint64_t file_id;
    uint64_t logical_size_after;
    uint64_t max_generation;
    uint64_t last_commit_offset;
} pqc_epoch_log_replay_summary_t;

int pqc_epoch_log_encode_record(const pqc_epoch_log_record_t *record,
                                uint8_t *out, size_t out_len,
                                size_t *written);
int pqc_epoch_log_decode_record(const uint8_t *buf, size_t buf_len,
                                pqc_epoch_log_record_t *out);
int pqc_epoch_log_append_record_fd(int fd,
                                   const pqc_epoch_log_record_t *record);
int pqc_epoch_log_append_records_fd(int fd,
                                    const pqc_epoch_log_record_t *records,
                                    size_t count);
int pqc_epoch_log_replay_fd(int fd, uint64_t expected_file_id,
                            pqc_epoch_log_replay_summary_t *out);
int pqc_epoch_log_open_replay_path(const char *marker_path,
                                   uint64_t expected_file_id,
                                   int *fd_out,
                                   pqc_epoch_log_replay_summary_t *out);
int pqc_epoch_log_replay_path(const char *marker_path,
                              uint64_t expected_file_id,
                              pqc_epoch_log_replay_summary_t *out);
int pqc_epoch_log_compact_checkpoint(const char *marker_path,
                                     uint64_t expected_file_id,
                                     int journal_fd,
                                     uint64_t journal_max_generation,
                                     pqc_epoch_log_replay_summary_t *out);
int pqc_epoch_log_lookup_mapping_committed(const char *marker_path,
                                           uint64_t expected_file_id,
                                           uint64_t logical_block,
                                           uint64_t max_generation,
                                           block_mapping_t *out);
int pqc_epoch_log_lookup_mapping_committed_fd(
    int fd,
    const pqc_epoch_log_replay_summary_t *summary,
    uint64_t expected_file_id,
    uint64_t logical_block,
    uint64_t max_generation,
    block_mapping_t *out);

#define PQC_EPOCH_LOG_LOOKUP_VIEW_MAX_BLOCKS PQC_WRITEBACK_MAX_BLOCKS

typedef struct {
    int initialized;
    int overflow;
    int available;
    uint64_t first_block;
    uint64_t last_block;
    uint64_t max_generation;
    uint64_t file_id;
    size_t slot_count;
    pqc_epoch_log_replay_summary_t summary;
    uint8_t present[PQC_EPOCH_LOG_LOOKUP_VIEW_MAX_BLOCKS];
    block_mapping_t mappings[PQC_EPOCH_LOG_LOOKUP_VIEW_MAX_BLOCKS];
} pqc_epoch_log_lookup_view_t;

void pqc_epoch_log_lookup_view_init(pqc_epoch_log_lookup_view_t *view,
                                    uint64_t first_block,
                                    uint64_t last_block,
                                    uint64_t max_generation,
                                    uint64_t file_id);
void pqc_epoch_log_lookup_view_clear(pqc_epoch_log_lookup_view_t *view);
int pqc_epoch_log_lookup_mapping_committed_view(
    const pqc_epoch_log_lookup_view_t *view,
    uint64_t expected_file_id,
    uint64_t logical_block,
    uint64_t max_generation,
    block_mapping_t *out);
int pqc_epoch_log_lookup_mapping_committed_fd_view(
    int fd,
    const pqc_epoch_log_replay_summary_t *summary,
    pqc_epoch_log_lookup_view_t *view,
    uint64_t expected_file_id,
    uint64_t logical_block,
    uint64_t max_generation,
    block_mapping_t *out);
void pqc_epoch_log_trace_shutdown(void);

#endif /* PQC_EPOCH_LOG_H */
