#ifndef PQC_STRICT_PUBLISH_H
#define PQC_STRICT_PUBLISH_H

#include <stddef.h>
#include <stdint.h>

#include "pqc_crypto.h"

/*
 * Strict-mode durable publication for authenticated data blocks.
 *
 * The caller owns encryption, scheduling, generation reservation, publish-turn
 * ordering, and final in-memory visibility.  This module owns the current
 * strict durable ordering from ciphertext sidecar write through journal
 * publication, logical-size xattr update, checkpoint staging, and final lower
 * file truncate.  It does not implement epoch mode or group commit.
 */

typedef struct pqc_strict_publish_request pqc_strict_publish_request_t;

typedef int (*pqc_strict_publish_after_data_fsync_fn)(
    const pqc_strict_publish_request_t *req,
    void *opaque);

typedef int (*pqc_strict_publish_after_metadata_fn)(
    const pqc_strict_publish_request_t *req,
    void *opaque);

struct pqc_strict_publish_request {
    int data_fd;
    int journal_fd;
    int epoch_log_fd;
    int epoch_log_syncfs_domain_known;
    int epoch_log_same_syncfs_domain;
    int storage_fd;
    const char *marker_path;
    const char *epoch_log_path;
    uint64_t file_id;
    uint64_t final_size;
    uint64_t reserved_generation;
    uint64_t data_sidecar_append_offset;
    int data_sidecar_append_offset_valid;
    uint64_t *data_sidecar_end_after_append;
    int *data_sidecar_end_after_append_valid;
    uint64_t journal_sidecar_append_offset;
    int journal_sidecar_append_offset_valid;
    uint64_t *journal_sidecar_end_after_append;
    int *journal_sidecar_end_after_append_valid;
    uint32_t algorithm_id;
    pqc_crypto_block_desc_t *blocks;
    size_t block_count;
    const uint8_t *cipher_batch;
    int *data_sidecar_dirty;
    int *journal_sidecar_dirty;
    int *epoch_log_dirty;
    uint64_t *data_sidecar_dirty_epoch;
    uint64_t *journal_sidecar_dirty_epoch;
    uint64_t *epoch_log_dirty_epoch;
    int *journal_sidecar_epoch_repairable;
    uint64_t *journal_sidecar_epoch_repairable_epoch;
    uint64_t *logical_size;
    pqc_strict_publish_after_data_fsync_fn after_data_fsync;
    void *after_data_fsync_opaque;
    pqc_strict_publish_after_metadata_fn after_metadata_publish;
    void *after_metadata_publish_opaque;
    int skip_data_fsync;
    int skip_journal_append;
    int skip_journal_fsync;
    int defer_data_fsync;
    int defer_journal_fsync;
};

int pqc_strict_publish_commit(const pqc_strict_publish_request_t *req);

#endif /* PQC_STRICT_PUBLISH_H */
