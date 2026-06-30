#ifndef PQC_RECOVERY_H
#define PQC_RECOVERY_H

#include <stddef.h>
#include <stdint.h>

#include "pqc_format.h"
#include "pqc_epoch_log.h"
#include "pqc_journal.h"

/*
 * Strict-mode block recovery.  This helper reconstructs one logical block from
 * the committed journal mapping and ciphertext sidecar, then authenticates it
 * with the existing AES-GCM block format.
 */

int pqc_recovery_load_authenticated_block(int journal_fd, int data_fd,
                                          const uint8_t *key, size_t key_len,
                                          uint64_t file_id, uint64_t block,
                                          uint8_t plain[PQC_LOGICAL_BLOCK_SIZE]);
int pqc_recovery_load_authenticated_block_committed(
    int journal_fd,
    int data_fd,
    const uint8_t *key,
    size_t key_len,
    uint64_t file_id,
    uint64_t block,
    uint64_t max_generation,
    uint8_t plain[PQC_LOGICAL_BLOCK_SIZE]);
int pqc_recovery_load_authenticated_block_committed_epoch_fallback(
    int journal_fd,
    int data_fd,
    const char *marker_path,
    const uint8_t *key,
    size_t key_len,
    uint64_t file_id,
    uint64_t block,
    uint64_t max_generation,
    uint8_t plain[PQC_LOGICAL_BLOCK_SIZE]);

typedef struct {
    int initialized;
    int fd;
    int available;
    pqc_epoch_log_replay_summary_t summary;
    pqc_epoch_log_lookup_view_t *lookup_view;
} pqc_recovery_epoch_fallback_view_t;

void pqc_recovery_epoch_fallback_view_init(
    pqc_recovery_epoch_fallback_view_t *view);
void pqc_recovery_epoch_fallback_view_set_lookup(
    pqc_recovery_epoch_fallback_view_t *view,
    pqc_epoch_log_lookup_view_t *lookup_view);
void pqc_recovery_epoch_fallback_view_close(
    pqc_recovery_epoch_fallback_view_t *view);
int pqc_recovery_lookup_mapping_committed_epoch_view(
    int journal_fd,
    const char *marker_path,
    pqc_journal_lookup_view_t *journal_view,
    pqc_recovery_epoch_fallback_view_t *view,
    uint64_t file_id,
    uint64_t block,
    uint64_t max_generation,
    block_mapping_t *out);
int pqc_recovery_load_authenticated_block_committed_epoch_view(
    int journal_fd,
    int data_fd,
    const char *marker_path,
    pqc_journal_lookup_view_t *journal_view,
    pqc_recovery_epoch_fallback_view_t *view,
    const uint8_t *key,
    size_t key_len,
    uint64_t file_id,
    uint64_t block,
    uint64_t max_generation,
    uint8_t plain[PQC_LOGICAL_BLOCK_SIZE]);

#endif /* PQC_RECOVERY_H */
