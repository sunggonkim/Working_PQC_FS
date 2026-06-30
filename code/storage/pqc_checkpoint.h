#ifndef PQC_CHECKPOINT_H
#define PQC_CHECKPOINT_H

#include <stdint.h>

#include "pqc_anchor.h"
#include "pqc_format.h"

/*
 * Strict-mode checkpoint orchestration.
 *
 * pqc_publish.[ch] owns the checkpoint xattr bytes.  pqc_anchor.[ch] owns the
 * committed-prefix map and freshness backend.  pqc_anchor_worker.[ch] owns the
 * background worker that persists staged anchor state.  This module keeps the
 * existing ordering between those boundaries explicit during mechanical
 * decomposition; it does not introduce epoch mode or a new durability policy.
 */

void pqc_checkpoint_make_anchor_state(uint64_t sequence,
                                      uint64_t logical_size,
                                      uint64_t max_generation,
                                      pqc_anchor_state_t *out);

int pqc_checkpoint_record_file_anchor(uint64_t file_id,
                                      uint64_t sequence,
                                      uint64_t logical_size,
                                      uint64_t max_generation,
                                      pqc_anchor_state_t *out_state);

int pqc_checkpoint_store_and_stage_anchor(const char *path,
                                          uint64_t file_id,
                                          uint64_t sequence,
                                          uint64_t logical_size,
                                          uint64_t max_generation);

int pqc_checkpoint_reserve_generation(const char *path,
                                      uint64_t file_id,
                                      uint64_t logical_size,
                                      uint64_t reserved_generation);

int pqc_checkpoint_load_and_verify_anchor(const char *path,
                                          uint64_t expected_file_id,
                                          pqc_checkpoint_t *out);

#endif /* PQC_CHECKPOINT_H */
