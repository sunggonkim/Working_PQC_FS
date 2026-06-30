#ifndef PQC_JOURNAL_H
#define PQC_JOURNAL_H

#include <stddef.h>
#include <stdint.h>

#include "pqc_format.h"

/*
 * Strict-mode journal record helpers.  These functions preserve the current
 * append-only record format and torn-tail recovery behavior; they do not
 * implement epoch publication or group commit.
 */

int pqc_journal_append_mapping(int journal_fd, const block_mapping_t *mapping);
int pqc_journal_append_mapping_unsynced(int journal_fd,
                                        const block_mapping_t *mapping);
int pqc_journal_append_highwater_unsynced(int journal_fd,
                                          uint64_t max_generation);
int pqc_journal_append_mappings_with_highwater_unsynced(
    int journal_fd,
    const block_mapping_t *mappings,
    size_t count,
    uint64_t max_generation);
int pqc_journal_append_mappings_with_highwater_at_unsynced(
    int journal_fd,
    const block_mapping_t *mappings,
    size_t count,
    uint64_t max_generation,
    uint64_t append_offset,
    uint64_t *end_after_append);
int pqc_journal_lookup_mapping(int journal_fd, uint64_t logical_block,
                               block_mapping_t *out);
int pqc_journal_lookup_mapping_committed(int journal_fd,
                                         uint64_t logical_block,
                                         uint64_t max_generation,
                                         block_mapping_t *out);

#define PQC_JOURNAL_LOOKUP_VIEW_MAX_BLOCKS PQC_WRITEBACK_MAX_BLOCKS

typedef struct {
    int initialized;
    int overflow;
    uint64_t first_block;
    uint64_t last_block;
    uint64_t max_generation;
    size_t slot_count;
    uint8_t present[PQC_JOURNAL_LOOKUP_VIEW_MAX_BLOCKS];
    block_mapping_t mappings[PQC_JOURNAL_LOOKUP_VIEW_MAX_BLOCKS];
} pqc_journal_lookup_view_t;

void pqc_journal_lookup_view_init(pqc_journal_lookup_view_t *view,
                                  uint64_t first_block,
                                  uint64_t last_block,
                                  uint64_t max_generation);
void pqc_journal_lookup_view_clear(pqc_journal_lookup_view_t *view);
int pqc_journal_lookup_mapping_committed_view(
    int journal_fd,
    pqc_journal_lookup_view_t *view,
    uint64_t logical_block,
    uint64_t max_generation,
    block_mapping_t *out);
int pqc_journal_tail_highwater_generation(int journal_fd,
                                          uint64_t *max_generation);
int pqc_journal_tail_highwater_generation_at(int journal_fd,
                                             uint64_t journal_end,
                                             uint64_t *max_generation);
uint64_t pqc_journal_max_generation(int journal_fd);

#endif /* PQC_JOURNAL_H */
