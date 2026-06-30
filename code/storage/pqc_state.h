#ifndef PQC_STATE_H
#define PQC_STATE_H

#include <stdint.h>
#include <pthread.h>
#include <sys/types.h>

#include "pqc_format.h"

#define PQC_FILE_STATE_MAPPING_CACHE_CAPACITY 65536U

typedef struct {
    int present;
    block_mapping_t mapping;
} pqc_file_state_mapping_slot_t;

/*
 * Per-backing-inode publication state.  next_generation is a high-water mark:
 * strict mode now reserves generation ranges before ciphertext can be written,
 * so it may be ahead of the journal's committed maximum after an interrupted
 * flush.  committed_generation is the reader-visible journal boundary; strict
 * publication advances it only after data, journal, checkpoint, and logical
 * size publication finish.  publish_ticket serializes read-modify-publish turns
 * without forcing the caller to hold commit_lock while doing crypto or durable
 * sidecar I/O.
 */
typedef struct file_state {
    dev_t dev;
    ino_t ino;
    unsigned refs;
    uint64_t next_generation;
    uint64_t committed_generation;
    uint64_t logical_size;
    int logical_size_valid;
    uint64_t data_sidecar_end;
    int data_sidecar_end_valid;
    uint64_t journal_sidecar_end;
    int journal_sidecar_end_valid;
    uint64_t next_publish_ticket;
    uint64_t publish_ticket;
    pqc_file_state_mapping_slot_t *mapping_cache;
    size_t mapping_cache_count;
    int mapping_cache_complete;
    pthread_mutex_t commit_lock;
    pthread_cond_t publish_cv;
    struct file_state *next;
} file_state_t;

file_state_t *pqc_file_state_acquire(int fd);
void pqc_file_state_release(file_state_t *state);
void pqc_file_state_mapping_cache_mark_complete_empty_locked(file_state_t *state);
void pqc_file_state_mapping_cache_mark_unknown_locked(file_state_t *state);
int pqc_file_state_mapping_cache_store_locked(file_state_t *state,
                                              const block_mapping_t *mappings,
                                              size_t count);
int pqc_file_state_mapping_cache_lookup(file_state_t *state,
                                        uint64_t logical_block,
                                        uint64_t max_generation,
                                        block_mapping_t *out);

#endif /* PQC_STATE_H */
