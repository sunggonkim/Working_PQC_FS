#ifndef PQC_FLUSH_BATCH_H
#define PQC_FLUSH_BATCH_H

#include "pqc_crypto.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t base;
    uint64_t end;
    uint64_t final_size;
    size_t write_size;
    size_t block_count;
    size_t packed_bytes;
    pqc_crypto_block_desc_t *blocks;
    pqc_crypto_block_desc_t block_storage[PQC_WRITEBACK_MAX_BLOCKS];
    uint8_t *plain_batch;
    uint8_t *cipher_batch;
} pqc_flush_batch_t;

void pqc_flush_batch_init(pqc_flush_batch_t *batch);
int pqc_flush_batch_prepare(pqc_flush_batch_t *batch,
                            int journal_fd,
                            int data_fd,
                            const char *epoch_fallback_marker_path,
                            const uint8_t *key,
                            size_t key_len,
                            uint64_t file_id,
                            uint64_t first_generation,
                            uint64_t logical_size,
                            uint64_t base,
                            size_t write_size,
                            const uint8_t *write_buf,
                            uint8_t *plain_scratch,
                            uint8_t *cipher_scratch,
                            size_t scratch_capacity);
void pqc_flush_batch_cleanup(pqc_flush_batch_t *batch);

#ifdef __cplusplus
}
#endif

#endif /* PQC_FLUSH_BATCH_H */
