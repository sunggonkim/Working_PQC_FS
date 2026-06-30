#include "pqc_flush_batch.h"

#include "cuda_aead.h"
#include "pqc_format.h"
#include "pqc_recovery.h"

#include <errno.h>
#include <openssl/crypto.h>
#include <string.h>

void pqc_flush_batch_init(pqc_flush_batch_t *batch)
{
    if (!batch)
        return;
    memset(batch, 0, sizeof(*batch));
}

static uint32_t block_plaintext_length(uint64_t final_size, uint64_t block_start)
{
    uint64_t remaining = final_size - block_start;
    if (remaining < PQC_LOGICAL_BLOCK_SIZE)
        return (uint32_t)remaining;
    return PQC_LOGICAL_BLOCK_SIZE;
}

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
                            size_t scratch_capacity)
{
    if (!batch || !key || key_len == 0 || !write_buf || write_size == 0 ||
        !plain_scratch || !cipher_scratch || scratch_capacity == 0)
        return -EINVAL;

    pqc_flush_batch_init(batch);
    batch->base = base;
    batch->write_size = write_size;
    if ((uint64_t)write_size > UINT64_MAX - base)
        return -EFBIG;
    batch->end = base + (uint64_t)write_size;
    batch->final_size = batch->end > logical_size ? batch->end : logical_size;

    uint64_t first = base / PQC_LOGICAL_BLOCK_SIZE;
    uint64_t last = (batch->end - 1) / PQC_LOGICAL_BLOCK_SIZE;
    uint64_t block_count = last - first + 1;
    if (block_count == 0 || block_count > PQC_WRITEBACK_MAX_BLOCKS)
        return -E2BIG;
    batch->block_count = (size_t)block_count;
    batch->blocks = batch->block_storage;
    memset(batch->blocks, 0, batch->block_count * sizeof(*batch->blocks));

    for (size_t bi = 0; bi < batch->block_count; ++bi) {
        uint64_t block = first + bi;
        uint64_t block_start = block * PQC_LOGICAL_BLOCK_SIZE;
        pqc_crypto_block_desc_t *desc = &batch->blocks[bi];
        desc->block = block;
        desc->generation = first_generation + bi;
        desc->length = block_plaintext_length(batch->final_size, block_start);
        desc->input_offset = batch->packed_bytes;
        desc->output_offset = batch->packed_bytes;
        int rc = pqc_crypto_derive_block_nonce(file_id, block, desc->generation,
                                               desc->nonce);
        if (rc != 0)
            return rc;
        pqc_crypto_build_block_aad(desc->aad, file_id, block,
                                   desc->generation, desc->length);
        batch->packed_bytes += desc->length;
    }

    if (batch->packed_bytes > scratch_capacity)
        return -E2BIG;
    batch->plain_batch = plain_scratch;
    batch->cipher_batch = cipher_scratch;

    batch->packed_bytes = 0;
    for (size_t bi = 0; bi < batch->block_count; ++bi) {
        pqc_crypto_block_desc_t *desc = &batch->blocks[bi];
        uint8_t plain[PQC_LOGICAL_BLOCK_SIZE];
        uint64_t block_start = desc->block * PQC_LOGICAL_BLOCK_SIZE;
        uint64_t from = base > block_start ? base - block_start : 0;
        uint64_t to = batch->end < block_start + PQC_LOGICAL_BLOCK_SIZE ?
                      batch->end - block_start : PQC_LOGICAL_BLOCK_SIZE;
        uint64_t source = block_start + from - base;
        uint64_t max_recovery_generation =
            first_generation > 0 ? first_generation - 1U : UINT64_MAX;
        int rc = 0;
        if (epoch_fallback_marker_path && epoch_fallback_marker_path[0]) {
            rc = pqc_recovery_load_authenticated_block_committed_epoch_fallback(
                journal_fd, data_fd, epoch_fallback_marker_path, key, key_len,
                file_id, desc->block, max_recovery_generation, plain);
        } else {
            rc = pqc_recovery_load_authenticated_block_committed(
                journal_fd, data_fd, key, key_len, file_id, desc->block,
                max_recovery_generation, plain);
        }
        if (rc != 0) {
            OPENSSL_cleanse(plain, sizeof(plain));
            return rc;
        }
        memcpy(plain + from, write_buf + source, to - from);
        memcpy(batch->plain_batch + batch->packed_bytes, plain, desc->length);
        desc->input_offset = batch->packed_bytes;
        desc->output_offset = batch->packed_bytes;
        batch->packed_bytes += desc->length;
        OPENSSL_cleanse(plain, sizeof(plain));
    }

    return 0;
}

void pqc_flush_batch_cleanup(pqc_flush_batch_t *batch)
{
    if (!batch)
        return;
    if (skim_cuda_aead_is_uma()) {
        if (batch->plain_batch)
            (void)skim_cuda_mem_prefetch_host(batch->plain_batch,
                                              batch->packed_bytes);
        if (batch->cipher_batch)
            (void)skim_cuda_mem_prefetch_host(batch->cipher_batch,
                                              batch->packed_bytes);
    }
    if (batch->cipher_batch)
        OPENSSL_cleanse(batch->cipher_batch, batch->packed_bytes);
    if (batch->plain_batch)
        OPENSSL_cleanse(batch->plain_batch, batch->packed_bytes);
    if (batch->blocks)
        OPENSSL_cleanse(batch->blocks,
                        batch->block_count * sizeof(*batch->blocks));
    pqc_flush_batch_init(batch);
}
