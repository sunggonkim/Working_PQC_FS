#include "pqc_flush_crypto.h"

#include "cuda_aead.h"
#include "pqc_crypto.h"

#include <errno.h>
#include <openssl/crypto.h>
#include <string.h>

static void flush_crypto_begin_job(const pqc_flush_crypto_hooks_t *hooks)
{
    if (hooks && hooks->begin_job)
        hooks->begin_job(hooks->ctx);
}

static void flush_crypto_end_job(const pqc_flush_crypto_hooks_t *hooks)
{
    if (hooks && hooks->end_job)
        hooks->end_job(hooks->ctx);
}

int pqc_flush_crypto_encrypt(const uint8_t *key,
                             size_t key_len,
                             uint64_t file_id,
                             pqc_flush_batch_t *batch,
                             int use_gpu_batch,
                             const pqc_flush_crypto_hooks_t *hooks)
{
    if (!key || !batch || !batch->blocks || !batch->plain_batch ||
        !batch->cipher_batch || batch->block_count == 0)
        return -EINVAL;

    int res = 0;
    if (use_gpu_batch) {
        flush_crypto_begin_job(hooks);
        if (skim_cuda_aead_is_uma()) {
            int dev = skim_cuda_current_device();
            (void)skim_cuda_mem_prefetch(batch->plain_batch,
                                         batch->packed_bytes, dev);
            (void)skim_cuda_mem_prefetch(batch->cipher_batch,
                                         batch->packed_bytes, dev);
        }
        res = pqc_crypto_crypt_block_batch_gcm(key, key_len, file_id,
                                               batch->blocks,
                                               batch->block_count,
                                               batch->plain_batch,
                                               batch->cipher_batch);
        flush_crypto_end_job(hooks);
    }

    if (!use_gpu_batch || res != 0) {
        res = 0;
        for (size_t bi = 0; bi < batch->block_count; ++bi) {
            uint8_t tag[PQC_AEAD_TAG_SIZE];
            flush_crypto_begin_job(hooks);
            res = pqc_crypto_crypt_block_gcm(
                key, key_len, file_id,
                batch->blocks[bi].block,
                batch->blocks[bi].generation,
                batch->blocks[bi].length,
                batch->plain_batch + batch->blocks[bi].input_offset,
                batch->cipher_batch + batch->blocks[bi].output_offset,
                tag, 1, 0);
            flush_crypto_end_job(hooks);
            if (res != 0) {
                OPENSSL_cleanse(tag, sizeof(tag));
                break;
            }
            memcpy(batch->blocks[bi].tag, tag, sizeof(tag));
            OPENSSL_cleanse(tag, sizeof(tag));
        }
    } else {
        for (size_t bi = 0; bi < batch->block_count; ++bi) {
            uint8_t tag[PQC_AEAD_TAG_SIZE];
            res = pqc_crypto_gcm_compute_tag(
                key, batch->blocks[bi].nonce, batch->blocks[bi].aad,
                sizeof(batch->blocks[bi].aad),
                batch->cipher_batch + batch->blocks[bi].output_offset,
                batch->blocks[bi].length, tag);
            if (res != 0) {
                OPENSSL_cleanse(tag, sizeof(tag));
                break;
            }
            memcpy(batch->blocks[bi].tag, tag, sizeof(tag));
            OPENSSL_cleanse(tag, sizeof(tag));
        }
    }

    return res;
}
