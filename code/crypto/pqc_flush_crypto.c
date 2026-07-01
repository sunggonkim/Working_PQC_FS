#include "pqc_flush_crypto.h"

#include "pqc_crypto.h"

#include <errno.h>

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

    int res;
    flush_crypto_begin_job(hooks);
    if (use_gpu_batch) {
        res = pqc_crypto_crypt_block_batch_gcm(key, key_len, file_id,
                                               batch->blocks,
                                               batch->block_count,
                                               batch->plain_batch,
                                               batch->cipher_batch, 1);
    } else {
        res = pqc_crypto_encrypt_block_batch_cpu_gcm(
            key, key_len, file_id, batch->blocks, batch->block_count,
            batch->plain_batch, batch->cipher_batch);
    }
    flush_crypto_end_job(hooks);

    return res;
}
