#ifndef PQC_FLUSH_CRYPTO_H
#define PQC_FLUSH_CRYPTO_H

#include "pqc_flush_batch.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void *ctx;
    void (*begin_job)(void *ctx);
    void (*end_job)(void *ctx);
} pqc_flush_crypto_hooks_t;

int pqc_flush_crypto_encrypt(const uint8_t *key,
                             size_t key_len,
                             uint64_t file_id,
                             pqc_flush_batch_t *batch,
                             int use_gpu_batch,
                             const pqc_flush_crypto_hooks_t *hooks);

#ifdef __cplusplus
}
#endif

#endif /* PQC_FLUSH_CRYPTO_H */
