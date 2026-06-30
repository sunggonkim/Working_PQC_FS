#ifndef PQC_CRYPTO_H
#define PQC_CRYPTO_H

#include <stddef.h>
#include <stdint.h>

#include "pqc_format.h"

typedef struct {
    uint64_t block;
    uint64_t generation;
    uint32_t length;
    size_t input_offset;
    size_t output_offset;
    uint64_t ciphertext_offset;
    uint8_t nonce[PQC_AEAD_NONCE_SIZE];
    uint8_t aad[28];
    uint8_t tag[PQC_AEAD_TAG_SIZE];
} pqc_crypto_block_desc_t;

int pqc_crypto_gcm_compute_tag(const uint8_t key[32],
                               const uint8_t nonce[PQC_AEAD_NONCE_SIZE],
                               const uint8_t *aad, size_t aad_length,
                               const uint8_t *ciphertext,
                               size_t ciphertext_length,
                               uint8_t tag[PQC_AEAD_TAG_SIZE]);
int pqc_crypto_derive_block_nonce(uint64_t file_id, uint64_t block,
                                  uint64_t generation,
                                  uint8_t nonce[PQC_AEAD_NONCE_SIZE]);
void pqc_crypto_build_block_aad(uint8_t aad[28], uint64_t file_id,
                                uint64_t block, uint64_t generation,
                                uint32_t length);
int pqc_crypto_crypt_block_gcm(const uint8_t *key, size_t key_len,
                               uint64_t file_id, uint64_t block,
                               uint64_t generation, uint32_t length,
                               const uint8_t *in, uint8_t *out,
                               uint8_t tag[PQC_AEAD_TAG_SIZE],
                               int encrypt, int prefer_gpu);
int pqc_crypto_crypt_block_batch_gcm(const uint8_t *key, size_t key_len,
                                     uint64_t file_id,
                                     pqc_crypto_block_desc_t *blocks,
                                     size_t count, const uint8_t *input,
                                     uint8_t *output);
int pqc_crypto_decrypt_block_batch_gcm(const uint8_t *key, size_t key_len,
                                       uint64_t file_id,
                                       pqc_crypto_block_desc_t *blocks,
                                       size_t count, const uint8_t *input,
                                       uint8_t *output, int prefer_gpu);

#endif /* PQC_CRYPTO_H */
