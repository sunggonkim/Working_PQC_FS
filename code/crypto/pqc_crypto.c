#include "pqc_crypto.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include <openssl/crypto.h>
#include <openssl/evp.h>

#include "cuda_aead.h"
#include "pqc_plane_trace.h"

static int aes256_ecb_block(const uint8_t key[32], const uint8_t in[16],
                            uint8_t out[16])
{
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    int produced = 0, final = 0;
    int ok = ctx &&
             EVP_EncryptInit_ex(ctx, EVP_aes_256_ecb(), NULL, key, NULL) == 1 &&
             EVP_CIPHER_CTX_set_padding(ctx, 0) == 1 &&
             EVP_EncryptUpdate(ctx, out, &produced, in, 16) == 1 &&
             EVP_EncryptFinal_ex(ctx, out + produced, &final) == 1 &&
             produced + final == 16;
    EVP_CIPHER_CTX_free(ctx);
    return ok ? 0 : -EIO;
}

static void gcm_shift_right(uint8_t value[16])
{
    uint8_t carry = 0;
    for (size_t i = 0; i < 16; ++i) {
        uint8_t next = (uint8_t)(value[i] & 1U);
        value[i] = (uint8_t)((value[i] >> 1) | (carry << 7));
        carry = next;
    }
}

static void gcm_multiply(const uint8_t x[16], const uint8_t y[16],
                         uint8_t out[16])
{
    uint8_t z[16] = {0}, v[16];
    memcpy(v, y, sizeof(v));
    for (size_t bit = 0; bit < 128; ++bit) {
        if ((x[bit / 8] >> (7 - (bit % 8))) & 1U)
            for (size_t i = 0; i < sizeof(z); ++i) z[i] ^= v[i];
        int lsb = v[15] & 1U;
        gcm_shift_right(v);
        if (lsb) v[0] ^= 0xe1U;
    }
    memcpy(out, z, sizeof(z));
    OPENSSL_cleanse(v, sizeof(v));
    OPENSSL_cleanse(z, sizeof(z));
}

static void gcm_absorb(uint8_t state[16], const uint8_t h[16],
                       const uint8_t *data, size_t length)
{
    while (length) {
        uint8_t block[16] = {0};
        size_t take = length < sizeof(block) ? length : sizeof(block);
        memcpy(block, data, take);
        for (size_t i = 0; i < sizeof(block); ++i) state[i] ^= block[i];
        gcm_multiply(state, h, state);
        OPENSSL_cleanse(block, sizeof(block));
        data += take;
        length -= take;
    }
}

static void store_u64_be(uint8_t out[8], uint64_t value)
{
    for (int i = 7; i >= 0; --i) {
        out[i] = (uint8_t)value;
        value >>= 8;
    }
}

int pqc_crypto_gcm_compute_tag(const uint8_t key[32],
                               const uint8_t nonce[PQC_AEAD_NONCE_SIZE],
                               const uint8_t *aad, size_t aad_length,
                               const uint8_t *ciphertext,
                               size_t ciphertext_length,
                               uint8_t tag[PQC_AEAD_TAG_SIZE])
{
    uint8_t zero[16] = {0}, h[16], state[16] = {0}, j0[16] = {0};
    uint8_t encrypted_j0[16], lengths[16] = {0};
    memcpy(j0, nonce, PQC_AEAD_NONCE_SIZE);
    j0[15] = 1;
    if (aes256_ecb_block(key, zero, h) ||
        aes256_ecb_block(key, j0, encrypted_j0))
        return -EIO;
    gcm_absorb(state, h, aad, aad_length);
    gcm_absorb(state, h, ciphertext, ciphertext_length);
    store_u64_be(lengths, (uint64_t)aad_length * 8U);
    store_u64_be(lengths + 8, (uint64_t)ciphertext_length * 8U);
    for (size_t i = 0; i < sizeof(state); ++i) state[i] ^= lengths[i];
    gcm_multiply(state, h, state);
    for (size_t i = 0; i < PQC_AEAD_TAG_SIZE; ++i)
        tag[i] = encrypted_j0[i] ^ state[i];
    OPENSSL_cleanse(h, sizeof(h));
    OPENSSL_cleanse(state, sizeof(state));
    OPENSSL_cleanse(encrypted_j0, sizeof(encrypted_j0));
    return 0;
}

int pqc_crypto_derive_block_nonce(uint64_t file_id, uint64_t block,
                                  uint64_t generation,
                                  uint8_t nonce[PQC_AEAD_NONCE_SIZE])
{
    uint8_t digest[32];
    uint8_t nonce_seed[24];
    EVP_MD_CTX *md = NULL;

    memcpy(nonce_seed, &file_id, sizeof(file_id));
    memcpy(nonce_seed + 8, &block, sizeof(block));
    memcpy(nonce_seed + 16, &generation, sizeof(generation));
    md = EVP_MD_CTX_new();
    if (!md) return -ENOMEM;
    if (EVP_DigestInit_ex(md, EVP_sha256(), NULL) != 1 ||
        EVP_DigestUpdate(md, nonce_seed, sizeof(nonce_seed)) != 1 ||
        EVP_DigestFinal_ex(md, digest, NULL) != 1) {
        EVP_MD_CTX_free(md);
        return -EIO;
    }
    EVP_MD_CTX_free(md);
    memcpy(nonce, digest, PQC_AEAD_NONCE_SIZE);
    OPENSSL_cleanse(digest, sizeof(digest));
    OPENSSL_cleanse(nonce_seed, sizeof(nonce_seed));
    return 0;
}

void pqc_crypto_build_block_aad(uint8_t aad[28], uint64_t file_id,
                                uint64_t block, uint64_t generation,
                                uint32_t length)
{
    memcpy(aad, &file_id, 8);
    memcpy(aad + 8, &block, 8);
    memcpy(aad + 16, &generation, 8);
    memcpy(aad + 24, &length, 4);
}

static uint64_t crypto_block_desc_total_bytes(
    const pqc_crypto_block_desc_t *blocks, size_t count)
{
    uint64_t total = 0;
    if (!blocks)
        return 0;
    for (size_t i = 0; i < count; ++i)
        total += blocks[i].length;
    return total;
}

int pqc_crypto_crypt_block_gcm(const uint8_t *key, size_t key_len,
                               uint64_t file_id, uint64_t block,
                               uint64_t generation, uint32_t length,
                               const uint8_t *in, uint8_t *out,
                               uint8_t tag[PQC_AEAD_TAG_SIZE],
                               int encrypt, int prefer_gpu)
{
    if (!key || key_len < 32 || !in || !out || length > PQC_LOGICAL_BLOCK_SIZE)
        return -EINVAL;
    uint8_t nonce[PQC_AEAD_NONCE_SIZE] = {0};
    uint8_t aad[28];
    int rc = pqc_crypto_derive_block_nonce(file_id, block, generation, nonce);
    if (rc) return rc;
    pqc_crypto_build_block_aad(aad, file_id, block, generation, length);
    int gpu_attempted = prefer_gpu && skim_cuda_aead_available();
    if (gpu_attempted) {
        if (encrypt) {
            if (skim_cuda_aes256_gcm_ctr(key, nonce, in, out, length) == 0 &&
                pqc_crypto_gcm_compute_tag(key, nonce, aad, sizeof(aad), out,
                                           length, tag) == 0) {
                pqc_plane_trace_record_data_encrypt(1, length, 1);
                return 0;
            }
        } else {
            uint8_t expected_tag[PQC_AEAD_TAG_SIZE];
            if (pqc_crypto_gcm_compute_tag(key, nonce, aad, sizeof(aad), in,
                                           length, expected_tag) != 0)
                return -EIO;
            int valid = CRYPTO_memcmp(expected_tag, tag, PQC_AEAD_TAG_SIZE) == 0;
            OPENSSL_cleanse(expected_tag, sizeof(expected_tag));
            if (!valid) return -EBADMSG;
            if (skim_cuda_aes256_gcm_ctr(key, nonce, in, out, length) == 0) {
                pqc_plane_trace_record_data_decrypt(1, length, 1);
                return 0;
            }
        }
        /* CUDA failure is a performance failure, never a format change. */
        pqc_plane_trace_record_data_gpu_fallback();
    }
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx) return -ENOMEM;
    int n = 0, total = 0;
    int ok;
    if (encrypt) {
        ok = EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, key, nonce) == 1 &&
             EVP_EncryptUpdate(ctx, NULL, &n, aad, sizeof(aad)) == 1 &&
             EVP_EncryptUpdate(ctx, out, &n, in, (int)length) == 1;
        total = n;
        ok = ok && EVP_EncryptFinal_ex(ctx, out + total, &n) == 1 &&
             EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG,
                                 PQC_AEAD_TAG_SIZE, tag) == 1;
    } else {
        ok = EVP_DecryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, key, nonce) == 1 &&
             EVP_DecryptUpdate(ctx, NULL, &n, aad, sizeof(aad)) == 1 &&
             EVP_DecryptUpdate(ctx, out, &n, in, (int)length) == 1 &&
             EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG,
                                 PQC_AEAD_TAG_SIZE, tag) == 1;
        total = n;
        ok = ok && EVP_DecryptFinal_ex(ctx, out + total, &n) == 1;
    }
    EVP_CIPHER_CTX_free(ctx);
    if (ok) {
        if (encrypt)
            pqc_plane_trace_record_data_encrypt(1, length, 0);
        else
            pqc_plane_trace_record_data_decrypt(1, length, 0);
    }
    return ok ? 0 : -EBADMSG;
}

static int pqc_crypto_crypt_block_batch_cpu(const uint8_t *key, size_t key_len,
                                            uint64_t file_id,
                                            pqc_crypto_block_desc_t *blocks,
                                            size_t count,
                                            const uint8_t *input,
                                            uint8_t *output)
{
    for (size_t i = 0; i < count; ++i) {
        uint8_t tag[PQC_AEAD_TAG_SIZE];
        int rc = pqc_crypto_crypt_block_gcm(key, key_len, file_id,
                                            blocks[i].block,
                                            blocks[i].generation,
                                            blocks[i].length,
                                            input + blocks[i].input_offset,
                                            output + blocks[i].output_offset,
                                            tag, 1, 0);
        if (rc != 0) {
            OPENSSL_cleanse(tag, sizeof(tag));
            return rc;
        }
        memcpy(blocks[i].tag, tag, sizeof(tag));
        OPENSSL_cleanse(tag, sizeof(tag));
    }
    return 0;
}

static int pqc_crypto_decrypt_block_batch_cpu(const uint8_t *key,
                                              size_t key_len,
                                              uint64_t file_id,
                                              pqc_crypto_block_desc_t *blocks,
                                              size_t count,
                                              const uint8_t *input,
                                              uint8_t *output)
{
    (void)file_id;
    if (!key || key_len < 32 || !blocks || !input || !output || count == 0)
        return -EINVAL;

    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx)
        return -ENOMEM;

    int rc = 0;
    for (size_t i = 0; i < count; ++i) {
        if (blocks[i].length > PQC_LOGICAL_BLOCK_SIZE) {
            rc = -EINVAL;
            break;
        }
        int n = 0, total = 0;
        int ok = EVP_CIPHER_CTX_reset(ctx) == 1 &&
                 EVP_DecryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, key,
                                    blocks[i].nonce) == 1 &&
                 EVP_DecryptUpdate(ctx, NULL, &n, blocks[i].aad,
                                   sizeof(blocks[i].aad)) == 1 &&
                 EVP_DecryptUpdate(ctx, output + blocks[i].output_offset, &n,
                                   input + blocks[i].input_offset,
                                   (int)blocks[i].length) == 1 &&
                 EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG,
                                     PQC_AEAD_TAG_SIZE,
                                     blocks[i].tag) == 1;
        total = n;
        ok = ok && EVP_DecryptFinal_ex(
            ctx, output + blocks[i].output_offset + total, &n) == 1;
        if (!ok) {
            rc = -EBADMSG;
            break;
        }
    }

    EVP_CIPHER_CTX_free(ctx);
    if (rc == 0) {
        pqc_plane_trace_record_data_decrypt(
            (uint64_t)count, crypto_block_desc_total_bytes(blocks, count), 0);
    }
    return rc;
}

int pqc_crypto_crypt_block_batch_gcm(const uint8_t *key, size_t key_len,
                                     uint64_t file_id,
                                     pqc_crypto_block_desc_t *blocks,
                                     size_t count, const uint8_t *input,
                                     uint8_t *output)
{
    if (!key || key_len < 32 || !blocks || !input || !output || count == 0)
        return -EINVAL;

    if (count > PQC_WRITEBACK_MAX_BLOCKS)
        return pqc_crypto_crypt_block_batch_cpu(key, key_len, file_id, blocks,
                                                count, input, output);

    size_t offsets[PQC_WRITEBACK_MAX_BLOCKS];
    size_t lengths[PQC_WRITEBACK_MAX_BLOCKS];
    uint8_t nonces[PQC_WRITEBACK_MAX_BLOCKS * PQC_AEAD_NONCE_SIZE];
    uint8_t aads[PQC_WRITEBACK_MAX_BLOCKS * sizeof(blocks[0].aad)];
    uint8_t tags[PQC_WRITEBACK_MAX_BLOCKS * PQC_AEAD_TAG_SIZE];
    int rc = 0;

    for (size_t i = 0; i < count; ++i) {
        offsets[i] = blocks[i].input_offset;
        lengths[i] = blocks[i].length;
        memcpy(nonces + i * PQC_AEAD_NONCE_SIZE, blocks[i].nonce,
               PQC_AEAD_NONCE_SIZE);
        memcpy(aads + i * sizeof(blocks[i].aad), blocks[i].aad,
               sizeof(blocks[i].aad));
    }

    if (skim_cuda_aead_available()) {
        int gpu_rc = skim_cuda_aes256_gcm_batch(key, nonces, aads, input,
                                                output, offsets, lengths,
                                                tags, count);
        if (gpu_rc == 0) {
        for (size_t i = 0; i < count; ++i) {
            memcpy(blocks[i].tag, tags + i * PQC_AEAD_TAG_SIZE,
                   PQC_AEAD_TAG_SIZE);
        }
        pqc_plane_trace_record_data_encrypt(
            (uint64_t)count, crypto_block_desc_total_bytes(blocks, count), 1);
        rc = 0;
        goto out;
        }
        pqc_plane_trace_record_data_gpu_fallback();
    }

    rc = pqc_crypto_crypt_block_batch_cpu(key, key_len, file_id, blocks,
                                          count, input, output);
out:
    OPENSSL_cleanse(tags, sizeof(tags));
    OPENSSL_cleanse(aads, sizeof(aads));
    OPENSSL_cleanse(nonces, sizeof(nonces));
    OPENSSL_cleanse(lengths, sizeof(lengths));
    OPENSSL_cleanse(offsets, sizeof(offsets));
    return rc;
}

int pqc_crypto_decrypt_block_batch_gcm(const uint8_t *key, size_t key_len,
                                       uint64_t file_id,
                                       pqc_crypto_block_desc_t *blocks,
                                       size_t count, const uint8_t *input,
                                       uint8_t *output, int prefer_gpu)
{
    if (!key || key_len < 32 || !blocks || !input || !output || count == 0)
        return -EINVAL;

    if (count > PQC_WRITEBACK_MAX_BLOCKS || !prefer_gpu)
        return pqc_crypto_decrypt_block_batch_cpu(key, key_len, file_id,
                                                  blocks, count, input,
                                                  output);

    size_t offsets[PQC_WRITEBACK_MAX_BLOCKS];
    size_t lengths[PQC_WRITEBACK_MAX_BLOCKS];
    uint8_t nonces[PQC_WRITEBACK_MAX_BLOCKS * PQC_AEAD_NONCE_SIZE];
    uint8_t expected_tag[PQC_AEAD_TAG_SIZE];
    int same_offsets = 1;

    for (size_t i = 0; i < count; ++i) {
        if (blocks[i].length > PQC_LOGICAL_BLOCK_SIZE) {
            OPENSSL_cleanse(expected_tag, sizeof(expected_tag));
            return -EINVAL;
        }
        offsets[i] = blocks[i].input_offset;
        lengths[i] = blocks[i].length;
        if (blocks[i].input_offset != blocks[i].output_offset)
            same_offsets = 0;
        memcpy(nonces + i * PQC_AEAD_NONCE_SIZE, blocks[i].nonce,
               PQC_AEAD_NONCE_SIZE);
        int rc = pqc_crypto_gcm_compute_tag(
            key, blocks[i].nonce, blocks[i].aad, sizeof(blocks[i].aad),
            input + blocks[i].input_offset, blocks[i].length, expected_tag);
        if (rc != 0) {
            OPENSSL_cleanse(expected_tag, sizeof(expected_tag));
            OPENSSL_cleanse(nonces, sizeof(nonces));
            OPENSSL_cleanse(lengths, sizeof(lengths));
            OPENSSL_cleanse(offsets, sizeof(offsets));
            return rc;
        }
        if (CRYPTO_memcmp(expected_tag, blocks[i].tag,
                          PQC_AEAD_TAG_SIZE) != 0) {
            OPENSSL_cleanse(expected_tag, sizeof(expected_tag));
            OPENSSL_cleanse(nonces, sizeof(nonces));
            OPENSSL_cleanse(lengths, sizeof(lengths));
            OPENSSL_cleanse(offsets, sizeof(offsets));
            return -EBADMSG;
        }
    }

    int rc = -1;
    if (same_offsets && skim_cuda_aead_available()) {
        rc = skim_cuda_aes256_gcm_ctr_batch(key, nonces, input, output,
                                            offsets, lengths, count);
        if (rc == 0) {
            pqc_plane_trace_record_data_decrypt(
                (uint64_t)count, crypto_block_desc_total_bytes(blocks, count),
                1);
            goto out;
        }
        pqc_plane_trace_record_data_gpu_fallback();
    }

    rc = pqc_crypto_decrypt_block_batch_cpu(key, key_len, file_id, blocks,
                                            count, input, output);
out:
    OPENSSL_cleanse(expected_tag, sizeof(expected_tag));
    OPENSSL_cleanse(nonces, sizeof(nonces));
    OPENSSL_cleanse(lengths, sizeof(lengths));
    OPENSSL_cleanse(offsets, sizeof(offsets));
    return rc;
}
