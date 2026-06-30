#ifndef PQC_FORMAT_H
#define PQC_FORMAT_H

#include <stdint.h>

/*
 * Shared on-disk/on-xattr format definitions.  This header owns magic values,
 * version values, and fixed binary record layouts; it intentionally contains
 * no publication or recovery logic.
 */

#define PQC_LOGICAL_BLOCK_SIZE 4096U
#define PQC_WRITEBACK_COALESCE_SIZE (512U * 1024U)
#define PQC_WRITEBACK_MAX_BLOCKS \
    (((PQC_WRITEBACK_COALESCE_SIZE + (PQC_LOGICAL_BLOCK_SIZE - 1U)) / \
      PQC_LOGICAL_BLOCK_SIZE) + 1U)
#define PQC_WRITEBACK_BATCH_SCRATCH_SIZE \
    (PQC_WRITEBACK_MAX_BLOCKS * PQC_LOGICAL_BLOCK_SIZE)
#define PQC_READ_BATCH_MAX_BLOCKS 16U
#define PQC_READ_BATCH_SCRATCH_SIZE \
    (PQC_READ_BATCH_MAX_BLOCKS * PQC_LOGICAL_BLOCK_SIZE)
#define PQC_AEAD_TAG_SIZE 16U
#define PQC_AEAD_NONCE_SIZE 12U

#define PQC_JOURNAL_MAGIC UINT64_C(0x5051434a4e4c3031) /* PQCJNL01 */
#define PQC_JOURNAL_VERSION 1U
#define PQC_JOURNAL_COMMITTED UINT32_C(0x434f4d4d)

#define PQC_EPOCH_LOG_MAGIC UINT64_C(0x50514345504c3031) /* PQCEPL01 */
#define PQC_EPOCH_LOG_VERSION 1U
#define PQC_EPOCH_LOG_RECORD_BLOCK 1U
#define PQC_EPOCH_LOG_RECORD_COMMIT 2U
#define PQC_EPOCH_LOG_RECORD_SIZE 136U
#define PQC_EPOCH_LOG_RECORD_VERSION_OFFSET 8U
#define PQC_EPOCH_LOG_RECORD_DIGEST_OFFSET 104U

#define PQC_ALGO_AES_256_GCM 0u

#define PQC_TIER_FULL 1
#define PQC_TIER_NONE 2
#define PQC_XATTR_TIER "user.pqc_tier"
#define PQC_XATTR_QOS_CLASS "user.pqc_qos_class"
#define PQC_XATTR_METADATA "user.pqc_metadata"
#define PQC_XATTR_LOGICAL_SIZE "user.pqc_logical_size"
#define PQC_XATTR_CHECKPOINT "user.pqc_checkpoint"
#define PQC_QOS_CLASS_ELASTIC 0
#define PQC_QOS_CLASS_LATENCY 1

#define PQC_KDF_METADATA_FILENAME ".pqc_kdf"
#define PQC_KDF_METADATA_MAGIC UINT64_C(0x5051434b44463131) /* PQCKDF11 */
#define PQC_KDF_METADATA_VERSION 1U
#define PQC_KDF_ALG_PBKDF2_SHA256 1U
#define PQC_KDF_ALG_SCRYPT 2U
#define PQC_KDF_SALT_SIZE 32U

#define PQC_METADATA_MAGIC UINT64_C(0x5043514d45544131) /* "PQC META1" */
#define PQC_METADATA_VERSION 1U
#define PQC_CHECKPOINT_MAGIC UINT64_C(0x504351434b505431) /* "PQCCPT1" */
#define PQC_CHECKPOINT_VERSION 1U

#define PQC_PREFIX_ANCHOR_MAGIC 0x50524546u /* PREF */
#define PQC_PREFIX_ANCHOR_VERSION 3u
#define PQC_FRESHNESS_ANCHOR_MAGIC 0x46524553u /* FRES, legacy */
#define PQC_FRESHNESS_ANCHOR_VERSION 2u
#define PQC_TPM_NV_DEFAULT_INDEX 0x01500010u

typedef struct {
    uint64_t magic;
    uint32_t version;
    uint32_t algorithm;
    uint32_t salt_len;
    uint32_t reserved;
    uint64_t scrypt_n;
    uint32_t scrypt_r;
    uint32_t scrypt_p;
    uint64_t scrypt_maxmem;
    uint8_t salt[PQC_KDF_SALT_SIZE];
    uint8_t digest[32];
} pqc_kdf_metadata_t;

typedef struct {
    uint64_t magic;
    uint32_t version;
    uint32_t ss_len;
    uint64_t file_id;
    uint8_t wrapped_ss[64];
    uint8_t digest[32];
} pqc_metadata_t;

typedef struct {
    uint64_t magic;
    uint32_t version;
    uint32_t reserved;
    uint64_t file_id;
    uint64_t sequence;
    uint64_t logical_size;
    uint64_t max_generation;
    uint8_t digest[32];
} pqc_checkpoint_t;

typedef struct {
    uint64_t logical_block;
    uint64_t generation;
    uint64_t ciphertext_offset;
    uint32_t plaintext_length;
    uint32_t algorithm_id;
    uint8_t tag[PQC_AEAD_TAG_SIZE];
} block_mapping_t;

typedef struct {
    uint64_t magic;
    uint32_t version;
    uint32_t committed;
    block_mapping_t mapping;
    uint8_t digest[32];
} journal_record_t;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t global_sequence;
    uint64_t file_count;
    uint8_t prefix_root[32];
    uint8_t digest[32];
} pqc_prefix_anchor_t;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t epoch;
    uint64_t sequence;
    uint64_t logical_size;
    uint8_t digest[32];
} pqc_freshness_anchor_t;

#endif /* PQC_FORMAT_H */
