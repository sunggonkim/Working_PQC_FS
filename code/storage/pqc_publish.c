#include "pqc_publish.h"

#include "pqc_keyring.h"

#include <errno.h>
#include <stddef.h>
#include <sys/xattr.h>

#include <openssl/crypto.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <oqs/oqs.h>

int pqc_publish_logical_size_load(const char *path, uint64_t *size)
{
    if (!path || !size)
        return -EINVAL;

    uint64_t value = 0;
    ssize_t n = getxattr(path, PQC_XATTR_LOGICAL_SIZE, &value, sizeof(value));
    if (n == -1 && errno == ENODATA) {
        *size = 0;
        return 0;
    }
    if (n != (ssize_t)sizeof(value))
        return -errno;
    *size = value;
    return 0;
}

int pqc_publish_logical_size_store(const char *path, uint64_t size)
{
    if (!path)
        return -EINVAL;
    return setxattr(path, PQC_XATTR_LOGICAL_SIZE, &size, sizeof(size), 0) == 0
        ? 0
        : -errno;
}

int pqc_publish_checkpoint_store_xattr(const char *path, uint64_t file_id,
                                       uint64_t sequence,
                                       uint64_t logical_size,
                                       uint64_t max_generation)
{
    if (!path)
        return -EINVAL;

    pqc_checkpoint_t ckpt = {
        .magic = PQC_CHECKPOINT_MAGIC,
        .version = PQC_CHECKPOINT_VERSION,
        .reserved = 0,
        .file_id = file_id,
        .sequence = sequence,
        .logical_size = logical_size,
        .max_generation = max_generation,
    };
    unsigned int out_len = 0;
    unsigned char *mac = HMAC(EVP_sha256(), g_master_key, sizeof(g_master_key),
                              (unsigned char *)&ckpt,
                              offsetof(pqc_checkpoint_t, digest),
                              ckpt.digest, &out_len);
    if (!mac || out_len != sizeof(ckpt.digest)) {
        OQS_MEM_cleanse(&ckpt, sizeof(ckpt));
        return -EIO;
    }
    int rc = setxattr(path, PQC_XATTR_CHECKPOINT, &ckpt, sizeof(ckpt), 0);
    OQS_MEM_cleanse(&ckpt, sizeof(ckpt));
    return rc == -1 ? -errno : 0;
}

int pqc_publish_checkpoint_load_xattr(const char *path,
                                      uint64_t expected_file_id,
                                      pqc_checkpoint_t *out)
{
    if (!path || !out)
        return -EINVAL;

    pqc_checkpoint_t ckpt = {0};
    ssize_t n = getxattr(path, PQC_XATTR_CHECKPOINT, &ckpt, sizeof(ckpt));
    if (n == -1)
        return -errno;
    if ((size_t)n != sizeof(ckpt) || ckpt.magic != PQC_CHECKPOINT_MAGIC ||
        ckpt.version != PQC_CHECKPOINT_VERSION ||
        ckpt.file_id != expected_file_id) {
        OQS_MEM_cleanse(&ckpt, sizeof(ckpt));
        return -EINVAL;
    }

    uint8_t digest[32];
    unsigned int out_len = 0;
    unsigned char *mac = HMAC(EVP_sha256(), g_master_key, sizeof(g_master_key),
                              (unsigned char *)&ckpt,
                              offsetof(pqc_checkpoint_t, digest),
                              digest, &out_len);
    if (!mac || out_len != sizeof(digest) ||
        CRYPTO_memcmp(digest, ckpt.digest, sizeof(digest)) != 0) {
        OQS_MEM_cleanse(digest, sizeof(digest));
        OQS_MEM_cleanse(&ckpt, sizeof(ckpt));
        return -EKEYREJECTED;
    }

    *out = ckpt;
    OQS_MEM_cleanse(digest, sizeof(digest));
    return 0;
}
