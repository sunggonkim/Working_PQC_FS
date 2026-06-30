#include "pqc_keyring.h"

#include "pqc_durability.h"
#include "pqc_format.h"
#include "pqc_storage_path.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/xattr.h>
#include <unistd.h>

#include <openssl/crypto.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/rand.h>
#include <oqs/oqs.h>

uint8_t g_master_key[32];
static int g_has_master_key = 0;
static const char *g_kdf_name = "unset";

#define PQC_LEGACY_PBKDF2_ITERATIONS 600000
#define PQC_SCRYPT_N UINT64_C(32768)
#define PQC_SCRYPT_R 8U
#define PQC_SCRYPT_P 1U
#define PQC_SCRYPT_MAXMEM UINT64_C(67108864)
#define PQC_KDF_METADATA_DIGEST_OFFSET \
    (8U + 4U + 4U + 4U + 4U + 8U + 4U + 4U + 8U + PQC_KDF_SALT_SIZE)
#define PQC_KDF_METADATA_ENCODED_SIZE \
    (PQC_KDF_METADATA_DIGEST_OFFSET + 32U)

static void store_le32(uint8_t out[4], uint32_t value)
{
    out[0] = (uint8_t)value;
    out[1] = (uint8_t)(value >> 8);
    out[2] = (uint8_t)(value >> 16);
    out[3] = (uint8_t)(value >> 24);
}

static void store_le64(uint8_t out[8], uint64_t value)
{
    for (size_t i = 0; i < 8U; ++i)
        out[i] = (uint8_t)(value >> (i * 8U));
}

static uint32_t load_le32(const uint8_t in[4])
{
    return (uint32_t)in[0] |
           ((uint32_t)in[1] << 8) |
           ((uint32_t)in[2] << 16) |
           ((uint32_t)in[3] << 24);
}

static uint64_t load_le64(const uint8_t in[8])
{
    uint64_t value = 0;
    for (size_t i = 0; i < 8U; ++i)
        value |= (uint64_t)in[i] << (i * 8U);
    return value;
}

static void encode_kdf_metadata_prefix(const pqc_kdf_metadata_t *meta,
                                       uint8_t out[PQC_KDF_METADATA_DIGEST_OFFSET])
{
    store_le64(out, meta->magic);
    store_le32(out + 8U, meta->version);
    store_le32(out + 12U, meta->algorithm);
    store_le32(out + 16U, meta->salt_len);
    store_le32(out + 20U, meta->reserved);
    store_le64(out + 24U, meta->scrypt_n);
    store_le32(out + 32U, meta->scrypt_r);
    store_le32(out + 36U, meta->scrypt_p);
    store_le64(out + 40U, meta->scrypt_maxmem);
    memcpy(out + 48U, meta->salt, PQC_KDF_SALT_SIZE);
}

static int encode_kdf_metadata(const pqc_kdf_metadata_t *meta,
                               uint8_t out[PQC_KDF_METADATA_ENCODED_SIZE])
{
    if (!meta || !out)
        return -EINVAL;
    memset(out, 0, PQC_KDF_METADATA_ENCODED_SIZE);
    encode_kdf_metadata_prefix(meta, out);
    memcpy(out + PQC_KDF_METADATA_DIGEST_OFFSET, meta->digest,
           sizeof(meta->digest));
    return 0;
}

static int decode_kdf_metadata(const uint8_t in[PQC_KDF_METADATA_ENCODED_SIZE],
                               size_t in_len,
                               pqc_kdf_metadata_t *meta)
{
    if (!in || !meta)
        return -EINVAL;
    if (in_len != PQC_KDF_METADATA_ENCODED_SIZE)
        return -EMSGSIZE;

    memset(meta, 0, sizeof(*meta));
    meta->magic = load_le64(in);
    meta->version = load_le32(in + 8U);
    meta->algorithm = load_le32(in + 12U);
    meta->salt_len = load_le32(in + 16U);
    meta->reserved = load_le32(in + 20U);
    meta->scrypt_n = load_le64(in + 24U);
    meta->scrypt_r = load_le32(in + 32U);
    meta->scrypt_p = load_le32(in + 36U);
    meta->scrypt_maxmem = load_le64(in + 40U);
    memcpy(meta->salt, in + 48U, sizeof(meta->salt));
    memcpy(meta->digest, in + PQC_KDF_METADATA_DIGEST_OFFSET,
           sizeof(meta->digest));
    return 0;
}

static int digest_kdf_metadata(const pqc_kdf_metadata_t *meta,
                               uint8_t digest[32])
{
    uint8_t encoded_prefix[PQC_KDF_METADATA_DIGEST_OFFSET];
    encode_kdf_metadata_prefix(meta, encoded_prefix);
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    if (!ctx) {
        OQS_MEM_cleanse(encoded_prefix, sizeof(encoded_prefix));
        return -ENOMEM;
    }
    int ok = EVP_DigestInit_ex(ctx, EVP_sha256(), NULL) == 1 &&
             EVP_DigestUpdate(ctx, encoded_prefix,
                              sizeof(encoded_prefix)) == 1 &&
             EVP_DigestFinal_ex(ctx, digest, NULL) == 1;
    EVP_MD_CTX_free(ctx);
    OQS_MEM_cleanse(encoded_prefix, sizeof(encoded_prefix));
    return ok ? 0 : -EIO;
}

static int validate_kdf_metadata(const pqc_kdf_metadata_t *meta)
{
    if (!meta)
        return -EINVAL;
    if (meta->magic != PQC_KDF_METADATA_MAGIC ||
        meta->version != PQC_KDF_METADATA_VERSION)
        return -EINVAL;
    if (meta->algorithm != PQC_KDF_ALG_SCRYPT)
        return -ENOTSUP;
    if (meta->salt_len == 0 || meta->salt_len > PQC_KDF_SALT_SIZE)
        return -EINVAL;
    if (meta->scrypt_n == 0 || meta->scrypt_r == 0 ||
        meta->scrypt_p == 0 || meta->scrypt_maxmem == 0)
        return -EINVAL;

    uint8_t digest[32];
    int rc = digest_kdf_metadata(meta, digest);
    if (rc != 0)
        return rc;
    int match = CRYPTO_memcmp(digest, meta->digest, sizeof(digest)) == 0;
    OQS_MEM_cleanse(digest, sizeof(digest));
    return match ? 0 : -EKEYREJECTED;
}

static int kdf_metadata_path(char *out, size_t out_size)
{
    const char *root = pqc_storage_path_root();
    if (!root || !*root)
        return -ENOENT;
    int n = snprintf(out, out_size, "%s/%s", root,
                     PQC_KDF_METADATA_FILENAME);
    return n < 0 || (size_t)n >= out_size ? -ENAMETOOLONG : 0;
}

static int storage_root_has_existing_payload(void)
{
    const char *root = pqc_storage_path_root();
    if (!root || !*root)
        return 0;

    DIR *dir = opendir(root);
    if (!dir)
        return -errno;
    int has_payload = 0;
    struct dirent *de;
    while ((de = readdir(dir)) != NULL) {
        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0 ||
            strcmp(de->d_name, PQC_KDF_METADATA_FILENAME) == 0)
            continue;
        has_payload = 1;
        break;
    }
    closedir(dir);
    return has_payload;
}

static int fsync_parent_dir(const char *path)
{
    if (!path)
        return -EINVAL;

    char dir_path[4096];
    const char *slash = strrchr(path, '/');
    if (!slash) {
        memcpy(dir_path, ".", 2);
    } else if (slash == path) {
        memcpy(dir_path, "/", 2);
    } else {
        size_t len = (size_t)(slash - path);
        if (len >= sizeof(dir_path))
            return -ENAMETOOLONG;
        memcpy(dir_path, path, len);
        dir_path[len] = '\0';
    }

    int fd = open(dir_path, O_RDONLY | O_DIRECTORY);
    if (fd < 0)
        return -errno;
    int rc = pqc_durability_fsync(
        fd, PQC_DURABILITY_SITE_KEYRING_METADATA);
    if (close(fd) != 0 && rc == 0)
        rc = -errno;
    return rc;
}

static int write_full(int fd, const void *buf, size_t len)
{
    const uint8_t *cursor = (const uint8_t *)buf;
    size_t remaining = len;
    while (remaining > 0) {
        ssize_t written = write(fd, cursor, remaining);
        if (written < 0) {
            if (errno == EINTR)
                continue;
            return -errno;
        }
        if (written == 0)
            return -EIO;
        cursor += (size_t)written;
        remaining -= (size_t)written;
    }
    return 0;
}

static int read_kdf_metadata(const char *path, pqc_kdf_metadata_t *meta)
{
    uint8_t encoded[PQC_KDF_METADATA_ENCODED_SIZE];
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return errno == ENOENT ? -ENOENT : -errno;

    ssize_t got = read(fd, encoded, sizeof(encoded));
    int saved_errno = errno;
    if (close(fd) != 0 && got == (ssize_t)sizeof(encoded))
        return -errno;
    if (got < 0)
        return -saved_errno;
    if (got != (ssize_t)sizeof(encoded))
        return -EIO;
    int rc = decode_kdf_metadata(encoded, sizeof(encoded), meta);
    OQS_MEM_cleanse(encoded, sizeof(encoded));
    if (rc != 0)
        return rc;
    return validate_kdf_metadata(meta);
}

static int write_kdf_metadata(const char *path, const pqc_kdf_metadata_t *meta)
{
    uint8_t encoded[PQC_KDF_METADATA_ENCODED_SIZE];
    int encode_rc = encode_kdf_metadata(meta, encoded);
    if (encode_rc != 0)
        return encode_rc;

    char tmp_path[4096];
    int n = snprintf(tmp_path, sizeof(tmp_path), "%s.tmp.XXXXXX", path);
    if (n < 0 || (size_t)n >= sizeof(tmp_path)) {
        OQS_MEM_cleanse(encoded, sizeof(encoded));
        return -ENAMETOOLONG;
    }

    int fd = mkstemp(tmp_path);
    if (fd < 0) {
        OQS_MEM_cleanse(encoded, sizeof(encoded));
        return -errno;
    }
    int rc = 0;
    int renamed = 0;
    if (fchmod(fd, 0600) != 0)
        rc = -errno;
    if (rc == 0)
        rc = write_full(fd, encoded, sizeof(encoded));
    if (rc == 0)
        rc = pqc_durability_fdatasync(
            fd, PQC_DURABILITY_SITE_KEYRING_METADATA);
    if (close(fd) != 0 && rc == 0)
        rc = -errno;
    if (rc == 0) {
        if (rename(tmp_path, path) != 0) {
            rc = -errno;
        } else {
            renamed = 1;
            rc = fsync_parent_dir(path);
        }
    }
    if (rc != 0 && !renamed)
        unlink(tmp_path);
    OQS_MEM_cleanse(encoded, sizeof(encoded));
    return rc;
}

static int create_kdf_metadata(const char *path, pqc_kdf_metadata_t *meta)
{
    memset(meta, 0, sizeof(*meta));
    meta->magic = PQC_KDF_METADATA_MAGIC;
    meta->version = PQC_KDF_METADATA_VERSION;
    meta->algorithm = PQC_KDF_ALG_SCRYPT;
    meta->salt_len = PQC_KDF_SALT_SIZE;
    meta->scrypt_n = PQC_SCRYPT_N;
    meta->scrypt_r = PQC_SCRYPT_R;
    meta->scrypt_p = PQC_SCRYPT_P;
    meta->scrypt_maxmem = PQC_SCRYPT_MAXMEM;
    if (RAND_bytes(meta->salt, sizeof(meta->salt)) != 1)
        return -EIO;
    int rc = digest_kdf_metadata(meta, meta->digest);
    if (rc != 0)
        return rc;
    rc = write_kdf_metadata(path, meta);
    if (rc != 0)
        OQS_MEM_cleanse(meta, sizeof(*meta));
    return rc;
}

static int derive_pbkdf2_legacy(const char *password)
{
    const uint8_t salt[] = "PQC_FUSE_SALT_NIST";
    if (PKCS5_PBKDF2_HMAC(password, strlen(password), salt,
                           sizeof(salt) - 1,
                           PQC_LEGACY_PBKDF2_ITERATIONS, EVP_sha256(), 32,
                           g_master_key) == 1) {
        g_has_master_key = 1;
        g_kdf_name = "PBKDF2-HMAC-SHA256-legacy";
        return 0;
    }
    return -EIO;
}

static int derive_scrypt_metadata(const char *password,
                                  const pqc_kdf_metadata_t *meta)
{
    if (EVP_PBE_scrypt(password, strlen(password), meta->salt,
                       meta->salt_len, meta->scrypt_n, meta->scrypt_r,
                       meta->scrypt_p, meta->scrypt_maxmem, g_master_key,
                       sizeof(g_master_key)) == 1) {
        g_has_master_key = 1;
        g_kdf_name = "scrypt";
        return 0;
    }
    return -EIO;
}

int pqc_keyring_derive_master_key(const char *password)
{
    if (!password)
        return -EINVAL;
    OQS_MEM_cleanse(g_master_key, sizeof(g_master_key));
    g_has_master_key = 0;
    g_kdf_name = "unset";

    pqc_kdf_metadata_t meta;
    char path[4096];
    int path_rc = kdf_metadata_path(path, sizeof(path));
    if (path_rc == -ENOENT) {
        memset(&meta, 0, sizeof(meta));
        meta.magic = PQC_KDF_METADATA_MAGIC;
        meta.version = PQC_KDF_METADATA_VERSION;
        meta.algorithm = PQC_KDF_ALG_SCRYPT;
        meta.salt_len = sizeof("PQC_SELFTEST_KDF_SALT") - 1;
        meta.scrypt_n = PQC_SCRYPT_N;
        meta.scrypt_r = PQC_SCRYPT_R;
        meta.scrypt_p = PQC_SCRYPT_P;
        meta.scrypt_maxmem = PQC_SCRYPT_MAXMEM;
        memcpy(meta.salt, "PQC_SELFTEST_KDF_SALT", meta.salt_len);
        int rc = derive_scrypt_metadata(password, &meta);
        OQS_MEM_cleanse(&meta, sizeof(meta));
        return rc;
    }
    if (path_rc != 0)
        return path_rc;

    int rc = read_kdf_metadata(path, &meta);
    if (rc == -ENOENT) {
        int has_payload = storage_root_has_existing_payload();
        if (has_payload < 0)
            return has_payload;
        if (has_payload > 0)
            return derive_pbkdf2_legacy(password);

        rc = create_kdf_metadata(path, &meta);
        if (rc != 0)
            return rc;
    }
    if (rc != 0)
        return rc;

    rc = derive_scrypt_metadata(password, &meta);
    OQS_MEM_cleanse(&meta, sizeof(meta));
    return rc;
}

const char *pqc_keyring_kdf_name(void)
{
    return g_kdf_name;
}

int pqc_keyring_kdf_self_test(void)
{
    char old_root[4096];
    snprintf(old_root, sizeof(old_root), "%s", pqc_storage_path_root());

    char tmp_root[] = "/tmp/pqc_kdf_selftest_XXXXXX";
    if (!mkdtemp(tmp_root))
        return -errno;

    int rc = 0;
    char metadata_path[4096] = {0};
    pqc_kdf_metadata_t meta;
    uint8_t first_key[sizeof(g_master_key)];
    uint8_t second_key[sizeof(g_master_key)];
    uint8_t different_key[sizeof(g_master_key)];

    memset(&meta, 0, sizeof(meta));
    memset(first_key, 0, sizeof(first_key));
    memset(second_key, 0, sizeof(second_key));
    memset(different_key, 0, sizeof(different_key));

    pqc_storage_path_set_root(tmp_root);
    rc = pqc_keyring_derive_master_key("kdf-self-test-password");
    if (rc != 0)
        goto out;
    memcpy(first_key, g_master_key, sizeof(first_key));
    if (strcmp(pqc_keyring_kdf_name(), "scrypt") != 0) {
        rc = -EINVAL;
        goto out;
    }

    rc = kdf_metadata_path(metadata_path, sizeof(metadata_path));
    if (rc != 0)
        goto out;
    rc = read_kdf_metadata(metadata_path, &meta);
    if (rc != 0)
        goto out;
    if (meta.algorithm != PQC_KDF_ALG_SCRYPT ||
        meta.salt_len != PQC_KDF_SALT_SIZE ||
        meta.scrypt_n != PQC_SCRYPT_N ||
        meta.scrypt_r != PQC_SCRYPT_R ||
        meta.scrypt_p != PQC_SCRYPT_P ||
        meta.scrypt_maxmem != PQC_SCRYPT_MAXMEM) {
        rc = -EINVAL;
        goto out;
    }

    pqc_keyring_clear_master_key();
    rc = pqc_keyring_derive_master_key("kdf-self-test-password");
    if (rc != 0)
        goto out;
    memcpy(second_key, g_master_key, sizeof(second_key));
    if (strcmp(pqc_keyring_kdf_name(), "scrypt") != 0 ||
        CRYPTO_memcmp(first_key, second_key, sizeof(first_key)) != 0) {
        rc = -EKEYREJECTED;
        goto out;
    }

    pqc_keyring_clear_master_key();
    rc = pqc_keyring_derive_master_key("kdf-self-test-different-password");
    if (rc != 0)
        goto out;
    memcpy(different_key, g_master_key, sizeof(different_key));
    if (CRYPTO_memcmp(first_key, different_key, sizeof(first_key)) == 0) {
        rc = -EKEYREJECTED;
        goto out;
    }

out:
    pqc_keyring_clear_master_key();
    OQS_MEM_cleanse(&meta, sizeof(meta));
    OQS_MEM_cleanse(first_key, sizeof(first_key));
    OQS_MEM_cleanse(second_key, sizeof(second_key));
    OQS_MEM_cleanse(different_key, sizeof(different_key));
    if (metadata_path[0] != '\0')
        (void)unlink(metadata_path);
    (void)rmdir(tmp_root);
    pqc_storage_path_set_root(old_root);
    return rc;
}

void pqc_keyring_clear_master_key(void)
{
    OQS_MEM_cleanse(g_master_key, sizeof(g_master_key));
    g_has_master_key = 0;
    g_kdf_name = "unset";
}

static int wrap_shared_secret(const uint8_t *in_ss, size_t ss_len,
                              uint64_t fid, uint64_t epoch,
                              uint8_t *out_wrapped)
{
    if (!g_has_master_key)
        return -EACCES;

    uint8_t seed[48];
    memcpy(seed, g_master_key, 32);
    memcpy(seed + 32, &fid, 8);
    memcpy(seed + 40, &epoch, 8);

    EVP_MD_CTX *md = EVP_MD_CTX_new();
    if (!md) {
        OQS_MEM_cleanse(seed, sizeof(seed));
        return -ENOMEM;
    }
    int ok = EVP_DigestInit_ex(md, EVP_shake128(), NULL) == 1 &&
             EVP_DigestUpdate(md, seed, sizeof(seed)) == 1 &&
             EVP_DigestFinalXOF(md, out_wrapped, ss_len) == 1;
    EVP_MD_CTX_free(md);
    OQS_MEM_cleanse(seed, sizeof(seed));
    if (!ok)
        return -EIO;

    for (size_t i = 0; i < ss_len; i++)
        out_wrapped[i] ^= in_ss[i];
    return 0;
}

static int unwrap_shared_secret(const uint8_t *in_wrapped, size_t ss_len,
                                uint64_t fid, uint64_t epoch, uint8_t *out_ss)
{
    return wrap_shared_secret(in_wrapped, ss_len, fid, epoch, out_ss);
}

int pqc_keyring_metadata_store(const char *phys_path, const uint8_t *ss,
                               size_t ss_len, uint64_t file_id)
{
    if (!phys_path || !ss)
        return -EINVAL;
    if (ss_len == 0 || ss_len > sizeof(((pqc_metadata_t *)0)->wrapped_ss))
        return -EINVAL;

    pqc_metadata_t meta = {0};
    meta.magic = PQC_METADATA_MAGIC;
    meta.version = PQC_METADATA_VERSION;
    meta.ss_len = (uint32_t)ss_len;
    meta.file_id = file_id;
    if (wrap_shared_secret(ss, ss_len, file_id, 0, meta.wrapped_ss) != 0)
        return -EIO;

    unsigned int digest_len = 0;
    if (!HMAC(EVP_sha256(), g_master_key, sizeof(g_master_key),
              (const unsigned char *)&meta, offsetof(pqc_metadata_t, digest),
              meta.digest, &digest_len) || digest_len != sizeof(meta.digest)) {
        OQS_MEM_cleanse(&meta, sizeof(meta));
        return -EIO;
    }
    int rc = setxattr(phys_path, PQC_XATTR_METADATA, &meta, sizeof(meta), 0);
    OQS_MEM_cleanse(&meta, sizeof(meta));
    return rc == -1 ? -errno : 0;
}

int pqc_keyring_metadata_load(const char *phys_path, uint8_t *ss,
                              size_t *ss_len, uint64_t *file_id)
{
    if (!phys_path || !ss || !ss_len || !file_id)
        return -EINVAL;

    pqc_metadata_t meta = {0};
    ssize_t n = getxattr(phys_path, PQC_XATTR_METADATA, &meta, sizeof(meta));
    if (n == -1)
        return -errno;
    if ((size_t)n != sizeof(meta) || meta.magic != PQC_METADATA_MAGIC ||
        meta.version != PQC_METADATA_VERSION || meta.ss_len == 0 ||
        meta.ss_len > sizeof(meta.wrapped_ss)) {
        OQS_MEM_cleanse(&meta, sizeof(meta));
        return -EINVAL;
    }

    uint8_t digest[sizeof(meta.digest)];
    unsigned int digest_len = 0;
    if (!HMAC(EVP_sha256(), g_master_key, sizeof(g_master_key),
              (const unsigned char *)&meta, offsetof(pqc_metadata_t, digest),
              digest, &digest_len) || digest_len != sizeof(digest) ||
        CRYPTO_memcmp(digest, meta.digest, sizeof(digest)) != 0) {
        OQS_MEM_cleanse(digest, sizeof(digest));
        OQS_MEM_cleanse(&meta, sizeof(meta));
        return -EKEYREJECTED;
    }
    OQS_MEM_cleanse(digest, sizeof(digest));

    int rc = unwrap_shared_secret(meta.wrapped_ss, meta.ss_len, meta.file_id,
                                  0, ss);
    if (rc == 0) {
        *ss_len = meta.ss_len;
        *file_id = meta.file_id;
    }
    OQS_MEM_cleanse(&meta, sizeof(meta));
    return rc == 0 ? 0 : -EIO;
}
