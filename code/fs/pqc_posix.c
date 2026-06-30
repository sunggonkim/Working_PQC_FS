#include "pqc_posix.h"

#include "pqc_config.h"
#include "pqc_format.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

int pqc_sidecar_path(char *out, size_t out_size,
                     const char *path, const char *suffix)
{
    int n = snprintf(out, out_size, "%s%s", path, suffix);
    return n < 0 || (size_t)n >= out_size ? -ENAMETOOLONG : 0;
}

int pqc_path_has_suffix(const char *path, const char *suffix)
{
    size_t path_len = strlen(path);
    size_t suffix_len = strlen(suffix);
    return path_len >= suffix_len &&
           strcmp(path + path_len - suffix_len, suffix) == 0;
}

int pqc_is_hidden_sidecar_path(const char *path)
{
    if (!path)
        return 0;
    if (path && (strcmp(path, "/" PQC_KDF_METADATA_FILENAME) == 0 ||
                 strcmp(path, PQC_KDF_METADATA_FILENAME) == 0))
        return 1;
    return pqc_path_has_suffix(path, ".pqcdata") ||
           pqc_path_has_suffix(path, ".pqcmeta") ||
           pqc_path_has_suffix(path, ".pqcepoch");
}

int pqc_is_sqlite_wal_sidecar_path(const char *path)
{
    size_t len = strlen(path);
    return (len >= 4 && strcmp(path + len - 4, "-wal") == 0) ||
           (len >= 4 && strcmp(path + len - 4, "-shm") == 0);
}

static uint64_t pqc_hash_path(const char *path)
{
    uint64_t h = 1469598103934665603ULL;
    for (const unsigned char *p = (const unsigned char *)path; *p; ++p) {
        h ^= (uint64_t)*p;
        h *= 1099511628211ULL;
    }
    return h;
}

int pqc_sqlite_sidecar_redirect_path(char *out, size_t out_size,
                                     const char *path)
{
    const char *compat = pqc_config_get_nonempty("PQC_ALLOW_SQLITE_MMAP");
    if (!compat)
        return -ENOENT;
    if (!pqc_is_sqlite_wal_sidecar_path(path))
        return -ENOENT;

    if (mkdir("/dev/shm/pqc_sqlite", 0700) != 0 && errno != EEXIST)
        return -errno;

    uint64_t h = pqc_hash_path(path);
    int n = snprintf(out, out_size, "/dev/shm/pqc_sqlite/%016llx%s",
                     (unsigned long long)h,
                     (strstr(path, "-shm") != NULL) ? ".shm" : ".wal");
    return n < 0 || (size_t)n >= out_size ? -ENAMETOOLONG : 0;
}

int pqc_is_internal_xattr_name(const char *name)
{
    if (!name)
        return 0;
    return strcmp(name, PQC_XATTR_METADATA) == 0 ||
           strcmp(name, PQC_XATTR_LOGICAL_SIZE) == 0 ||
           strcmp(name, PQC_XATTR_CHECKPOINT) == 0;
}

ssize_t pqc_filter_xattr_list(const char *raw, size_t raw_len,
                              char *out, size_t out_size)
{
    size_t used = 0;
    for (size_t pos = 0; pos < raw_len;) {
        size_t name_len = strnlen(raw + pos, raw_len - pos);
        if (pos + name_len >= raw_len)
            return -EIO;
        const char *name = raw + pos;
        size_t record_len = name_len + 1;
        if (!pqc_is_internal_xattr_name(name)) {
            if (out && used + record_len <= out_size)
                memmove(out + used, name, record_len);
            used += record_len;
        }
        pos += record_len;
    }
    if (out && used > out_size)
        return -ERANGE;
    return (ssize_t)used;
}
