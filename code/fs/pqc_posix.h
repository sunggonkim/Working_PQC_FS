#ifndef PQC_POSIX_H
#define PQC_POSIX_H

#include <stddef.h>
#include <sys/types.h>

int pqc_sidecar_path(char *out, size_t out_size,
                     const char *path, const char *suffix);
int pqc_path_has_suffix(const char *path, const char *suffix);
int pqc_is_hidden_sidecar_path(const char *path);
int pqc_is_internal_xattr_name(const char *name);
ssize_t pqc_filter_xattr_list(const char *raw, size_t raw_len,
                              char *out, size_t out_size);

#endif /* PQC_POSIX_H */
