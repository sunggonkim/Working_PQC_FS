#ifndef PQC_XATTR_H
#define PQC_XATTR_H

#include <stddef.h>

int pqc_setxattr(const char *path, const char *name,
                 const char *value, size_t size, int flags);
int pqc_getxattr(const char *path, const char *name,
                 char *value, size_t size);
int pqc_listxattr(const char *path, char *list, size_t size);

#endif /* PQC_XATTR_H */
