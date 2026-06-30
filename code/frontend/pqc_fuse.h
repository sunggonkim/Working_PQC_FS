#ifndef PQC_FUSE_H
#define PQC_FUSE_H

#ifndef FUSE_USE_VERSION
#define FUSE_USE_VERSION 31
#endif

#include <fuse.h>

const struct fuse_operations *pqc_fuse_operations(void);

#endif /* PQC_FUSE_H */
