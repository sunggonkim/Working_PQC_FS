#ifndef PQC_FUSE_TRACE_H
#define PQC_FUSE_TRACE_H

#include <stdint.h>

typedef enum {
    PQC_FUSE_TRACE_GETATTR = 0,
    PQC_FUSE_TRACE_READDIR,
    PQC_FUSE_TRACE_OPEN,
    PQC_FUSE_TRACE_READ,
    PQC_FUSE_TRACE_WRITE,
    PQC_FUSE_TRACE_FLUSH,
    PQC_FUSE_TRACE_FSYNC,
    PQC_FUSE_TRACE_CREATE,
    PQC_FUSE_TRACE_TRUNCATE,
    PQC_FUSE_TRACE_FALLOCATE,
    PQC_FUSE_TRACE_UNLINK,
    PQC_FUSE_TRACE_MKDIR,
    PQC_FUSE_TRACE_RMDIR,
    PQC_FUSE_TRACE_RELEASE,
    PQC_FUSE_TRACE_LOCK,
    PQC_FUSE_TRACE_FLOCK,
    PQC_FUSE_TRACE_RENAME,
    PQC_FUSE_TRACE_LINK,
    PQC_FUSE_TRACE_READLINK,
    PQC_FUSE_TRACE_SYMLINK,
    PQC_FUSE_TRACE_FSYNCDIR,
    PQC_FUSE_TRACE_UTIMENS,
    PQC_FUSE_TRACE_SETXATTR,
    PQC_FUSE_TRACE_GETXATTR,
    PQC_FUSE_TRACE_LISTXATTR,
    PQC_FUSE_TRACE_COUNT
} pqc_fuse_trace_op_t;

uint64_t pqc_fuse_trace_begin(void);
void pqc_fuse_trace_end(pqc_fuse_trace_op_t op, uint64_t start_ns, int rc);
int pqc_fuse_trace_is_enabled(void);
void pqc_fuse_trace_reset(void);
int pqc_fuse_trace_dump_if_requested(void);
const char *pqc_fuse_trace_op_name(pqc_fuse_trace_op_t op);

#endif /* PQC_FUSE_TRACE_H */
