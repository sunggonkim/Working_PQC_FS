/**
 * ============================================================================
 *  pqc_fuse.c — PQC-based FUSE Filesystem for Edge AI Bottleneck Profiling
 * ============================================================================
 *
 *  Purpose:
 *    Transparent FUSE prototype for authenticated encrypted block storage.
 *    Every file receives a random data-encryption key (DEK), stored in an
 *    HMAC-authenticated envelope under a mount-derived key.  AES-256-GCM
 *    protects data records; ciphertext is synchronized before journal
 *    publication.  ML-KEM-768 is initialized for optional key-plane batch
 *    work and microbenchmarks, not for per-block encryption.
 *
 *  Architecture:
 *    mnt_secure/ (FUSE mount) ──write()──► AES-GCM + journal ──► storage_physical/
 *
 *  Build:
 *    mkdir build && cd build && cmake .. && make
 *
 *  Usage:
 *    ./pqc_fuse  -f  <mountpoint>
 *    (the storage_physical directory is set via -o storage_dir=<path> or defaults
 *     to ../storage_physical relative to the mountpoint)
 *
 *  Author : PQC Edge Research Team
 *  License: MIT
 * ============================================================================
 */

#define FUSE_USE_VERSION 31

#include <fuse.h>

#include "pqc_fuse.h"
#include "pqc_file_io.h"
#include "pqc_file_lock.h"
#include "pqc_fuse_trace.h"
#include "pqc_lifecycle.h"
#include "pqc_namespace.h"
#include "pqc_xattr.h"

/* ════════════════════════════════════════════════════════════════════════════
 *  FUSE Operations
 * ════════════════════════════════════════════════════════════════════════════ */

static int trace_getattr(const char *path, struct stat *stbuf,
                         struct fuse_file_info *fi)
{
    uint64_t t0 = pqc_fuse_trace_begin();
    int rc = pqc_getattr(path, stbuf, fi);
    pqc_fuse_trace_end(PQC_FUSE_TRACE_GETATTR, t0, rc);
    return rc;
}

static int trace_readdir(const char *path, void *buf, fuse_fill_dir_t filler,
                         off_t offset, struct fuse_file_info *fi,
                         enum fuse_readdir_flags flags)
{
    uint64_t t0 = pqc_fuse_trace_begin();
    int rc = pqc_readdir(path, buf, filler, offset, fi, flags);
    pqc_fuse_trace_end(PQC_FUSE_TRACE_READDIR, t0, rc);
    return rc;
}

static int trace_open(const char *path, struct fuse_file_info *fi)
{
    uint64_t t0 = pqc_fuse_trace_begin();
    int rc = pqc_open(path, fi);
    pqc_fuse_trace_end(PQC_FUSE_TRACE_OPEN, t0, rc);
    return rc;
}

static int trace_read(const char *path, char *buf, size_t size, off_t offset,
                      struct fuse_file_info *fi)
{
    uint64_t t0 = pqc_fuse_trace_begin();
    int rc = pqc_read(path, buf, size, offset, fi);
    pqc_fuse_trace_end(PQC_FUSE_TRACE_READ, t0, rc);
    return rc;
}

static int trace_write(const char *path, const char *buf, size_t size,
                       off_t offset, struct fuse_file_info *fi)
{
    uint64_t t0 = pqc_fuse_trace_begin();
    int rc = pqc_write(path, buf, size, offset, fi);
    pqc_fuse_trace_end(PQC_FUSE_TRACE_WRITE, t0, rc);
    return rc;
}

static int trace_flush(const char *path, struct fuse_file_info *fi)
{
    uint64_t t0 = pqc_fuse_trace_begin();
    int rc = pqc_flush(path, fi);
    pqc_fuse_trace_end(PQC_FUSE_TRACE_FLUSH, t0, rc);
    return rc;
}

static int trace_fsync(const char *path, int datasync,
                       struct fuse_file_info *fi)
{
    uint64_t t0 = pqc_fuse_trace_begin();
    int rc = pqc_fsync(path, datasync, fi);
    pqc_fuse_trace_end(PQC_FUSE_TRACE_FSYNC, t0, rc);
    return rc;
}

static int trace_create(const char *path, mode_t mode,
                        struct fuse_file_info *fi)
{
    uint64_t t0 = pqc_fuse_trace_begin();
    int rc = pqc_create(path, mode, fi);
    pqc_fuse_trace_end(PQC_FUSE_TRACE_CREATE, t0, rc);
    return rc;
}

static int trace_truncate(const char *path, off_t size,
                          struct fuse_file_info *fi)
{
    uint64_t t0 = pqc_fuse_trace_begin();
    int rc = pqc_truncate(path, size, fi);
    pqc_fuse_trace_end(PQC_FUSE_TRACE_TRUNCATE, t0, rc);
    return rc;
}

static int trace_fallocate(const char *path, int mode, off_t offset,
                           off_t length, struct fuse_file_info *fi)
{
    uint64_t t0 = pqc_fuse_trace_begin();
    int rc = pqc_fallocate(path, mode, offset, length, fi);
    pqc_fuse_trace_end(PQC_FUSE_TRACE_FALLOCATE, t0, rc);
    return rc;
}

static int trace_unlink(const char *path)
{
    uint64_t t0 = pqc_fuse_trace_begin();
    int rc = pqc_unlink(path);
    pqc_fuse_trace_end(PQC_FUSE_TRACE_UNLINK, t0, rc);
    return rc;
}

static int trace_mkdir(const char *path, mode_t mode)
{
    uint64_t t0 = pqc_fuse_trace_begin();
    int rc = pqc_mkdir(path, mode);
    pqc_fuse_trace_end(PQC_FUSE_TRACE_MKDIR, t0, rc);
    return rc;
}

static int trace_rmdir(const char *path)
{
    uint64_t t0 = pqc_fuse_trace_begin();
    int rc = pqc_rmdir(path);
    pqc_fuse_trace_end(PQC_FUSE_TRACE_RMDIR, t0, rc);
    return rc;
}

static int trace_release(const char *path, struct fuse_file_info *fi)
{
    uint64_t t0 = pqc_fuse_trace_begin();
    int rc = pqc_release(path, fi);
    pqc_fuse_trace_end(PQC_FUSE_TRACE_RELEASE, t0, rc);
    return rc;
}

static int trace_lock(const char *path, struct fuse_file_info *fi,
                      int cmd, struct flock *lock)
{
    uint64_t t0 = pqc_fuse_trace_begin();
    int rc = pqc_lock(path, fi, cmd, lock);
    pqc_fuse_trace_end(PQC_FUSE_TRACE_LOCK, t0, rc);
    return rc;
}

static int trace_flock(const char *path, struct fuse_file_info *fi, int op)
{
    uint64_t t0 = pqc_fuse_trace_begin();
    int rc = pqc_flock(path, fi, op);
    pqc_fuse_trace_end(PQC_FUSE_TRACE_FLOCK, t0, rc);
    return rc;
}

static int trace_rename(const char *from, const char *to, unsigned int flags)
{
    uint64_t t0 = pqc_fuse_trace_begin();
    int rc = pqc_rename(from, to, flags);
    pqc_fuse_trace_end(PQC_FUSE_TRACE_RENAME, t0, rc);
    return rc;
}

static int trace_link(const char *from, const char *to)
{
    uint64_t t0 = pqc_fuse_trace_begin();
    int rc = pqc_link(from, to);
    pqc_fuse_trace_end(PQC_FUSE_TRACE_LINK, t0, rc);
    return rc;
}

static int trace_readlink(const char *path, char *buf, size_t size)
{
    uint64_t t0 = pqc_fuse_trace_begin();
    int rc = pqc_readlink(path, buf, size);
    pqc_fuse_trace_end(PQC_FUSE_TRACE_READLINK, t0, rc);
    return rc;
}

static int trace_symlink(const char *from, const char *to)
{
    uint64_t t0 = pqc_fuse_trace_begin();
    int rc = pqc_symlink(from, to);
    pqc_fuse_trace_end(PQC_FUSE_TRACE_SYMLINK, t0, rc);
    return rc;
}

static int trace_fsyncdir(const char *path, int datasync,
                          struct fuse_file_info *fi)
{
    uint64_t t0 = pqc_fuse_trace_begin();
    int rc = pqc_fsyncdir(path, datasync, fi);
    pqc_fuse_trace_end(PQC_FUSE_TRACE_FSYNCDIR, t0, rc);
    return rc;
}

static int trace_utimens(const char *path, const struct timespec tv[2],
                         struct fuse_file_info *fi)
{
    uint64_t t0 = pqc_fuse_trace_begin();
    int rc = pqc_utimens(path, tv, fi);
    pqc_fuse_trace_end(PQC_FUSE_TRACE_UTIMENS, t0, rc);
    return rc;
}

static int trace_setxattr(const char *path, const char *name,
                          const char *value, size_t size, int flags)
{
    uint64_t t0 = pqc_fuse_trace_begin();
    int rc = pqc_setxattr(path, name, value, size, flags);
    pqc_fuse_trace_end(PQC_FUSE_TRACE_SETXATTR, t0, rc);
    return rc;
}

static int trace_getxattr(const char *path, const char *name,
                          char *value, size_t size)
{
    uint64_t t0 = pqc_fuse_trace_begin();
    int rc = pqc_getxattr(path, name, value, size);
    pqc_fuse_trace_end(PQC_FUSE_TRACE_GETXATTR, t0, rc);
    return rc;
}

static int trace_listxattr(const char *path, char *list, size_t size)
{
    uint64_t t0 = pqc_fuse_trace_begin();
    int rc = pqc_listxattr(path, list, size);
    pqc_fuse_trace_end(PQC_FUSE_TRACE_LISTXATTR, t0, rc);
    return rc;
}

/* ── FUSE operations table ── */
static const struct fuse_operations pqc_oper = {
    .init       = pqc_fuse_init,
    .getattr    = trace_getattr,
    .readdir    = trace_readdir,
    .open       = trace_open,
    .read       = trace_read,
    .write      = trace_write,
    .flush      = trace_flush,
    .fsync      = trace_fsync,
    .create     = trace_create,
    .truncate   = trace_truncate,
    .fallocate  = trace_fallocate,
    .unlink     = trace_unlink,
    .mkdir      = trace_mkdir,
    .rmdir      = trace_rmdir,
    .release    = trace_release,
    .lock       = trace_lock,
    .flock      = trace_flock,
    .rename     = trace_rename,
    .link       = trace_link,
    .readlink   = trace_readlink,
    .symlink    = trace_symlink,
    .fsyncdir   = trace_fsyncdir,
    .utimens    = trace_utimens,
    .destroy    = pqc_destroy,
    .setxattr   = trace_setxattr,
    .getxattr   = trace_getxattr,
    .listxattr  = trace_listxattr,
};

const struct fuse_operations *pqc_fuse_operations(void)
{
    return &pqc_oper;
}
