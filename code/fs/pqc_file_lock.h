#ifndef PQC_FILE_LOCK_H
#define PQC_FILE_LOCK_H

#ifndef FUSE_USE_VERSION
#define FUSE_USE_VERSION 31
#endif

#include <fuse.h>
#include <fcntl.h>

int pqc_lock(const char *path, struct fuse_file_info *fi,
             int cmd, struct flock *lock);
int pqc_flock(const char *path, struct fuse_file_info *fi, int op);

#endif /* PQC_FILE_LOCK_H */
