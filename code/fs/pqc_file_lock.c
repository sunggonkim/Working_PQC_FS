#include "pqc_file_lock.h"

#include <errno.h>
#include <sys/file.h>

int pqc_lock(const char *path, struct fuse_file_info *fi,
             int cmd, struct flock *lock)
{
    (void)path;
    if (!fi || !lock)
        return -EINVAL;
    int fd = (int)fi->fh;
    if (fcntl(fd, cmd, lock) == -1)
        return -errno;
    return 0;
}

int pqc_flock(const char *path, struct fuse_file_info *fi, int op)
{
    (void)path;
    if (!fi)
        return -EINVAL;
    int fd = (int)fi->fh;
    if (flock(fd, op) == -1)
        return -errno;
    return 0;
}
