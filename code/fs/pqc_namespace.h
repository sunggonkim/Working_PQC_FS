#ifndef PQC_NAMESPACE_H
#define PQC_NAMESPACE_H

#ifndef FUSE_USE_VERSION
#define FUSE_USE_VERSION 31
#endif

#include <fuse.h>
#include <sys/stat.h>
#include <time.h>

int pqc_getattr(const char *path, struct stat *stbuf,
                struct fuse_file_info *fi);
int pqc_readdir(const char *path, void *buf, fuse_fill_dir_t filler,
                off_t offset, struct fuse_file_info *fi,
                enum fuse_readdir_flags flags);
int pqc_unlink(const char *path);
int pqc_mkdir(const char *path, mode_t mode);
int pqc_rmdir(const char *path);
int pqc_rename(const char *from, const char *to, unsigned int flags);
int pqc_link(const char *from, const char *to);
int pqc_readlink(const char *path, char *buf, size_t size);
int pqc_symlink(const char *from, const char *to);
int pqc_fsyncdir(const char *path, int datasync,
                 struct fuse_file_info *fi);
int pqc_utimens(const char *path, const struct timespec tv[2],
                struct fuse_file_info *fi);

#endif /* PQC_NAMESPACE_H */
