#ifndef PQC_FILE_IO_H
#define PQC_FILE_IO_H

#ifndef FUSE_USE_VERSION
#define FUSE_USE_VERSION 31
#endif

#include <fuse.h>
#include <stddef.h>
#include <sys/stat.h>
#include <sys/types.h>

int pqc_open(const char *path, struct fuse_file_info *fi);
int pqc_read(const char *path, char *buf, size_t size, off_t offset,
             struct fuse_file_info *fi);
int pqc_create(const char *path, mode_t mode,
               struct fuse_file_info *fi);
int pqc_write(const char *path, const char *buf, size_t size,
              off_t offset, struct fuse_file_info *fi);
int pqc_fsync(const char *path, int datasync,
              struct fuse_file_info *fi);
int pqc_flush(const char *path, struct fuse_file_info *fi);
int pqc_truncate(const char *path, off_t size,
                 struct fuse_file_info *fi);
int pqc_fallocate(const char *path, int mode, off_t offset,
                  off_t length, struct fuse_file_info *fi);
int pqc_release(const char *path, struct fuse_file_info *fi);

#endif /* PQC_FILE_IO_H */
