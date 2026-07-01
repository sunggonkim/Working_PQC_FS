#include "pqc_namespace.h"

#include "pqc_fd_context.h"
#include "pqc_format.h"
#include "pqc_posix.h"
#include "pqc_publish.h"
#include "pqc_storage_path.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/xattr.h>
#include <unistd.h>

int pqc_getattr(const char *path, struct stat *stbuf,
                struct fuse_file_info *fi)
{
    (void)fi;
    if (pqc_is_hidden_sidecar_path(path))
        return -ENOENT;
    char phys_path[4096];
    pqc_storage_path_resolve(phys_path, sizeof(phys_path), path);

    int res = lstat(phys_path, stbuf);
    if (res == -1)
        return -errno;

    uint64_t logical_size = 0;
    if (S_ISREG(stbuf->st_mode) &&
        pqc_publish_logical_size_load(phys_path, &logical_size) == 0 &&
        getxattr(phys_path, PQC_XATTR_METADATA, NULL, 0) >= 0)
        stbuf->st_size = (off_t)logical_size;

    return 0;
}

int pqc_readdir(const char *path, void *buf, fuse_fill_dir_t filler,
                off_t offset, struct fuse_file_info *fi,
                enum fuse_readdir_flags flags)
{
    (void)offset;
    (void)fi;
    (void)flags;

    char phys_path[4096];
    pqc_storage_path_resolve(phys_path, sizeof(phys_path), path);

    DIR *dp = opendir(phys_path);
    if (!dp)
        return -errno;

    struct dirent *de;
    while ((de = readdir(dp)) != NULL) {
        size_t name_len = strlen(de->d_name);
        if (strcmp(de->d_name, PQC_KDF_METADATA_FILENAME) == 0)
            continue;
    if ((name_len >= 8 && strcmp(de->d_name + name_len - 8, ".pqcdata") == 0) ||
            (name_len >= 8 && strcmp(de->d_name + name_len - 8, ".pqcmeta") == 0) ||
            (name_len >= 9 && strcmp(de->d_name + name_len - 9, ".pqcepoch") == 0))
            continue;
        if (strncmp(de->d_name, ".fuse_hidden", strlen(".fuse_hidden")) == 0)
            continue;
        struct stat st;
        memset(&st, 0, sizeof(st));
        st.st_ino  = de->d_ino;
        st.st_mode = de->d_type << 12;

        if (filler(buf, de->d_name, &st, 0, 0))
            break;
    }

    closedir(dp);
    return 0;
}

int pqc_unlink(const char *path)
{
    if (pqc_is_hidden_sidecar_path(path))
        return -ENOENT;
    char phys_path[4096];
    pqc_storage_path_resolve(phys_path, sizeof(phys_path), path);
    char data_path[4096 + 16], journal_path[4096 + 16];
    char epoch_path[4096 + 16];
    int res = pqc_sidecar_path(data_path, sizeof(data_path), phys_path, ".pqcdata");
    if (res == 0)
        res = pqc_sidecar_path(journal_path, sizeof(journal_path), phys_path, ".pqcmeta");
    if (res == 0)
        res = pqc_sidecar_path(epoch_path, sizeof(epoch_path), phys_path, ".pqcepoch");
    if (res != 0)
        return res;

    if (unlink(data_path) == -1 && errno != ENOENT)
        return -errno;
    if (unlink(journal_path) == -1 && errno != ENOENT)
        return -errno;
    if (unlink(epoch_path) == -1 && errno != ENOENT)
        return -errno;
    if (unlink(phys_path) == -1)
        return -errno;
    return 0;
}

int pqc_mkdir(const char *path, mode_t mode)
{
    char phys_path[4096];
    pqc_storage_path_resolve(phys_path, sizeof(phys_path), path);

    int res = mkdir(phys_path, mode);
    if (res == -1)
        return -errno;

    return 0;
}

int pqc_rmdir(const char *path)
{
    char phys_path[4096];
    pqc_storage_path_resolve(phys_path, sizeof(phys_path), path);

    int res = rmdir(phys_path);
    if (res == -1)
        return -errno;

    return 0;
}

static int path_exists(const char *path)
{
    struct stat st;
    return lstat(path, &st) == 0;
}

static int path_is_regular(const char *path, int *exists_out)
{
    struct stat st;
    if (lstat(path, &st) != 0) {
        if (errno == ENOENT) {
            if (exists_out)
                *exists_out = 0;
            return 0;
        }
        return -errno;
    }
    if (exists_out)
        *exists_out = 1;
    return S_ISREG(st.st_mode) ? 0 : -ENOTSUP;
}

static int is_fuse_hidden_path(const char *path)
{
    if (!path)
        return 0;
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    return strncmp(base, ".fuse_hidden", strlen(".fuse_hidden")) == 0;
}

static int rename_optional_sidecar(const char *from_phys, const char *to_phys,
                                   const char *suffix)
{
    char from_path[4096 + 16];
    char to_path[4096 + 16];
    int rc = pqc_sidecar_path(from_path, sizeof(from_path), from_phys, suffix);
    if (rc != 0)
        return rc;
    rc = pqc_sidecar_path(to_path, sizeof(to_path), to_phys, suffix);
    if (rc != 0)
        return rc;
    if (!path_exists(from_path))
        return 0;
    if (path_exists(to_path))
        return -EEXIST;
    if (rename(from_path, to_path) != 0)
        return -errno;
    return 0;
}

static int link_optional_sidecar(const char *from_phys, const char *to_phys,
                                 const char *suffix)
{
    char from_path[4096 + 16];
    char to_path[4096 + 16];
    int rc = pqc_sidecar_path(from_path, sizeof(from_path), from_phys, suffix);
    if (rc != 0)
        return rc;
    rc = pqc_sidecar_path(to_path, sizeof(to_path), to_phys, suffix);
    if (rc != 0)
        return rc;
    if (!path_exists(from_path))
        return 0;
    if (path_exists(to_path))
        return -EEXIST;
    if (link(from_path, to_path) != 0)
        return -errno;
    return 0;
}

static void unlink_linked_sidecars(const char *phys_path)
{
    char sidecar_path[4096 + 16];
    if (pqc_sidecar_path(sidecar_path, sizeof(sidecar_path),
                         phys_path, ".pqcdata") == 0)
        (void)unlink(sidecar_path);
    if (pqc_sidecar_path(sidecar_path, sizeof(sidecar_path),
                         phys_path, ".pqcmeta") == 0)
        (void)unlink(sidecar_path);
    if (pqc_sidecar_path(sidecar_path, sizeof(sidecar_path),
                         phys_path, ".pqcepoch") == 0)
        (void)unlink(sidecar_path);
}

static int build_hidden_logical_path(char *out, size_t out_size,
                                     const char *logical_path)
{
    static unsigned long hidden_counter;
    if (!out || !logical_path)
        return -EINVAL;

    const char *slash = strrchr(logical_path, '/');
    size_t dir_len = slash ? (size_t)(slash - logical_path) : 0;
    unsigned long seq = __sync_add_and_fetch(&hidden_counter, 1);
    int written;
    if (dir_len == 0) {
        written = snprintf(out, out_size, "/.fuse_hidden_pqc_%ld_%lu",
                           (long)getpid(), seq);
    } else {
        written = snprintf(out, out_size, "%.*s/.fuse_hidden_pqc_%ld_%lu",
                           (int)dir_len, logical_path, (long)getpid(), seq);
    }
    if (written < 0 || (size_t)written >= out_size)
        return -ENAMETOOLONG;
    return 0;
}

static int move_optional_sidecar(const char *from_phys, const char *to_phys,
                                 const char *suffix)
{
    return rename_optional_sidecar(from_phys, to_phys, suffix);
}

static void rollback_moved_sidecars(const char *hidden_phys,
                                    const char *to_phys)
{
    (void)rename_optional_sidecar(hidden_phys, to_phys, ".pqcdata");
    (void)rename_optional_sidecar(hidden_phys, to_phys, ".pqcmeta");
    (void)rename_optional_sidecar(hidden_phys, to_phys, ".pqcepoch");
}

static int move_open_target_to_hidden(const char *to,
                                      const char *to_phys,
                                      char *hidden_logical,
                                      size_t hidden_logical_size,
                                      char *hidden_phys,
                                      size_t hidden_phys_size)
{
    int rc = build_hidden_logical_path(hidden_logical, hidden_logical_size, to);
    if (rc != 0)
        return rc;
    pqc_storage_path_resolve(hidden_phys, hidden_phys_size, hidden_logical);
    if (path_exists(hidden_phys))
        return -EEXIST;

    rc = move_optional_sidecar(to_phys, hidden_phys, ".pqcdata");
    if (rc == 0)
        rc = move_optional_sidecar(to_phys, hidden_phys, ".pqcmeta");
    if (rc == 0)
        rc = move_optional_sidecar(to_phys, hidden_phys, ".pqcepoch");
    if (rc == 0 && rename(to_phys, hidden_phys) != 0)
        rc = -errno;
    if (rc != 0) {
        rollback_moved_sidecars(hidden_phys, to_phys);
        (void)rename(hidden_phys, to_phys);
    }
    return rc;
}

int pqc_rename(const char *from, const char *to, unsigned int flags)
{
    if (flags != 0)
        return -ENOTSUP;
    if (pqc_is_hidden_sidecar_path(from) || pqc_is_hidden_sidecar_path(to))
        return -ENOENT;
    if (is_fuse_hidden_path(from))
        return -ENOTSUP;

    char from_phys[4096];
    char to_phys[4096];
    pqc_storage_path_resolve(from_phys, sizeof(from_phys), from);
    pqc_storage_path_resolve(to_phys, sizeof(to_phys), to);

    struct stat from_st;
    if (lstat(from_phys, &from_st) != 0)
        return errno == ENOENT ? -ENOENT : -errno;
    if (S_ISDIR(from_st.st_mode)) {
        struct stat to_st;
        if (lstat(to_phys, &to_st) == 0)
            return -ENOTSUP;
        if (errno != ENOENT)
            return -errno;
        if (pqc_fd_context_any_open())
            return -ENOTSUP;
        if (rename(from_phys, to_phys) != 0)
            return -errno;
        return 0;
    }
    if (!S_ISREG(from_st.st_mode))
        return -ENOTSUP;

    int to_exists = 0;
    int rc = path_is_regular(to_phys, &to_exists);
    if (rc != 0)
        return rc;
    int from_open = pqc_fd_context_path_is_open(from_phys) ||
        pqc_fd_context_logical_path_is_open(from);
    int to_open = pqc_fd_context_path_is_open(to_phys) ||
        pqc_fd_context_logical_path_is_open(to);
    int to_fuse_hidden = is_fuse_hidden_path(to);
    if (to_fuse_hidden && !from_open)
        return -ENOTSUP;
    if (pqc_fd_context_any_open() && !from_open && !to_open &&
        !pqc_fd_context_all_open_markers_hidden())
        return -ENOTSUP;

    char hidden_logical[4096];
    char hidden_phys[4096];
    hidden_logical[0] = '\0';
    hidden_phys[0] = '\0';
    if (to_exists && to_open) {
        rc = move_open_target_to_hidden(to, to_phys, hidden_logical,
                                        sizeof(hidden_logical), hidden_phys,
                                        sizeof(hidden_phys));
        if (rc != 0)
            return rc;
        rc = pqc_fd_context_rename_path(to_phys, hidden_phys);
        if (rc != 0)
            return rc;
        rc = pqc_fd_context_rename_logical_path(to, hidden_logical);
        if (rc != 0)
            return rc;
        to_exists = 0;
    }
    if (to_exists) {
        rc = pqc_unlink(to);
        if (rc != 0)
            return rc;
    }

    rc = rename_optional_sidecar(from_phys, to_phys, ".pqcdata");
    if (rc == 0)
        rc = rename_optional_sidecar(from_phys, to_phys, ".pqcmeta");
    if (rc == 0)
        rc = rename_optional_sidecar(from_phys, to_phys, ".pqcepoch");
    if (rc != 0)
        return rc;
    if (rename(from_phys, to_phys) != 0)
        return -errno;
    if (from_open) {
        rc = pqc_fd_context_rename_path(from_phys, to_phys);
        if (rc != 0)
            return rc;
        rc = pqc_fd_context_rename_logical_path(from, to);
        if (rc != 0)
            return rc;
    }
    return 0;
}

int pqc_link(const char *from, const char *to)
{
    if (!from || !to)
        return -EINVAL;
    if (pqc_is_hidden_sidecar_path(from) || pqc_is_hidden_sidecar_path(to))
        return -ENOENT;
    if (is_fuse_hidden_path(from) || is_fuse_hidden_path(to))
        return -ENOTSUP;
    if (pqc_fd_context_any_open())
        return -ENOTSUP;

    char from_phys[4096];
    char to_phys[4096];
    pqc_storage_path_resolve(from_phys, sizeof(from_phys), from);
    pqc_storage_path_resolve(to_phys, sizeof(to_phys), to);

    struct stat st;
    if (lstat(from_phys, &st) != 0)
        return errno == ENOENT ? -ENOENT : -errno;
    if (!S_ISREG(st.st_mode))
        return -ENOTSUP;
    if (path_exists(to_phys))
        return -EEXIST;

    int rc = link_optional_sidecar(from_phys, to_phys, ".pqcdata");
    if (rc == 0)
        rc = link_optional_sidecar(from_phys, to_phys, ".pqcmeta");
    if (rc == 0)
        rc = link_optional_sidecar(from_phys, to_phys, ".pqcepoch");
    if (rc == 0 && link(from_phys, to_phys) != 0)
        rc = -errno;
    if (rc != 0) {
        unlink_linked_sidecars(to_phys);
        (void)unlink(to_phys);
        return rc;
    }
    return 0;
}

int pqc_readlink(const char *path, char *buf, size_t size)
{
    if (!buf || size == 0)
        return -EINVAL;
    if (pqc_is_hidden_sidecar_path(path))
        return -ENOENT;

    char phys_path[4096];
    pqc_storage_path_resolve(phys_path, sizeof(phys_path), path);
    ssize_t n = readlink(phys_path, buf, size - 1);
    if (n < 0)
        return -errno;
    buf[n] = '\0';
    return 0;
}

static int symlink_target_is_supported(const char *target)
{
    if (!target || target[0] == '\0' || target[0] == '/')
        return 0;
    if (pqc_is_hidden_sidecar_path(target))
        return 0;

    const char *seg = target;
    while (*seg) {
        const char *end = strchr(seg, '/');
        size_t len = end ? (size_t)(end - seg) : strlen(seg);
        if (len == 2 && seg[0] == '.' && seg[1] == '.')
            return 0;
        seg = end ? end + 1 : seg + len;
    }
    return 1;
}

int pqc_symlink(const char *from, const char *to)
{
    if (!from || !to)
        return -EINVAL;
    if (!symlink_target_is_supported(from))
        return -ENOTSUP;
    if (pqc_is_hidden_sidecar_path(to))
        return -ENOENT;

    char to_phys[4096];
    pqc_storage_path_resolve(to_phys, sizeof(to_phys), to);
    if (path_exists(to_phys))
        return -EEXIST;

    if (symlink(from, to_phys) != 0)
        return -errno;
    return 0;
}

int pqc_fsyncdir(const char *path, int datasync,
                 struct fuse_file_info *fi)
{
    (void)datasync;
    (void)fi;

    char phys_path[4096];
    pqc_storage_path_resolve(phys_path, sizeof(phys_path), path);
    int fd = open(phys_path, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0)
        return -errno;
    int rc = fsync(fd);
    int saved_errno = errno;
    close(fd);
    if (rc != 0)
        return -saved_errno;
    return 0;
}

int pqc_utimens(const char *path, const struct timespec tv[2],
                struct fuse_file_info *fi)
{
    (void)fi;
    char phys_path[4096];
    pqc_storage_path_resolve(phys_path, sizeof(phys_path), path);

    int res = utimensat(AT_FDCWD, phys_path, tv, AT_SYMLINK_NOFOLLOW);
    if (res == -1)
        return -errno;

    return 0;
}
