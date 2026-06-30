#include "pqc_xattr.h"

#include "pqc_admission.h"
#include "pqc_anchor.h"
#include "pqc_fd_context.h"
#include "pqc_format.h"
#include "pqc_lock_profile.h"
#include "pqc_metrics.h"
#include "pqc_posix.h"
#include "pqc_storage_path.h"

#include <errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/xattr.h>

#define PQC_XATTR_LIST_STACK_CAP (64u * 1024u)

int pqc_setxattr(const char *path, const char *name,
                 const char *value, size_t size, int flags)
{
    if (pqc_is_hidden_sidecar_path(path))
        return -ENOENT;
    char phys_path[4096];
    pqc_storage_path_resolve(phys_path, sizeof(phys_path), path);

    if (pqc_is_internal_xattr_name(name))
        return -EPERM;

    if (strcmp(name, "user.pqc.freshness_window") == 0 && size > 0) {
        char tmp[16] = {0};
        memcpy(tmp, value, size < 15 ? size : 15);
        int n = atoi(tmp);
        if (n > 0) {
            pqc_anchor_set_freshness_window(n);
            return 0;
        }
        return -EINVAL;
    }

    if (strcmp(name, PQC_XATTR_TIER) == 0 && size > 0) {
        char tmp[8] = {0};
        memcpy(tmp, value, size < 7 ? size : 7);
        int new_tier = atoi(tmp);
        if (new_tier != PQC_TIER_FULL && new_tier != PQC_TIER_NONE)
            return -EINVAL;
        if (setxattr(phys_path, name, value, size, flags) == -1)
            return -errno;
        for (size_t i = 0; i < pqc_fd_context_capacity(); i++) {
            pqc_fd_ctx_t *ctx = pqc_fd_context_at_index(i);
            if (!ctx)
                continue;
            pqc_lock_profile_scope_t fd_scope;
            (void)pqc_profiled_mutex_lock(&ctx->fd_lock, "fd_lock",
                                          __func__, &fd_scope);
            if (ctx->valid && strcmp(ctx->marker_path, phys_path) == 0)
                ctx->tier = new_tier;
            (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock",
                                            __func__, &fd_scope);
        }
        pqc_log("SETXATTR %s tier=%d", path, new_tier);
        return 0;
    }

    if (strcmp(name, PQC_XATTR_QOS_CLASS) == 0 && size > 0) {
        int qos_class = PQC_QOS_CLASS_ELASTIC;
        int rc = pqc_parse_qos_class_value(value, size, &qos_class);
        if (rc != 0)
            return rc;
        if (setxattr(phys_path, name, value, size, flags) == -1)
            return -errno;
        for (size_t i = 0; i < pqc_fd_context_capacity(); i++) {
            pqc_fd_ctx_t *ctx = pqc_fd_context_at_index(i);
            if (!ctx)
                continue;
            pqc_lock_profile_scope_t fd_scope;
            (void)pqc_profiled_mutex_lock(&ctx->fd_lock, "fd_lock",
                                          __func__, &fd_scope);
            if (ctx->valid && strcmp(ctx->marker_path, phys_path) == 0)
                ctx->qos_class = qos_class;
            (void)pqc_profiled_mutex_unlock(&ctx->fd_lock, "fd_lock",
                                            __func__, &fd_scope);
        }
        pqc_log("SETXATTR %s qos_class=%s", path, pqc_qos_class_name(qos_class));
        return 0;
    }

    if (setxattr(phys_path, name, value, size, flags) == -1)
        return -errno;
    return 0;
}

int pqc_getxattr(const char *path, const char *name,
                 char *value, size_t size)
{
    if (pqc_is_hidden_sidecar_path(path))
        return -ENOENT;
    char phys_path[4096];
    pqc_storage_path_resolve(phys_path, sizeof(phys_path), path);

    if (pqc_is_internal_xattr_name(name))
        return -ENODATA;

    ssize_t res = getxattr(phys_path, name, value, size);
    return res == -1 ? -errno : (int)res;
}

int pqc_listxattr(const char *path, char *list, size_t size)
{
    if (pqc_is_hidden_sidecar_path(path))
        return -ENOENT;
    char phys_path[4096];
    pqc_storage_path_resolve(phys_path, sizeof(phys_path), path);

    ssize_t raw_len = listxattr(phys_path, NULL, 0);
    if (raw_len <= 0)
        return raw_len == -1 ? -errno : 0;
    if ((size_t)raw_len > PQC_XATTR_LIST_STACK_CAP)
        return -ERANGE;
    char raw[PQC_XATTR_LIST_STACK_CAP];
    ssize_t got = listxattr(phys_path, raw, (size_t)raw_len);
    if (got == -1)
        return -errno;
    ssize_t filtered = pqc_filter_xattr_list(raw, (size_t)got, list, size);
    return filtered < 0 ? (int)filtered : (int)filtered;
}
