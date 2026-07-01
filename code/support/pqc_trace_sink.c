#include "pqc_trace_sink.h"

#include "pqc_config.h"
#include "pqc_lock_profile.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static int trace_sink_lock(pqc_trace_sink_t *sink,
                           pqc_lock_profile_scope_t *scope,
                           const char *site)
{
    return pqc_profiled_mutex_lock(&sink->lock, "trace_sink_lock",
                                   site, scope);
}

static int trace_sink_unlock(pqc_trace_sink_t *sink,
                             pqc_lock_profile_scope_t *scope,
                             const char *site)
{
    return pqc_profiled_mutex_unlock(&sink->lock, "trace_sink_lock",
                                     site, scope);
}

static int trace_sink_write_all(int fd, const char *line, size_t len)
{
    size_t off = 0;
    while (off < len) {
        ssize_t written = write(fd, line + off, len - off);
        if (written < 0) {
            if (errno == EINTR)
                continue;
            return -(errno ? errno : EIO);
        }
        if (written == 0)
            return -EIO;
        off += (size_t)written;
    }
    return 0;
}

static void trace_sink_close_locked(pqc_trace_sink_t *sink)
{
    if (!sink)
        return;
    if (sink->fd >= 0) {
        (void)close(sink->fd);
        sink->fd = -1;
    }
    sink->path[0] = '\0';
    atomic_store_explicit(&sink->enabled, 0, memory_order_release);
    atomic_store_explicit(&sink->configured, 0, memory_order_release);
}

static int trace_sink_configure_locked(pqc_trace_sink_t *sink,
                                       const char *env_name)
{
    if (!sink || !env_name)
        return -EINVAL;
    if (atomic_load_explicit(&sink->configured, memory_order_acquire))
        return atomic_load_explicit(&sink->enabled, memory_order_acquire) ?
            1 : 0;

    const char *path = pqc_config_get_nonempty(env_name);
    if (!path) {
        sink->path[0] = '\0';
        sink->fd = -1;
        atomic_store_explicit(&sink->enabled, 0, memory_order_release);
        atomic_store_explicit(&sink->configured, 1, memory_order_release);
        return 0;
    }

    int n = snprintf(sink->path, sizeof(sink->path), "%s", path);
    if (n < 0 || (size_t)n >= sizeof(sink->path)) {
        sink->path[0] = '\0';
        sink->fd = -1;
        atomic_store_explicit(&sink->enabled, 0, memory_order_release);
        atomic_store_explicit(&sink->configured, 1, memory_order_release);
        return -ENAMETOOLONG;
    }

    int fd = open(sink->path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    if (fd < 0) {
        sink->fd = -1;
        atomic_store_explicit(&sink->enabled, 0, memory_order_release);
        atomic_store_explicit(&sink->configured, 1, memory_order_release);
        return -(errno ? errno : EIO);
    }
    sink->fd = fd;
    atomic_store_explicit(&sink->enabled, 1, memory_order_release);
    atomic_store_explicit(&sink->configured, 1, memory_order_release);
    return 1;
}

int pqc_trace_sink_enabled_env(pqc_trace_sink_t *sink,
                               const char *env_name)
{
    if (!sink || !env_name)
        return 0;
    if (atomic_load_explicit(&sink->configured, memory_order_acquire))
        return atomic_load_explicit(&sink->enabled, memory_order_acquire) != 0;

    pqc_lock_profile_scope_t scope;
    if (trace_sink_lock(sink, &scope, "trace_sink_enabled") != 0)
        return 0;
    int rc = trace_sink_configure_locked(sink, env_name);
    (void)trace_sink_unlock(sink, &scope, "trace_sink_enabled");
    return rc > 0;
}

int pqc_trace_sink_write_env(pqc_trace_sink_t *sink,
                             const char *env_name,
                             const char *line,
                             size_t len)
{
    if (!sink || !env_name || !line || len == 0)
        return 0;

    if (!pqc_trace_sink_enabled_env(sink, env_name))
        return 0;

    pqc_lock_profile_scope_t scope;
    if (trace_sink_lock(sink, &scope, "trace_sink_write") != 0)
        return 0;
    int rc = 0;
    if (sink->fd < 0) {
        rc = -EIO;
    } else {
        rc = trace_sink_write_all(sink->fd, line, len);
        if (rc != 0)
            trace_sink_close_locked(sink);
    }
    (void)trace_sink_unlock(sink, &scope, "trace_sink_write");
    return rc;
}

void pqc_trace_sink_close(pqc_trace_sink_t *sink)
{
    if (!sink)
        return;
    pqc_lock_profile_scope_t scope;
    if (trace_sink_lock(sink, &scope, "trace_sink_close") != 0)
        return;
    trace_sink_close_locked(sink);
    (void)trace_sink_unlock(sink, &scope, "trace_sink_close");
}
