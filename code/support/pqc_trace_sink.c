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
}

int pqc_trace_sink_write_env(pqc_trace_sink_t *sink,
                             const char *env_name,
                             const char *line,
                             size_t len)
{
    (void)sink;
    if (!env_name || !line || len == 0)
        return 0;

    const char *path = pqc_config_get_nonempty(env_name);
    if (!path)
        return 0;

    int write_fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC,
                        0600);
    if (write_fd < 0)
        return -(errno ? errno : EIO);
    int rc = trace_sink_write_all(write_fd, line, len);
    (void)close(write_fd);
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
