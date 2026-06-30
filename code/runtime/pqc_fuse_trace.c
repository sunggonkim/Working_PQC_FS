#include "pqc_fuse_trace.h"

#include "pqc_config.h"

#include <errno.h>
#include <fcntl.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    atomic_ullong calls;
    atomic_ullong errors;
    atomic_ullong total_ns;
    atomic_ullong max_ns;
} pqc_fuse_trace_counter_t;

static pqc_fuse_trace_counter_t g_fuse_trace[PQC_FUSE_TRACE_COUNT];

static uint64_t monotonic_ns(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        return 0;
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) +
           (uint64_t)ts.tv_nsec;
}

static uint64_t elapsed_ns(uint64_t end_ns, uint64_t start_ns)
{
    return end_ns >= start_ns ? end_ns - start_ns : 0;
}

static void atomic_max(atomic_ullong *target, uint64_t value)
{
    unsigned long long old =
        atomic_load_explicit(target, memory_order_relaxed);
    while (value > old &&
           !atomic_compare_exchange_weak_explicit(
               target, &old, (unsigned long long)value,
               memory_order_relaxed, memory_order_relaxed)) {
    }
}

static uint64_t load_counter(const atomic_ullong *counter)
{
    return (uint64_t)atomic_load_explicit((atomic_ullong *)counter,
                                         memory_order_relaxed);
}

uint64_t pqc_fuse_trace_begin(void)
{
    return monotonic_ns();
}

void pqc_fuse_trace_end(pqc_fuse_trace_op_t op, uint64_t start_ns, int rc)
{
    if (op < 0 || op >= PQC_FUSE_TRACE_COUNT)
        return;
    uint64_t duration_ns = elapsed_ns(monotonic_ns(), start_ns);
    pqc_fuse_trace_counter_t *counter = &g_fuse_trace[op];
    atomic_fetch_add_explicit(&counter->calls, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&counter->total_ns,
                              (unsigned long long)duration_ns,
                              memory_order_relaxed);
    atomic_max(&counter->max_ns, duration_ns);
    if (rc < 0)
        atomic_fetch_add_explicit(&counter->errors, 1,
                                  memory_order_relaxed);
}

void pqc_fuse_trace_reset(void)
{
    for (size_t i = 0; i < PQC_FUSE_TRACE_COUNT; ++i) {
        atomic_store_explicit(&g_fuse_trace[i].calls, 0,
                              memory_order_relaxed);
        atomic_store_explicit(&g_fuse_trace[i].errors, 0,
                              memory_order_relaxed);
        atomic_store_explicit(&g_fuse_trace[i].total_ns, 0,
                              memory_order_relaxed);
        atomic_store_explicit(&g_fuse_trace[i].max_ns, 0,
                              memory_order_relaxed);
    }
}

const char *pqc_fuse_trace_op_name(pqc_fuse_trace_op_t op)
{
    switch (op) {
    case PQC_FUSE_TRACE_GETATTR:
        return "getattr";
    case PQC_FUSE_TRACE_READDIR:
        return "readdir";
    case PQC_FUSE_TRACE_OPEN:
        return "open";
    case PQC_FUSE_TRACE_READ:
        return "read";
    case PQC_FUSE_TRACE_WRITE:
        return "write";
    case PQC_FUSE_TRACE_FLUSH:
        return "flush";
    case PQC_FUSE_TRACE_FSYNC:
        return "fsync";
    case PQC_FUSE_TRACE_CREATE:
        return "create";
    case PQC_FUSE_TRACE_TRUNCATE:
        return "truncate";
    case PQC_FUSE_TRACE_FALLOCATE:
        return "fallocate";
    case PQC_FUSE_TRACE_UNLINK:
        return "unlink";
    case PQC_FUSE_TRACE_MKDIR:
        return "mkdir";
    case PQC_FUSE_TRACE_RMDIR:
        return "rmdir";
    case PQC_FUSE_TRACE_RELEASE:
        return "release";
    case PQC_FUSE_TRACE_LOCK:
        return "lock";
    case PQC_FUSE_TRACE_FLOCK:
        return "flock";
    case PQC_FUSE_TRACE_RENAME:
        return "rename";
    case PQC_FUSE_TRACE_LINK:
        return "link";
    case PQC_FUSE_TRACE_READLINK:
        return "readlink";
    case PQC_FUSE_TRACE_SYMLINK:
        return "symlink";
    case PQC_FUSE_TRACE_FSYNCDIR:
        return "fsyncdir";
    case PQC_FUSE_TRACE_UTIMENS:
        return "utimens";
    case PQC_FUSE_TRACE_SETXATTR:
        return "setxattr";
    case PQC_FUSE_TRACE_GETXATTR:
        return "getxattr";
    case PQC_FUSE_TRACE_LISTXATTR:
        return "listxattr";
    case PQC_FUSE_TRACE_COUNT:
        break;
    }
    return "unknown";
}

static int write_all(int fd, const char *buf, size_t len)
{
    size_t off = 0;
    while (off < len) {
        ssize_t written = write(fd, buf + off, len - off);
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

int pqc_fuse_trace_dump_if_requested(void)
{
    const char *path = pqc_config_get_nonempty("PQC_FUSE_TRACE_PATH");
    if (!path)
        return 0;

    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0)
        return -(errno ? errno : EIO);

    char line[512];
    int n = snprintf(line, sizeof(line),
                     "{\n"
                     "  \"version\": 1,\n"
                     "  \"scope\": \"mounted FUSE operation latency counters\",\n"
                     "  \"operations\": [\n");
    if (n < 0 || (size_t)n >= sizeof(line) ||
        write_all(fd, line, (size_t)n) != 0) {
        close(fd);
        return -EIO;
    }

    for (size_t i = 0; i < PQC_FUSE_TRACE_COUNT; ++i) {
        const pqc_fuse_trace_counter_t *counter = &g_fuse_trace[i];
        uint64_t calls = load_counter(&counter->calls);
        uint64_t errors = load_counter(&counter->errors);
        uint64_t total_ns = load_counter(&counter->total_ns);
        uint64_t max_ns = load_counter(&counter->max_ns);
        n = snprintf(line, sizeof(line),
                     "    {\"op\":\"%s\",\"calls\":%llu,"
                     "\"errors\":%llu,\"total_ns\":%llu,"
                     "\"max_ns\":%llu}%s\n",
                     pqc_fuse_trace_op_name((pqc_fuse_trace_op_t)i),
                     (unsigned long long)calls,
                     (unsigned long long)errors,
                     (unsigned long long)total_ns,
                     (unsigned long long)max_ns,
                     i + 1 == PQC_FUSE_TRACE_COUNT ? "" : ",");
        if (n < 0 || (size_t)n >= sizeof(line) ||
            write_all(fd, line, (size_t)n) != 0) {
            close(fd);
            return -EIO;
        }
    }

    n = snprintf(line, sizeof(line), "  ]\n}\n");
    int rc = (n > 0 && (size_t)n < sizeof(line))
        ? write_all(fd, line, (size_t)n)
        : -EIO;
    if (close(fd) != 0 && rc == 0)
        rc = -(errno ? errno : EIO);
    return rc;
}
