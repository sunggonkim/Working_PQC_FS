#include "pqc_durability.h"

#include "pqc_config.h"
#include "pqc_metrics.h"

#include <errno.h>
#include <fcntl.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

typedef enum {
    PQC_DURABILITY_OP_FDATASYNC = 0,
    PQC_DURABILITY_OP_FSYNC,
    PQC_DURABILITY_OP_SYNCFS,
    PQC_DURABILITY_OP_COUNT
} pqc_durability_op_t;

typedef struct {
    atomic_ullong calls[PQC_DURABILITY_OP_COUNT];
    atomic_ullong failures[PQC_DURABILITY_OP_COUNT];
    atomic_ullong total_ns[PQC_DURABILITY_OP_COUNT];
    atomic_ullong site_calls[PQC_DURABILITY_SITE_COUNT];
    atomic_ullong site_failures[PQC_DURABILITY_SITE_COUNT];
} pqc_durability_state_t;

static pqc_durability_state_t g_durability;
static pqc_durability_state_t g_mounted_durability;
static atomic_int g_mounted_operations_active = ATOMIC_VAR_INIT(0);
static atomic_int g_timing_enabled = ATOMIC_VAR_INIT(-1);

static int durability_timing_enabled(void)
{
    int cached = atomic_load_explicit(&g_timing_enabled,
                                      memory_order_relaxed);
    if (cached >= 0)
        return cached;
    int enabled = pqc_config_enabled("PQC_DURABILITY_TIMING");
    int expected = -1;
    if (atomic_compare_exchange_strong_explicit(
            &g_timing_enabled, &expected, enabled,
            memory_order_relaxed, memory_order_relaxed))
        return enabled;
    return atomic_load_explicit(&g_timing_enabled, memory_order_relaxed);
}

static uint64_t monotonic_ns(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        return 0;
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static uint64_t elapsed_ns(uint64_t end_ns, uint64_t start_ns)
{
    return end_ns >= start_ns ? end_ns - start_ns : 0;
}

static pqc_durability_site_t normalize_site(pqc_durability_site_t site)
{
    if (site < 0 || site >= PQC_DURABILITY_SITE_COUNT)
        return PQC_DURABILITY_SITE_OTHER;
    return site;
}

static void counter_add(atomic_ullong *counter, uint64_t value)
{
    if (value == 0)
        return;
    atomic_fetch_add_explicit(counter, (unsigned long long)value,
                              memory_order_relaxed);
}

static uint64_t counter_load(atomic_ullong *counter)
{
    return (uint64_t)atomic_load_explicit(counter, memory_order_relaxed);
}

static void state_record(pqc_durability_state_t *state,
                         pqc_durability_op_t op,
                         pqc_durability_site_t site,
                         int rc, uint64_t duration_ns)
{
    counter_add(&state->calls[op], 1);
    counter_add(&state->total_ns[op], duration_ns);
    counter_add(&state->site_calls[site], 1);
    if (rc != 0) {
        counter_add(&state->failures[op], 1);
        counter_add(&state->site_failures[site], 1);
    }
}

static int record_result(pqc_durability_op_t op,
                         pqc_durability_site_t site,
                         int rc, int saved_errno,
                         uint64_t start_ns, uint64_t end_ns,
                         int accounting_enabled)
{
    if (accounting_enabled) {
        site = normalize_site(site);
        uint64_t duration_ns = elapsed_ns(end_ns, start_ns);
        state_record(&g_durability, op, site, rc, duration_ns);
        if (atomic_load_explicit(&g_mounted_operations_active,
                                 memory_order_relaxed)) {
            state_record(&g_mounted_durability, op, site, rc, duration_ns);
        }
    }
    if (rc == 0)
        return 0;

    return -(saved_errno ? saved_errno : EIO);
}

const char *pqc_durability_site_name(pqc_durability_site_t site)
{
    switch (normalize_site(site)) {
    case PQC_DURABILITY_SITE_USER_FILE:
        return "user_file";
    case PQC_DURABILITY_SITE_DATA_SIDECAR:
        return "data_sidecar";
    case PQC_DURABILITY_SITE_JOURNAL_SIDECAR:
        return "journal_sidecar";
    case PQC_DURABILITY_SITE_EPOCH_LOG:
        return "epoch_log";
    case PQC_DURABILITY_SITE_ANCHOR_FILE:
        return "anchor_file";
    case PQC_DURABILITY_SITE_MARKER_METADATA:
        return "marker_metadata";
    case PQC_DURABILITY_SITE_KEYRING_METADATA:
        return "keyring_metadata";
    case PQC_DURABILITY_SITE_PARENT_DIR:
        return "parent_dir";
    case PQC_DURABILITY_SITE_OTHER:
    case PQC_DURABILITY_SITE_COUNT:
        return "other";
    }
    return "other";
}

int pqc_durability_fdatasync(int fd, pqc_durability_site_t site)
{
    int accounting = durability_timing_enabled();
    uint64_t start_ns = accounting ? monotonic_ns() : 0;
    int rc = fdatasync(fd);
    int saved_errno = errno;
    uint64_t end_ns = accounting ? monotonic_ns() : 0;
    return record_result(PQC_DURABILITY_OP_FDATASYNC, site, rc,
                         saved_errno, start_ns, end_ns, accounting);
}

int pqc_durability_fsync(int fd, pqc_durability_site_t site)
{
    int accounting = durability_timing_enabled();
    uint64_t start_ns = accounting ? monotonic_ns() : 0;
    int rc = fsync(fd);
    int saved_errno = errno;
    uint64_t end_ns = accounting ? monotonic_ns() : 0;
    return record_result(PQC_DURABILITY_OP_FSYNC, site, rc,
                         saved_errno, start_ns, end_ns, accounting);
}

int pqc_durability_syncfs(int fd, pqc_durability_site_t site)
{
    int rc = -1;
    int saved_errno = ENOTSUP;
    int accounting = durability_timing_enabled();
    uint64_t start_ns = accounting ? monotonic_ns() : 0;
#ifdef SYS_syncfs
    rc = (int)syscall(SYS_syncfs, fd);
    saved_errno = errno;
#endif
    uint64_t end_ns = accounting ? monotonic_ns() : 0;
    return record_result(PQC_DURABILITY_OP_SYNCFS, site, rc,
                         saved_errno, start_ns, end_ns, accounting);
}

void pqc_durability_init_from_config(void)
{
    (void)durability_timing_enabled();
}

void pqc_durability_begin_mounted_operations(void)
{
    if (!durability_timing_enabled()) {
        atomic_store_explicit(&g_mounted_operations_active, 0,
                              memory_order_relaxed);
        return;
    }
    memset(&g_mounted_durability, 0, sizeof(g_mounted_durability));
    atomic_store_explicit(&g_mounted_operations_active, 1,
                          memory_order_relaxed);
}

void pqc_durability_end_mounted_operations(void)
{
    atomic_store_explicit(&g_mounted_operations_active, 0,
                          memory_order_relaxed);
}

static void stats_snapshot_from_state(pqc_durability_state_t *state,
                                      pqc_durability_stats_t *out)
{
    if (!out)
        return;
    memset(out, 0, sizeof(*out));
    out->fdatasync_calls =
        counter_load(&state->calls[PQC_DURABILITY_OP_FDATASYNC]);
    out->fdatasync_failures =
        counter_load(&state->failures[PQC_DURABILITY_OP_FDATASYNC]);
    out->fdatasync_total_ns =
        counter_load(&state->total_ns[PQC_DURABILITY_OP_FDATASYNC]);
    out->fsync_calls =
        counter_load(&state->calls[PQC_DURABILITY_OP_FSYNC]);
    out->fsync_failures =
        counter_load(&state->failures[PQC_DURABILITY_OP_FSYNC]);
    out->fsync_total_ns =
        counter_load(&state->total_ns[PQC_DURABILITY_OP_FSYNC]);
    out->syncfs_calls =
        counter_load(&state->calls[PQC_DURABILITY_OP_SYNCFS]);
    out->syncfs_failures =
        counter_load(&state->failures[PQC_DURABILITY_OP_SYNCFS]);
    out->syncfs_total_ns =
        counter_load(&state->total_ns[PQC_DURABILITY_OP_SYNCFS]);
    for (size_t i = 0; i < PQC_DURABILITY_SITE_COUNT; ++i) {
        out->site_calls[i] = counter_load(&state->site_calls[i]);
        out->site_failures[i] =
            counter_load(&state->site_failures[i]);
    }
}

void pqc_durability_stats_snapshot(pqc_durability_stats_t *out)
{
    stats_snapshot_from_state(&g_durability, out);
}

void pqc_durability_mounted_stats_snapshot(pqc_durability_stats_t *out)
{
    stats_snapshot_from_state(&g_mounted_durability, out);
}

void pqc_durability_stats_reset(void)
{
    for (size_t i = 0; i < PQC_DURABILITY_OP_COUNT; ++i) {
        atomic_store_explicit(&g_durability.calls[i], 0,
                              memory_order_relaxed);
        atomic_store_explicit(&g_durability.failures[i], 0,
                              memory_order_relaxed);
        atomic_store_explicit(&g_durability.total_ns[i], 0,
                              memory_order_relaxed);
    }
    for (size_t i = 0; i < PQC_DURABILITY_SITE_COUNT; ++i) {
        atomic_store_explicit(&g_durability.site_calls[i], 0,
                              memory_order_relaxed);
        atomic_store_explicit(&g_durability.site_failures[i], 0,
                              memory_order_relaxed);
    }
    memset(&g_mounted_durability, 0, sizeof(g_mounted_durability));
    atomic_store_explicit(&g_mounted_operations_active, 0,
                          memory_order_relaxed);
    atomic_store_explicit(&g_timing_enabled, -1, memory_order_relaxed);
}

void pqc_durability_log_summary(void)
{
    pqc_durability_stats_t s;
    pqc_durability_stats_t mounted;
    pqc_durability_stats_snapshot(&s);
    pqc_durability_mounted_stats_snapshot(&mounted);
    pqc_log("Durability sync stats: fdatasync=%llu failures=%llu total_ns=%llu "
            "fsync=%llu failures=%llu total_ns=%llu syncfs=%llu failures=%llu "
            "total_ns=%llu user_file=%llu data_sidecar=%llu journal_sidecar=%llu "
            "epoch_log=%llu anchor_file=%llu marker_metadata=%llu "
            "keyring_metadata=%llu parent_dir=%llu other=%llu",
            (unsigned long long)s.fdatasync_calls,
            (unsigned long long)s.fdatasync_failures,
            (unsigned long long)s.fdatasync_total_ns,
            (unsigned long long)s.fsync_calls,
            (unsigned long long)s.fsync_failures,
            (unsigned long long)s.fsync_total_ns,
            (unsigned long long)s.syncfs_calls,
            (unsigned long long)s.syncfs_failures,
            (unsigned long long)s.syncfs_total_ns,
            (unsigned long long)s.site_calls[PQC_DURABILITY_SITE_USER_FILE],
            (unsigned long long)s.site_calls[PQC_DURABILITY_SITE_DATA_SIDECAR],
            (unsigned long long)
                s.site_calls[PQC_DURABILITY_SITE_JOURNAL_SIDECAR],
            (unsigned long long)s.site_calls[PQC_DURABILITY_SITE_EPOCH_LOG],
            (unsigned long long)s.site_calls[PQC_DURABILITY_SITE_ANCHOR_FILE],
            (unsigned long long)
                s.site_calls[PQC_DURABILITY_SITE_MARKER_METADATA],
            (unsigned long long)
                s.site_calls[PQC_DURABILITY_SITE_KEYRING_METADATA],
            (unsigned long long)s.site_calls[PQC_DURABILITY_SITE_PARENT_DIR],
            (unsigned long long)s.site_calls[PQC_DURABILITY_SITE_OTHER]);
    pqc_log("Durability mounted-operation sync stats: fdatasync=%llu failures=%llu total_ns=%llu "
            "fsync=%llu failures=%llu total_ns=%llu syncfs=%llu failures=%llu "
            "total_ns=%llu user_file=%llu data_sidecar=%llu journal_sidecar=%llu "
            "epoch_log=%llu anchor_file=%llu marker_metadata=%llu "
            "keyring_metadata=%llu parent_dir=%llu other=%llu",
            (unsigned long long)mounted.fdatasync_calls,
            (unsigned long long)mounted.fdatasync_failures,
            (unsigned long long)mounted.fdatasync_total_ns,
            (unsigned long long)mounted.fsync_calls,
            (unsigned long long)mounted.fsync_failures,
            (unsigned long long)mounted.fsync_total_ns,
            (unsigned long long)mounted.syncfs_calls,
            (unsigned long long)mounted.syncfs_failures,
            (unsigned long long)mounted.syncfs_total_ns,
            (unsigned long long)
                mounted.site_calls[PQC_DURABILITY_SITE_USER_FILE],
            (unsigned long long)
                mounted.site_calls[PQC_DURABILITY_SITE_DATA_SIDECAR],
            (unsigned long long)
                mounted.site_calls[PQC_DURABILITY_SITE_JOURNAL_SIDECAR],
            (unsigned long long)
                mounted.site_calls[PQC_DURABILITY_SITE_EPOCH_LOG],
            (unsigned long long)
                mounted.site_calls[PQC_DURABILITY_SITE_ANCHOR_FILE],
            (unsigned long long)
                mounted.site_calls[PQC_DURABILITY_SITE_MARKER_METADATA],
            (unsigned long long)
                mounted.site_calls[PQC_DURABILITY_SITE_KEYRING_METADATA],
            (unsigned long long)
                mounted.site_calls[PQC_DURABILITY_SITE_PARENT_DIR],
            (unsigned long long)
                mounted.site_calls[PQC_DURABILITY_SITE_OTHER]);
}

int pqc_durability_self_test(void)
{
    char path[] = "/tmp/pqc_durability_selftest.XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0)
        return -1;

    int rc = 0;
    const char *old_timing = getenv("PQC_DURABILITY_TIMING");
    int had_old_timing = old_timing != NULL;
    int changed_timing_env = 0;
    char *old_timing_copy = old_timing ? strdup(old_timing) : NULL;
    if (old_timing && !old_timing_copy)
        rc = -1;
    if (rc == 0) {
        if (setenv("PQC_DURABILITY_TIMING", "1", 1) != 0)
            rc = -1;
        else
            changed_timing_env = 1;
    }
    pqc_durability_stats_reset();
    if (write(fd, "x", 1) != 1)
        rc = -1;
    if (rc == 0 &&
        pqc_durability_fdatasync(fd,
                                 PQC_DURABILITY_SITE_DATA_SIDECAR) != 0)
        rc = -1;
    if (rc == 0 &&
        pqc_durability_fsync(fd, PQC_DURABILITY_SITE_USER_FILE) != 0)
        rc = -1;

    pqc_durability_stats_t s;
    pqc_durability_stats_snapshot(&s);
    if (rc == 0 &&
        (s.fdatasync_calls != 1 || s.fsync_calls != 1 ||
         s.site_calls[PQC_DURABILITY_SITE_DATA_SIDECAR] != 1 ||
         s.site_calls[PQC_DURABILITY_SITE_USER_FILE] != 1))
        rc = -1;
    pqc_durability_mounted_stats_snapshot(&s);
    if (rc == 0 && (s.fdatasync_calls != 0 || s.fsync_calls != 0))
        rc = -1;

    pqc_durability_begin_mounted_operations();
    if (rc == 0 &&
        pqc_durability_fdatasync(fd,
                                 PQC_DURABILITY_SITE_JOURNAL_SIDECAR) != 0)
        rc = -1;
    pqc_durability_end_mounted_operations();
    pqc_durability_mounted_stats_snapshot(&s);
    if (rc == 0 &&
        (s.fdatasync_calls != 1 ||
         s.site_calls[PQC_DURABILITY_SITE_JOURNAL_SIDECAR] != 1))
        rc = -1;

    if (close(fd) != 0 && rc == 0)
        rc = -1;
    unlink(path);
    pqc_durability_stats_reset();
    if (changed_timing_env) {
        if (had_old_timing && old_timing_copy)
            (void)setenv("PQC_DURABILITY_TIMING", old_timing_copy, 1);
        else
            (void)unsetenv("PQC_DURABILITY_TIMING");
    }
    free(old_timing_copy);
    return rc;
}
