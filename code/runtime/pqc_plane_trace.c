#include "pqc_plane_trace.h"

#include "pqc_config.h"

#include <errno.h>
#include <fcntl.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    atomic_ullong data_aes_gcm_encrypt_blocks;
    atomic_ullong data_aes_gcm_encrypt_bytes;
    atomic_ullong data_aes_gcm_decrypt_blocks;
    atomic_ullong data_aes_gcm_decrypt_bytes;
    atomic_ullong data_route_cpu_blocks;
    atomic_ullong data_route_gpu_blocks;
    atomic_ullong data_gpu_fallback_events;
    atomic_ullong keyplane_batches;
    atomic_ullong keyplane_candidate_files;
    atomic_ullong keyplane_refreshed_files;
    atomic_ullong keyplane_work_bytes;
    atomic_ullong keyplane_cpu_batches;
    atomic_ullong keyplane_gpu_batches;
    atomic_ullong keyplane_gpu_fallback_events;
    atomic_ullong keyplane_failed_batches;
    atomic_ullong freshness_anchor_events;
    atomic_ullong freshness_anchor_successes;
    atomic_ullong freshness_anchor_failures;
    atomic_ullong freshness_anchor_file_backend;
    atomic_ullong freshness_anchor_hardware_backend;
} pqc_plane_trace_state_t;

static pqc_plane_trace_state_t g_plane_trace;
static atomic_int g_plane_trace_enabled = ATOMIC_VAR_INIT(-1);
static char g_plane_trace_path[4096] = {0};

static int path_is_absolute(const char *path)
{
    return path && path[0] == '/';
}

static void plane_trace_set_path_locked(const char *raw)
{
    if (!raw || !*raw) {
        g_plane_trace_path[0] = '\0';
        return;
    }
    if (path_is_absolute(raw)) {
        snprintf(g_plane_trace_path, sizeof(g_plane_trace_path), "%s", raw);
        return;
    }
    char cwd[sizeof(g_plane_trace_path)];
    if (getcwd(cwd, sizeof(cwd))) {
        size_t cwd_len = strlen(cwd);
        size_t raw_len = strlen(raw);
        if (cwd_len + 1U + raw_len >= sizeof(g_plane_trace_path)) {
            g_plane_trace_path[0] = '\0';
            return;
        }
        memcpy(g_plane_trace_path, cwd, cwd_len);
        g_plane_trace_path[cwd_len] = '/';
        memcpy(g_plane_trace_path + cwd_len + 1U, raw, raw_len + 1U);
    } else {
        snprintf(g_plane_trace_path, sizeof(g_plane_trace_path), "%s", raw);
    }
}

static int plane_trace_configure_from_env(int force)
{
    for (;;) {
        int enabled = atomic_load_explicit(&g_plane_trace_enabled,
                                           memory_order_acquire);
        if (!force && enabled >= 0)
            return enabled;
        if (enabled == -2)
            continue;
        int expected = enabled;
        if (atomic_compare_exchange_strong_explicit(
                &g_plane_trace_enabled, &expected, -2,
                memory_order_acq_rel, memory_order_acquire))
            break;
    }

    const char *path = pqc_config_get_nonempty("PQC_PLANE_TRACE_PATH");
    int enabled = 0;
    if (path) {
        plane_trace_set_path_locked(path);
        enabled = g_plane_trace_path[0] != '\0' ? 1 : 0;
    } else {
        g_plane_trace_path[0] = '\0';
    }
    atomic_store_explicit(&g_plane_trace_enabled, enabled,
                          memory_order_release);
    return enabled;
}

int pqc_plane_trace_enabled(void)
{
    return plane_trace_configure_from_env(0);
}

static void trace_add(atomic_ullong *counter, uint64_t value)
{
    if (value == 0)
        return;
    atomic_fetch_add_explicit(counter, (unsigned long long)value,
                              memory_order_relaxed);
}

static uint64_t trace_load(atomic_ullong *counter)
{
    return (uint64_t)atomic_load_explicit(counter, memory_order_relaxed);
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

void pqc_plane_trace_record_data_encrypt(uint64_t blocks, uint64_t bytes,
                                         int gpu_used)
{
    if (!pqc_plane_trace_enabled())
        return;
    trace_add(&g_plane_trace.data_aes_gcm_encrypt_blocks, blocks);
    trace_add(&g_plane_trace.data_aes_gcm_encrypt_bytes, bytes);
    trace_add(gpu_used ? &g_plane_trace.data_route_gpu_blocks
                       : &g_plane_trace.data_route_cpu_blocks,
              blocks);
}

void pqc_plane_trace_record_data_decrypt(uint64_t blocks, uint64_t bytes,
                                         int gpu_used)
{
    if (!pqc_plane_trace_enabled())
        return;
    trace_add(&g_plane_trace.data_aes_gcm_decrypt_blocks, blocks);
    trace_add(&g_plane_trace.data_aes_gcm_decrypt_bytes, bytes);
    trace_add(gpu_used ? &g_plane_trace.data_route_gpu_blocks
                       : &g_plane_trace.data_route_cpu_blocks,
              blocks);
}

void pqc_plane_trace_record_data_gpu_fallback(void)
{
    if (!pqc_plane_trace_enabled())
        return;
    trace_add(&g_plane_trace.data_gpu_fallback_events, 1);
}

void pqc_plane_trace_record_keyplane_batch(uint64_t candidate_files,
                                           uint64_t refreshed_files,
                                           uint64_t work_bytes,
                                           int target_gpu,
                                           int gpu_used,
                                           int success)
{
    if (!pqc_plane_trace_enabled())
        return;
    trace_add(&g_plane_trace.keyplane_batches, 1);
    trace_add(&g_plane_trace.keyplane_candidate_files, candidate_files);
    trace_add(&g_plane_trace.keyplane_refreshed_files, refreshed_files);
    trace_add(&g_plane_trace.keyplane_work_bytes, work_bytes);
    trace_add(gpu_used ? &g_plane_trace.keyplane_gpu_batches
                       : &g_plane_trace.keyplane_cpu_batches,
              1);
    if (target_gpu && !gpu_used)
        trace_add(&g_plane_trace.keyplane_gpu_fallback_events, 1);
    if (!success)
        trace_add(&g_plane_trace.keyplane_failed_batches, 1);
}

void pqc_plane_trace_record_freshness_anchor(uint32_t backend, int rc)
{
    if (!pqc_plane_trace_enabled())
        return;
    trace_add(&g_plane_trace.freshness_anchor_events, 1);
    if (rc == 0)
        trace_add(&g_plane_trace.freshness_anchor_successes, 1);
    else
        trace_add(&g_plane_trace.freshness_anchor_failures, 1);
    if (backend == 1)
        trace_add(&g_plane_trace.freshness_anchor_file_backend, 1);
    else if (backend == 2)
        trace_add(&g_plane_trace.freshness_anchor_hardware_backend, 1);
}

void pqc_plane_trace_snapshot(pqc_plane_trace_snapshot_t *out)
{
    if (!out)
        return;
    out->data_aes_gcm_encrypt_blocks =
        trace_load(&g_plane_trace.data_aes_gcm_encrypt_blocks);
    out->data_aes_gcm_encrypt_bytes =
        trace_load(&g_plane_trace.data_aes_gcm_encrypt_bytes);
    out->data_aes_gcm_decrypt_blocks =
        trace_load(&g_plane_trace.data_aes_gcm_decrypt_blocks);
    out->data_aes_gcm_decrypt_bytes =
        trace_load(&g_plane_trace.data_aes_gcm_decrypt_bytes);
    out->data_route_cpu_blocks =
        trace_load(&g_plane_trace.data_route_cpu_blocks);
    out->data_route_gpu_blocks =
        trace_load(&g_plane_trace.data_route_gpu_blocks);
    out->data_gpu_fallback_events =
        trace_load(&g_plane_trace.data_gpu_fallback_events);
    out->keyplane_batches = trace_load(&g_plane_trace.keyplane_batches);
    out->keyplane_candidate_files =
        trace_load(&g_plane_trace.keyplane_candidate_files);
    out->keyplane_refreshed_files =
        trace_load(&g_plane_trace.keyplane_refreshed_files);
    out->keyplane_work_bytes =
        trace_load(&g_plane_trace.keyplane_work_bytes);
    out->keyplane_cpu_batches =
        trace_load(&g_plane_trace.keyplane_cpu_batches);
    out->keyplane_gpu_batches =
        trace_load(&g_plane_trace.keyplane_gpu_batches);
    out->keyplane_gpu_fallback_events =
        trace_load(&g_plane_trace.keyplane_gpu_fallback_events);
    out->keyplane_failed_batches =
        trace_load(&g_plane_trace.keyplane_failed_batches);
    out->freshness_anchor_events =
        trace_load(&g_plane_trace.freshness_anchor_events);
    out->freshness_anchor_successes =
        trace_load(&g_plane_trace.freshness_anchor_successes);
    out->freshness_anchor_failures =
        trace_load(&g_plane_trace.freshness_anchor_failures);
    out->freshness_anchor_file_backend =
        trace_load(&g_plane_trace.freshness_anchor_file_backend);
    out->freshness_anchor_hardware_backend =
        trace_load(&g_plane_trace.freshness_anchor_hardware_backend);
}

void pqc_plane_trace_reset(void)
{
    (void)plane_trace_configure_from_env(1);
    atomic_store_explicit(&g_plane_trace.data_aes_gcm_encrypt_blocks, 0,
                          memory_order_relaxed);
    atomic_store_explicit(&g_plane_trace.data_aes_gcm_encrypt_bytes, 0,
                          memory_order_relaxed);
    atomic_store_explicit(&g_plane_trace.data_aes_gcm_decrypt_blocks, 0,
                          memory_order_relaxed);
    atomic_store_explicit(&g_plane_trace.data_aes_gcm_decrypt_bytes, 0,
                          memory_order_relaxed);
    atomic_store_explicit(&g_plane_trace.data_route_cpu_blocks, 0,
                          memory_order_relaxed);
    atomic_store_explicit(&g_plane_trace.data_route_gpu_blocks, 0,
                          memory_order_relaxed);
    atomic_store_explicit(&g_plane_trace.data_gpu_fallback_events, 0,
                          memory_order_relaxed);
    atomic_store_explicit(&g_plane_trace.keyplane_batches, 0,
                          memory_order_relaxed);
    atomic_store_explicit(&g_plane_trace.keyplane_candidate_files, 0,
                          memory_order_relaxed);
    atomic_store_explicit(&g_plane_trace.keyplane_refreshed_files, 0,
                          memory_order_relaxed);
    atomic_store_explicit(&g_plane_trace.keyplane_work_bytes, 0,
                          memory_order_relaxed);
    atomic_store_explicit(&g_plane_trace.keyplane_cpu_batches, 0,
                          memory_order_relaxed);
    atomic_store_explicit(&g_plane_trace.keyplane_gpu_batches, 0,
                          memory_order_relaxed);
    atomic_store_explicit(&g_plane_trace.keyplane_gpu_fallback_events, 0,
                          memory_order_relaxed);
    atomic_store_explicit(&g_plane_trace.keyplane_failed_batches, 0,
                          memory_order_relaxed);
    atomic_store_explicit(&g_plane_trace.freshness_anchor_events, 0,
                          memory_order_relaxed);
    atomic_store_explicit(&g_plane_trace.freshness_anchor_successes, 0,
                          memory_order_relaxed);
    atomic_store_explicit(&g_plane_trace.freshness_anchor_failures, 0,
                          memory_order_relaxed);
    atomic_store_explicit(&g_plane_trace.freshness_anchor_file_backend, 0,
                          memory_order_relaxed);
    atomic_store_explicit(&g_plane_trace.freshness_anchor_hardware_backend, 0,
                          memory_order_relaxed);
}

int pqc_plane_trace_dump_if_requested(void)
{
    if (!plane_trace_configure_from_env(0))
        return 0;
    const char *path = g_plane_trace_path;
    if (!path || !*path)
        return 0;

    pqc_plane_trace_snapshot_t s;
    pqc_plane_trace_snapshot(&s);
    char line[4096];
    int n = snprintf(
        line, sizeof(line),
        "{\n"
        "  \"version\": 1,\n"
        "  \"data_plane_algorithm\": \"AES-256-GCM\",\n"
        "  \"key_plane_algorithm\": \"ML-KEM-768/Kyber768 envelope refresh\",\n"
        "  \"freshness_plane_algorithm\": \"committed-prefix anchor\",\n"
        "  \"data_aes_gcm_encrypt_blocks\": %llu,\n"
        "  \"data_aes_gcm_encrypt_bytes\": %llu,\n"
        "  \"data_aes_gcm_decrypt_blocks\": %llu,\n"
        "  \"data_aes_gcm_decrypt_bytes\": %llu,\n"
        "  \"data_route_cpu_blocks\": %llu,\n"
        "  \"data_route_gpu_blocks\": %llu,\n"
        "  \"data_gpu_fallback_events\": %llu,\n"
        "  \"keyplane_batches\": %llu,\n"
        "  \"keyplane_candidate_files\": %llu,\n"
        "  \"keyplane_refreshed_files\": %llu,\n"
        "  \"keyplane_work_bytes\": %llu,\n"
        "  \"keyplane_cpu_batches\": %llu,\n"
        "  \"keyplane_gpu_batches\": %llu,\n"
        "  \"keyplane_gpu_fallback_events\": %llu,\n"
        "  \"keyplane_failed_batches\": %llu,\n"
        "  \"freshness_anchor_events\": %llu,\n"
        "  \"freshness_anchor_successes\": %llu,\n"
        "  \"freshness_anchor_failures\": %llu,\n"
        "  \"freshness_anchor_file_backend\": %llu,\n"
        "  \"freshness_anchor_hardware_backend\": %llu\n"
        "}\n",
        (unsigned long long)s.data_aes_gcm_encrypt_blocks,
        (unsigned long long)s.data_aes_gcm_encrypt_bytes,
        (unsigned long long)s.data_aes_gcm_decrypt_blocks,
        (unsigned long long)s.data_aes_gcm_decrypt_bytes,
        (unsigned long long)s.data_route_cpu_blocks,
        (unsigned long long)s.data_route_gpu_blocks,
        (unsigned long long)s.data_gpu_fallback_events,
        (unsigned long long)s.keyplane_batches,
        (unsigned long long)s.keyplane_candidate_files,
        (unsigned long long)s.keyplane_refreshed_files,
        (unsigned long long)s.keyplane_work_bytes,
        (unsigned long long)s.keyplane_cpu_batches,
        (unsigned long long)s.keyplane_gpu_batches,
        (unsigned long long)s.keyplane_gpu_fallback_events,
        (unsigned long long)s.keyplane_failed_batches,
        (unsigned long long)s.freshness_anchor_events,
        (unsigned long long)s.freshness_anchor_successes,
        (unsigned long long)s.freshness_anchor_failures,
        (unsigned long long)s.freshness_anchor_file_backend,
        (unsigned long long)s.freshness_anchor_hardware_backend);
    if (n < 0 || (size_t)n >= sizeof(line))
        return -EOVERFLOW;

    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0)
        return -(errno ? errno : EIO);
    int rc = write_all(fd, line, (size_t)n);
    if (close(fd) != 0 && rc == 0)
        rc = -(errno ? errno : EIO);
    return rc;
}
