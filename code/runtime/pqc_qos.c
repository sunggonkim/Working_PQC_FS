#include "pqc_qos.h"

#include "pqc_admission.h"
#include "pqc_config.h"
#include "pqc_format.h"
#include "pqc_lock_profile.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static pthread_mutex_t g_gpu_load_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_t g_gpu_load_thread;
static int g_gpu_load_thread_started = 0;
static int g_gpu_load_stop = 0;
static double g_gpu_load_ewma = 0.0;

static pthread_t g_admission_telemetry_thread;
static int g_admission_telemetry_thread_started = 0;
static int g_admission_telemetry_stop = 0;
static char g_admission_telemetry_path[4096] = {0};

static pthread_mutex_t g_qos_throttle_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_qos_throttle_state = 0;
static double g_qos_pressure_value = 0.0;
static unsigned g_qos_below_exit_count = 0;

static int qos_gpu_load_lock(pqc_lock_profile_scope_t *scope,
                             const char *site)
{
    return pqc_profiled_mutex_lock(&g_gpu_load_lock, "qos_gpu_load_lock",
                                   site, scope);
}

static int qos_gpu_load_unlock(pqc_lock_profile_scope_t *scope,
                               const char *site)
{
    return pqc_profiled_mutex_unlock(&g_gpu_load_lock, "qos_gpu_load_lock",
                                     site, scope);
}

static int qos_throttle_lock(pqc_lock_profile_scope_t *scope,
                             const char *site)
{
    return pqc_profiled_mutex_lock(&g_qos_throttle_lock, "qos_throttle_lock",
                                   site, scope);
}

static int qos_throttle_unlock(pqc_lock_profile_scope_t *scope,
                               const char *site)
{
    return pqc_profiled_mutex_unlock(&g_qos_throttle_lock,
                                     "qos_throttle_lock", site, scope);
}

static const char *gpu_load_path(void)
{
    return pqc_config_nonempty_or_default("PQC_GPU_LOAD_PATH",
                                          "/sys/devices/gpu.0/load");
}

static int qos_runtime_throttle_enabled(void)
{
    return pqc_config_enabled("PQC_ENABLE_QOS_THROTTLE_ON_WRITE");
}

static void qos_trace_emit_line(const char *path, const char *line)
{
    if (!path || !line)
        return;

    size_t len = strlen(line);
    if (len == 0)
        return;

    int fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    if (fd < 0)
        return;

    size_t off = 0;
    while (off < len) {
        ssize_t written = write(fd, line + off, len - off);
        if (written < 0) {
            if (errno == EINTR)
                continue;
            break;
        }
        if (written == 0)
            break;
        off += (size_t)written;
    }
    (void)close(fd);
}

static void qos_update_runtime_pressure(double mem_util, double tensor_util)
{
    if (!qos_runtime_throttle_enabled())
        return;

    const double enter = pqc_config_double_prefix_or_default("PQC_QOS_MEM_ENTER_UTIL", 0.70);
    const double exit = pqc_config_double_prefix_or_default("PQC_QOS_MEM_EXIT_UTIL", 0.60);
    const unsigned hold = (unsigned)pqc_config_positive_long_or_default("PQC_QOS_HOLD_SAMPLES", 2);
    const double pressure = mem_util > tensor_util ? mem_util : tensor_util;

    pqc_lock_profile_scope_t throttle_scope;
    if (qos_throttle_lock(&throttle_scope, __func__) != 0)
        return;
    g_qos_pressure_value = pressure;
    if (!g_qos_throttle_state) {
        g_qos_below_exit_count = 0;
        if (pressure >= enter)
            g_qos_throttle_state = 1;
    } else {
        if (pressure <= exit) {
            if (++g_qos_below_exit_count >= hold) {
                g_qos_throttle_state = 0;
                g_qos_below_exit_count = 0;
            }
        } else {
            g_qos_below_exit_count = 0;
        }
    }
    (void)qos_throttle_unlock(&throttle_scope, __func__);
}

void pqc_qos_apply_runtime_throttle(size_t bytes, int qos_class)
{
    if (!qos_runtime_throttle_enabled())
        return;

    int throttled = 0;
    double pressure = 0.0;
    pqc_lock_profile_scope_t throttle_scope;
    if (qos_throttle_lock(&throttle_scope, __func__) == 0) {
        throttled = g_qos_throttle_state;
        pressure = g_qos_pressure_value;
        (void)qos_throttle_unlock(&throttle_scope, __func__);
    }

    const int eligible = qos_class != PQC_QOS_CLASS_LATENCY;
    const long sleep_us = (throttled && eligible) ?
        pqc_config_positive_long_or_default("PQC_QOS_THROTTLE_SLEEP_US", 50000) : 0;
    if (sleep_us > 0)
        usleep((useconds_t)sleep_us);

    const char *trace_path = pqc_config_get_nonempty("PQC_QOS_THROTTLE_TRACE_PATH");
    if (trace_path) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        const uint64_t now_ns = (uint64_t)ts.tv_sec * 1000000000ULL +
                                (uint64_t)ts.tv_nsec;
        char trace_line[512];
        snprintf(trace_line, sizeof(trace_line),
                 "{\"timestamp_ns\":%llu,\"bytes\":%zu,"
                 "\"pressure\":%.4f,\"qos_class\":\"%s\","
                 "\"eligible\":%d,\"throttled\":%d,\"sleep_us\":%ld}\n",
                 (unsigned long long)now_ns, bytes, pressure,
                 pqc_qos_class_name(qos_class), eligible,
                 throttled && eligible, sleep_us);
        qos_trace_emit_line(trace_path, trace_line);
    }
}

static void *gpu_load_monitor_main(void *arg)
{
    (void)arg;
    const char *path = gpu_load_path();
    int fd = open(path, O_RDONLY);
    if (fd < 0)
        return NULL;
    char buf[32];
    while (!g_gpu_load_stop) {
        ssize_t n = pread(fd, buf, sizeof(buf) - 1, 0);
        if (n > 0) {
            buf[n] = '\0';
            double sample = (double)strtod(buf, NULL);
            if (sample > 1.0)
                sample /= 10.0;
            if (sample < 0.0)
                sample = 0.0;
            if (sample > 100.0)
                sample = 100.0;
            pqc_lock_profile_scope_t gpu_scope;
            if (qos_gpu_load_lock(&gpu_scope, __func__) == 0) {
                g_gpu_load_ewma = 0.2 * sample + 0.8 * g_gpu_load_ewma;
                (void)qos_gpu_load_unlock(&gpu_scope, __func__);
            }
        }
        usleep(5000);
    }
    close(fd);
    return NULL;
}

static int qos_read_telemetry_file(const char *path,
                                   double *mem,
                                   double *tensor,
                                   unsigned long long *budget_ns,
                                   unsigned long long *queue_depth)
{
    if (!path || !mem || !tensor || !budget_ns || !queue_depth)
        return -EINVAL;

    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return -errno;

    char buf[256];
    ssize_t n;
    do {
        n = pread(fd, buf, sizeof(buf) - 1, 0);
    } while (n < 0 && errno == EINTR);

    int saved_errno = errno;
    (void)close(fd);
    if (n < 0)
        return -saved_errno;
    if (n == 0)
        return 0;

    buf[n] = '\0';
    return sscanf(buf, "%lf %lf %llu %llu",
                  mem, tensor, budget_ns, queue_depth);
}

static void *admission_telemetry_file_main(void *arg)
{
    const char *path = (const char *)arg;
    const long poll_ms = pqc_config_positive_long_or_default("PQC_TELEMETRY_POLL_MS", 50);
    uint64_t last_budget_ns = UINT64_MAX;
    uint64_t last_queue_depth = UINT64_MAX;
    double last_mem = -1.0;
    double last_tensor = -1.0;

    while (!g_admission_telemetry_stop) {
        double mem = 0.0;
        double tensor = 0.0;
        unsigned long long budget_ns = 0;
        unsigned long long queue_depth = 0;
        int fields = qos_read_telemetry_file(path, &mem, &tensor,
                                             &budget_ns, &queue_depth);
        if (fields >= 2) {
            if (mem < 0.0) mem = 0.0;
            if (mem > 1.0) mem = 1.0;
            if (tensor < 0.0) tensor = 0.0;
            if (tensor > 1.0) tensor = 1.0;
            if (mem != last_mem || tensor != last_tensor) {
                pqc_admission_update_telemetry(mem, tensor);
                qos_update_runtime_pressure(mem, tensor);
                last_mem = mem;
                last_tensor = tensor;
            }
            if (fields >= 3 && budget_ns != last_budget_ns) {
                uint64_t qd = (fields >= 4) ? (uint64_t)queue_depth : 0;
                pqc_admission_update_ai_budget((uint64_t)budget_ns, qd);
                last_budget_ns = (uint64_t)budget_ns;
                last_queue_depth = qd;
            } else if (fields >= 4 && queue_depth != last_queue_depth) {
                pqc_admission_update_ai_budget(last_budget_ns == UINT64_MAX ? 0 : last_budget_ns,
                                                (uint64_t)queue_depth);
                last_queue_depth = (uint64_t)queue_depth;
            }
        }
        usleep((useconds_t)poll_ms * 1000U);
    }
    return NULL;
}

void pqc_qos_start_monitors(pqc_qos_monitor_status_t *status)
{
    if (status)
        memset(status, 0, sizeof(*status));

    const char *gpu_path = gpu_load_path();
    if (status && gpu_path)
        snprintf(status->gpu_load_path, sizeof(status->gpu_load_path), "%s", gpu_path);
    g_gpu_load_stop = 0;
    int rc = pthread_create(&g_gpu_load_thread, NULL, gpu_load_monitor_main, NULL);
    if (rc == 0) {
        g_gpu_load_thread_started = 1;
        if (status)
            status->gpu_monitor_started = 1;
    } else if (status) {
        status->gpu_monitor_errno = rc;
    }

    const char *telemetry_path = pqc_config_get_nonempty("PQC_TELEMETRY_FILE");
    if (!telemetry_path)
        return;
    if (status) {
        status->telemetry_monitor_configured = 1;
        snprintf(status->telemetry_path, sizeof(status->telemetry_path), "%s", telemetry_path);
    }
    snprintf(g_admission_telemetry_path, sizeof(g_admission_telemetry_path), "%s", telemetry_path);
    g_admission_telemetry_stop = 0;
    rc = pthread_create(&g_admission_telemetry_thread, NULL,
                        admission_telemetry_file_main, g_admission_telemetry_path);
    if (rc == 0) {
        g_admission_telemetry_thread_started = 1;
        if (status)
            status->telemetry_monitor_started = 1;
    } else if (status) {
        status->telemetry_monitor_errno = rc;
    }
}

int pqc_qos_stop_gpu_monitor(void)
{
    if (!g_gpu_load_thread_started)
        return 0;
    g_gpu_load_stop = 1;
    pthread_join(g_gpu_load_thread, NULL);
    g_gpu_load_thread_started = 0;
    return 1;
}

int pqc_qos_stop_admission_telemetry(void)
{
    if (!g_admission_telemetry_thread_started)
        return 0;
    g_admission_telemetry_stop = 1;
    pthread_join(g_admission_telemetry_thread, NULL);
    g_admission_telemetry_thread_started = 0;
    return 1;
}

double pqc_qos_gpu_load_ewma_read(void)
{
    double v = 0.0;
    pqc_lock_profile_scope_t gpu_scope;
    if (qos_gpu_load_lock(&gpu_scope, __func__) == 0) {
        v = g_gpu_load_ewma;
        (void)qos_gpu_load_unlock(&gpu_scope, __func__);
    }
    return v;
}
