#include "pqc_metrics.h"

#include "pqc_posix.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define PQC_METRICS_MESSAGE_CAP 1024
#define PQC_METRICS_LINE_CAP 1280

static int g_log_fd = -1;
static const char *PQC_LOG_FILENAME = "pqc_fuse_latency.log";

static void pqc_metrics_write_all(int fd, const char *line)
{
    if (fd < 0 || !line)
        return;
    size_t len = strlen(line);
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
}

static int pqc_metrics_open_append(const char *path)
{
    if (!path || path[0] == '\0')
        return -EINVAL;
    int fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    if (fd < 0)
        return -(errno ? errno : EIO);
    return fd;
}

double pqc_metrics_time_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

int pqc_metrics_open_for_storage(const char *storage_dir)
{
    char log_path[4096];
    int log_rc = pqc_sidecar_path(log_path, sizeof(log_path), storage_dir, "/../");
    if (log_rc == 0) {
        size_t used = strlen(log_path);
        int n = snprintf(log_path + used, sizeof(log_path) - used, "%s",
                         PQC_LOG_FILENAME);
        if (n < 0 || (size_t)n >= sizeof(log_path) - used)
            log_rc = -ENAMETOOLONG;
    }
    if (log_rc != 0) {
        fprintf(stderr, "[PQC-FUSE] WARNING: log path too long; falling back to current directory\n");
        log_path[0] = '\0';
    }
    g_log_fd = pqc_metrics_open_append(log_path);
    if (g_log_fd < 0) {
        int saved_errno = -g_log_fd;
        fprintf(stderr, "[PQC-FUSE] WARNING: Cannot open log file '%s': %s\n",
                log_path, strerror(saved_errno));
        g_log_fd = pqc_metrics_open_append(PQC_LOG_FILENAME);
        if (g_log_fd < 0)
            return -saved_errno;
    }
    return 0;
}

void pqc_metrics_close(void)
{
    if (g_log_fd >= 0) {
        (void)close(g_log_fd);
        g_log_fd = -1;
    }
}

void pqc_log(const char *fmt, ...)
{
    va_list args;
    char timebuf[64];
    char message[PQC_METRICS_MESSAGE_CAP];
    char line[PQC_METRICS_LINE_CAP];
    char file_line[PQC_METRICS_LINE_CAP];
    struct timeval tv;
    struct tm tm_info;

    gettimeofday(&tv, NULL);
    localtime_r(&tv.tv_sec, &tm_info);
    strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S", &tm_info);

    va_start(args, fmt);
    int msg_n = vsnprintf(message, sizeof(message), fmt ? fmt : "", args);
    va_end(args);
    if (msg_n < 0)
        snprintf(message, sizeof(message), "log-format-error");
    message[sizeof(message) - 1] = '\0';

    int line_n = snprintf(line, sizeof(line), "[PQC-FUSE %s.%03ld] %s\n",
                          timebuf, tv.tv_usec / 1000, message);
    if (line_n >= 0) {
        line[sizeof(line) - 1] = '\0';
        pqc_metrics_write_all(STDERR_FILENO, line);
    }

    if (g_log_fd < 0)
        return;
    int file_line_n = snprintf(file_line, sizeof(file_line),
                               "[%s.%03ld] %s\n",
                               timebuf, tv.tv_usec / 1000, message);
    if (file_line_n >= 0) {
        file_line[sizeof(file_line) - 1] = '\0';
        pqc_metrics_write_all(g_log_fd, file_line);
    }
}
