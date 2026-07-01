#include "pqc_config.h"
#include "pqc_lock_profile.h"
#include "pqc_test_hooks.h"

#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static pthread_mutex_t g_fault_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t g_fault_config_once = PTHREAD_ONCE_INIT;
static int g_fault_triggered = 0;
static int g_pause_triggered = 0;
atomic_int g_pqc_fault_cutpoint_enabled = ATOMIC_VAR_INIT(-1);

typedef struct {
    int enabled;
    char pause_cutpoint[128];
    char fault_cutpoint[128];
    char pause_marker_path[4096];
    char fault_marker_path[4096];
    unsigned int pause_us;
} pqc_fault_config_t;

static pqc_fault_config_t g_fault_config = {
    .enabled = 0,
    .pause_us = 250000U,
};

static int fault_cutpoint_lock(pqc_lock_profile_scope_t *scope,
                               const char *site)
{
    return pqc_profiled_mutex_lock(&g_fault_lock, "fault_cutpoint_lock",
                                   site, scope);
}

static int fault_cutpoint_unlock(pqc_lock_profile_scope_t *scope,
                                 const char *site)
{
    return pqc_profiled_mutex_unlock(&g_fault_lock, "fault_cutpoint_lock",
                                     site, scope);
}

static void test_hook_copy(char *dst, size_t dst_len, const char *src)
{
    if (!dst || dst_len == 0)
        return;
    if (!src)
        src = "";
    snprintf(dst, dst_len, "%s", src);
}

static unsigned int test_hook_parse_pause_us(const char *text)
{
    if (!text)
        return 250000U;
    char *end = NULL;
    unsigned long value = strtoul(text, &end, 10);
    if (!end || *end != '\0' || value > 60000000UL)
        return 250000U;
    return (unsigned int)value;
}

static void fault_config_init_once(void)
{
    const char *pause_target = pqc_config_get_nonempty("PQC_PAUSE_CUTPOINT");
    const char *fault_target = pqc_config_get_nonempty("PQC_FAULT_CUTPOINT");
    test_hook_copy(g_fault_config.pause_cutpoint,
                   sizeof(g_fault_config.pause_cutpoint), pause_target);
    test_hook_copy(g_fault_config.fault_cutpoint,
                   sizeof(g_fault_config.fault_cutpoint), fault_target);
    if (pause_target && *pause_target) {
        test_hook_copy(g_fault_config.pause_marker_path,
                       sizeof(g_fault_config.pause_marker_path),
                       pqc_config_get_nonempty("PQC_PAUSE_MARKER_PATH"));
        g_fault_config.pause_us = test_hook_parse_pause_us(
            pqc_config_get_nonempty("PQC_PAUSE_US"));
    }
    if (fault_target && *fault_target) {
        test_hook_copy(g_fault_config.fault_marker_path,
                       sizeof(g_fault_config.fault_marker_path),
                       pqc_config_get_nonempty("PQC_FAULT_MARKER_PATH"));
    }
    g_fault_config.enabled =
        (g_fault_config.pause_cutpoint[0] != '\0' ||
         g_fault_config.fault_cutpoint[0] != '\0');
    atomic_store_explicit(&g_pqc_fault_cutpoint_enabled,
                          g_fault_config.enabled ? 1 : 0,
                          memory_order_release);
}

static void test_hook_marker_write(const char *path,
                                   const char *event,
                                   const char *name)
{
    if (!path || !event || !name)
        return;
    int fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    if (fd < 0)
        return;
    time_t now = time(NULL);
    (void)dprintf(fd,
                  "{\"event\":\"%s\",\"name\":\"%s\","
                  "\"pid\":%ld,\"unix_time\":%lld}\n",
                  event, name, (long)getpid(), (long long)now);
    (void)fsync(fd);
    (void)close(fd);
}

int pqc_fault_cutpoint_enabled_slow(void)
{
    (void)pthread_once(&g_fault_config_once, fault_config_init_once);
    return atomic_load_explicit(&g_pqc_fault_cutpoint_enabled,
                                memory_order_acquire) > 0;
}

void pqc_fault_cutpoint_slow(const char *name)
{
    if (!name)
        return;
    (void)pthread_once(&g_fault_config_once, fault_config_init_once);
    if (!g_fault_config.enabled)
        return;

    if (g_fault_config.pause_cutpoint[0] &&
        strcmp(g_fault_config.pause_cutpoint, name) == 0) {
        int should_pause = 0;
        pqc_lock_profile_scope_t scope;
        if (fault_cutpoint_lock(&scope, "fault_pause_cutpoint") == 0) {
            if (!g_pause_triggered) {
                g_pause_triggered = 1;
                should_pause = 1;
            }
            (void)fault_cutpoint_unlock(&scope, "fault_pause_cutpoint");
        }
        if (should_pause) {
            test_hook_marker_write(g_fault_config.pause_marker_path,
                                   "pause_cutpoint", name);
            usleep((useconds_t)g_fault_config.pause_us);
        }
    }

    if (!g_fault_config.fault_cutpoint[0] ||
        strcmp(g_fault_config.fault_cutpoint, name) != 0)
        return;

    pqc_lock_profile_scope_t scope;
    if (fault_cutpoint_lock(&scope, "fault_kill_cutpoint") != 0)
        return;
    if (g_fault_triggered) {
        (void)fault_cutpoint_unlock(&scope, "fault_kill_cutpoint");
        return;
    }
    g_fault_triggered = 1;
    (void)fault_cutpoint_unlock(&scope, "fault_kill_cutpoint");

    test_hook_marker_write(g_fault_config.fault_marker_path,
                           "fault_cutpoint", name);

    raise(SIGKILL);
    _exit(128 + SIGKILL);
}
