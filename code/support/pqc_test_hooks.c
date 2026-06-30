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
static int g_fault_triggered = 0;
static int g_pause_triggered = 0;

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

static unsigned int test_hook_pause_us(void)
{
    const char *text = pqc_config_get_nonempty("PQC_PAUSE_US");
    if (!text)
        return 250000U;
    char *end = NULL;
    unsigned long value = strtoul(text, &end, 10);
    if (!end || *end != '\0' || value > 60000000UL)
        return 250000U;
    return (unsigned int)value;
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

void pqc_fault_cutpoint(const char *name)
{
    const char *pause_target = pqc_config_get_nonempty("PQC_PAUSE_CUTPOINT");
    if (pause_target && strcmp(pause_target, name) == 0) {
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
            test_hook_marker_write(
                pqc_config_get_nonempty("PQC_PAUSE_MARKER_PATH"),
                "pause_cutpoint", name);
            usleep((useconds_t)test_hook_pause_us());
        }
    }

    const char *target = pqc_config_get_nonempty("PQC_FAULT_CUTPOINT");
    if (!target || strcmp(target, name) != 0)
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

    const char *marker = pqc_config_get_nonempty("PQC_FAULT_MARKER_PATH");
    test_hook_marker_write(marker, "fault_cutpoint", name);

    raise(SIGKILL);
    _exit(128 + SIGKILL);
}
