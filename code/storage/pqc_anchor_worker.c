#include "pqc_anchor_worker.h"

#include "pqc_config.h"
#include "pqc_lock_profile.h"
#include "pqc_test_hooks.h"

#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>

static pthread_mutex_t   g_anchor_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t   g_anchor_lifecycle_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t    g_anchor_cv = PTHREAD_COND_INITIALIZER;
static pthread_t         g_anchor_thread;
static int               g_anchor_thread_started = 0;
static int               g_anchor_thread_joining = 0;
static int               g_anchor_stop = 0;
static int               g_anchor_dirty = 0;
static pqc_anchor_state_t g_anchor_state = {0};
static time_t            g_anchor_last_commit = 0;
static atomic_int        g_windowed_file_anchor_policy = ATOMIC_VAR_INIT(-1);
static atomic_int        g_freshness_window_policy = ATOMIC_VAR_INIT(-1);

static void anchor_lifecycle_lock(pqc_lock_profile_scope_t *scope,
                                  const char *site)
{
    (void)pqc_profiled_mutex_lock(&g_anchor_lifecycle_lock,
                                  "anchor_lifecycle_lock", site, scope);
}

static void anchor_lifecycle_unlock(pqc_lock_profile_scope_t *scope,
                                    const char *site)
{
    (void)pqc_profiled_mutex_unlock(&g_anchor_lifecycle_lock,
                                    "anchor_lifecycle_lock", site, scope);
}

static void anchor_worker_lock(pqc_lock_profile_scope_t *scope,
                               const char *site)
{
    (void)pqc_profiled_mutex_lock(&g_anchor_lock,
                                  "anchor_worker_lock", site, scope);
}

static void anchor_worker_unlock(pqc_lock_profile_scope_t *scope,
                                 const char *site)
{
    (void)pqc_profiled_mutex_unlock(&g_anchor_lock,
                                    "anchor_worker_lock", site, scope);
}

static int anchor_worker_timedwait(pqc_lock_profile_scope_t *scope,
                                   const char *site,
                                   const struct timespec *deadline)
{
    return pqc_profiled_cond_timedwait(&g_anchor_cv, &g_anchor_lock,
                                       "anchor_worker_lock", site, scope,
                                       deadline);
}

static int anchor_worker_freshness_window_enabled(void)
{
    int cached = atomic_load_explicit(&g_freshness_window_policy,
                                      memory_order_acquire);
    if (cached >= 0)
        return cached;

    int enabled = pqc_config_present("PQC_FRESHNESS_WINDOW_N") ? 1 : 0;
    int expected = -1;
    if (atomic_compare_exchange_strong_explicit(
            &g_freshness_window_policy, &expected, enabled,
            memory_order_release, memory_order_acquire))
        return enabled;
    return atomic_load_explicit(&g_freshness_window_policy,
                                memory_order_acquire);
}

void pqc_anchor_worker_init_from_config(void)
{
    if (pqc_anchor_backend() == PQC_ANCHOR_BACKEND_DISABLED) {
        atomic_store_explicit(&g_freshness_window_policy, 0,
                              memory_order_release);
        atomic_store_explicit(&g_windowed_file_anchor_policy, 0,
                              memory_order_release);
        return;
    }
    (void)anchor_worker_freshness_window_enabled();
    (void)pqc_anchor_worker_windowed_file_anchor_enabled();
}

static void anchor_worker_mark_committed(void)
{
    pqc_lock_profile_scope_t scope;
    anchor_worker_lock(&scope, __func__);
    g_anchor_last_commit = time(NULL);
    anchor_worker_unlock(&scope, __func__);
}

static void anchor_worker_restore_dirty_if_current(
    const pqc_anchor_state_t *state)
{
    if (!state)
        return;

    pqc_lock_profile_scope_t scope;
    anchor_worker_lock(&scope, __func__);
    if (!g_anchor_dirty) {
        g_anchor_state = *state;
        g_anchor_dirty = 1;
        pthread_cond_signal(&g_anchor_cv);
    }
    anchor_worker_unlock(&scope, __func__);
}

static int anchor_worker_take_dirty(pqc_anchor_state_t *state)
{
    if (!state)
        return 0;

    pqc_lock_profile_scope_t scope;
    anchor_worker_lock(&scope, __func__);
    if (!g_anchor_dirty) {
        anchor_worker_unlock(&scope, __func__);
        return 0;
    }

    *state = g_anchor_state;
    g_anchor_dirty = 0;
    pthread_cond_signal(&g_anchor_cv);
    anchor_worker_unlock(&scope, __func__);
    return 1;
}

static int anchor_worker_flush_dirty(int fault_cutpoint)
{
    pqc_anchor_state_t state = {0};
    if (!anchor_worker_take_dirty(&state))
        return 0;

    if (fault_cutpoint)
        pqc_fault_cutpoint("anchor_update_before");

    int rc = pqc_anchor_store_force(&state);
    if (rc == 0) {
        anchor_worker_mark_committed();
    } else {
        anchor_worker_restore_dirty_if_current(&state);
    }
    return rc;
}

static void *anchor_worker_main(void *arg)
{
    (void)arg;
    while (1) {
        pqc_lock_profile_scope_t scope;
        anchor_worker_lock(&scope, __func__);
        while (!g_anchor_stop && !g_anchor_dirty) {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_sec += 10;
            (void)anchor_worker_timedwait(&scope, __func__, &ts);
        }
        if (g_anchor_stop) {
            anchor_worker_unlock(&scope, __func__);
            break;
        }
        struct timespec batch_deadline;
        clock_gettime(CLOCK_REALTIME, &batch_deadline);
        if (anchor_worker_freshness_window_enabled()) {
            batch_deadline.tv_sec += 86400; /* Wait effectively forever */
        } else {
            batch_deadline.tv_nsec += 250000000L;
            if (batch_deadline.tv_nsec >= 1000000000L) {
                batch_deadline.tv_sec += 1;
                batch_deadline.tv_nsec -= 1000000000L;
            }
        }
        while (!g_anchor_stop && g_anchor_dirty) {
            int wait_rc = anchor_worker_timedwait(&scope, __func__,
                                                  &batch_deadline);
            if (wait_rc == ETIMEDOUT)
                break;
        }
        if (!g_anchor_dirty) {
            int stopping = g_anchor_stop;
            anchor_worker_unlock(&scope, __func__);
            if (stopping)
                break;
            continue;
        }
        int stopping = g_anchor_stop;
        pqc_anchor_state_t state = g_anchor_state;
        g_anchor_dirty = 0;
        anchor_worker_unlock(&scope, __func__);

        int rc = stopping
            ? pqc_anchor_store_force(&state)
            : pqc_anchor_store(&state);
        if (rc == 0 && !stopping)
            rc = pqc_anchor_flush();
        if (rc == 0)
            anchor_worker_mark_committed();
        else
            anchor_worker_restore_dirty_if_current(&state);
    }
    return NULL;
}

void pqc_anchor_worker_stage(const pqc_anchor_state_t *state)
{
    if (!state)
        return;

    pqc_lock_profile_scope_t scope;
    anchor_worker_lock(&scope, __func__);
    g_anchor_state = *state;
    g_anchor_dirty = 1;
    pthread_cond_signal(&g_anchor_cv);
    anchor_worker_unlock(&scope, __func__);
}

int pqc_anchor_worker_start_if_configured(void)
{
    if (pqc_anchor_backend() == PQC_ANCHOR_BACKEND_DISABLED)
        return 0;
    const char *anchor_path = pqc_config_get_nonempty("PQC_FRESHNESS_ANCHOR_PATH");
    if (!anchor_path)
        return 0;

    pqc_lock_profile_scope_t lifecycle_scope;
    anchor_lifecycle_lock(&lifecycle_scope, __func__);
    if (g_anchor_thread_started || g_anchor_thread_joining) {
        anchor_lifecycle_unlock(&lifecycle_scope, __func__);
        return 0;
    }

    pqc_lock_profile_scope_t worker_scope;
    anchor_worker_lock(&worker_scope, __func__);
    g_anchor_stop = 0;
    g_anchor_last_commit = time(NULL);
    anchor_worker_unlock(&worker_scope, __func__);

    int rc = pthread_create(&g_anchor_thread, NULL, anchor_worker_main, NULL);
    if (rc != 0) {
        anchor_worker_lock(&worker_scope, __func__);
        g_anchor_stop = 1;
        anchor_worker_unlock(&worker_scope, __func__);
        anchor_lifecycle_unlock(&lifecycle_scope, __func__);
        return -rc;
    }

    g_anchor_thread_started = 1;
    anchor_lifecycle_unlock(&lifecycle_scope, __func__);
    return 1;
}

void pqc_anchor_worker_stop(void)
{
    pqc_lock_profile_scope_t lifecycle_scope;
    anchor_lifecycle_lock(&lifecycle_scope, __func__);
    if (!g_anchor_thread_started || g_anchor_thread_joining) {
        anchor_lifecycle_unlock(&lifecycle_scope, __func__);
        return;
    }
    g_anchor_thread_joining = 1;
    pthread_t thread_to_join = g_anchor_thread;

    pqc_lock_profile_scope_t worker_scope;
    anchor_worker_lock(&worker_scope, __func__);
    g_anchor_stop = 1;
    pthread_cond_broadcast(&g_anchor_cv);
    anchor_worker_unlock(&worker_scope, __func__);
    anchor_lifecycle_unlock(&lifecycle_scope, __func__);

    (void)pthread_join(thread_to_join, NULL);

    anchor_lifecycle_lock(&lifecycle_scope, __func__);
    g_anchor_thread_started = 0;
    g_anchor_thread_joining = 0;
    anchor_lifecycle_unlock(&lifecycle_scope, __func__);
}

int pqc_anchor_worker_flush_now(void)
{
    return anchor_worker_flush_dirty(1);
}

int pqc_anchor_worker_flush_now_external_sync(
    pqc_anchor_external_sync_fn sync_fn,
    void *opaque)
{
    if (!sync_fn)
        return pqc_anchor_worker_flush_now();

    pqc_anchor_state_t state = {0};
    if (!anchor_worker_take_dirty(&state))
        return sync_fn(opaque);

    pqc_fault_cutpoint("anchor_update_before");

    int rc = pqc_anchor_store_force_external_sync(&state, sync_fn, opaque);
    if (rc == 0) {
        anchor_worker_mark_committed();
    } else {
        anchor_worker_restore_dirty_if_current(&state);
    }
    return rc;
}

int pqc_anchor_worker_flush_windowed_external_sync(
    pqc_anchor_external_sync_fn sync_fn,
    void *opaque)
{
    if (!sync_fn)
        return pqc_anchor_worker_flush_now();

    pqc_anchor_state_t state = {0};
    if (!anchor_worker_take_dirty(&state))
        return sync_fn(opaque);

    pqc_fault_cutpoint("anchor_update_before");

    int rc = pqc_anchor_store_windowed_external_sync(&state, sync_fn, opaque);
    if (rc == 0) {
        anchor_worker_mark_committed();
    } else {
        anchor_worker_restore_dirty_if_current(&state);
    }
    return rc;
}

int pqc_anchor_worker_windowed_file_anchor_enabled(void)
{
    int cached = atomic_load_explicit(&g_windowed_file_anchor_policy,
                                      memory_order_acquire);
    if (cached >= 0)
        return cached;

    int enabled = pqc_config_enabled("PQC_FSYNC_WINDOWED_FILE_ANCHOR") ? 1 : 0;
    int expected = -1;
    if (atomic_compare_exchange_strong_explicit(
            &g_windowed_file_anchor_policy, &expected, enabled,
            memory_order_release, memory_order_acquire))
        return enabled;
    return atomic_load_explicit(&g_windowed_file_anchor_policy,
                                memory_order_acquire);
}

int pqc_anchor_worker_flush_pending_on_shutdown(void)
{
    return anchor_worker_flush_dirty(0);
}

int pqc_anchor_worker_lifecycle_self_test(void)
{
    if (pqc_anchor_backend() == PQC_ANCHOR_BACKEND_DISABLED) {
        pqc_anchor_worker_stop();
        return 0;
    }

    pqc_anchor_worker_stop();
    int rc = pqc_anchor_worker_start_if_configured();
    if (rc < 0)
        return rc;
    rc = pqc_anchor_worker_start_if_configured();
    if (rc < 0) {
        pqc_anchor_worker_stop();
        return rc;
    }

    pqc_anchor_state_t state = {
        .epoch = 13,
        .sequence = 17,
        .logical_size = 4096,
    };
    pqc_anchor_worker_stage(&state);
    rc = pqc_anchor_worker_flush_now();
    if (rc != 0) {
        pqc_anchor_worker_stop();
        return rc;
    }

    pqc_anchor_worker_stop();
    pqc_anchor_worker_stop();

    state.epoch = 14;
    state.sequence = 18;
    state.logical_size = 8192;
    pqc_anchor_worker_stage(&state);
    rc = pqc_anchor_worker_flush_pending_on_shutdown();
    pqc_anchor_worker_stop();
    return rc;
}
