#ifndef PQC_LOCK_PROFILE_H
#define PQC_LOCK_PROFILE_H

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

typedef struct {
    uint64_t wait_start_ns;
    uint64_t acquired_ns;
    int enabled;
} pqc_lock_profile_scope_t;

int pqc_lock_profile_init_from_config(void);
void pqc_lock_profile_shutdown(void);
int pqc_lock_profile_enabled(void);
void pqc_lock_profile_dump(FILE *out);
int pqc_lock_profile_self_test(FILE *out);

int pqc_profiled_mutex_lock(pthread_mutex_t *mutex, const char *lock_name,
                            const char *site,
                            pqc_lock_profile_scope_t *scope);
int pqc_profiled_mutex_unlock(pthread_mutex_t *mutex, const char *lock_name,
                              const char *site,
                              pqc_lock_profile_scope_t *scope);
int pqc_profiled_cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex,
                           const char *lock_name, const char *site,
                           pqc_lock_profile_scope_t *scope);
int pqc_profiled_cond_timedwait(pthread_cond_t *cond,
                                pthread_mutex_t *mutex,
                                const char *lock_name,
                                const char *site,
                                pqc_lock_profile_scope_t *scope,
                                const struct timespec *abstime);

#endif /* PQC_LOCK_PROFILE_H */
