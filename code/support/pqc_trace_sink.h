#ifndef PQC_TRACE_SINK_H
#define PQC_TRACE_SINK_H

#include <pthread.h>
#include <stddef.h>
#include <stdatomic.h>

typedef struct {
    pthread_mutex_t lock;
    int             fd;
    char            path[4096];
    atomic_int      configured;
    atomic_int      enabled;
} pqc_trace_sink_t;

#define PQC_TRACE_SINK_INITIALIZER \
    { PTHREAD_MUTEX_INITIALIZER, -1, {0}, ATOMIC_VAR_INIT(0), ATOMIC_VAR_INIT(0) }

int pqc_trace_sink_enabled_env(pqc_trace_sink_t *sink,
                               const char *env_name);
int pqc_trace_sink_write_env(pqc_trace_sink_t *sink,
                             const char *env_name,
                             const char *line,
                             size_t len);
void pqc_trace_sink_close(pqc_trace_sink_t *sink);

#endif /* PQC_TRACE_SINK_H */
