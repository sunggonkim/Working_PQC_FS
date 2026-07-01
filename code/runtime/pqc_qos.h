#ifndef PQC_QOS_H
#define PQC_QOS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int gpu_monitor_configured;
    int gpu_monitor_started;
    int gpu_monitor_errno;
    char gpu_load_path[4096];
    int telemetry_monitor_configured;
    int telemetry_monitor_started;
    int telemetry_monitor_errno;
    char telemetry_path[4096];
} pqc_qos_monitor_status_t;

void pqc_qos_start_monitors(pqc_qos_monitor_status_t *status);
void pqc_qos_disable_monitors_for_mount(void);
int pqc_qos_stop_gpu_monitor(void);
int pqc_qos_stop_admission_telemetry(void);
double pqc_qos_gpu_load_ewma_read(void);
int pqc_qos_runtime_throttle_enabled(void);
void pqc_qos_apply_runtime_throttle(size_t bytes, int qos_class);
void pqc_qos_apply_runtime_throttle_enabled(size_t bytes, int qos_class);

#ifdef __cplusplus
}
#endif

#endif /* PQC_QOS_H */
