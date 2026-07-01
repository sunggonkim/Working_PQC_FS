#ifndef PQC_REKEY_H
#define PQC_REKEY_H

#include <oqs/oqs.h>
#include <stdint.h>

typedef struct {
    uint64_t accepted;
    uint64_t duplicate;
    uint64_t invalid_fd;
    uint64_t queue_full;
} pqc_rekey_queue_stats_t;

typedef enum {
    PQC_REKEY_ENQUEUE_ACCEPTED = 0,
    PQC_REKEY_ENQUEUE_DUPLICATE,
    PQC_REKEY_ENQUEUE_INVALID_FD,
    PQC_REKEY_ENQUEUE_FULL
} pqc_rekey_enqueue_status_t;

int pqc_rekey_worker_start(OQS_KEM *kem, const uint8_t *public_key);
int pqc_rekey_worker_stop(void);
pqc_rekey_enqueue_status_t pqc_rekey_queue_push(int fd);
void pqc_rekey_queue_stats_snapshot(pqc_rekey_queue_stats_t *out);
int pqc_rekey_rotation_interval_s(void);
int pqc_rekey_force_on_write_enabled(void);
int pqc_rekey_write_trigger_enabled(void);
void pqc_rekey_write_policy_snapshot(int *trigger_enabled,
                                     int *force_on_write,
                                     int *rotation_interval_s);
void pqc_rekey_init_from_config(void);
int pqc_rekey_queue_accounting_selftest(void);

#endif /* PQC_REKEY_H */
