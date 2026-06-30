#ifndef PQC_TRUST_BOUNDARY_H
#define PQC_TRUST_BOUNDARY_H

#include <stddef.h>

typedef enum {
    PQC_TRUST_SUBJECT_DAEMON_PROCESS = 0,
    PQC_TRUST_SUBJECT_KERNEL_DRIVER_FUSE = 1,
    PQC_TRUST_SUBJECT_BACKING_STORAGE = 2,
    PQC_TRUST_SUBJECT_PRIVILEGED_LOCAL_ATTACKER = 3,
    PQC_TRUST_SUBJECT_MULTI_TENANT_GPU = 4,
    PQC_TRUST_SUBJECT_GPU_SIDE_CHANNEL = 5,
    PQC_TRUST_SUBJECT_DEPLOYMENT_READINESS = 6,
    PQC_TRUST_SUBJECT_COUNT = 7,
} pqc_trust_subject_t;

typedef enum {
    PQC_TRUST_STATUS_IMPLEMENTED_BOUNDARY = 1,
    PQC_TRUST_STATUS_EXTERNAL_TCB = 2,
    PQC_TRUST_STATUS_NON_CLAIM = 3,
} pqc_trust_status_t;

typedef struct {
    pqc_trust_subject_t subject;
    pqc_trust_status_t status;
    const char *name;
    const char *trusted_component;
    const char *implemented_boundary;
    const char *excluded_attacker;
    const char *failure_boundary;
    const char *non_claim_guard;
    int claims_defense;
    int deployment_ready;
} pqc_trust_boundary_entry_t;

const pqc_trust_boundary_entry_t *pqc_trust_boundary_entries(
    size_t *count_out);
const pqc_trust_boundary_entry_t *pqc_trust_boundary_find(
    pqc_trust_subject_t subject);
const char *pqc_trust_subject_name(pqc_trust_subject_t subject);
const char *pqc_trust_status_name(pqc_trust_status_t status);
int pqc_trust_boundary_self_test(void);

#endif /* PQC_TRUST_BOUNDARY_H */
