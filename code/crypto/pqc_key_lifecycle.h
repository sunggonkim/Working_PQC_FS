#ifndef PQC_KEY_LIFECYCLE_H
#define PQC_KEY_LIFECYCLE_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    PQC_KEY_LIFECYCLE_PLANE_DATA = 1,
    PQC_KEY_LIFECYCLE_PLANE_KEY = 2,
    PQC_KEY_LIFECYCLE_PLANE_FRESHNESS = 3,
} pqc_key_lifecycle_plane_t;

typedef enum {
    PQC_KEY_LIFECYCLE_STATUS_IMPLEMENTED = 1,
    PQC_KEY_LIFECYCLE_STATUS_EXTERNAL = 2,
    PQC_KEY_LIFECYCLE_STATUS_NON_CLAIM = 3,
} pqc_key_lifecycle_status_t;

typedef enum {
    PQC_KEY_MATERIAL_MOUNT_KEY = 0,
    PQC_KEY_MATERIAL_FILE_ENVELOPE_SECRET = 1,
    PQC_KEY_MATERIAL_DATA_BLOCK_KEY = 2,
    PQC_KEY_MATERIAL_MOUNT_KEM_KEYPAIR = 3,
    PQC_KEY_MATERIAL_FRESHNESS_ANCHOR = 4,
    PQC_KEY_MATERIAL_TPM_PCR_POLICY = 5,
    PQC_KEY_MATERIAL_COUNT = 6,
} pqc_key_material_t;

typedef struct {
    pqc_key_material_t material;
    pqc_key_lifecycle_plane_t plane;
    pqc_key_lifecycle_status_t status;
    const char *name;
    const char *owner_module;
    const char *producer;
    const char *storage;
    const char *protector;
    const char *rotation;
    const char *recovery;
    const char *failure_boundary;
    int hardware_released;
    int data_plane_critical;
} pqc_key_lifecycle_entry_t;

const pqc_key_lifecycle_entry_t *pqc_key_lifecycle_entries(size_t *count_out);
const pqc_key_lifecycle_entry_t *pqc_key_lifecycle_find(
    pqc_key_material_t material);
const char *pqc_key_lifecycle_plane_name(pqc_key_lifecycle_plane_t plane);
const char *pqc_key_lifecycle_status_name(pqc_key_lifecycle_status_t status);
const char *pqc_key_material_name(pqc_key_material_t material);
int pqc_key_lifecycle_self_test(void);

#endif /* PQC_KEY_LIFECYCLE_H */
