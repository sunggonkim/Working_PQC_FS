#ifndef PQC_RUNTIME_H
#define PQC_RUNTIME_H

#include <oqs/oqs.h>
#include <stdint.h>

int pqc_runtime_init(void);
void pqc_runtime_cleanup(void);
OQS_KEM *pqc_runtime_kem(void);
const uint8_t *pqc_runtime_public_key(void);
const uint8_t *pqc_runtime_secret_key(void);

#endif /* PQC_RUNTIME_H */
