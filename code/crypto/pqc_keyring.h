#ifndef PQC_KEYRING_H
#define PQC_KEYRING_H

#include <stddef.h>
#include <stdint.h>

/*
 * Mount-key and per-file envelope metadata ownership.  g_master_key remains
 * exported only for legacy anchor/checkpoint HMAC compatibility during the
 * mechanical decomposition phase.
 */
extern uint8_t g_master_key[32];

int pqc_keyring_derive_master_key(const char *password);
const char *pqc_keyring_kdf_name(void);
void pqc_keyring_clear_master_key(void);
int pqc_keyring_kdf_self_test(void);

int pqc_keyring_metadata_store(const char *phys_path, const uint8_t *ss,
                               size_t ss_len, uint64_t file_id);
int pqc_keyring_metadata_load(const char *phys_path, uint8_t *ss,
                              size_t *ss_len, uint64_t *file_id);

#endif /* PQC_KEYRING_H */
