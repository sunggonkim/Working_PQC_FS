#ifndef PQC_SELFTEST_H
#define PQC_SELFTEST_H

int pqc_selftest_crypto(void);
int pqc_selftest_journal(void);
int pqc_selftest_generation_replay(void);
int pqc_selftest_checkpoint(void);
int pqc_selftest_key_lifecycle(void);
int pqc_selftest_trust_boundary(void);
int pqc_selftest_keyring_kdf(void);
int pqc_selftest_scheduler(void);
int pqc_selftest_parallel_commit(void);
int pqc_selftest_rekey_lifecycle(void);
int pqc_selftest_rekey_queue_accounting(void);
int pqc_selftest_anchor_worker_lifecycle(void);
int pqc_selftest_anchor_committed_map_overflow(void);
int pqc_selftest_anchor_pending_clear(void);
int pqc_selftest_lock_profile(void);
int pqc_selftest_durability(void);
int pqc_selftest_epoch_log(void);
int pqc_selftest_epoch_publish(void);
int pqc_selftest_admission_init_modes(void);
int pqc_selftest_admission_telemetry_smoke(void);

#endif /* PQC_SELFTEST_H */
