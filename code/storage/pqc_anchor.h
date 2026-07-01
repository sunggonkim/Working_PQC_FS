#ifndef PQC_ANCHOR_H
#define PQC_ANCHOR_H

#include <stdint.h>

/* ── Per-file state that contributes to the committed-prefix root ────────── */
typedef struct {
    uint64_t epoch;        /* = max_generation for the file */
    uint64_t sequence;     /* monotonically increasing commit counter */
    uint64_t logical_size;
} pqc_anchor_state_t;

typedef enum {
    PQC_ANCHOR_BACKEND_DISABLED = 0,
    PQC_ANCHOR_BACKEND_FILE = 1,
    PQC_ANCHOR_BACKEND_HARDWARE = 2,
} pqc_anchor_backend_t;

typedef enum {
    PQC_ANCHOR_EPOCH_STATUS_NONE = 0,
    PQC_ANCHOR_EPOCH_STATUS_PENDING = 1,
    PQC_ANCHOR_EPOCH_STATUS_COMMITTED = 2,
    PQC_ANCHOR_EPOCH_STATUS_FAILED = 3,
} pqc_anchor_epoch_status_t;

typedef enum {
    PQC_ANCHOR_EPOCH_FLUSH_DISABLED = 0,
    PQC_ANCHOR_EPOCH_FLUSH_FILE_WINDOW = 1,
    PQC_ANCHOR_EPOCH_FLUSH_FILE_FORCE = 2,
    PQC_ANCHOR_EPOCH_FLUSH_HARDWARE_STAGE = 3,
    PQC_ANCHOR_EPOCH_FLUSH_HARDWARE_FORCE = 4,
    PQC_ANCHOR_EPOCH_FLUSH_FILE_EXTERNAL_SYNCFS = 5,
} pqc_anchor_epoch_flush_policy_t;

typedef struct {
    uint32_t version;
    uint32_t backend;
    uint32_t status;
    uint32_t flush_policy;
    uint64_t epoch_interval;
    uint64_t global_sequence;
    uint64_t file_count;
    uint64_t blocks_since_commit;
    int32_t last_rc;
    uint8_t prefix_root[32];
} pqc_anchor_epoch_record_t;

typedef int (*pqc_anchor_external_sync_fn)(void *opaque);

pqc_anchor_backend_t pqc_anchor_backend(void);
const char *pqc_anchor_epoch_status_name(uint32_t status);
int pqc_anchor_epoch_record_snapshot(pqc_anchor_epoch_record_t *out);

void pqc_anchor_init_from_config(void);
int pqc_anchor_probe(void);
int pqc_anchor_self_test(void);

/* ── Batch-anchor API ────────────────────────────────────────────────────────
 *
 * pqc_anchor_record_file(): called by checkpoint_store() to register the
 *   current committed generation for a file_id.  Updates the in-memory
 *   committed-prefix map; does NOT write to TPM yet.
 *
 * pqc_anchor_store(): windowed background path.  It records the legacy state,
 *   stages hardware anchors immediately, and writes to the configured backend
 *   only when the freshness window is reached.
 *
 * pqc_anchor_store_force(): synchronous durability path.  It records the
 *   legacy state and forces the current committed-prefix root to the configured
 *   backend before returning.
 *
 * pqc_anchor_load(): called at mount.  Re-reads the stored prefix root and
 *   verifies it is not ahead of the locally reconstructed root (fail-closed).
 *
 * pqc_anchor_flush(): blocks until any pending background commit completes.
 * pqc_anchor_finalize(): called at unmount.
 */
int pqc_anchor_record_file(uint64_t file_id, uint64_t max_generation,
                           uint64_t sequence, uint64_t logical_size);
int pqc_anchor_reconstruct_committed_map_from_storage(const char *root);
int pqc_anchor_committed_map_overflow_self_test(void);
int pqc_anchor_pending_clear_self_test(void);

int pqc_anchor_store(const pqc_anchor_state_t *state);   /* legacy compat */
int pqc_anchor_store_force(const pqc_anchor_state_t *state);
int pqc_anchor_store_force_external_sync(
    const pqc_anchor_state_t *state,
    pqc_anchor_external_sync_fn sync_fn,
    void *opaque);
int pqc_anchor_store_windowed_external_sync(
    const pqc_anchor_state_t *state,
    pqc_anchor_external_sync_fn sync_fn,
    void *opaque);
int pqc_anchor_load(const pqc_anchor_state_t *expected_state);
int pqc_anchor_flush(void);
int pqc_anchor_finalize(void);
void pqc_anchor_trace_shutdown(void);
int pqc_anchor_set_freshness_window(int n);

#endif
