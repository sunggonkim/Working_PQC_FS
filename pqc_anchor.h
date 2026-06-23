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

pqc_anchor_backend_t pqc_anchor_backend(void);

int pqc_anchor_probe(void);
int pqc_anchor_self_test(void);

/* ── Batch-anchor API ────────────────────────────────────────────────────────
 *
 * pqc_anchor_record_file(): called by checkpoint_store() to register the
 *   current committed generation for a file_id.  Updates the in-memory
 *   committed-prefix map; does NOT write to TPM yet.
 *
 * pqc_anchor_store(): called by the background anchor worker.  Recomputes the
 *   SHA-256 committed-prefix root over all registered (file_id, max_gen) pairs
 *   and writes a pqc_prefix_anchor_t to the configured backend.
 *
 * pqc_anchor_load(): called at mount.  Re-reads the stored prefix root and
 *   verifies it is not ahead of the locally reconstructed root (fail-closed).
 *
 * pqc_anchor_flush(): blocks until any pending background commit completes.
 * pqc_anchor_finalize(): called at unmount.
 */
int pqc_anchor_record_file(uint64_t file_id, uint64_t max_generation,
                           uint64_t sequence, uint64_t logical_size);

int pqc_anchor_store(const pqc_anchor_state_t *state);   /* legacy compat */
int pqc_anchor_load(const pqc_anchor_state_t *expected_state);
int pqc_anchor_flush(void);
int pqc_anchor_finalize(void);

#endif
