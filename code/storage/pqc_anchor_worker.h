#ifndef PQC_ANCHOR_WORKER_H
#define PQC_ANCHOR_WORKER_H

#include "pqc_anchor.h"

/*
 * Mount-lifetime freshness-anchor worker state.
 *
 * This module owns only the existing background worker, dirty-state staging,
 * and synchronous flush/shutdown behavior.  It does not change the anchor
 * backend format or introduce a stronger freshness model.
 */

void pqc_anchor_worker_stage(const pqc_anchor_state_t *state);

void pqc_anchor_worker_init_from_config(void);
int pqc_anchor_worker_start_if_configured(void);
void pqc_anchor_worker_stop(void);

int pqc_anchor_worker_flush_now(void);
int pqc_anchor_worker_flush_now_external_sync(
    pqc_anchor_external_sync_fn sync_fn,
    void *opaque);
int pqc_anchor_worker_flush_windowed_external_sync(
    pqc_anchor_external_sync_fn sync_fn,
    void *opaque);
int pqc_anchor_worker_windowed_file_anchor_enabled(void);
int pqc_anchor_worker_flush_pending_on_shutdown(void);
int pqc_anchor_worker_lifecycle_self_test(void);

#endif /* PQC_ANCHOR_WORKER_H */
