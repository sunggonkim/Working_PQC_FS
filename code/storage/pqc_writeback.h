#ifndef PQC_WRITEBACK_H
#define PQC_WRITEBACK_H

#include "pqc_fd_context.h"
#include "pqc_lock_profile.h"

/*
 * Flushes ctx->wbuf through the current strict publication path.
 *
 * The caller must hold ctx->fd_lock and pass its lock-profile scope/site.
 * This function snapshots ctx->wbuf, clears the live buffer, marks a pending
 * job, releases fd_lock for prepare/crypto/publish, then reacquires fd_lock
 * before returning.  Strict mode still uses generation reservation and ordered
 * publication; it is not epoch/group commit.
 */
int pqc_writeback_flush_locked(int storage_fd,
                               pqc_fd_ctx_t *ctx,
                               pqc_lock_profile_scope_t *fd_scope,
                               const char *fd_site,
                               int require_durable);

#endif /* PQC_WRITEBACK_H */
