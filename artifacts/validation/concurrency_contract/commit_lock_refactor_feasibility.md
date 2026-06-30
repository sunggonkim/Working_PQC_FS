# Commit Lock Refactor Feasibility

- Decision: `strict_prepare_crypto_outside_fd_and_commit_locks_but_gate_open`

## Blockers

- Read path does not use the pending-job lifetime guard while recovery runs outside fd_lock.
- Release/lifecycle teardown is not yet proven to detach fds, buffers, and file-state references before close/free/release work.
- Reader visibility is not yet bounded by a committed-generation journal lookup.

## Required Before Safe Refactor

- expand strict durable-publication fault coverage after commit_lock splitting
- fault matrix covering reserved-but-unpublished generations, crypto failure, data write failure, journal append failure, and remount
- production lock-hold evidence before and after the refactor
