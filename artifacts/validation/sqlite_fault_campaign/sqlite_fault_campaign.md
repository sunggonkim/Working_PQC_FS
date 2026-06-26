# SQLite fault-injection campaign

This report executes deterministic SQLite file-state fault injection against the durable-boundary oracle.
It does not claim syscall-exact crash timing or complete application crash certification.

## Reference states

- previous: rows=1, sha256=c36c821190d228587a9f173eee751e801f764fbcdf912466c824aad4e0e650d1
- latest: rows=2, sha256=283eaf711d14e229b9323ace640c3d6c64ff0abf7e57f1b0f4f85bd29a87ff0d

## Summary

- after_db_fsync_before_journal_unlink: trials=5, acceptable=5, silent_corruption=0, verdicts={'previous_committed': 5}
- after_journal_header_before_db_sync: trials=5, acceptable=5, silent_corruption=0, verdicts={'previous_committed': 5}
- before_sqlite_journal_header_sync: trials=5, acceptable=5, silent_corruption=0, verdicts={'previous_committed': 5}
- wal_file_created_before_checkpoint: trials=5, acceptable=5, silent_corruption=0, verdicts={'previous_committed': 5}

## Conservative interpretation

- This closes a SQLite-only per-cut oracle execution for the selected durable-boundary states.
- It does not close the broader two-workload app-recovery requirement.
- It does not prove crash behavior for arbitrary interruption times inside the filesystem.