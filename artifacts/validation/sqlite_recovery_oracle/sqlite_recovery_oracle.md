# SQLite recovery oracle

This report defines durable-boundary cut points and the recovery oracle for a future SQLite fault-injection campaign.
It is derived from retained SQLite samples and `artifacts/sqlite_strace.log`; it does not claim crash certification.

## Retained SQLite samples

### workload
- Source: `artifacts/motivation/sqlite_latency.csv`
- Rows: `2`
- SHA-256: `46fb07eb56fc20cb07b715cf849ba2ad79ce4d2b23af75002370e031687b54f3`
- full: WAL/WAL FULL, integrity=ok, samples=20
- plain: WAL/WAL FULL, integrity=ok, samples=20

### contention
- Source: `artifacts/motivation/sqlite_contention_latency.csv`
- Rows: `2`
- SHA-256: `31d74f1a4131d2be683357f3189b2c6bc385ff0db6074429553cd9a1c00c394f`
- full: WAL/WAL FULL, integrity=ok, samples=2
- plain: WAL/WAL FULL, integrity=ok, samples=2

## Strace durable-boundary observations

- open_db: `1`
- open_journal: `1`
- open_wal: `1`
- open_shm: `1`
- fdatasync: `4`
- fsync: `0`
- pwrite: `12`
- unlink_journal: `1`

## Cut points

### before_sqlite_journal_header_sync
- Observed basis: journal file open + pwrite64 + first fdatasync
- Observed in strace: `True`
- Permitted recovery: previous committed DB state or fail closed
- Bug signal: new DB contents reachable without durable journal header

### after_journal_header_before_db_sync
- Observed basis: journal fdatasync before database fdatasync
- Observed in strace: `True`
- Permitted recovery: previous state, latest fully committed state, or fail closed
- Bug signal: SQLite integrity_check != ok or unexpected row digest

### after_db_fsync_before_journal_unlink
- Observed basis: database pwrite/fdatasync followed by journal unlink
- Observed in strace: `True`
- Permitted recovery: latest committed state or fail closed
- Bug signal: journal removed while DB content is partial

### wal_file_created_before_checkpoint
- Observed basis: WAL and SHM sidecar open after journal bootstrap
- Observed in strace: `True`
- Permitted recovery: SQLite WAL recovery to a state with integrity_check=ok
- Bug signal: WAL exists but integrity_check fails or expected rows disappear silently

## Oracle

- Open SQLite database read-only and read-write.
- Run PRAGMA integrity_check and require result exactly 'ok'.
- Record expected table names, row counts, and ORDER BY primary-key content digest before the fault.
- After replay, accept only previous committed digest, latest committed digest, or explicit fail-closed error.
- Classify any other readable digest as silent corruption.

## Conservative interpretation

- This closes the oracle/cut-point definition step for the current SQLite path.
- It does not close the fault-injection campaign requirement because no per-cut replay was executed here.
