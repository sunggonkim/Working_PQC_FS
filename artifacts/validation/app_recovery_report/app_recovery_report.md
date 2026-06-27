# App recovery report

This report packages retained crash audit outputs, the executable SQLite oracle campaign, syscall-exact SQLite app-crash timing, and combined SQLite/dbm.dumb stale-snapshot replay.

- Input directory: `/home/thor/skim/pqc_encrpyted_fs/artifacts/validation/app_recovery_bundle`
- Checks: `4`
- Bundle present: `True`

- SQLite fault campaign present: `True`
- SQLite unacceptable oracle verdicts: `0`
- SQLite syscall crash campaign present: `True`
- SQLite syscall crash unacceptable verdicts: `0`
- Combined SQLite replay: `fail_closed` / acceptable `True`
- Combined dbm.dumb replay: `fail_closed` / acceptable `True`

This report does not claim power-loss or FUSE-daemon crash certification.