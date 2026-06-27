# Daemon power-fault campaign

Scope: final-binary daemon SIGKILL cutpoints plus retained application recovery artifacts. This is not physical power-loss or kernel-crash certification.

- Overall pass: `True`
- Daemon rows: `9`
- Application rows: `4`
- Missing daemon cut points: `none`
- Missing application modes: `none`

## Daemon cutpoints

- data write (`data_write_after_pwrite`): verdict `previous_committed`, acceptable `True`, marker `True`, daemon_killed `True`
- journal append (`journal_append_after`): verdict `latest_committed`, acceptable `True`, marker `True`, daemon_killed `True`
- journal fsync barrier (`journal_fsync_after`): verdict `latest_committed`, acceptable `True`, marker `True`, daemon_killed `True`
- xattr/checkpoint update (`logical_size_xattr_after`): verdict `latest_committed`, acceptable `True`, marker `True`, daemon_killed `True`
- checkpoint write (`checkpoint_xattr_after`): verdict `latest_committed`, acceptable `True`, marker `True`, daemon_killed `True`
- anchor update (`anchor_update_before`): verdict `latest_committed`, acceptable `True`, marker `True`, daemon_killed `True`
- fsync (`fsync_before_return`): verdict `latest_committed`, acceptable `True`, marker `True`, daemon_killed `True`
- remount (`remount_after_checkpoint_load`): verdict `latest_committed`, acceptable `True`, marker `True`, daemon_killed `True`
- application read (`read_after_auth`): verdict `latest_committed`, acceptable `True`, marker `True`, daemon_killed `True`

## Application modes

- SQLite WAL/FULL: verdict `previous_committed`, acceptable `True`, trials `20`, source `artifacts/validation/sqlite_fault_campaign/sqlite_fault_campaign.json`
- SQLite rollback DELETE/EXTRA: verdict `previous_committed`, acceptable `True`, trials `3`, source `artifacts/validation/sqlite_syscall_crash_tpm/sqlite_syscall_crash_tpm.json`
- dbm.dumb key-value: verdict `fail_closed`, acceptable `True`, trials `1`, source `artifacts/validation/combined_durability_bundle/combined_durability_bundle.json`
- SQLite DELETE/EXTRA TPM replay: verdict `fail_closed`, acceptable `True`, trials `1`, source `artifacts/validation/combined_durability_bundle/combined_durability_bundle.json`
