# Combined durability bundle

This bundle retains the previous TPM-only and app-only orchestration checks and adds same-backing-store SQLite+TPM and dbm.dumb+TPM campaigns.

## Orchestration checks

- `tpm_only_bundle`: returncode `0`
- `app_recovery_bundle`: returncode `0`

## Unified same-backing-store campaign

- Present: `True`
- Verdict: `fail_closed`
- Acceptable: `True`
- Detail: `stale snapshot mounted but SQLite read failed closed rc=1`
- Replay mounted: `True`
- Baseline rows: `1`
- Advanced rows: `3`

## Unified second-workload campaign

- Present: `True`
- Workload: `dbm.dumb` key-value store
- Verdict: `fail_closed`
- Acceptable: `True`
- Detail: `stale dbm snapshot mounted but dbm read failed closed rc=1`
- Replay mounted: `True`
- Baseline rows: `1`
- Advanced rows: `3`

Conservative interpretation: this supports fail-closed stale-snapshot results for SQLite and dbm.dumb on TPM-backed FUSE backing stores. It does not establish RocksDB coverage, syscall-exact crash timing, or arbitrary interruption safety.
