# Epoch Redo-Log Mounted Smoke

- Generated: `2026-06-28T06:36:24Z`
- Overall pass: `true`

## strict
- Pass: `true`
- Read matches write: `true`
- Epoch log files: `0`
- Block records: `0`
- Commit records: `0`

## epoch_redo_log
- Pass: `true`
- Read matches write: `true`
- Epoch log files: `1`
- Block records: `5`
- Commit records: `1`

## Negative Claim Guard

This smoke proves that epoch-redo-log mode appends block and commit records to a per-file .pqcepoch sidecar and reaches a log fdatasync barrier before strict journal publication continues. It does not prove checkpoint compaction, crash replay, sync-count reduction, throughput improvement, or rollback resistance.
