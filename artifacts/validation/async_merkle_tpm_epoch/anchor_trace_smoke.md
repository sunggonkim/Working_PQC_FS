# Anchor Trace Smoke

- Overall pass: `true`
- Trace path: `artifacts/validation/async_merkle_tpm_epoch/anchor_trace_smoke.jsonl`
- Event count: `2`
- Valid epoch records: `1`
- Mounted smoke pass: `true`
- Next code edit: Exercise the hardware TPM path with latency, anti-hammering, PCR drift, stale snapshot replay, reboot recovery, and mount refusal metadata before any rollback-resistance claim.

## Checks

- `self_test_passed`: `true`
- `trace_exists`: `true`
- `trace_jsonl_well_formed`: `true`
- `commit_event_valid`: `true`
- `epoch_record_event_valid`: `true`
- `anchor_file_written`: `true`
- `source_checks_pass`: `true`
- `mounted_smoke_pass`: `true`

## Boundary

C6 epoch-record smoke: production anchor store emits committed-prefix root transitions plus an explicit epoch freshness record carrying epoch interval, flush policy, pending/committed status, and prefix root.  This is still not async Merkle+TPM epoch freshness or PCR-bound rollback resistance.
