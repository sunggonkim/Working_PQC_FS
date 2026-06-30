# TPM Epoch Freshness Probe

This is a bounded C6-S2 probe of the production hardware anchor path.
It does not claim async Merkle maintenance, PCR-bound rollback resistance, or full replay protection.

## Verdict

- Overall pass: `true`
- Verdict: `environment-blocked`
- Environment blocked: `true`
- Hardware committed: `false`

## Checks

- `source_checks_pass`: `true`
- `trace_jsonl_well_formed`: `true`
- `hardware_pending_observed`: `true`
- `hardware_flush_attempted`: `true`
- `tpm_latency_recorded`: `true`
- `anti_hammering_boundary_recorded`: `true`
- `committed_or_environment_blocked`: `true`
- `rollback_claim_not_allowed`: `true`

## C6-S2 Requirement Status

- `tpm_latency`: `recorded`
- `anti_hammering_wear_boundary`: `recorded`
  - reason: `probe is intentionally single-shot and does not provision or loop TPM NV writes`
- `stale_snapshot_replay`: `not_run_environment_blocked`
  - reason: `hardware TPM epoch did not commit on this run; replay test would not measure a provisioned epoch`
- `pcr_drift`: `not_run_environment_blocked`
  - reason: `PCR read failed or anchor path has no persistent PCR policy; PCR-bound freshness remains a non-claim`
- `reboot_recovery`: `not_run_environment_blocked`
  - reason: `hardware TPM epoch did not commit on this run; replay test would not measure a provisioned epoch`
- `mount_refusal_logs`: `not_run_environment_blocked`
  - reason: `hardware TPM epoch did not commit on this run; replay test would not measure a provisioned epoch`

## Trace

- Trace path: `artifacts/validation/async_merkle_tpm_epoch/tpm_epoch_freshness_probe.trace.jsonl`
- Hardware pending observed: `true`
- Hardware flush attempted: `true`
- Hardware committed observed: `false`

## Conservative Interpretation

- A pending hardware epoch record proves the production path reached hardware-backend staging.
- A failed hardware-force flush with TPM command failures is environment/provisioning evidence, not rollback protection.
- Stale snapshot replay, PCR drift, reboot recovery, and mount refusal stay unclaimed unless a provisioned TPM run completes.
