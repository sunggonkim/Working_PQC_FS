# Statistical and Thermal Methodology Audit

- Overall pass: `true`
- Paper scope gate pass: `true`
- Methodology complete: `true`
- Scope: this audit defines the required method and verifies retained benchmark-result coverage without turning scoped rows into cross-system claims.

## Required Methodology

- Warmup: Each performance mode must run an untimed warmup: at least one full workload pass for fio-style profiles or explicit workload-specific warmup transactions before measured samples.
- Run count: at least `5` independent repetitions for headline results.
- Confidence intervals: nonparametric bootstrap at 95% with `10000` resamples.
- Outlier policy: Retain all completed repetitions. Exclude only predeclared infrastructure failures with raw log, exit status, and reason; do not winsorize or silently drop slow runs.
- Clocks/power: CPU governor `performance`, plus nvpmodel and clock-state capture.
- Thermal logging: tegrastats or equivalent raw thermal/power log for the whole measured interval
- Background control: Record process snapshot and declare unrelated CPU/GPU/storage jobs absent.
- Cache policy: warm and cold rows must follow the declared procedures.
- Failure handling: A mode with command failure, missing raw logs, failed integrity check, missing mount log, or missing metadata is invalid. Scripts must not synthesize zeros for unsupported configurations.

## Current Artifacts

- `verified_microbench`: `diagnostic_scope_gate_passed`; missing `0` metadata fields.
- `verified_microbench_methodology_run`: `methodology_progress_host_not_ready`; missing `2` metadata fields.
- `qos_sqlite_hero_bundle`: `single_workflow_scope_gate_passed`; missing `0` metadata fields.
- `qos_sqlite_hero_methodology`: `methodology_progress_recovery_unstable`; missing `3` metadata fields.
- `keyplane_rekey_workflow`: `single_workflow_scope_gate_passed`; missing `0` metadata fields.
- `keyplane_rekey_methodology`: `methodology_progress_host_not_ready`; missing `2` metadata fields.
- `frozen_workload_contract`: `contract_complete_not_executed`; missing `1` metadata fields.
- `frozen_aegisq_contract`: `methodology_metadata_retained_cold_invalid`; missing `0` metadata fields.
- `frozen_gocryptfs_contract`: `baseline_metadata_retained_cold_invalid`; missing `0` metadata fields.
- `frozen_plaintext_contract`: `baseline_metadata_retained_cold_invalid`; missing `0` metadata fields.
- `kernel_baseline_feasibility`: `kernel_baseline_feasibility_retained`; missing `0` metadata fields.
- `fscrypt_dmcrypt_reference_fio`: `baseline_scope_gate_passed`; missing `0` metadata fields.

## Paper Gate

- diagnostic_placement: `true`
- single_workflow_not_statistical: `true`
- future_contract: `true`
- limitations_three_run: `true`
- abstract headline violations: `0`

## Completion Blockers

- none
