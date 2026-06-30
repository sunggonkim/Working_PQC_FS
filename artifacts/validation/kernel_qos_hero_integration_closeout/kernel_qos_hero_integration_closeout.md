# Kernel QoS Hero Integration Closeout

- Overall pass: `true`
- Generated: `2026-06-30T13:42:43.270199+00:00`

## Close Conditions

- `required_inputs_present`: `true`
- `e1_closeout_passes`: `true`
- `b3_comparison_passes`: `true`
- `b3_parent_gate_closed`: `true`
- `two_kernel_baselines_measured`: `true`
- `hero_modes_available`: `true`
- `hero_rows_are_five_run`: `true`
- `kernel_rows_are_measured`: `true`
- `kernel_source_artifacts_exist`: `true`
- `paper_compares_both_kernel_baselines`: `true`
- `generated_table_complete`: `true`
- `paper_claim_guard_passes`: `true`
- `sqlite_uniqueness_claim_guard_allows_only_bounded_claim`: `true`

## Rows

- `app_only`: runs=5, p99_ms=7.253253040000001, misses=0.0, background_mb_s=0.0
- `unthrottled_storage`: runs=5, p99_ms=9.62087615, misses=0.0, background_mb_s=6.984360530500289
- `simple_controller`: runs=5, p99_ms=7.544277000000001, misses=0.0, background_mb_s=1.4970013728032954
- `aegis_policy`: runs=5, p99_ms=8.15053043, misses=0.0, background_mb_s=3.0160640171203785
- `ionice`: verdict=measured, p99_ms=16.217910019999998, misses=6.0, background_mb_s=7.248431018854942
- `systemd_io_weight`: verdict=measured, p99_ms=12.574790109999999, misses=7.0, background_mb_s=6.73069491922091

## Claim Boundary

- Allowed: bounded SQLite foreground p99 comparison under the retained mounted secure-storage pressure workflow
- Allowed: AEGIS-Q versus app-only, unthrottled storage, simple controller, ionice, and systemd IOWeight rows
- Allowed: statement that kernel controls preserve more background throughput in the retained one-run controls
- Allowed: statement that AEGIS-Q offers storage-visible policy/observability in this workload envelope
- Forbidden: SQLite p99 recovery uniqueness without qualification
- Forbidden: claim that kernel QoS cannot recover SQLite p99 in general
- Forbidden: claim that AEGIS-Q beats standard kernel throttling without noting repetition mismatch
- Forbidden: claim that the result generalizes to all workloads or all kernel QoS controls
