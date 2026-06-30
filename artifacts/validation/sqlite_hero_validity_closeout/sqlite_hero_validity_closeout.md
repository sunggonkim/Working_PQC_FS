# SQLite Hero Validity Closeout

- Overall pass: `true`
- Generated: `2026-06-30T13:42:43.242916+00:00`

## Close Conditions

- `required_inputs_present`: `true`
- `single_bundle_passes`: `true`
- `four_single_modes_present`: `true`
- `four_single_modes_acceptable`: `true`
- `single_raw_logs_complete`: `true`
- `repeated_warmup_and_five_runs`: `true`
- `repeated_mode_summaries_complete`: `true`
- `repeated_raw_logs_complete`: `true`
- `methodology_component_coverage_complete`: `true`
- `required_recovery_checks_pass`: `true`
- `unstable_recovery_directions_are_scoped`: `true`
- `kernel_qos_comparison_available`: `true`
- `stat_thermal_audit_passes`: `true`
- `thermal_log_retained`: `true`
- `paper_scope_guard_passes`: `true`

## Claim Boundary

- Allowed: SQLite WAL/FULL foreground transaction p99 under the retained mounted secure-storage pressure workflow
- Allowed: four-mode comparison: app-only, unthrottled storage, simple controller, and AEGIS-Q policy
- Allowed: AEGIS-Q storage-visible throttling records policy/telemetry/daemon-throttle decisions in every repeated run
- Allowed: kernel-QoS rows are compared with an explicit repetition mismatch warning
- Forbidden: broad workload generality
- Forbidden: SQLite recovery uniqueness without qualification
- Forbidden: external application scheduler recovery
- Forbidden: general-purpose filesystem performance
- Forbidden: deployed-filesystem or peak-throughput superiority

## Repeated Mode Summary

- `app_only`: runs=5, p99_median=7.253253040000001, miss_median=0.0, storage_mb_s_median=0.0
- `unthrottled_storage`: runs=5, p99_median=9.62087615, miss_median=0.0, storage_mb_s_median=6.984360530500289
- `simple_controller`: runs=5, p99_median=7.544277000000001, miss_median=0.0, storage_mb_s_median=1.4970013728032954
- `aegis_policy`: runs=5, p99_median=8.15053043, miss_median=0.0, storage_mb_s_median=3.0160640171203785

## Scoped Instability

- `aegis_recovers_p99`: {"all_true": false, "runs": 5, "true_runs": 4}
- `aegis_removes_deadline_misses`: {"all_true": false, "runs": 5, "true_runs": 3}
- `pressure_causes_deadline_miss`: {"all_true": false, "runs": 5, "true_runs": 2}
- `pressure_raises_p99`: {"all_true": false, "runs": 5, "true_runs": 3}

## Source Artifacts

- `single_bundle`: `artifacts/validation/qos_sqlite_hero_bundle/qos_sqlite_hero_bundle.json`
- `methodology_bundle`: `artifacts/validation/qos_sqlite_hero_methodology/qos_sqlite_hero_bundle.json`
- `kernel_qos_comparison`: `artifacts/validation/sqlite_kernel_qos_comparison/sqlite_kernel_qos_comparison.json`
- `stat_thermal_audit`: `artifacts/validation/stat_thermal_methodology/stat_thermal_methodology_audit.json`
- `thermal_tegrastats_log`: `artifacts/validation/qos_sqlite_hero_methodology/thermal_tegrastats.log`
