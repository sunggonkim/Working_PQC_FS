# SQLite QoS Methodology Bundle

- Overall pass: `false`
- Measured repetitions: `5`
- Warmup runs: `1`
- Scope: repeated mounted SQLite/FUSE QoS workflow methodology evidence.
- Non-claim: this is not foreground AI/TensorRT p99 recovery.

## Mode Summaries

- `app_only` runs=`5`, acceptable_runs=`5`, p99_median_ms=`7.253`, p99_ci95_ms=`[6.243, 11.361]`, storage_median_mb_s=`0.000`
- `unthrottled_storage` runs=`5`, acceptable_runs=`5`, p99_median_ms=`9.621`, p99_ci95_ms=`[8.270, 11.262]`, storage_median_mb_s=`6.984`
- `simple_controller` runs=`5`, acceptable_runs=`5`, p99_median_ms=`7.544`, p99_ci95_ms=`[7.033, 8.252]`, storage_median_mb_s=`1.497`
- `aegis_policy` runs=`5`, acceptable_runs=`5`, p99_median_ms=`8.151`, p99_ci95_ms=`[7.041, 10.001]`, storage_median_mb_s=`3.016`

## Recovery Checks

- `aegis_keeps_more_storage_than_simple`: true_runs=`5/5`, all_true=`true`
- `aegis_records_throttle_decisions`: true_runs=`5/5`, all_true=`true`
- `aegis_recovers_p99`: true_runs=`4/5`, all_true=`false`
- `aegis_removes_deadline_misses`: true_runs=`3/5`, all_true=`false`
- `pressure_causes_deadline_miss`: true_runs=`2/5`, all_true=`false`
- `pressure_raises_p99`: true_runs=`3/5`, all_true=`false`
- `required_modes_available`: true_runs=`5/5`, all_true=`true`
- `simple_recovers_p99`: true_runs=`5/5`, all_true=`true`
- `simple_removes_deadline_misses`: true_runs=`5/5`, all_true=`true`

## Component Coverage

- `aegis_daemon_throttle_trace`: true_runs=`5/5`, all_true=`true`
- `background_secure_writer_logs`: true_runs=`5/5`, all_true=`true`
- `foreground_sqlite_logs`: true_runs=`5/5`, all_true=`true`
- `mounted_fuse_logs`: true_runs=`5/5`, all_true=`true`
- `policy_trace_logs`: true_runs=`5/5`, all_true=`true`
- `required_modes_present`: true_runs=`5/5`, all_true=`true`
- `simple_controller_trace`: true_runs=`5/5`, all_true=`true`
- `telemetry_sampler_logs`: true_runs=`5/5`, all_true=`true`

## Methodology Metadata

- Run count meets headline minimum: `true`
- Full workload warmup retained: `true`
- CPU governor ready: `false`
- Thermal log nonempty: `true`
