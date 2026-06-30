# Second Macrobenchmark Closeout

- Overall pass: `true`
- Generated: `2026-06-30T08:20:44.984845+00:00`

## Close Conditions

- `runner_exists`: `true`
- `fuse_binary_exists`: `true`
- `strict_and_epoch_runs_present`: `true`
- `all_macro_runs_pass`: `true`
- `raw_latency_throughput_retained`: `true`
- `durability_recovery_retained`: `true`
- `process_resource_usage_retained`: `true`
- `thermal_metadata_linked`: `true`
- `paper_scope_guard_passes`: `true`

## Runs

- `strict`: pass=True, records=4, p99_us=6587.416, payload_mib_s=3.8171176779514306, raw=`artifacts/validation/second_macrobenchmark_closeout/strict_raw.json`
- `epoch-redo-log`: pass=True, records=2, p99_us=4403.1759999999995, payload_mib_s=3.3858550784684103, raw=`artifacts/validation/second_macrobenchmark_closeout/epoch_redo_log_raw.json`

## Claim Boundary

- Allowed: secure inference logging exists as a second mounted macrobenchmark smoke
- Allowed: strict and epoch-redo-log modes append, sync, read back, remount, and verify ordered records
- Allowed: latency, throughput, process resource usage, and linked Jetson thermal metadata are retained
- Forbidden: broad workload diversity
- Forbidden: SOSP/OSDI readiness from the second macrobenchmark alone
- Forbidden: foreground AI QoS recovery
- Forbidden: general application compatibility or database certification
