# Energy/Thermal Metadata Closeout

- Overall pass: `true`
- Generated: `2026-06-30T02:44:04.101264+00:00`

## Close Conditions

- `jetson_contract_complete`: `true`
- `frozen_rows_have_metadata_or_unavailable_status`: `true`
- `sqlite_hero_has_thermal_methodology`: `true`
- `kernel_qos_context_links_sqlite_closeout`: `true`
- `second_macro_has_linked_platform_metadata`: `true`
- `diagnostic_and_keyplane_rows_are_scoped_by_stat_thermal_audit`: `true`
- `paper_guard_passes`: `true`

## Frozen Filesystem Rows

- `aegisq`: status=`measured`, complete=`true`, thermal=`artifacts/validation/frozen_aegisq_contract/thermal_tegrastats.log`
- `plaintext`: status=`measured`, complete=`true`, thermal=`artifacts/validation/frozen_plaintext_contract/thermal_tegrastats.log`
- `gocryptfs`: status=`measured`, complete=`true`, thermal=`artifacts/validation/frozen_gocryptfs_contract/thermal_tegrastats.log`
- `dmcrypt`: status=`environment-blocked`, complete=`true`, thermal=`n/a`

## Headline Context

- SQLite thermal/methodology complete: `true`
- Kernel QoS context complete: `true`
- Second macro thermal metadata complete: `true`
- Diagnostic/key-plane stat-thermal audit complete: `true`

## Paper Guard

- Required phrases complete: `true`
- Unguarded forbidden hits: `0`

## Claim Boundary

- Allowed: measured filesystem headline rows carry retained platform and thermal logs
- Allowed: SQLite headline numbers are tied to repeated methodology and retained thermal logs
- Allowed: second macrobenchmark latency/throughput rows link to Jetson platform-state metadata
- Allowed: primitive and key-plane numbers remain diagnostic or progress-scoped when thermal methodology is incomplete
- Forbidden: energy-efficiency or power-efficiency claims
- Forbidden: throttling-free or fixed-clock claims without per-run proof
- Forbidden: treating diagnostic primitive results as full headline comparisons
- Forbidden: treating environment-blocked dm-crypt/fscrypt rows as measured
