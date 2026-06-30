# SQLite Kernel QoS Comparison

- Overall pass: `true`
- B3 code/artifact ready: `true`
- Parent B3 gate closed: `true`
- Kernel baselines measured: `true`
- Hero modes available: `true`

## Claim Guard

- SQLite p99 uniqueness claim allowed: `true`
- Paper mentions ionice: `true`
- Paper mentions systemd IOWeight: `true`
- Paper states repetition mismatch: `true`
- Paper states no-uniqueness boundary: `true`
- Paper SQLite/p99 candidate lines: `11`

## Comparison Rows

| row | type | p99 ms median | deadline misses median | bg MB/s median | runs |
|---|---|---:|---:|---:|---:|
| `app_only` | `hero_mode` | 7.253253040000001 | 0.0 | 0.0 | 5 |
| `unthrottled_storage` | `hero_mode` | 9.62087615 | 0.0 | 6.984360530500289 | 5 |
| `simple_controller` | `hero_mode` | 7.544277000000001 | 0.0 | 1.4970013728032954 | 5 |
| `aegis_policy` | `hero_mode` | 8.15053043 | 0.0 | 3.0160640171203785 | 5 |
| `kernel_ionice_idle` | `kernel_baseline` | 16.217910019999998 | 6.0 | 7.248431018854942 | 3 |
| `kernel_systemd_io_weight` | `kernel_baseline` | 12.574790109999999 | 7.0 | 6.73069491922091 | 3 |

## Conservative Boundary

- The kernel-control rows currently have retained repetitions shown in the comparison table.
- The repeated hero artifact and kernel baselines may still have different repetition counts and foreground sample counts, so paper text must avoid broad superiority wording.
- The table includes both kernel controls, but no broad SQLite p99 uniqueness claim is allowed without broader platform and workload coverage.
