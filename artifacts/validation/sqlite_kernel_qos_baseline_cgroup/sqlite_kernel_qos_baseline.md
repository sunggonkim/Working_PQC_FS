# SQLite Kernel QoS Baseline

- Overall pass: `true`
- Verdict: `measured`
- Control: `systemd_io_weight`
- Repetitions: `3`
- Scope: first kernel-control baseline for SQLite foreground p99 under mounted secure-storage pressure.
- Non-claim: this is not yet the full Gate B3 two-baseline comparison.

## Summary

- Foreground p99 median ms: `12.574790109999999`
- Deadline misses median: `7.0`
- Background throughput median MB/s: `6.73069491922091`
- Kernel-control successful rows: `3/3`

## Repetition Rows

| repetition | acceptable | p99 ms | deadline misses | bg MB/s |
|---:|---:|---:|---:|---:|
| 0 | `true` | 12.65173145 | 10 | 6.685623817701344 |
| 1 | `true` | 11.40027003 | 5 | 6.73069491922091 |
| 2 | `true` | 12.574790109999999 | 7 | 6.79095433316294 |

## Raw Logs

- repetition `0`: `artifacts/validation/sqlite_kernel_qos_baseline_cgroup/rep_00/kernel_systemd_io_weight/mode_summary.json`
- repetition `1`: `artifacts/validation/sqlite_kernel_qos_baseline_cgroup/rep_01/kernel_systemd_io_weight/mode_summary.json`
- repetition `2`: `artifacts/validation/sqlite_kernel_qos_baseline_cgroup/rep_02/kernel_systemd_io_weight/mode_summary.json`
