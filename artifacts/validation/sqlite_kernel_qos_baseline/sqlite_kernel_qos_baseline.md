# SQLite Kernel QoS Baseline

- Overall pass: `true`
- Verdict: `measured`
- Control: `ionice`
- Repetitions: `3`
- Scope: first kernel-control baseline for SQLite foreground p99 under mounted secure-storage pressure.
- Non-claim: this is not yet the full Gate B3 two-baseline comparison.

## Summary

- Foreground p99 median ms: `16.217910019999998`
- Deadline misses median: `6.0`
- Background throughput median MB/s: `7.248431018854942`
- Kernel-control successful rows: `3/3`

## Repetition Rows

| repetition | acceptable | p99 ms | deadline misses | bg MB/s |
|---:|---:|---:|---:|---:|
| 0 | `true` | 16.537148539999997 | 17 | 7.0084336827733695 |
| 1 | `true` | 11.325034330000001 | 5 | 7.36872842829315 |
| 2 | `true` | 16.217910019999998 | 6 | 7.248431018854942 |

## Raw Logs

- repetition `0`: `artifacts/validation/sqlite_kernel_qos_baseline/rep_00/kernel_ionice_idle/mode_summary.json`
- repetition `1`: `artifacts/validation/sqlite_kernel_qos_baseline/rep_01/kernel_ionice_idle/mode_summary.json`
- repetition `2`: `artifacts/validation/sqlite_kernel_qos_baseline/rep_02/kernel_ionice_idle/mode_summary.json`
