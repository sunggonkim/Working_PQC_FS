# SQLite Kernel QoS Baseline

- Overall pass: `true`
- Verdict: `measured`
- Control: `ionice`
- Repetitions: `3`
- Scope: first kernel-control baseline for SQLite foreground p99 under mounted secure-storage pressure.
- Non-claim: this is not yet the full Gate B3 two-baseline comparison.

## Summary

- Foreground p99 median ms: `11.52369313`
- Deadline misses median: `3.0`
- Background throughput median MB/s: `22.525423409232523`
- Kernel-control successful rows: `3/3`

## Repetition Rows

| repetition | acceptable | p99 ms | deadline misses | bg MB/s |
|---:|---:|---:|---:|---:|
| 0 | `true` | 12.173229379999999 | 3 | 22.525423409232523 |
| 1 | `true` | 11.52369313 | 3 | 23.36153342337365 |
| 2 | `true` | 9.890588139999998 | 0 | 22.015962810099943 |

## Raw Logs

- repetition `0`: `artifacts/validation/sqlite_kernel_qos_ionice/rep_00/kernel_ionice_idle/mode_summary.json`
- repetition `1`: `artifacts/validation/sqlite_kernel_qos_ionice/rep_01/kernel_ionice_idle/mode_summary.json`
- repetition `2`: `artifacts/validation/sqlite_kernel_qos_ionice/rep_02/kernel_ionice_idle/mode_summary.json`
