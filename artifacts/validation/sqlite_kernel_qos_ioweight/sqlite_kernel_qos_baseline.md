# SQLite Kernel QoS Baseline

- Overall pass: `false`
- Verdict: `invalid-run`
- Control: `systemd_io_weight`
- Repetitions: `3`
- Scope: first kernel-control baseline for SQLite foreground p99 under mounted secure-storage pressure.
- Non-claim: this is not yet the full Gate B3 two-baseline comparison.

## Summary

- Foreground p99 median ms: `None`
- Deadline misses median: `0.0`
- Background throughput median MB/s: `0.0`
- Kernel-control successful rows: `0/0`

## Repetition Rows

| repetition | acceptable | p99 ms | deadline misses | bg MB/s |
|---:|---:|---:|---:|---:|
| 0 | `false` | None | 0 | 0.0 |
| 1 | `false` | None | 0 | 0.0 |
| 2 | `false` | None | 0 | 0.0 |

## Raw Logs

- repetition `0`: `artifacts/validation/sqlite_kernel_qos_ioweight/rep_00/kernel_systemd_io_weight/mode_summary.json`
- repetition `1`: `artifacts/validation/sqlite_kernel_qos_ioweight/rep_01/kernel_systemd_io_weight/mode_summary.json`
- repetition `2`: `artifacts/validation/sqlite_kernel_qos_ioweight/rep_02/kernel_systemd_io_weight/mode_summary.json`
