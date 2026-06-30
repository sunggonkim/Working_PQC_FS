# Blocking Syscall Profile

- Overall pass: `true`
- Raw trace: `artifacts/validation/concurrency_contract/blocking_syscall_profile/strace_raw.txt`
- Syscalls observed: `close, fdatasync, openat, pread64, pwrite64`

| Syscall | Count | p50 ns | p95 ns | p99 ns | max ns | total ns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `close` | 63 | 19000 | 25000 | 48000 | 53000 | 1278000 |
| `fdatasync` | 8 | 2315000 | 5674000 | 5674000 | 5674000 | 22165000 |
| `openat` | 80 | 25000 | 42000 | 1487000 | 3886000 | 7367000 |
| `pread64` | 8 | 84000 | 120000 | 120000 | 120000 | 660999 |
| `pwrite64` | 8 | 50000 | 99000 | 99000 | 99000 | 461000 |
