# Epoch Publication Comparison

- Generated: `2026-06-30T05:17:56Z`
- Overall pass: `true`
- Workload contract: `{'repetitions': 5, 'warmup': 1, 'payload_bytes': 16384, 'operation': 'unique-file write + fdatasync + readback', 'mount_options': '-f foreground FUSE via helper', 'cache_state': 'not dropped; same mounted-path contract for both modes', 'modes': ['strict', 'epoch-redo-log'], 'group_clients': 8, 'group_wait_ns': 10000000}`

## strict
- Pass: `true`
- Throughput MiB/s: `2.011975794997629`
- Client p99 ns: `14465241.08`
- Client p99.9 ns: `14467973.408`
- Commit p99 ns: `747342.88`
- Sync count total: `10`
- Lock hold events: `256`
- Lock-order violations: `0`

## epoch_redo_log
- Pass: `true`
- Throughput MiB/s: `2.08808838388347`
- Client p99 ns: `14365237.2`
- Client p99.9 ns: `14366549.22`
- Commit p99 ns: `3067015.2`
- Sync count total: `10`
- Lock hold events: `261`
- Lock-order violations: `0`

## strict_grouped
- Pass: `true`
- Throughput MiB/s: `1.5987223727024984`
- Client p99 ns: `61439279.6`
- Client p99.9 ns: `61535110.160000004`
- Commit p99 ns: `1510080.22`
- Sync count total: `16`
- Lock hold events: `338`
- Lock-order violations: `0`
- Workload kind: `concurrent_unique_file_fdatasync`
- Client count: `8`
- Max epoch append group size: `0`
- Epoch sync primitives: `[]`

## epoch_redo_log_grouped
- Pass: `true`
- Throughput MiB/s: `1.556238648873107`
- Client p99 ns: `67866167.55`
- Client p99.9 ns: `67900876.455`
- Commit p99 ns: `66394908.129999995`
- Sync count total: `1`
- Lock hold events: `373`
- Lock-order violations: `0`
- fd_lock hold p99 <= ns: `131072`
- commit_lock hold p99 <= ns: `524288`
- epoch_barrier_lock hold p99 <= ns: `524288`
- parallel_runtime_lock hold p99 <= ns: `512`
- Workload kind: `concurrent_unique_file_fdatasync`
- Client count: `8`
- Max epoch append group size: `8`
- Epoch sync primitives: `['none', 'syncfs']`

## Verdict

Grouped epoch redo-log amortized the metadata barrier in the concurrent mounted workload: total traced sync count is lower than strict while journal fsync remains off the foreground epoch path. Throughput is lower_or_equal and client p99 is higher_or_equal versus strict in this run, so claims must be scoped to the metrics that actually improve.

## Negative Claim Guard

This artifact is a narrow mounted-path comparison. It does not support any claim that epoch mode reduces total sync count, improves throughput, improves p99 latency, or is SOSP-ready unless the reported metrics show that result under the stated workload. A zero journal-fsync count only supports the narrower claim that strict journal publication has moved out of the foreground epoch-mode write path.
