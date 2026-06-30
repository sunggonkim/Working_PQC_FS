# Epoch Publication Comparison

- Generated: `2026-06-28T06:59:21Z`
- Overall pass: `true`
- Workload contract: `{'repetitions': 5, 'warmup': 1, 'payload_bytes': 16384, 'operation': 'unique-file write + fdatasync + readback', 'mount_options': '-f foreground FUSE via helper', 'cache_state': 'not dropped; same mounted-path contract for both modes', 'modes': ['strict', 'epoch-redo-log'], 'group_clients': 4, 'group_wait_ns': 100000000}`

## strict
- Pass: `true`
- Throughput MiB/s: `12.959676619366542`
- Client p99 ns: `1359286.32`
- Client p99.9 ns: `1363045.332`
- Commit p99 ns: `787508.88`
- Sync count total: `10`

## epoch_redo_log
- Pass: `true`
- Throughput MiB/s: `10.880134274435354`
- Client p99 ns: `1552775.52`
- Client p99.9 ns: `1555177.152`
- Commit p99 ns: `857367.52`
- Sync count total: `10`

## strict_grouped
- Pass: `true`
- Throughput MiB/s: `6.4461233014465105`
- Client p99 ns: `2669886.0199999996`
- Client p99.9 ns: `2687130.002`
- Commit p99 ns: `1401824.6`
- Sync count total: `8`
- Workload kind: `concurrent_unique_file_fdatasync`
- Client count: `4`
- Max epoch append group size: `0`
- Epoch sync primitives: `[]`

## epoch_redo_log_grouped
- Pass: `true`
- Throughput MiB/s: `1.0764902470241977`
- Client p99 ns: `53136288.72`
- Client p99.9 ns: `53149280.472`
- Commit p99 ns: `51631281.49`
- Sync count total: `5`
- Workload kind: `concurrent_unique_file_fdatasync`
- Client count: `4`
- Max epoch append group size: `4`
- Epoch sync primitives: `['none', 'syncfs']`

## Verdict

Grouped epoch redo-log amortized the metadata barrier in the concurrent mounted workload: total traced sync count is lower than strict while journal fsync remains off the foreground epoch path.

## Negative Claim Guard

This artifact is a narrow mounted-path comparison. It does not support any claim that epoch mode reduces total sync count, improves throughput, improves p99 latency, or is SOSP-ready unless the reported metrics show that result under the stated workload. A zero journal-fsync count only supports the narrower claim that strict journal publication has moved out of the foreground epoch-mode write path.
