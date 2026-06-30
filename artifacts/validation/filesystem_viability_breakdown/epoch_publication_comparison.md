# Epoch Publication Comparison

- Generated: `2026-06-28T13:33:11Z`
- Overall pass: `true`
- Workload contract: `{'repetitions': 5, 'warmup': 1, 'payload_bytes': 262144, 'operation': 'unique-file write + fdatasync + readback', 'mount_options': '-f foreground FUSE via helper', 'cache_state': 'not dropped; same mounted-path contract for both modes', 'modes': ['strict', 'epoch-redo-log'], 'group_clients': 4, 'group_wait_ns': 1000000}`

## strict
- Pass: `true`
- Throughput MiB/s: `74.1881721856435`
- Client p99 ns: `3739443.04`
- Client p99.9 ns: `3743360.704`
- Commit p99 ns: `1188612.6400000001`
- Sync count total: `10`
- Lock hold events: `159`
- Lock-order violations: `0`

## epoch_redo_log
- Pass: `true`
- Throughput MiB/s: `68.78409799697955`
- Client p99 ns: `3701727.48`
- Client p99.9 ns: `3703589.148`
- Commit p99 ns: `1304686.96`
- Sync count total: `10`
- Lock hold events: `162`
- Lock-order violations: `0`

## strict_grouped
- Pass: `true`
- Throughput MiB/s: `9.051860598631222`
- Client p99 ns: `47612792.88999999`
- Client p99.9 ns: `47630411.389`
- Commit p99 ns: `1562458.19`
- Sync count total: `8`
- Lock hold events: `109`
- Lock-order violations: `0`
- Workload kind: `concurrent_unique_file_fdatasync`
- Client count: `4`
- Max epoch append group size: `0`
- Epoch sync primitives: `[]`

## epoch_redo_log_grouped
- Pass: `true`
- Throughput MiB/s: `9.721212539283178`
- Client p99 ns: `40138994.73`
- Client p99.9 ns: `40180183.473000005`
- Commit p99 ns: `36222448.160000004`
- Sync count total: `6`
- Lock hold events: `110`
- Lock-order violations: `0`
- Workload kind: `concurrent_unique_file_fdatasync`
- Client count: `4`
- Max epoch append group size: `3`
- Epoch sync primitives: `['fdatasync', 'none', 'syncfs']`

## Verdict

Grouped epoch redo-log amortized the metadata barrier in the concurrent mounted workload: total traced sync count is lower than strict while journal fsync remains off the foreground epoch path. Throughput is higher and client p99 is lower versus strict in this run, so claims must be scoped to the metrics that actually improve.

## Negative Claim Guard

This artifact is a narrow mounted-path comparison. It does not support any claim that epoch mode reduces total sync count, improves throughput, improves p99 latency, or is SOSP-ready unless the reported metrics show that result under the stated workload. A zero journal-fsync count only supports the narrower claim that strict journal publication has moved out of the foreground epoch-mode write path.
