# Epoch Publication Comparison

- Generated: `2026-06-28T13:30:38Z`
- Overall pass: `true`
- Workload contract: `{'repetitions': 5, 'warmup': 1, 'payload_bytes': 262144, 'operation': 'unique-file write + fdatasync + readback', 'mount_options': '-f foreground FUSE via helper', 'cache_state': 'not dropped; same mounted-path contract for both modes', 'modes': ['strict', 'epoch-redo-log'], 'group_clients': 4, 'group_wait_ns': 1000000}`

## strict
- Pass: `true`
- Throughput MiB/s: `69.59634620295977`
- Client p99 ns: `3954430.84`
- Client p99.9 ns: `3968568.1840000004`
- Commit p99 ns: `1276974.96`
- Sync count total: `10`
- Lock hold events: `159`
- Lock-order violations: `0`

## epoch_redo_log
- Pass: `true`
- Throughput MiB/s: `67.28669834022017`
- Client p99 ns: `3784076.48`
- Client p99.9 ns: `3786790.4480000003`
- Commit p99 ns: `1296818.96`
- Sync count total: `10`
- Lock hold events: `165`
- Lock-order violations: `0`

## strict_grouped
- Pass: `true`
- Throughput MiB/s: `9.217535487442495`
- Client p99 ns: `45141936.230000004`
- Client p99.9 ns: `45162368.723000005`
- Commit p99 ns: `1916834.56`
- Sync count total: `8`
- Lock hold events: `104`
- Lock-order violations: `0`
- Workload kind: `concurrent_unique_file_fdatasync`
- Client count: `4`
- Max epoch append group size: `0`
- Epoch sync primitives: `[]`

## epoch_redo_log_grouped
- Pass: `true`
- Throughput MiB/s: `9.596901628792862`
- Client p99 ns: `40910277.78`
- Client p99.9 ns: `40914019.278`
- Commit p99 ns: `39287651.44`
- Sync count total: `6`
- Lock hold events: `111`
- Lock-order violations: `0`
- Workload kind: `concurrent_unique_file_fdatasync`
- Client count: `4`
- Max epoch append group size: `3`
- Epoch sync primitives: `['fdatasync', 'none', 'syncfs']`

## Verdict

Grouped epoch redo-log amortized the metadata barrier in the concurrent mounted workload: total traced sync count is lower than strict while journal fsync remains off the foreground epoch path.

## Negative Claim Guard

This artifact is a narrow mounted-path comparison. It does not support any claim that epoch mode reduces total sync count, improves throughput, improves p99 latency, or is SOSP-ready unless the reported metrics show that result under the stated workload. A zero journal-fsync count only supports the narrower claim that strict journal publication has moved out of the foreground epoch-mode write path.
