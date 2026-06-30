# A4 Hidden Overhead Trace Smoke

- Overall pass: `True`
- Scope: This is daemon-side mounted-path overhead accounting. FUSE operation latency is a daemon-side proxy for the FUSE crossing; it is not a kernel scheduler context-switch trace.

## FUSE Operation Counters

- `create` calls `1`, errors `0`, total_ns `1644658`, max_ns `1644658`
- `write` calls `1`, errors `0`, total_ns `38287`, max_ns `38287`
- `fsync` calls `1`, errors `0`, total_ns `6801778`, max_ns `6801778`
- `read` calls `1`, errors `0`, total_ns `227037`, max_ns `227037`
- `release` calls `1`, errors `0`, total_ns `492296`, max_ns `492296`
- `readdir` calls `1`, errors `0`, total_ns `19454`, max_ns `19454`

## Proof Checks

- `workload_passed`: `True`
- `fuse_trace_has_required_ops`: `True`
- `fuse_trace_has_latency`: `True`
- `plane_trace_has_data_path`: `True`
- `publication_trace_has_timing`: `True`
- `durability_has_sites`: `True`
- `fuse_operations_wrapped`: `True`
- `fuse_trace_dump_env`: `True`
- `fuse_trace_latency_counters`: `True`
- `runtime_dumps_fuse_trace`: `True`
