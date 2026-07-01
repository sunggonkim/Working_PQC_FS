# A4 Hidden Overhead Trace Smoke

- Overall pass: `True`
- Scope: This is daemon-side mounted-path overhead accounting. FUSE operation latency is a daemon-side proxy for the FUSE crossing; it is not a kernel scheduler context-switch trace.

## FUSE Operation Counters

- `create` calls `1`, errors `0`, total_ns `319130`, max_ns `319130`
- `write` calls `1`, errors `0`, total_ns `45871`, max_ns `45871`
- `fsync` calls `1`, errors `0`, total_ns `113655141`, max_ns `113655141`
- `read` calls `1`, errors `0`, total_ns `228268`, max_ns `228268`
- `release` calls `1`, errors `0`, total_ns `789324`, max_ns `789324`
- `readdir` calls `1`, errors `0`, total_ns `14046`, max_ns `14046`

## Publication Trace Counters

- Publication events: `28`
- Publication elapsed_ns total: `37573287`
- Publication sync_count total: `56`
- Data fdatasync count: `28`
- Journal fdatasync count: `28`
- Epoch-log fdatasync count: `0`

## Mounted Durability Counters

- fdatasync: `2`
- fsync: `1`
- syncfs: `0`
- data_sidecar: `1`
- journal_sidecar: `1`
- marker_metadata: `1`
- failures: `0`

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
