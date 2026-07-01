# A4 Hidden Overhead Trace Smoke

- Overall pass: `True`
- Scope: This is daemon-side mounted-path overhead accounting. FUSE operation latency is a daemon-side proxy for the FUSE crossing; it is not a kernel scheduler context-switch trace.

## FUSE Operation Counters

- `create` calls `1`, errors `0`, total_ns `295945`, max_ns `295945`
- `write` calls `1`, errors `0`, total_ns `31389`, max_ns `31389`
- `fsync` calls `1`, errors `0`, total_ns `110509441`, max_ns `110509441`
- `read` calls `1`, errors `0`, total_ns `237648`, max_ns `237648`
- `release` calls `1`, errors `0`, total_ns `789509`, max_ns `789509`
- `readdir` calls `1`, errors `0`, total_ns `27028`, max_ns `27028`

## Publication Trace Counters

- Publication events: `27`
- Publication elapsed_ns total: `36355851`
- Publication sync_count total: `54`
- Data fdatasync count: `27`
- Journal fdatasync count: `27`
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
