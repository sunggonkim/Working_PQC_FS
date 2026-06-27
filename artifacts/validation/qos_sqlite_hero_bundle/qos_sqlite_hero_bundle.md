# SQLite QoS Recovery Bundle

- Overall pass: `true`
- Scope: synchronized SQLite foreground QoS recovery under mounted secure-storage pressure.
- Foreground: SQLite DELETE/FULL transactions on mounted AEGIS-Q FUSE.
- Telemetry source: foreground SQLite slack plus background-storage pressure, not hardware PMU/CUPTI/TensorRT.
- Non-claim: this is not foreground AI/TensorRT p99 recovery.

## Recovery checks

- required_modes_available: `true`
- pressure_raises_p99: `true`
- pressure_causes_deadline_miss: `true`
- simple_recovers_p99: `true`
- simple_removes_deadline_misses: `true`
- aegis_recovers_p99: `true`
- aegis_removes_deadline_misses: `true`
- aegis_keeps_more_storage_than_simple: `true`
- aegis_records_throttle_decisions: `true`

## Modes

| mode | acceptable | fg p99 ms | misses | bg MB | telemetry rows | policy throttled rows | daemon throttled rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| `app_only` | `true` | 6.43596154 | 0 | 0.000 | 28 | 0 | 0 |
| `unthrottled_storage` | `true` | 13.822082770000009 | 1 | 4.562 | 31 | 0 | 0 |
| `simple_controller` | `true` | 8.185088930000001 | 0 | 1.375 | 28 | 26 | 0 |
| `aegis_policy` | `true` | 8.75331646 | 0 | 1.625 | 29 | 28 | 16 |

## Required components

- required_modes_present: `true`
- foreground_sqlite_logs: `true`
- mounted_fuse_logs: `true`
- background_secure_writer_logs: `true`
- telemetry_sampler_logs: `true`
- policy_trace_logs: `true`
- aegis_daemon_throttle_trace: `true`
- simple_controller_trace: `true`

## Raw logs

- `app_only`
  - foreground_jsonl: `artifacts/validation/qos_sqlite_hero_bundle/app_only/foreground_sqlite_latency.jsonl`
  - foreground_csv: `artifacts/validation/qos_sqlite_hero_bundle/app_only/foreground_sqlite_latency.csv`
  - background_jsonl: `artifacts/validation/qos_sqlite_hero_bundle/app_only/background_writer.jsonl`
  - telemetry_jsonl: `artifacts/validation/qos_sqlite_hero_bundle/app_only/telemetry_trace.jsonl`
  - policy_jsonl: `artifacts/validation/qos_sqlite_hero_bundle/app_only/policy_trace.jsonl`
  - runtime_telemetry: `artifacts/validation/qos_sqlite_hero_bundle/app_only/runtime_telemetry.txt`
  - runtime_fuse_admission_trace: `artifacts/validation/qos_sqlite_hero_bundle/app_only/runtime_fuse_admission_trace.jsonl`
  - runtime_fuse_throttle_trace: `artifacts/validation/qos_sqlite_hero_bundle/app_only/runtime_fuse_throttle_trace.jsonl`
  - fuse_stdout: `artifacts/validation/qos_sqlite_hero_bundle/app_only/mount_logs/pqc_fuse.stdout.txt`
  - fuse_stderr: `artifacts/validation/qos_sqlite_hero_bundle/app_only/mount_logs/pqc_fuse.stderr.txt`
- `unthrottled_storage`
  - foreground_jsonl: `artifacts/validation/qos_sqlite_hero_bundle/unthrottled_storage/foreground_sqlite_latency.jsonl`
  - foreground_csv: `artifacts/validation/qos_sqlite_hero_bundle/unthrottled_storage/foreground_sqlite_latency.csv`
  - background_jsonl: `artifacts/validation/qos_sqlite_hero_bundle/unthrottled_storage/background_writer.jsonl`
  - telemetry_jsonl: `artifacts/validation/qos_sqlite_hero_bundle/unthrottled_storage/telemetry_trace.jsonl`
  - policy_jsonl: `artifacts/validation/qos_sqlite_hero_bundle/unthrottled_storage/policy_trace.jsonl`
  - runtime_telemetry: `artifacts/validation/qos_sqlite_hero_bundle/unthrottled_storage/runtime_telemetry.txt`
  - runtime_fuse_admission_trace: `artifacts/validation/qos_sqlite_hero_bundle/unthrottled_storage/runtime_fuse_admission_trace.jsonl`
  - runtime_fuse_throttle_trace: `artifacts/validation/qos_sqlite_hero_bundle/unthrottled_storage/runtime_fuse_throttle_trace.jsonl`
  - fuse_stdout: `artifacts/validation/qos_sqlite_hero_bundle/unthrottled_storage/mount_logs/pqc_fuse.stdout.txt`
  - fuse_stderr: `artifacts/validation/qos_sqlite_hero_bundle/unthrottled_storage/mount_logs/pqc_fuse.stderr.txt`
- `simple_controller`
  - foreground_jsonl: `artifacts/validation/qos_sqlite_hero_bundle/simple_controller/foreground_sqlite_latency.jsonl`
  - foreground_csv: `artifacts/validation/qos_sqlite_hero_bundle/simple_controller/foreground_sqlite_latency.csv`
  - background_jsonl: `artifacts/validation/qos_sqlite_hero_bundle/simple_controller/background_writer.jsonl`
  - telemetry_jsonl: `artifacts/validation/qos_sqlite_hero_bundle/simple_controller/telemetry_trace.jsonl`
  - policy_jsonl: `artifacts/validation/qos_sqlite_hero_bundle/simple_controller/policy_trace.jsonl`
  - runtime_telemetry: `artifacts/validation/qos_sqlite_hero_bundle/simple_controller/runtime_telemetry.txt`
  - runtime_fuse_admission_trace: `artifacts/validation/qos_sqlite_hero_bundle/simple_controller/runtime_fuse_admission_trace.jsonl`
  - runtime_fuse_throttle_trace: `artifacts/validation/qos_sqlite_hero_bundle/simple_controller/runtime_fuse_throttle_trace.jsonl`
  - fuse_stdout: `artifacts/validation/qos_sqlite_hero_bundle/simple_controller/mount_logs/pqc_fuse.stdout.txt`
  - fuse_stderr: `artifacts/validation/qos_sqlite_hero_bundle/simple_controller/mount_logs/pqc_fuse.stderr.txt`
- `aegis_policy`
  - foreground_jsonl: `artifacts/validation/qos_sqlite_hero_bundle/aegis_policy/foreground_sqlite_latency.jsonl`
  - foreground_csv: `artifacts/validation/qos_sqlite_hero_bundle/aegis_policy/foreground_sqlite_latency.csv`
  - background_jsonl: `artifacts/validation/qos_sqlite_hero_bundle/aegis_policy/background_writer.jsonl`
  - telemetry_jsonl: `artifacts/validation/qos_sqlite_hero_bundle/aegis_policy/telemetry_trace.jsonl`
  - policy_jsonl: `artifacts/validation/qos_sqlite_hero_bundle/aegis_policy/policy_trace.jsonl`
  - runtime_telemetry: `artifacts/validation/qos_sqlite_hero_bundle/aegis_policy/runtime_telemetry.txt`
  - runtime_fuse_admission_trace: `artifacts/validation/qos_sqlite_hero_bundle/aegis_policy/runtime_fuse_admission_trace.jsonl`
  - runtime_fuse_throttle_trace: `artifacts/validation/qos_sqlite_hero_bundle/aegis_policy/runtime_fuse_throttle_trace.jsonl`
  - fuse_stdout: `artifacts/validation/qos_sqlite_hero_bundle/aegis_policy/mount_logs/pqc_fuse.stdout.txt`
  - fuse_stderr: `artifacts/validation/qos_sqlite_hero_bundle/aegis_policy/mount_logs/pqc_fuse.stderr.txt`
