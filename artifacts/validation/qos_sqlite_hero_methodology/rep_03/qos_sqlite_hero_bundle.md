# SQLite QoS Recovery Bundle

- Overall pass: `false`
- Scope: synchronized SQLite foreground QoS recovery under mounted secure-storage pressure.
- Foreground: SQLite DELETE/FULL transactions on mounted AEGIS-Q FUSE.
- Telemetry source: foreground SQLite slack plus background-storage pressure, not hardware PMU/CUPTI/TensorRT.
- Non-claim: this is not foreground AI/TensorRT p99 recovery.

## Recovery checks

- required_modes_available: `true`
- pressure_raises_p99: `false`
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
| `app_only` | `true` | 11.361421180000008 | 1 | 0.000 | 30 | 0 | 0 |
| `unthrottled_storage` | `true` | 11.261585520000004 | 1 | 4.312 | 30 | 0 | 0 |
| `simple_controller` | `true` | 7.824026110000002 | 0 | 0.625 | 28 | 27 | 0 |
| `aegis_policy` | `true` | 7.04147598 | 0 | 1.750 | 29 | 27 | 15 |

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
  - foreground_jsonl: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/app_only/foreground_sqlite_latency.jsonl`
  - foreground_csv: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/app_only/foreground_sqlite_latency.csv`
  - background_jsonl: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/app_only/background_writer.jsonl`
  - telemetry_jsonl: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/app_only/telemetry_trace.jsonl`
  - policy_jsonl: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/app_only/policy_trace.jsonl`
  - runtime_telemetry: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/app_only/runtime_telemetry.txt`
  - runtime_fuse_admission_trace: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/app_only/runtime_fuse_admission_trace.jsonl`
  - runtime_fuse_throttle_trace: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/app_only/runtime_fuse_throttle_trace.jsonl`
  - fuse_stdout: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/app_only/mount_logs/pqc_fuse.stdout.txt`
  - fuse_stderr: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/app_only/mount_logs/pqc_fuse.stderr.txt`
- `unthrottled_storage`
  - foreground_jsonl: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/unthrottled_storage/foreground_sqlite_latency.jsonl`
  - foreground_csv: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/unthrottled_storage/foreground_sqlite_latency.csv`
  - background_jsonl: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/unthrottled_storage/background_writer.jsonl`
  - telemetry_jsonl: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/unthrottled_storage/telemetry_trace.jsonl`
  - policy_jsonl: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/unthrottled_storage/policy_trace.jsonl`
  - runtime_telemetry: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/unthrottled_storage/runtime_telemetry.txt`
  - runtime_fuse_admission_trace: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/unthrottled_storage/runtime_fuse_admission_trace.jsonl`
  - runtime_fuse_throttle_trace: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/unthrottled_storage/runtime_fuse_throttle_trace.jsonl`
  - fuse_stdout: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/unthrottled_storage/mount_logs/pqc_fuse.stdout.txt`
  - fuse_stderr: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/unthrottled_storage/mount_logs/pqc_fuse.stderr.txt`
- `simple_controller`
  - foreground_jsonl: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/simple_controller/foreground_sqlite_latency.jsonl`
  - foreground_csv: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/simple_controller/foreground_sqlite_latency.csv`
  - background_jsonl: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/simple_controller/background_writer.jsonl`
  - telemetry_jsonl: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/simple_controller/telemetry_trace.jsonl`
  - policy_jsonl: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/simple_controller/policy_trace.jsonl`
  - runtime_telemetry: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/simple_controller/runtime_telemetry.txt`
  - runtime_fuse_admission_trace: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/simple_controller/runtime_fuse_admission_trace.jsonl`
  - runtime_fuse_throttle_trace: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/simple_controller/runtime_fuse_throttle_trace.jsonl`
  - fuse_stdout: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/simple_controller/mount_logs/pqc_fuse.stdout.txt`
  - fuse_stderr: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/simple_controller/mount_logs/pqc_fuse.stderr.txt`
- `aegis_policy`
  - foreground_jsonl: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/aegis_policy/foreground_sqlite_latency.jsonl`
  - foreground_csv: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/aegis_policy/foreground_sqlite_latency.csv`
  - background_jsonl: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/aegis_policy/background_writer.jsonl`
  - telemetry_jsonl: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/aegis_policy/telemetry_trace.jsonl`
  - policy_jsonl: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/aegis_policy/policy_trace.jsonl`
  - runtime_telemetry: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/aegis_policy/runtime_telemetry.txt`
  - runtime_fuse_admission_trace: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/aegis_policy/runtime_fuse_admission_trace.jsonl`
  - runtime_fuse_throttle_trace: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/aegis_policy/runtime_fuse_throttle_trace.jsonl`
  - fuse_stdout: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/aegis_policy/mount_logs/pqc_fuse.stdout.txt`
  - fuse_stderr: `artifacts/validation/qos_sqlite_hero_methodology/rep_03/aegis_policy/mount_logs/pqc_fuse.stderr.txt`
