# QoS Sensitivity Analysis

- Overall pass: `true`
- Scope: mounted SQLite AEGIS-Q policy sensitivity plus final-binary no-slack admission fallback.
- Non-claim: not TensorRT/AI p99 recovery and not a statistical confidence study.

## Checks

- all_cases_acceptable: `true`
- required_variables_covered: `true`
- no_throttle_fallback: `true`
- no_slack_mounted_zero_budget: `true`
- no_slack_admission_cpu_fallback: `true`
- hysteresis_enter_exit_trace: `true`
- oscillation_count_recorded: `true`
- daemon_trace_retained: `true`

## Cases

| case | variable | p99 ms | misses | MB/s | daemon thr. | transitions | osc. |
|---|---|---:|---:|---:|---:|---:|---:|
| `baseline` | default | 6.432 | 0 | 3.832 | 13 | 1 | 0 |
| `tight_budget` | budget | 7.788 | 1 | 3.752 | 13 | 1 | 0 |
| `no_slack_mounted` | budget | 7.599 | 40 | 3.696 | 13 | 1 | 0 |
| `slow_sampling` | sampling_interval | 6.971 | 0 | 4.240 | 12 | 1 | 0 |
| `high_threshold` | controller_threshold | 8.832 | 0 | 7.236 | 0 | 0 | 0 |
| `queue_depth_2` | queue_depth | 12.311 | 1 | 6.463 | 27 | 1 | 0 |
| `background_128k` | background_intensity | 7.218 | 0 | 4.997 | 11 | 1 | 0 |
| `low_pressure_no_throttle` | no_throttle_fallback | 7.372 | 0 | 7.600 | 0 | 0 | 0 |
| `hysteresis_wave` | hysteresis | 6.952 | 0 | 3.754 | 13 | 9 | 8 |

## Raw Logs

- `baseline`
  - foreground_jsonl: `artifacts/validation/qos_sensitivity_analysis/baseline/aegis_policy/foreground_sqlite_latency.jsonl`
  - foreground_csv: `artifacts/validation/qos_sensitivity_analysis/baseline/aegis_policy/foreground_sqlite_latency.csv`
  - background_jsonl: `artifacts/validation/qos_sensitivity_analysis/baseline/aegis_policy/background_writer.jsonl`
  - telemetry_jsonl: `artifacts/validation/qos_sensitivity_analysis/baseline/aegis_policy/telemetry_trace.jsonl`
  - policy_jsonl: `artifacts/validation/qos_sensitivity_analysis/baseline/aegis_policy/policy_trace.jsonl`
  - runtime_telemetry: `artifacts/validation/qos_sensitivity_analysis/baseline/aegis_policy/runtime_telemetry.txt`
  - runtime_fuse_admission_trace: `artifacts/validation/qos_sensitivity_analysis/baseline/aegis_policy/runtime_fuse_admission_trace.jsonl`
  - runtime_fuse_throttle_trace: `artifacts/validation/qos_sensitivity_analysis/baseline/aegis_policy/runtime_fuse_throttle_trace.jsonl`
  - fuse_stdout: `artifacts/validation/qos_sensitivity_analysis/baseline/aegis_policy/mount_logs/pqc_fuse.stdout.txt`
  - fuse_stderr: `artifacts/validation/qos_sensitivity_analysis/baseline/aegis_policy/mount_logs/pqc_fuse.stderr.txt`
- `tight_budget`
  - foreground_jsonl: `artifacts/validation/qos_sensitivity_analysis/tight_budget/aegis_policy/foreground_sqlite_latency.jsonl`
  - foreground_csv: `artifacts/validation/qos_sensitivity_analysis/tight_budget/aegis_policy/foreground_sqlite_latency.csv`
  - background_jsonl: `artifacts/validation/qos_sensitivity_analysis/tight_budget/aegis_policy/background_writer.jsonl`
  - telemetry_jsonl: `artifacts/validation/qos_sensitivity_analysis/tight_budget/aegis_policy/telemetry_trace.jsonl`
  - policy_jsonl: `artifacts/validation/qos_sensitivity_analysis/tight_budget/aegis_policy/policy_trace.jsonl`
  - runtime_telemetry: `artifacts/validation/qos_sensitivity_analysis/tight_budget/aegis_policy/runtime_telemetry.txt`
  - runtime_fuse_admission_trace: `artifacts/validation/qos_sensitivity_analysis/tight_budget/aegis_policy/runtime_fuse_admission_trace.jsonl`
  - runtime_fuse_throttle_trace: `artifacts/validation/qos_sensitivity_analysis/tight_budget/aegis_policy/runtime_fuse_throttle_trace.jsonl`
  - fuse_stdout: `artifacts/validation/qos_sensitivity_analysis/tight_budget/aegis_policy/mount_logs/pqc_fuse.stdout.txt`
  - fuse_stderr: `artifacts/validation/qos_sensitivity_analysis/tight_budget/aegis_policy/mount_logs/pqc_fuse.stderr.txt`
- `no_slack_mounted`
  - foreground_jsonl: `artifacts/validation/qos_sensitivity_analysis/no_slack_mounted/aegis_policy/foreground_sqlite_latency.jsonl`
  - foreground_csv: `artifacts/validation/qos_sensitivity_analysis/no_slack_mounted/aegis_policy/foreground_sqlite_latency.csv`
  - background_jsonl: `artifacts/validation/qos_sensitivity_analysis/no_slack_mounted/aegis_policy/background_writer.jsonl`
  - telemetry_jsonl: `artifacts/validation/qos_sensitivity_analysis/no_slack_mounted/aegis_policy/telemetry_trace.jsonl`
  - policy_jsonl: `artifacts/validation/qos_sensitivity_analysis/no_slack_mounted/aegis_policy/policy_trace.jsonl`
  - runtime_telemetry: `artifacts/validation/qos_sensitivity_analysis/no_slack_mounted/aegis_policy/runtime_telemetry.txt`
  - runtime_fuse_admission_trace: `artifacts/validation/qos_sensitivity_analysis/no_slack_mounted/aegis_policy/runtime_fuse_admission_trace.jsonl`
  - runtime_fuse_throttle_trace: `artifacts/validation/qos_sensitivity_analysis/no_slack_mounted/aegis_policy/runtime_fuse_throttle_trace.jsonl`
  - fuse_stdout: `artifacts/validation/qos_sensitivity_analysis/no_slack_mounted/aegis_policy/mount_logs/pqc_fuse.stdout.txt`
  - fuse_stderr: `artifacts/validation/qos_sensitivity_analysis/no_slack_mounted/aegis_policy/mount_logs/pqc_fuse.stderr.txt`
- `slow_sampling`
  - foreground_jsonl: `artifacts/validation/qos_sensitivity_analysis/slow_sampling/aegis_policy/foreground_sqlite_latency.jsonl`
  - foreground_csv: `artifacts/validation/qos_sensitivity_analysis/slow_sampling/aegis_policy/foreground_sqlite_latency.csv`
  - background_jsonl: `artifacts/validation/qos_sensitivity_analysis/slow_sampling/aegis_policy/background_writer.jsonl`
  - telemetry_jsonl: `artifacts/validation/qos_sensitivity_analysis/slow_sampling/aegis_policy/telemetry_trace.jsonl`
  - policy_jsonl: `artifacts/validation/qos_sensitivity_analysis/slow_sampling/aegis_policy/policy_trace.jsonl`
  - runtime_telemetry: `artifacts/validation/qos_sensitivity_analysis/slow_sampling/aegis_policy/runtime_telemetry.txt`
  - runtime_fuse_admission_trace: `artifacts/validation/qos_sensitivity_analysis/slow_sampling/aegis_policy/runtime_fuse_admission_trace.jsonl`
  - runtime_fuse_throttle_trace: `artifacts/validation/qos_sensitivity_analysis/slow_sampling/aegis_policy/runtime_fuse_throttle_trace.jsonl`
  - fuse_stdout: `artifacts/validation/qos_sensitivity_analysis/slow_sampling/aegis_policy/mount_logs/pqc_fuse.stdout.txt`
  - fuse_stderr: `artifacts/validation/qos_sensitivity_analysis/slow_sampling/aegis_policy/mount_logs/pqc_fuse.stderr.txt`
- `high_threshold`
  - foreground_jsonl: `artifacts/validation/qos_sensitivity_analysis/high_threshold/aegis_policy/foreground_sqlite_latency.jsonl`
  - foreground_csv: `artifacts/validation/qos_sensitivity_analysis/high_threshold/aegis_policy/foreground_sqlite_latency.csv`
  - background_jsonl: `artifacts/validation/qos_sensitivity_analysis/high_threshold/aegis_policy/background_writer.jsonl`
  - telemetry_jsonl: `artifacts/validation/qos_sensitivity_analysis/high_threshold/aegis_policy/telemetry_trace.jsonl`
  - policy_jsonl: `artifacts/validation/qos_sensitivity_analysis/high_threshold/aegis_policy/policy_trace.jsonl`
  - runtime_telemetry: `artifacts/validation/qos_sensitivity_analysis/high_threshold/aegis_policy/runtime_telemetry.txt`
  - runtime_fuse_admission_trace: `artifacts/validation/qos_sensitivity_analysis/high_threshold/aegis_policy/runtime_fuse_admission_trace.jsonl`
  - runtime_fuse_throttle_trace: `artifacts/validation/qos_sensitivity_analysis/high_threshold/aegis_policy/runtime_fuse_throttle_trace.jsonl`
  - fuse_stdout: `artifacts/validation/qos_sensitivity_analysis/high_threshold/aegis_policy/mount_logs/pqc_fuse.stdout.txt`
  - fuse_stderr: `artifacts/validation/qos_sensitivity_analysis/high_threshold/aegis_policy/mount_logs/pqc_fuse.stderr.txt`
- `queue_depth_2`
  - foreground_jsonl: `artifacts/validation/qos_sensitivity_analysis/queue_depth_2/aegis_policy/foreground_sqlite_latency.jsonl`
  - foreground_csv: `artifacts/validation/qos_sensitivity_analysis/queue_depth_2/aegis_policy/foreground_sqlite_latency.csv`
  - background_jsonl: `artifacts/validation/qos_sensitivity_analysis/queue_depth_2/aegis_policy/background_writer.jsonl`
  - telemetry_jsonl: `artifacts/validation/qos_sensitivity_analysis/queue_depth_2/aegis_policy/telemetry_trace.jsonl`
  - policy_jsonl: `artifacts/validation/qos_sensitivity_analysis/queue_depth_2/aegis_policy/policy_trace.jsonl`
  - runtime_telemetry: `artifacts/validation/qos_sensitivity_analysis/queue_depth_2/aegis_policy/runtime_telemetry.txt`
  - runtime_fuse_admission_trace: `artifacts/validation/qos_sensitivity_analysis/queue_depth_2/aegis_policy/runtime_fuse_admission_trace.jsonl`
  - runtime_fuse_throttle_trace: `artifacts/validation/qos_sensitivity_analysis/queue_depth_2/aegis_policy/runtime_fuse_throttle_trace.jsonl`
  - fuse_stdout: `artifacts/validation/qos_sensitivity_analysis/queue_depth_2/aegis_policy/mount_logs/pqc_fuse.stdout.txt`
  - fuse_stderr: `artifacts/validation/qos_sensitivity_analysis/queue_depth_2/aegis_policy/mount_logs/pqc_fuse.stderr.txt`
- `background_128k`
  - foreground_jsonl: `artifacts/validation/qos_sensitivity_analysis/background_128k/aegis_policy/foreground_sqlite_latency.jsonl`
  - foreground_csv: `artifacts/validation/qos_sensitivity_analysis/background_128k/aegis_policy/foreground_sqlite_latency.csv`
  - background_jsonl: `artifacts/validation/qos_sensitivity_analysis/background_128k/aegis_policy/background_writer.jsonl`
  - telemetry_jsonl: `artifacts/validation/qos_sensitivity_analysis/background_128k/aegis_policy/telemetry_trace.jsonl`
  - policy_jsonl: `artifacts/validation/qos_sensitivity_analysis/background_128k/aegis_policy/policy_trace.jsonl`
  - runtime_telemetry: `artifacts/validation/qos_sensitivity_analysis/background_128k/aegis_policy/runtime_telemetry.txt`
  - runtime_fuse_admission_trace: `artifacts/validation/qos_sensitivity_analysis/background_128k/aegis_policy/runtime_fuse_admission_trace.jsonl`
  - runtime_fuse_throttle_trace: `artifacts/validation/qos_sensitivity_analysis/background_128k/aegis_policy/runtime_fuse_throttle_trace.jsonl`
  - fuse_stdout: `artifacts/validation/qos_sensitivity_analysis/background_128k/aegis_policy/mount_logs/pqc_fuse.stdout.txt`
  - fuse_stderr: `artifacts/validation/qos_sensitivity_analysis/background_128k/aegis_policy/mount_logs/pqc_fuse.stderr.txt`
- `low_pressure_no_throttle`
  - foreground_jsonl: `artifacts/validation/qos_sensitivity_analysis/low_pressure_no_throttle/aegis_policy/foreground_sqlite_latency.jsonl`
  - foreground_csv: `artifacts/validation/qos_sensitivity_analysis/low_pressure_no_throttle/aegis_policy/foreground_sqlite_latency.csv`
  - background_jsonl: `artifacts/validation/qos_sensitivity_analysis/low_pressure_no_throttle/aegis_policy/background_writer.jsonl`
  - telemetry_jsonl: `artifacts/validation/qos_sensitivity_analysis/low_pressure_no_throttle/aegis_policy/telemetry_trace.jsonl`
  - policy_jsonl: `artifacts/validation/qos_sensitivity_analysis/low_pressure_no_throttle/aegis_policy/policy_trace.jsonl`
  - runtime_telemetry: `artifacts/validation/qos_sensitivity_analysis/low_pressure_no_throttle/aegis_policy/runtime_telemetry.txt`
  - runtime_fuse_admission_trace: `artifacts/validation/qos_sensitivity_analysis/low_pressure_no_throttle/aegis_policy/runtime_fuse_admission_trace.jsonl`
  - runtime_fuse_throttle_trace: `artifacts/validation/qos_sensitivity_analysis/low_pressure_no_throttle/aegis_policy/runtime_fuse_throttle_trace.jsonl`
  - fuse_stdout: `artifacts/validation/qos_sensitivity_analysis/low_pressure_no_throttle/aegis_policy/mount_logs/pqc_fuse.stdout.txt`
  - fuse_stderr: `artifacts/validation/qos_sensitivity_analysis/low_pressure_no_throttle/aegis_policy/mount_logs/pqc_fuse.stderr.txt`
- `hysteresis_wave`
  - foreground_jsonl: `artifacts/validation/qos_sensitivity_analysis/hysteresis_wave/aegis_policy/foreground_sqlite_latency.jsonl`
  - foreground_csv: `artifacts/validation/qos_sensitivity_analysis/hysteresis_wave/aegis_policy/foreground_sqlite_latency.csv`
  - background_jsonl: `artifacts/validation/qos_sensitivity_analysis/hysteresis_wave/aegis_policy/background_writer.jsonl`
  - telemetry_jsonl: `artifacts/validation/qos_sensitivity_analysis/hysteresis_wave/aegis_policy/telemetry_trace.jsonl`
  - policy_jsonl: `artifacts/validation/qos_sensitivity_analysis/hysteresis_wave/aegis_policy/policy_trace.jsonl`
  - runtime_telemetry: `artifacts/validation/qos_sensitivity_analysis/hysteresis_wave/aegis_policy/runtime_telemetry.txt`
  - runtime_fuse_admission_trace: `artifacts/validation/qos_sensitivity_analysis/hysteresis_wave/aegis_policy/runtime_fuse_admission_trace.jsonl`
  - runtime_fuse_throttle_trace: `artifacts/validation/qos_sensitivity_analysis/hysteresis_wave/aegis_policy/runtime_fuse_throttle_trace.jsonl`
  - fuse_stdout: `artifacts/validation/qos_sensitivity_analysis/hysteresis_wave/aegis_policy/mount_logs/pqc_fuse.stdout.txt`
  - fuse_stderr: `artifacts/validation/qos_sensitivity_analysis/hysteresis_wave/aegis_policy/mount_logs/pqc_fuse.stderr.txt`
- `admission_no_slack`
  - stdout: `artifacts/validation/qos_sensitivity_analysis/admission_no_slack.stdout.txt`
  - stderr: `artifacts/validation/qos_sensitivity_analysis/admission_no_slack.stderr.txt`
  - trace: `artifacts/validation/qos_sensitivity_analysis/admission_no_slack_trace.jsonl`
