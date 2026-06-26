# QoS mounted-FUSE live bridge

This bundle writes through a real FUSE mount while sampling live `tegrastats`.
Each live sample is fed into `pqc_fuse --admission-telemetry-smoke`, and the same sample drives a harness-level hysteresis throttle for the mounted writer.
This is stronger than a pure smoke path because real FUSE work is slowed in the same execution, but it is still not a PMU/CUPTI/Nsight-backed in-daemon controller proof.

- Samples recorded: `6` / requested `6`
- GPU burner started: `True`
- Admission decision counts: `{'CPU': 6}`
- Throttle state counts: `{'open': 0, 'throttled': 6}`
- Writer stats: `{'chunks_written': 48, 'bytes_written': 6291456, 'throttle_sleeps': 26, 'sleep_time_s': 1.3000000000000005, 'elapsed_s': 2.1980561101809144, 'throughput_mb_s': 2.729684639172455}`
- FUSE stderr: `artifacts/validation/qos_fuse_live_bridge/mount_logs/pqc_fuse.stderr.txt`

| sample | gpu_power_mw | gr3d_percent | mem_util | tensor_util | admission_target | throttle_state | trace |
|---:|---:|---:|---:|---:|---|---|---|
| 1 | 28081 | None | 0.8023 | 0.0000 | CPU | throttled | `artifacts/validation/qos_fuse_live_bridge/sample_01.jsonl` |
| 2 | 30455 | None | 0.8701 | 0.0000 | CPU | throttled | `artifacts/validation/qos_fuse_live_bridge/sample_02.jsonl` |
| 3 | 31641 | None | 0.9040 | 0.0000 | CPU | throttled | `artifacts/validation/qos_fuse_live_bridge/sample_03.jsonl` |
| 4 | 31641 | None | 0.9040 | 0.0000 | CPU | throttled | `artifacts/validation/qos_fuse_live_bridge/sample_04.jsonl` |
| 5 | 32432 | None | 0.9266 | 0.0000 | CPU | throttled | `artifacts/validation/qos_fuse_live_bridge/sample_05.jsonl` |
| 6 | 32828 | None | 0.9379 | 0.0000 | CPU | throttled | `artifacts/validation/qos_fuse_live_bridge/sample_06.jsonl` |
