# QoS mounted-FUSE live bridge

This bundle writes through a real FUSE mount while sampling live `tegrastats`.
The mounted daemon reads the live sample stream through `PQC_TELEMETRY_FILE`; the real write flush path applies telemetry-derived throttle and writes `runtime_fuse_throttle_trace.jsonl`.
The DATA plane is structurally CPU-only in this codebase, so `runtime_fuse_admission_trace.jsonl` can remain empty without invalidating the throttle result.
The per-sample smoke process is retained only as a diagnostic replay of the same telemetry values.
This is stronger than a pure smoke path because real FUSE flushes are delayed inside the mounted daemon, but it is still not a PMU/CUPTI/Nsight-backed controller proof.

- Samples recorded: `8` / requested `8`
- GPU burner started: `True`
- Smoke admission decision counts: `{'CPU': 8}`
- Runtime FUSE admission decision counts: `{}`
- Runtime low-pressure targets: `{}`
- Runtime high-pressure targets: `{}`
- Runtime trace rows: `0`
- Runtime throttle counts: `{'open': 12, 'throttled': 8}`
- Runtime throttle trace rows: `20`
- Runtime throttle sleep total: `400000` us
- Throttle state counts: `{'open': 0, 'throttled': 8}`
- Writer stats: `{'chunks_written': 80, 'bytes_written': 10485760, 'throttle_sleeps': 0, 'sleep_time_s': 0.0, 'elapsed_s': 2.9004736952483654, 'throughput_mb_s': 3.4477127016812017}`
- Runtime trace: `artifacts/validation/qos_fuse_live_bridge/runtime_fuse_admission_trace.jsonl`
- Runtime throttle trace: `artifacts/validation/qos_fuse_live_bridge/runtime_fuse_throttle_trace.jsonl`
- FUSE stderr: `artifacts/validation/qos_fuse_live_bridge/mount_logs/pqc_fuse.stderr.txt`

| sample | gpu_power_mw | gr3d_percent | mem_util | tensor_util | admission_target | throttle_state | trace |
|---:|---:|---:|---:|---:|---|---|---|
| 1 | 28081 | None | 0.8023 | 0.0000 | CPU | throttled | `artifacts/validation/qos_fuse_live_bridge/sample_01.jsonl` |
| 2 | 30455 | None | 0.8701 | 0.0000 | CPU | throttled | `artifacts/validation/qos_fuse_live_bridge/sample_02.jsonl` |
| 3 | 32037 | None | 0.9153 | 0.0000 | CPU | throttled | `artifacts/validation/qos_fuse_live_bridge/sample_03.jsonl` |
| 4 | 31641 | None | 0.9040 | 0.0000 | CPU | throttled | `artifacts/validation/qos_fuse_live_bridge/sample_04.jsonl` |
| 5 | 32432 | None | 0.9266 | 0.0000 | CPU | throttled | `artifacts/validation/qos_fuse_live_bridge/sample_05.jsonl` |
| 6 | 32828 | None | 0.9379 | 0.0000 | CPU | throttled | `artifacts/validation/qos_fuse_live_bridge/sample_06.jsonl` |
| 7 | 33223 | None | 0.9492 | 0.0000 | CPU | throttled | `artifacts/validation/qos_fuse_live_bridge/sample_07.jsonl` |
| 8 | 33223 | None | 0.9492 | 0.0000 | CPU | throttled | `artifacts/validation/qos_fuse_live_bridge/sample_08.jsonl` |
