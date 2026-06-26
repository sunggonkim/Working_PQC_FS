# QoS live telemetry admission bridge

This bundle samples live `tegrastats` output while optional GPU pressure is active and feeds each sample into `pqc_fuse --admission-telemetry-smoke`.
It is path evidence only, not full closed-loop QoS certification.

- Samples requested: `6`
- Samples recorded: `6`
- GPU burner started: `True`
- Decision counts: `{'CPU': 6}`

| sample | gpu_power_mw | gr3d_percent | mem_util | tensor_util | target | trace |
|---:|---:|---:|---:|---:|---|---|
| 1 | 27301 | None | 0.7800 | 0.0000 | CPU | `artifacts/validation/qos_live_telemetry_admission/sample_01.jsonl` |
| 2 | 30071 | None | 0.8592 | 0.0000 | CPU | `artifacts/validation/qos_live_telemetry_admission/sample_02.jsonl` |
| 3 | 31641 | None | 0.9040 | 0.0000 | CPU | `artifacts/validation/qos_live_telemetry_admission/sample_03.jsonl` |
| 4 | 31246 | None | 0.8927 | 0.0000 | CPU | `artifacts/validation/qos_live_telemetry_admission/sample_04.jsonl` |
| 5 | 32037 | None | 0.9153 | 0.0000 | CPU | `artifacts/validation/qos_live_telemetry_admission/sample_05.jsonl` |
| 6 | 32828 | None | 0.9379 | 0.0000 | CPU | `artifacts/validation/qos_live_telemetry_admission/sample_06.jsonl` |
