# QoS measured-pressure adapter report

This bundle feeds retained Nsight-derived pressure metrics into the existing admission controller path.
It proves controller-path integration only; it is not an end-to-end PMU-backed QoS result.

## Source metrics

- Source CSV: `artifacts/validation/uma_storage_dma_profile_combined/ncu/io_uring_uvm_checksum.csv`
- Memory bandwidth util: `0.0453`
- Tensor-core/SM util: `0.3374`

## Admission cases

| Case | mem util | tensor util | target | trace |
|---|---:|---:|---|---|
| idle_control | 0.0000 | 0.0000 | CPU | `artifacts/validation/qos_measured_pressure_adapter/idle_control.jsonl` |
| nsight_derived | 0.0453 | 0.3374 | GPU | `artifacts/validation/qos_measured_pressure_adapter/nsight_derived.jsonl` |

## Interpretation

- `idle_control` records the deterministic baseline with no measured pressure.
- `nsight_derived` records the same admission context after mapping retained NCU metrics into the admission telemetry inputs.
- The JSONL trace contains `telemetry_mem_bandwidth_util` and `telemetry_tensor_core_util`, so the artifact directly ties the measured adapter values to the controller decision.
