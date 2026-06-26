# Remaining gap map

This report consolidates the currently open checklist items and the evidence that is already retained.

## storage-DMA proof: raw NVMe O_DIRECT path crosses the intended NVMe-to-UVM boundary
Current evidence:
- artifacts/evidence/repro_malloc_register.out
- artifacts/evidence/io_uring_uvm.out
- artifacts/evidence/io_uring_uvm_nvme_sudo.out
- artifacts/validation/uma_storage_dma_same_buffer/
- artifacts/validation/uma_storage_dma_profile_combined_report/
- artifacts/validation/uma_storage_dma_profile_combined/ncu/
- artifacts/validation/uma_counter_availability/
- artifacts/validation/um_smoke.json
- artifacts/motivation/uvm_proxy_report/
- experiments/run_uma_storage_dma_probe.py
Missing evidence:
- full NVMe-to-UVM DMA semantics
- a final FUSE data-path proof that the production path itself crosses the intended NVMe-to-UVM boundary

## PMU counter integration under workload pressure
Current evidence:
- pqc_admission.c/h software proxy path: pqc_admission_record_uma_event(), pqc_admission_update_telemetry()
- pqc_admission.c threshold-softening branch in pqc_admit()
- artifacts/validation/tegra_qos_daemon_trace.jsonl
- artifacts/validation/run_qos_gpu_trace.jsonl
- artifacts/validation/telemetry_trace_report/
- artifacts/validation/qos_repeated_run/
- artifacts/validation/qos_repeated_report/
- artifacts/validation/qos_measured_pressure_adapter/
- artifacts/validation/qos_live_telemetry_admission/
- artifacts/m3_qos_ci_report/
- artifacts/tensorrt_ci_report/
Missing evidence:
- PMU/CUPTI/Nsight inputs sampled during the foreground workload rather than tegrastats power proxy or offline NCU CSV
- controller path that consumes measured PMU/CUPTI/Nsight input and throttles/admission-controls real FUSE work during the same end-to-end workload

## App-level recovery and fault injection across durable boundaries
Current evidence:
- pqc_anchor.c / pqc_fuse.c file and hardware anchor paths with fail-closed loading
- artifacts/sqlite_strace.log
- artifacts/sqlite_ci_report/
- artifacts/validation/sqlite_recovery_oracle/
- artifacts/validation/sqlite_fault_campaign/
- artifacts/crash_replay_e8_test_summary.json
- artifacts/crash_audit_report/
Missing evidence:
- broader application workload beyond SQLite
- syscall-exact crash timing inside SQLite/FUSE rather than deterministic file-state mutation

## Second hardware platform / driver-version matrix
Current evidence:
- artifacts/validation/microbench/summary.json current platform manifest
- artifacts/platform_inventory_report/
- artifacts/repro_bundle/
Missing evidence:
- preserved raw outputs from the same scripts on at least one additional Jetson, kernel, or JetPack revision

## Combined durability campaign
Current evidence:
- experiments/run_tpm_only_bundle.py (bundle checks returncode 0)
- experiments/run_combined_durability_bundle.py (bundle checks returncode 0)
- artifacts/validation/tpm_only_bundle/
- artifacts/validation/combined_durability_bundle/
- artifacts/validation/tpm_recovery_verdict/
- artifacts/validation/tpm_freshness_bundle/
- artifacts/validation/app_recovery_bundle/
Missing evidence:
- a hardware-backed combined flow that proves TPM freshness validation and app-level recovery on the same retained backing store
- a single retained study that ties the TPM replay-after-advance path to the SQLite selected-boundary campaign on the same backing store
