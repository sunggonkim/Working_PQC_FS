# Remaining gap map

This report consolidates the currently open checklist items and the evidence that is already retained.

## storage-DMA proof: raw NVMe O_DIRECT path crosses the intended NVMe-to-UVM boundary
Current evidence:
- artifacts/probes/evidence/repro_malloc_register.out
- artifacts/probes/evidence/io_uring_uvm.out
- artifacts/probes/evidence/io_uring_uvm_nvme_sudo.out
- artifacts/validation/uma_storage_dma_same_buffer/
- artifacts/reports/uma_storage_dma_profile_combined_report/
- artifacts/validation/uma_storage_dma_profile_combined/ncu/
- artifacts/validation/uma_counter_availability/
- artifacts/validation/um_smoke.json
- artifacts/results/motivation/uvm_proxy_report/
- code/experiments/run_uma_storage_dma_probe.py
Missing evidence:
- full NVMe-to-UVM DMA semantics
- a final FUSE data-path proof that the production path itself crosses the intended NVMe-to-UVM boundary

## Foreground AI p99 QoS restoration
Current evidence:
- code/runtime/pqc_admission.c/h software proxy path: pqc_admission_record_uma_event(), pqc_admission_update_telemetry()
- code/runtime/pqc_admission.c threshold-softening branch in pqc_admit()
- artifacts/validation/tegra_qos_daemon_trace.jsonl
- artifacts/validation/run_qos_gpu_trace.jsonl
- artifacts/validation/telemetry_trace_report/
- artifacts/validation/qos_repeated_run/
- artifacts/validation/qos_repeated_report/
- artifacts/validation/qos_measured_pressure_adapter/
- artifacts/validation/qos_live_telemetry_admission/
- artifacts/validation/qos_fuse_live_bridge/ (live tegrastats reaches mounted daemon; runtime_fuse_throttle_trace.jsonl records in-daemon throttle)
- artifacts/validation/qos_cupti_pm_fuse_bridge/ (live CUPTI PM samples drive the mounted daemon in the same execution)
- artifacts/reports/m3_qos_ci_report/
- artifacts/reports/tensorrt_ci_report/
Missing evidence:
- a single foreground inference co-run that reports p50/p95/p99 latency under no-background, naïve-background, and AEGIS-Q-controlled background I/O
- repeated-run confidence intervals showing that PM/CUPTI-driven throttling recovers foreground inference tail latency rather than only throttling the background FUSE path

## Real crash/power evidence beyond lower-block interruption
Current evidence:
- code/storage/pqc_anchor.c and code/frontend/pqc_fuse.c file/hardware anchor paths with fail-closed loading
- artifacts/reports/sqlite_ci_report/
- artifacts/validation/sqlite_recovery_oracle/
- artifacts/validation/sqlite_fault_campaign/
- artifacts/validation/combined_durability_bundle/ (same-backing-store SQLite+TPM and dbm.dumb+TPM fail-closed stale-snapshot replay)
- artifacts/validation/sqlite_syscall_crash_tpm/ (fdatasync-exact SQLite app-process SIGKILL timing on TPM-backed FUSE)
- artifacts/reports/crash_audit_report/
- artifacts/validation/x1_block_fault_campaign/x1_block_fault_campaign.json (three loopback ext4/device-mapper lower-block interruption rows: pre-commit rows recover previous_committed; post-fsync accepted-state row recovers latest_committed)
Missing evidence:
- physical power-loss, VM power-cut, reboot, kernel-crash, or drive-cache-sensitive campaign
- FUSE-daemon or whole-system crash timing rather than app-process SIGKILL only
- arbitrary interruption safety across all cut points remains a non-claim
Not sufficient evidence:
- code/experiments/run_power_fail_test.py is only an analytical freshness-window model and must not be used as X1B evidence
- the lower-block interruption campaign supports storage-error recovery but is not physical power-loss, kernel-crash, or drive-cache certification

## Second hardware platform / driver-version matrix
Current evidence:
- artifacts/validation/microbench/summary.json current platform manifest
- artifacts/reports/platform_inventory_report/
- artifacts/repro_bundle/
Missing evidence:
- preserved raw outputs from the same scripts on at least one additional Jetson, kernel, or JetPack revision

## Combined durability beyond app-process crash timing
Current evidence:
- code/experiments/run_combined_durability_bundle.py (TPM-only and app-only checks returncode 0)
- artifacts/validation/combined_durability_bundle/combined_durability_bundle.json
- same-backing-store SQLite baseline row count 1, advanced row count 3, stale replay fail_closed at SQLite open
- same-backing-store dbm.dumb baseline row count 1, advanced row count 3, stale replay fail_closed at dbm read
- fdatasync-exact SQLite app-process SIGKILL timing: 3 trials, 0 unacceptable verdicts
Missing evidence:
- power-loss or FUSE-daemon crash timing for the combined SQLite+TPM path
Not sufficient evidence:
- analytical freshness-window outputs do not close combined durability or power-loss objections
