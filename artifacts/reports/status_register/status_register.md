# Status register

This register lists retained packaging artifacts and broader claims that remain out of scope.

## Present reports
- crash_audit_report: True
- platform_inventory_report: True
- tpm_freshness_bundle: True
- tpm_freshness_report: True
- tpm_only_bundle: True
- tpm_monotonic_replay: True
- tpm_recovery_verdict: True
- tpm_provisioning_probe: True
- tpm_pcr_policy_probe: True
- qos_repeated_run: True
- qos_repeated_report: True
- qos_live_telemetry_admission: True
- qos_fuse_live_bridge: True
- qos_cupti_pm_fuse_bridge: True
- uma_storage_dma_repeated: False
- uma_storage_dma_report: True
- evidence_dashboard: True
- app_recovery_bundle: True
- app_recovery_report: True
- combined_durability_bundle: True
- sqlite_recovery_oracle: True
- sqlite_fault_campaign: True
- sqlite_syscall_crash_tpm: True

## Broader claims still out of scope
- storage-DMA proof
- foreground AI p99 QoS restoration
- power-loss / FUSE-daemon crash timing
- second-platform portability

This register does not upgrade any out-of-scope claim into a verified claim.