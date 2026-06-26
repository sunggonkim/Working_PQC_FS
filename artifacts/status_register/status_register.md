# Status register

This register lists retained packaging artifacts and the open items that remain open.

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
- uma_storage_dma_repeated: False
- uma_storage_dma_report: True
- evidence_dashboard: True
- app_recovery_bundle: True
- app_recovery_report: True
- combined_durability_bundle: True
- sqlite_recovery_oracle: True
- sqlite_fault_campaign: True

## Open items
- storage-DMA proof
- QoS PMU + hysteresis
- full multi-workload app recovery classification
- second-platform portability
- combined durability campaign

This register does not upgrade any open item into a verified claim.