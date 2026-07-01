# Related-work and applicability audit

- Overall pass: `True`
- Paper pages: `13`
- Dangerous unguarded claim hits: `0`

## Checks

| Check | Pass |
| --- | ---: |
| `paper_pdf_pages_le_13` | `True` |
| `systems_literature_gate_passes` | `True` |
| `technique_transfer_matrix_passes` | `True` |
| `all_related_topics_present` | `True` |
| `applicability_boundary_present` | `True` |
| `win_loss_nonapply_language_present` | `True` |
| `checklist_competitor_strategy_present` | `True` |
| `bibliography_competitor_set_present` | `True` |
| `dangerous_claims_guarded` | `True` |

## Related Topics

| Topic | Complete | Required terms |
| --- | ---: | --- |
| `kernel_file_encryption` | `True` | `fscrypt`, `dm-crypt`, `kernel` |
| `userspace_encryption` | `True` | `gocryptfs`, `FUSE` |
| `integrity_boundaries` | `True` | `fs-verity`, `dm-integrity` |
| `kernel_qos_controls` | `True` | `kernel QoS`, `ionice`, `IOWeight` |
| `log_structured_storage` | `True` | `F2FS`, `ScaleFS`, `FastCommit` |
| `journaling_group_commit` | `True` | `journal`, `epoch`, `barrier` |
| `tpm_tee_freshness` | `True` | `TPM`, `TEE`, `OP-TEE`, `PCR` |
| `gpu_crypto_storage_staging` | `True` | `GPU`, `fscrypt-GPU`, `FlashNeuron`, `Fastensor`, `SnuQS`, `Speculative GPU encryption` |
| `gpu_filesystem_boundary` | `True` | `GPUstore`, `GPUfs`, `GeminiFS`, `GPU4FS`, `GPU-side execution` |
| `storage_bypass_nonclaims` | `True` | `SPDK`, `GPUDirect Storage`, `cuFile`, `direct storage/GPU data path`, `FUSE` |
| `same_contract_baseline_boundary` | `True` | `same-contract rows`, `artificial throughput row`, `mode-aligned measured rows` |

## Applicability Boundary

| Row | Complete | Required terms |
| --- | ---: | --- |
| `first_page_boundary` | `True` | `tab:applicability_boundary` |
| `protection_domain` | `True` | `authenticated FUSE`, `mounted runtime` |
| `trust_assumptions` | `True` | `kernel/FUSE-daemon trust` |
| `platform_dependencies` | `True` | `Jetson/CUDA/TPM dependencies` |
| `known_loss_cases` | `True` | `gocryptfs`, `free storage path` |
| `unsupported_guarantees` | `True` | `not a general POSIX`, `PCR-bound`, `power-loss` |

## Win/Loss/Non-Apply Language

| Row | Complete | Required terms |
| --- | ---: | --- |
| `wins_or_value` | `True` | `storage-visible`, `SQLite`, `replay-after-advance` |
| `loses` | `True` | `gocryptfs`, `free storage path`, `not kernel-replacement maturity` |
| `does_not_apply` | `True` | `not a full fscrypt`, `not as a replacement`, `not a general POSIX` |

## Checklist Competitor Strategy

| Row | Complete | Missing terms |
| --- | ---: | --- |
| `baseline_deletion_forbidden` | `True` | - |
| `direct_measured_rows_preserved` | `True` | - |
| `fscrypt_unavailable_not_speedup` | `True` | - |
| `gpu_storage_related_work_boundary` | `True` | - |
| `sufficiency_scope` | `True` | - |

## Bibliography Competitor Set

| Row | Complete | Missing terms |
| --- | ---: | --- |
| `storage_encryption_baselines` | `True` | - |
| `fuse_cost_boundary` | `True` | - |
| `gpu_crypto_file_system` | `True` | - |
| `gpu_storage_systems` | `True` | - |
| `direct_storage_gpu_path` | `True` | - |
| `pqc_cost_model` | `True` | - |
