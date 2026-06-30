# Novelty-isolation audit

- Overall pass: `True`
- Paper pages: `12`
- Direct answer present: `True`

## Direct Answer

- Source: `Paper/2_Background.tex:13`
- Text: AEGIS-Q distinguishes itself by making placement and recovery policy visible at the same boundary as file encryption. It is not merely gocryptfs/fscrypt plus CUDA/TPM scripts: the research question is whether one mounted runtime can keep authenticated-block publication, storage-visible QoS/admission, optional batched GPU key-plane work, and external replay-after-advance checks under one evidence contract. In that setting, the storage layer exposes policy decisions that conventional encryption layers usually hide. A FUSE daemon can decide whether a maintenance job is latency-sensitive, batch-shaped, or replay-critical; it can also fail closed before exposing a file whose envelope, checkpoint, or external anchor is inconsistent.

## Combined Capability

| Term | Present |
| --- | ---: |
| `authenticated-block publication` | `True` |
| `storage-visible QoS/admission` | `True` |
| `optional batched GPU key-plane work` | `True` |
| `external replay-after-advance checks` | `True` |
| `one evidence contract` | `True` |

## Deployed-Baseline Classes

| Class | Present in first-page table |
| --- | ---: |
| `Plaintext` | `True` |
| `gocryptfs` | `True` |
| `fscrypt` | `True` |
| `dm-crypt` | `True` |
| `fs-verity/dm-integrity` | `True` |
| `TPM/TEE-backed` | `True` |
| `GPU-storage systems` | `True` |
| `AEGIS-Q` | `True` |

## Evidence Gates

| Gate | Present | Overall pass | Violations |
| --- | ---: | ---: | ---: |
| `artifacts/reports/paper_spine_gate/paper_spine_gate.json` | `True` | `True` | `0` |
| `artifacts/reports/hero_result_contract/hero_result_contract.json` | `True` | `True` | `0` |
| `artifacts/reports/design_eval_isomorphism/design_eval_isomorphism.json` | `True` | `True` | `0` |
| `artifacts/validation/mechanism_ablation_manifest/mechanism_ablation_manifest.json` | `True` | `True` | `n/a` |
| `artifacts/validation/integrity_comparison_manifest/integrity_comparison_manifest.json` | `True` | `True` | `n/a` |
| `artifacts/validation/kernel_baseline_feasibility/kernel_baseline_feasibility.json` | `True` | `True` | `n/a` |
| `artifacts/validation/frozen_gocryptfs_contract/frozen_gocryptfs_contract.json` | `True` | `True` | `n/a` |
| `artifacts/validation/frozen_aegisq_contract/frozen_aegisq_contract.json` | `True` | `True` | `n/a` |

## Checks

| Check | Pass |
| --- | ---: |
| `direct_answer_present` | `True` |
| `combined_capability_terms_present` | `True` |
| `related_boundary_present` | `True` |
| `baseline_classes_present` | `True` |
| `evidence_gates_pass` | `True` |
| `no_broad_deployed_superiority_claim` | `True` |
| `paper_pages_12` | `True` |
| `maturity_boundary_present` | `True` |
| `full_kernel_matrix_boundary_present` | `True` |
