# Recurring-review elimination audit

- Overall pass: `False`
- Paper pages: `13`
- Themes checked: `10`

| Theme | First page | Design | Evaluation | Limits | Artifacts | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no end-to-end benefit | `True` | `True` | `True` | `True` | `True` | `True` |
| disconnected GPU/KEM story | `True` | `True` | `True` | `True` | `True` | `True` |
| incomplete baselines | `True` | `True` | `True` | `True` | `True` | `True` |
| wiring-only QoS | `True` | `True` | `True` | `True` | `True` | `True` |
| partial TPM replay | `True` | `True` | `True` | `False` | `True` | `False` |
| narrow POSIX semantics | `True` | `True` | `True` | `True` | `True` | `True` |
| password-derived credential boundary | `True` | `True` | `True` | `False` | `True` | `False` |
| single-platform evidence | `True` | `True` | `True` | `True` | `True` | `True` |
| energy/thermal missing | `True` | `True` | `True` | `False` | `True` | `False` |
| microbenchmark-only methodology | `True` | `True` | `True` | `False` | `True` | `False` |

## Theme Details

### no end-to-end benefit

- First page: `Paper/main.tex:91`
- Design: `Paper/3_Design.tex:139`
- Evaluation: `Paper/4_Evaluation.tex:61`
- Limits: `Paper/10_Discussion_and_Limitations.tex:4`
- Artifacts: `artifacts/reports/hero_result_contract/hero_result_contract.json`, `artifacts/validation/qos_sqlite_hero_bundle/qos_sqlite_hero_bundle.json`

### disconnected GPU/KEM story

- First page: `Paper/main.tex:90`
- Design: `Paper/3_Design.tex:97`
- Evaluation: `Paper/4_Evaluation.tex:37`
- Limits: `Paper/10_Discussion_and_Limitations.tex:4`
- Artifacts: `artifacts/validation/keyplane_rekey_workflow/keyplane_rekey_workflow.json`, `artifacts/validation/keyplane_rekey_methodology/keyplane_rekey_workflow.json`, `artifacts/reports/design_eval_isomorphism/design_eval_isomorphism.json`

### incomplete baselines

- First page: `Paper/1_Introduction.tex:37`
- Design: `Paper/2_Background.tex:7`
- Evaluation: `Paper/4_Evaluation.tex:28`
- Limits: `Paper/10_Discussion_and_Limitations.tex:8`
- Artifacts: `artifacts/reports/novelty_isolation/novelty_isolation.json`, `artifacts/validation/kernel_baseline_feasibility/kernel_baseline_feasibility.json`, `artifacts/validation/frozen_dmcrypt_contract/frozen_dmcrypt_contract.json`, `artifacts/validation/frozen_plaintext_contract/frozen_plaintext_contract.json`, `artifacts/validation/frozen_gocryptfs_contract/frozen_gocryptfs_contract.json`, `artifacts/validation/frozen_aegisq_contract/frozen_aegisq_contract.json`

### wiring-only QoS

- First page: `Paper/1_Introduction.tex:39`
- Design: `Paper/3_Design.tex:139`
- Evaluation: `Paper/4_Evaluation.tex:61`
- Limits: `Paper/10_Discussion_and_Limitations.tex:11`
- Artifacts: `artifacts/validation/qos_sqlite_hero_bundle/qos_sqlite_hero_bundle.json`, `artifacts/validation/qos_sensitivity_analysis/qos_sensitivity_analysis.json`, `artifacts/reports/case_study_takeaway/case_study_takeaway.json`

### partial TPM replay

- First page: `Paper/main.tex:90`
- Design: `Paper/3_Design.tex:151`
- Evaluation: `Paper/4_Evaluation.tex:54`
- Limits: `Paper/10_Discussion_and_Limitations.tex:None`
- Artifacts: `artifacts/validation/hardware_freshness_recovery_matrix/hardware_freshness_recovery_matrix.json`, `artifacts/validation/tpm_freshness_policy/tpm_freshness_policy.json`, `artifacts/validation/pcr_anchor_decision/pcr_anchor_decision.json`

### narrow POSIX semantics

- First page: `Paper/1_Introduction.tex:30`
- Design: `Paper/3_Design.tex:88`
- Evaluation: `Paper/4_Evaluation.tex:52`
- Limits: `Paper/10_Discussion_and_Limitations.tex:8`
- Artifacts: `artifacts/validation/posix_scope_audit/posix_scope_audit.json`, `artifacts/validation/generation_fault_matrix/generation_fault_matrix.json`

### password-derived credential boundary

- First page: `Paper/1_Introduction.tex:37`
- Design: `Paper/3_Design.tex:86`
- Evaluation: `Paper/4_Evaluation.tex:50`
- Limits: `Paper/10_Discussion_and_Limitations.tex:None`
- Artifacts: `artifacts/validation/mount_key_lifecycle/mount_key_lifecycle.json`, `artifacts/validation/fuse_tamper_rejection.json`, `artifacts/validation/keyplane_rekey_workflow/keyplane_rekey_workflow.json`

### single-platform evidence

- First page: `Paper/main.tex:91`
- Design: `Paper/7_Implementation_Details.tex:4`
- Evaluation: `Paper/4_Evaluation.tex:18`
- Limits: `Paper/10_Discussion_and_Limitations.tex:11`
- Artifacts: `artifacts/validation/stat_thermal_methodology/stat_thermal_methodology_audit.json`, `artifacts/validation/qos_sqlite_hero_bundle/qos_sqlite_hero_bundle.json`

### energy/thermal missing

- First page: `Paper/main.tex:91`
- Design: `Paper/7_Implementation_Details.tex:52`
- Evaluation: `Paper/4_Evaluation.tex:18`
- Limits: `Paper/10_Discussion_and_Limitations.tex:None`
- Artifacts: `artifacts/reports/o4_energy_thermal_result/o4_energy_thermal_result.json`, `artifacts/validation/jetson_power_thermal_contract/jetson_power_thermal_contract.json`

### microbenchmark-only methodology

- First page: `Paper/main.tex:91`
- Design: `Paper/3_Design.tex:93`
- Evaluation: `Paper/4_Evaluation.tex:33`
- Limits: `Paper/10_Discussion_and_Limitations.tex:None`
- Artifacts: `artifacts/reports/hero_result_contract/hero_result_contract.json`, `artifacts/validation/stat_thermal_methodology/stat_thermal_methodology_audit.json`, `artifacts/reports/evaluation_rq_audit/evaluation_rq_audit.json`

## Violations

- partial TPM replay
- password-derived credential boundary
- energy/thermal missing
- microbenchmark-only methodology
