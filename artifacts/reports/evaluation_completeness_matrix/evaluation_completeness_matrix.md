# Evaluation completeness matrix

- Overall pass: `True`
- Paper pages: `13`
- Required rows: `7`
- Figure/table labels mapped: `12`

## Matrix

| Row | Required | Status | Primary labels | Evidence | Paper anchors | Scoped gaps |
| --- | ---: | --- | --- | ---: | ---: | --- |
| `baseline_sota_comparison` | `True` | `scoped_partial` | `tab:capability_matrix` | `True` | `True` | fscrypt/fs-verity/dm-integrity rows are not current matched throughput measurements; cold-cache rows remain invalid without privileged cache-drop control |
| `scalability_pressure_behavior` | `True` | `implemented_scoped` | `fig:first_page_qos`, `fig:evaluation_summary` | `True` | `True` | pressure result is mounted SQLite storage pressure, not non-storage application p99 recovery |
| `workload_diversity` | `True` | `implemented_scoped` | - | `True` | `True` | workloads do not establish broad workload generalization or full crash certification |
| `time_overhead_breakdown` | `True` | `implemented_scoped` | `fig:dataplane_negative_control`, `fig:verified_microbench`, `fig:publication_cost_detail` | `True` | `True` | primitive placement is not reported as an end-to-end FUSE write speedup |
| `sensitivity_analysis` | `True` | `implemented_scoped` | `fig:recovery_qos_detail` | `True` | `True` | sensitivity is controller-parameter coverage, not a statistical confidence study |
| `stability_variance` | `True` | `scoped_partial` | - | `True` | `True` | SQLite QoS headline remains a single retained workflow artifact; future headline comparisons require the methodology gate before generalization |
| `case_study` | `True` | `implemented_scoped` | - | `True` | `True` | deployment takeaway is the SQLite/FUSE hero contract, not a separate anecdote |
| `protocol_correctness_and_security_obligations` | `False` | `implemented_scoped` | `fig:problem_boundary`, `tab:component_contracts`, `fig:overall_procedure`, `fig:djc_state_machine`, `tab:threat_boundary` | `True` | `True` | protocol rows close design/security obligations rather than adding evaluation completeness rows |

## Label ownership

| Figure/table label | Matrix row |
| --- | --- |
| `fig:dataplane_negative_control` | `time_overhead_breakdown` |
| `fig:djc_state_machine` | `protocol_correctness_and_security_obligations` |
| `fig:evaluation_summary` | `scalability_pressure_behavior` |
| `fig:first_page_qos` | `scalability_pressure_behavior` |
| `fig:overall_procedure` | `protocol_correctness_and_security_obligations` |
| `fig:problem_boundary` | `protocol_correctness_and_security_obligations` |
| `fig:publication_cost_detail` | `time_overhead_breakdown` |
| `fig:recovery_qos_detail` | `sensitivity_analysis` |
| `fig:verified_microbench` | `time_overhead_breakdown` |
| `tab:capability_matrix` | `baseline_sota_comparison` |
| `tab:component_contracts` | `protocol_correctness_and_security_obligations` |
| `tab:threat_boundary` | `protocol_correctness_and_security_obligations` |

## Evidence contracts

| Matrix row | Artifact | Expected pass | Actual pass | Matrix pass | Role |
| --- | --- | ---: | ---: | ---: | --- |
| `baseline_sota_comparison` | `artifacts/validation/frozen_plaintext_contract/frozen_plaintext_contract.json` | `True` | `True` | `True` | retained evidence |
| `baseline_sota_comparison` | `artifacts/validation/frozen_gocryptfs_contract/frozen_gocryptfs_contract.json` | `True` | `True` | `True` | retained evidence |
| `baseline_sota_comparison` | `artifacts/validation/frozen_dmcrypt_contract/frozen_dmcrypt_contract.json` | `True` | `True` | `True` | retained evidence |
| `baseline_sota_comparison` | `artifacts/validation/frozen_aegisq_contract/frozen_aegisq_contract.json` | `True` | `True` | `True` | retained evidence |
| `baseline_sota_comparison` | `artifacts/validation/kernel_baseline_feasibility/kernel_baseline_feasibility.json` | `True` | `True` | `True` | retained evidence |
| `baseline_sota_comparison` | `artifacts/validation/kernel_baseline_feasibility/paper_fscrypt_verdict.json` | `True` | `True` | `True` | retained evidence |
| `baseline_sota_comparison` | `artifacts/reports/novelty_isolation/novelty_isolation.json` | `True` | `True` | `True` | retained evidence |
| `scalability_pressure_behavior` | `artifacts/reports/hero_result_contract/hero_result_contract.json` | `True` | `True` | `True` | retained evidence |
| `scalability_pressure_behavior` | `artifacts/validation/qos_sqlite_hero_bundle/qos_sqlite_hero_bundle.json` | `True` | `True` | `True` | retained evidence |
| `workload_diversity` | `artifacts/validation/workload_diversity_matrix/workload_diversity_matrix.json` | `True` | `True` | `True` | retained evidence |
| `time_overhead_breakdown` | `artifacts/validation/mechanism_ablation_manifest/mechanism_ablation_manifest.json` | `True` | `True` | `True` | retained evidence |
| `time_overhead_breakdown` | `artifacts/validation/keyplane_rekey_workflow/keyplane_rekey_workflow.json` | `True` | `True` | `True` | retained evidence |
| `time_overhead_breakdown` | `artifacts/reports/x11_mlkem_break_even_model/x11_mlkem_break_even_model.json` | `True` | `True` | `True` | retained evidence |
| `sensitivity_analysis` | `artifacts/validation/qos_sensitivity_analysis/qos_sensitivity_analysis.json` | `True` | `True` | `True` | retained evidence |
| `stability_variance` | `artifacts/validation/stat_thermal_methodology/stat_thermal_methodology_audit.json` | `True` | `True` | `True` | retained evidence |
| `stability_variance` | `artifacts/validation/qos_sqlite_hero_methodology/qos_sqlite_hero_bundle.json` | `False` | `False` | `True` | negative repeated-run stability artifact |
| `case_study` | `artifacts/reports/case_study_takeaway/case_study_takeaway.json` | `True` | `True` | `True` | retained evidence |
| `case_study` | `artifacts/reports/hero_result_contract/hero_result_contract.json` | `True` | `True` | `True` | retained evidence |
| `protocol_correctness_and_security_obligations` | `artifacts/reports/design_eval_isomorphism/design_eval_isomorphism.json` | `True` | `True` | `True` | retained evidence |
| `protocol_correctness_and_security_obligations` | `artifacts/validation/recovery_oracle_audit/recovery_oracle_audit.json` | `True` | `True` | `True` | retained evidence |
| `protocol_correctness_and_security_obligations` | `artifacts/validation/hardware_freshness_recovery_matrix/hardware_freshness_recovery_matrix.json` | `True` | `True` | `True` | retained evidence |

## Checks

| Check | Pass |
| --- | ---: |
| `figure_table_audit_pass` | `True` |
| `all_required_rows_present` | `True` |
| `required_rows_have_allowed_status` | `True` |
| `all_rows_have_retained_evidence` | `True` |
| `all_rows_have_paper_scope_anchors` | `True` |
| `all_figure_table_labels_mapped_once` | `True` |
| `paper_pages_le_13` | `True` |
