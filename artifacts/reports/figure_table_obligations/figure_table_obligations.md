# Figure/table obligation audit

- Overall pass: `True`
- Paper pages: `12`
- Figure/table count: `12`

## Obligations

| Label | Source | Kind | Obligation | Caption terms | References | Pass |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `fig:first_page_qos` | `Paper/1_Introduction.tex:6` | `RQ3` | first-page SQLite pressure/hero result | `True` | `4` | `True` |
| `tab:capability_matrix` | `Paper/1_Introduction.tex:14` | `design` | capability table defining the design gap | `True` | `1` | `True` |
| `tab:design_goals` | `Paper/3_Design.tex:11` | `design` | formal storage-protocol invariant table | `True` | `1` | `True` |
| `fig:overall_procedure` | `Paper/3_Design.tex:32` | `design` | architecture and plane-separation figure | `True` | `1` | `True` |
| `fig:djc_state_machine` | `Paper/3_Design.tex:57` | `design` | D/J/C/xattr publication state machine | `True` | `1` | `True` |
| `tab:memory_compat` | `Paper/3_Design.tex:110` | `design/RQ2/RQ5` | memory-path claim boundary | `True` | `1` | `True` |
| `tab:impl_boundaries` | `Paper/7_Implementation_Details.tex:8` | `design` | implementation boundary summary | `True` | `1` | `True` |
| `tab:threat_boundary` | `Paper/8_Security_Analysis.tex:11` | `RQ1/RQ4` | threat-model and recovery boundary | `True` | `1` | `True` |
| `tab:benchmark_workloads` | `Paper/4_Evaluation.tex:24` | `RQ1-RQ5` | evaluation scope by research question | `True` | `1` | `True` |
| `fig:baseline_comparison` | `Paper/4_Evaluation.tex:58` | `RQ2/RQ5` | CPU/GPU placement and batch-lane evidence | `True` | `2` | `True` |
| `tab:recovery_scope` | `Paper/4_Evaluation.tex:83` | `RQ4` | recovery/freshness interpretation | `True` | `1` | `True` |
| `tab:qos_sqlite_recovery` | `Paper/generated_qos_recovery_table.tex:1` | `RQ3` | SQLite QoS recovery hero result | `True` | `2` | `True` |

## Checks

| Check | Pass |
| --- | ---: |
| `all_expected_labels_present` | `True` |
| `no_duplicate_labels` | `True` |
| `all_figure_tables_have_obligations` | `True` |
| `no_rendered_artifact_path_lists` | `True` |
| `paper_pages_12` | `True` |
