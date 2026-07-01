# Figure/table obligation audit

- Overall pass: `True`
- Paper pages: `13`
- Figure/table count: `13`

## Obligations

| Label | Source | Kind | Obligation | Caption terms | References | Pass |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `fig:first_page_qos` | `Paper/1_Introduction.tex:6` | `RQ2` | first-page SQLite pressure/hero result | `True` | `3` | `True` |
| `tab:capability_matrix` | `Paper/1_Introduction.tex:14` | `design` | capability table defining the design gap | `True` | `1` | `True` |
| `fig:problem_boundary` | `Paper/2_Background.tex:15` | `motivation` | problem boundary for the edge-runtime thesis | `True` | `1` | `True` |
| `fig:dataplane_negative_control` | `Paper/2_Background.tex:34` | `motivation/RQ2` | AES-GCM data-plane size sweep motivating CPU-first placement | `True` | `3` | `True` |
| `fig:overall_procedure` | `Paper/3_Design.tex:7` | `design` | architecture and plane-separation figure | `True` | `8` | `True` |
| `tab:component_contracts` | `Paper/3_Design.tex:21` | `design` | architecture-indexed component contract map | `True` | `1` | `True` |
| `fig:djc_state_machine` | `Paper/3_Design.tex:59` | `design` | D/J/C/xattr publication state machine | `True` | `1` | `True` |
| `tab:design_goals` | `Paper/3_Design.tex:135` | `design` | formal storage-protocol invariant table | `True` | `1` | `True` |
| `tab:impl_boundaries` | `Paper/7_Implementation_Details.tex:8` | `design` | implementation boundary summary | `True` | `1` | `True` |
| `tab:threat_boundary` | `Paper/8_Security_Analysis.tex:11` | `RQ1/RQ4` | threat-model and recovery boundary | `True` | `1` | `True` |
| `fig:evaluation_summary` | `Paper/4_Evaluation.tex:22` | `RQ2/RQ3/RQ5` | evaluation spine across cost, QoS, and key-plane placement | `True` | `2` | `True` |
| `tab:benchmark_workloads` | `Paper/4_Evaluation.tex:34` | `RQ1-RQ5` | evaluation scope by research question | `True` | `1` | `True` |
| `tab:qos_sqlite_recovery` | `Paper/generated_qos_recovery_table.tex:1` | `RQ3` | SQLite QoS recovery hero result | `True` | `1` | `True` |

## Checks

| Check | Pass |
| --- | ---: |
| `all_expected_labels_present` | `True` |
| `no_duplicate_labels` | `True` |
| `all_figure_tables_have_obligations` | `True` |
| `no_rendered_artifact_path_lists` | `True` |
| `paper_pages_le_13` | `True` |
