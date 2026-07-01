# Figure/table obligation audit

- Overall pass: `True`
- Paper pages: `13`
- Figure/table count: `12`

## Obligations

| Label | Source | Kind | Obligation | Caption terms | References | Pass |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `fig:first_page_qos` | `Paper/1_Introduction.tex:6` | `RQ2` | first-page SQLite pressure/hero result | `True` | `4` | `True` |
| `tab:capability_matrix` | `Paper/1_Introduction.tex:14` | `design` | capability table defining the design gap | `True` | `1` | `True` |
| `fig:problem_boundary` | `Paper/2_Background.tex:15` | `motivation` | problem boundary for the edge-runtime thesis | `True` | `1` | `True` |
| `fig:dataplane_negative_control` | `Paper/2_Background.tex:34` | `motivation/RQ2` | AES-GCM data-plane size sweep motivating CPU-first placement | `True` | `3` | `True` |
| `fig:overall_procedure` | `Paper/3_Design.tex:10` | `design` | architecture and plane-separation figure | `True` | `8` | `True` |
| `tab:component_contracts` | `Paper/3_Design.tex:24` | `design` | architecture-indexed component contract map | `True` | `2` | `True` |
| `fig:djc_state_machine` | `Paper/3_Design.tex:60` | `design` | D/J/C/xattr publication state machine | `True` | `1` | `True` |
| `tab:threat_boundary` | `Paper/8_Security_Analysis.tex:15` | `RQ1/RQ4` | threat-model and recovery boundary | `True` | `1` | `True` |
| `fig:evaluation_summary` | `Paper/4_Evaluation.tex:20` | `RQ2/RQ3/RQ5` | evaluation spine across cost, QoS, and key-plane placement | `True` | `4` | `True` |
| `fig:publication_cost_detail` | `Paper/4_Evaluation.tex:33` | `RQ5` | graph-driven strict-publication cost and grouped-publication boundary | `True` | `1` | `True` |
| `fig:verified_microbench` | `Paper/4_Evaluation.tex:41` | `RQ1` | verified primitive placement graph for CPU/GPU/PQC policy | `True` | `1` | `True` |
| `fig:recovery_qos_detail` | `Paper/4_Evaluation.tex:60` | `RQ2/RQ3/RQ4` | recovery oracle, QoS sensitivity, and elastic GPU admission detail | `True` | `3` | `True` |

## Checks

| Check | Pass |
| --- | ---: |
| `all_expected_labels_present` | `True` |
| `no_duplicate_labels` | `True` |
| `all_figure_tables_have_obligations` | `True` |
| `no_rendered_artifact_path_lists` | `True` |
| `paper_pages_le_13` | `True` |
