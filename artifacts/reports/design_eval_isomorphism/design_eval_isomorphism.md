# Design-evaluation isomorphism audit

- Overall pass: `True`
- Paper pages: `13`
- Mapping text present: `True`
- Architecture component map present: `True`

## Mechanism closures

| Mechanism | Closure id | RQ | Design present | Evaluation present | Artifacts present |
| --- | --- | --- | ---: | ---: | ---: |
| authenticated block format | `E1-generation-and-envelope-authentication` | RQ1 | `True` | `True` | `True` |
| D/J/C publication | `E2-oracle-labeled-publication-cutpoints` | RQ4 | `True` | `True` | `True` |
| TPM replay check | `E3-hardware-freshness-verdict-matrix` | RQ4 | `True` | `True` | `True` |
| QoS controller | `E4-sqlite-recovery-and-controller-sensitivity` | RQ3/RQ5 | `True` | `True` | `True` |
| CPU/GPU placement | `E5-data-plane-placement-asymmetry` | RQ2/RQ5 | `True` | `True` | `True` |
| optional ML-KEM batch lane | `E6-mounted-open-file-rekey-workflow` | RQ5 | `True` | `True` | `True` |

## Figure/table obligations

| Label | Obligation | Label present | Referenced |
| --- | --- | ---: | ---: |
| `fig:first_page_qos` | first-page QoS pressure/hero result | `True` | `True` |
| `tab:capability_matrix` | design-gap capability comparison | `True` | `True` |
| `tab:design_goals` | formal invariant table | `True` | `True` |
| `fig:dataplane_negative_control` | data-plane placement motivation | `True` | `True` |
| `fig:overall_procedure` | architecture and plane separation | `True` | `True` |
| `fig:djc_state_machine` | publication protocol state machine | `True` | `True` |
| `tab:impl_boundaries` | implementation boundary summary | `True` | `True` |
| `tab:threat_boundary` | security threat boundary | `True` | `True` |
| `tab:benchmark_workloads` | evaluation provenance | `True` | `True` |
| `fig:evaluation_summary` | evaluation spine | `True` | `True` |
| `tab:qos_sqlite_recovery` | SQLite QoS hero result | `True` | `True` |
