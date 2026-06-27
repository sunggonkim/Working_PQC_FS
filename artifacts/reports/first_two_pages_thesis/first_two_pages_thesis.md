# First-two-pages positive-thesis audit

- Overall pass: `True`
- Paper pages: `12`

## Source Anchors

| Anchor | Source | Present |
| --- | --- | ---: |
| `pressure_result` | `Paper/1_Introduction.tex:38` | `True` |
| `capability_table` | `Paper/1_Introduction.tex:16` | `True` |
| `concrete_gap` | `Paper/1_Introduction.tex:36` | `True` |
| `central_thesis` | `Paper/1_Introduction.tex:40` | `True` |
| `design_insight` | `Paper/1_Introduction.tex:42` | `True` |
| `contribution_intro` | `Paper/1_Introduction.tex:44` | `True` |
| `contribution_c1` | `Paper/1_Introduction.tex:46` | `True` |
| `contribution_c2` | `Paper/1_Introduction.tex:47` | `True` |
| `contribution_c3` | `Paper/1_Introduction.tex:48` | `True` |
| `contribution_c4` | `Paper/1_Introduction.tex:49` | `True` |
| `not_weaker_component_stack` | `Paper/2_Background.tex:13` | `True` |

## Compiled PDF Checks

| Check | Present |
| --- | ---: |
| `first_page_pressure_figure` | `True` |
| `first_page_capability_table` | `True` |
| `first_two_pages_gap` | `True` |
| `first_two_pages_thesis` | `True` |
| `first_two_pages_contributions` | `True` |
| `first_two_pages_component_stack_answer` | `True` |

## Support Gates

| Gate | Present | Overall pass | Violations |
| --- | ---: | ---: | ---: |
| `artifacts/reports/paper_spine_gate/paper_spine_gate.json` | `True` | `True` | `0` |
| `artifacts/reports/hero_result_contract/hero_result_contract.json` | `True` | `True` | `0` |
| `artifacts/reports/novelty_isolation/novelty_isolation.json` | `True` | `True` | `0` |
| `artifacts/reports/accepted_paper_structure_audit/accepted_paper_structure_audit.json` | `True` | `True` | `0` |

## Checks

| Check | Pass |
| --- | ---: |
| `paper_pages_12` | `True` |
| `source_anchors_present` | `True` |
| `compiled_first_two_pages_present` | `True` |
| `positive_contributions_before_defensive_scope` | `True` |
| `support_gates_pass` | `True` |
