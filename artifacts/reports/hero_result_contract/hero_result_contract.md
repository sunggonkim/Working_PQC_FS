# Hero-result contract

- Overall pass: `True`
- Hero id: `sqlite-mounted-qos-recovery-2026-06-27`
- Paper pages: `12`
- Artifact: `artifacts/validation/qos_sqlite_hero_bundle/qos_sqlite_hero_bundle.json`

## Headline claim

For foreground SQLite transactions on mounted AEGIS-Q FUSE, unthrottled secure-storage pressure raises p99 from 6.436 ms to 13.822 ms and causes 1 deadline miss; AEGIS-Q policy recovers p99 to 8.753 ms with 0 misses while retaining 2.736 MB/s of background progress.

## Metrics

| Mode | p99 ms | Deadline misses | Background MB/s |
| --- | ---: | ---: | ---: |
| `app_only` | 6.436 | 0 | 0.000 |
| `unthrottled_storage` | 13.822 | 1 | 6.793 |
| `simple_controller` | 8.185 | 0 | 2.268 |
| `aegis_policy` | 8.753 | 0 | 2.736 |

## Paper locations

| Location | Source line | Present |
| --- | ---: | ---: |
| abstract | `Paper/main.tex:89` | `True` |
| introduction | `Paper/1_Introduction.tex:38` | `True` |
| evaluation | `Paper/4_Evaluation.tex:108` | `True` |
| scope_boundary | `Paper/4_Evaluation.tex:108` | `True` |

## Checks

| Check | Pass |
| --- | ---: |
| `bundle_overall_pass` | `True` |
| `required_modes_available` | `True` |
| `recovery_checks_pass` | `True` |
| `component_coverage_pass` | `True` |
| `figure_data_matches_bundle` | `True` |
| `figure_script_exists` | `True` |
| `figure_pdf_exists` | `True` |
| `paper_locations_present` | `True` |
| `paper_pages_12` | `True` |
| `aegis_recovers_p99` | `True` |
| `aegis_removes_deadline_misses` | `True` |
| `aegis_retains_more_background_than_simple` | `True` |
