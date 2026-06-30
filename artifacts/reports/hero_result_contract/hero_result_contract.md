# Hero-result contract

- Overall pass: `True`
- Hero id: `sqlite-mounted-qos-recovery-2026-06-27`
- Paper pages: `12`
- Artifact: `artifacts/validation/sqlite_hero_validity_closeout/sqlite_hero_validity_closeout.json`

## Headline claim

For foreground SQLite transactions on mounted AEGIS-Q FUSE repeated-run medians, app-only reports 7.253 ms p99, unthrottled secure-storage pressure reports 9.621 ms p99 and 6.984 MB/s background writes, and AEGIS-Q policy reports 8.151 ms p99 with 0 median misses while retaining 3.016 MB/s of background progress.

## Metrics

| Mode | p99 ms | Deadline misses | Background MB/s |
| --- | ---: | ---: | ---: |
| `app_only` | 7.253 | 0 | 0.000 |
| `unthrottled_storage` | 9.621 | 0 | 6.984 |
| `simple_controller` | 7.544 | 0 | 1.497 |
| `aegis_policy` | 8.151 | 0 | 3.016 |

## Paper locations

| Location | Source line | Present |
| --- | ---: | ---: |
| abstract | `Paper/main.tex:91` | `True` |
| introduction | `Paper/1_Introduction.tex:39` | `True` |
| evaluation | `Paper/4_Evaluation.tex:92` | `True` |
| scope_boundary | `Paper/4_Evaluation.tex:92` | `True` |

## Checks

| Check | Pass |
| --- | ---: |
| `bundle_overall_pass` | `True` |
| `closeout_overall_pass` | `True` |
| `required_modes_available` | `True` |
| `recovery_checks_pass` | `True` |
| `component_coverage_pass` | `True` |
| `figure_data_matches_bundle` | `True` |
| `figure_script_exists` | `True` |
| `figure_pdf_exists` | `True` |
| `paper_locations_present` | `True` |
| `paper_pages_12` | `True` |
| `aegis_recovers_p99` | `True` |
| `aegis_retains_more_background_than_simple` | `True` |
