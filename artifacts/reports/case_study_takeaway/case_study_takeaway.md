# Case-study takeaway audit

- Overall pass: `True`
- Paper pages: `12`
- Hero id: `sqlite-mounted-qos-recovery-2026-06-27`
- Hero artifact: `artifacts/validation/qos_sqlite_hero_bundle/qos_sqlite_hero_bundle.json`

## Takeaway

The evaluated deployment envelope is a local edge service whose foreground SQLite database shares mounted secure storage with elastic background writes. This case study is the same \texttt{sqlite-mounted-qos-recovery-2026-06-27} contract used by Figure~\ref{fig:first_page_qos} and Table~\ref{tab:qos_sqlite_recovery}, not a separate anecdotal demo. The useful property is policy separation plus TPM replay-after-advance fail-closed behavior; it does not extend to kernel-encryption replacement, TensorRT p99 recovery, unmanaged power loss, or CUDA-independent deployment.

## Required Terms

| Term | Present |
| --- | ---: |
| `local edge service` | `True` |
| `foreground SQLite database` | `True` |
| `mounted secure storage` | `True` |
| `elastic background writes` | `True` |
| `sqlite-mounted-qos-recovery-2026-06-27` | `True` |
| `Figure~\ref{fig:first_page_qos}` | `True` |
| `Table~\ref{tab:qos_sqlite_recovery}` | `True` |
| `not a separate anecdotal demo` | `True` |

## Boundary Terms

| Term | Present |
| --- | ---: |
| `kernel-encryption replacement` | `True` |
| `TensorRT p99 recovery` | `True` |
| `unmanaged power loss` | `True` |
| `CUDA-independent deployment` | `True` |

## Checks

| Check | Pass |
| --- | ---: |
| `takeaway_subsection_present` | `True` |
| `takeaway_terms_present` | `True` |
| `boundary_terms_present` | `True` |
| `hero_contract_pass` | `True` |
| `hero_artifact_present` | `True` |
| `hero_id_matches_takeaway` | `True` |
| `figure_label_present` | `True` |
| `table_label_present` | `True` |
| `evaluation_uses_same_table` | `True` |
| `paper_pages_12` | `True` |
