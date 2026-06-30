# Dangerous Claim Lint

- Overall pass: `True`
- Scanned files: `336`
- Candidate hits: `291`
- Violations: `0`

## Checks

| Check | Pass |
| --- | ---: |
| `scan_roots_present` | `True` |
| `required_terms_represented` | `True` |
| `no_unguarded_dangerous_claims` | `True` |
| `guarded_contexts_not_serialized` | `True` |

## Term Summary

| Term | Candidates | Guarded | Violations |
| --- | ---: | ---: | ---: |
| `direct_nvme_to_uvm` | `50` | `50` | `0` |
| `gpudirect` | `60` | `60` | `0` |
| `dma_buf` | `23` | `23` | `0` |
| `ebpf_iouring_completion_bypass` | `18` | `18` | `0` |
| `persistent_pcr_bound` | `40` | `40` | `0` |
| `foreground_nonstorage_qos_recovery` | `19` | `19` | `0` |
| `full_crash_certification` | `25` | `25` | `0` |
| `side_channel_protection` | `12` | `12` | `0` |
| `general_purpose_posix` | `15` | `15` | `0` |
| `general_purpose_filesystem` | `17` | `17` | `0` |
| `ready_for_deployment` | `12` | `12` | `0` |

## Negative Claim Guard

Dangerous architecture and deployment phrases may appear only as explicit non-claims, limitations, blocked/future-work items, probe fields, related-work comparisons, or lint/checklist rules.

Guarded candidate contexts are intentionally not serialized; the reviewer-facing report contains only aggregate counts and unguarded violations.

