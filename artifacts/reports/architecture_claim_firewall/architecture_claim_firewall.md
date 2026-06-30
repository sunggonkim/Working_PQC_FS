# Architecture Claim Firewall

- Overall pass: `True`
- Scanned files: `336`
- Candidate hits: `306`
- Violations: `0`

## Checks

| Check | Pass |
| --- | ---: |
| `scan_roots_present` | `True` |
| `all_terms_configured` | `True` |
| `no_unguarded_architecture_claims` | `True` |
| `guarded_contexts_not_serialized` | `True` |

## Term Summary

| Term | Candidates | Guarded | Violations |
| --- | ---: | ---: | ---: |
| `ebpf_passthrough` | `17` | `17` | `0` |
| `async_epoch_fdatasync` | `3` | `3` | `0` |
| `shadow_mmap` | `4` | `4` | `0` |
| `out_of_place_update` | `1` | `1` | `0` |
| `argon2id` | `7` | `7` | `0` |
| `async_merkle` | `11` | `11` | `0` |
| `merkle_root` | `13` | `13` | `0` |
| `tpm_epoch` | `17` | `17` | `0` |
| `tpm_rollback_resistance` | `42` | `42` | `0` |
| `pcr` | `170` | `170` | `0` |
| `jetson_zero_copy` | `15` | `15` | `0` |
| `cuda_stream_priority` | `6` | `6` | `0` |

## Negative Claim Guard

Architecture mechanism phrases may appear only when they are tied to an implemented gate, explicit future/blocked work, a probe or diagnostic, related-work comparison, or an explicit non-claim.

Guarded candidate contexts are intentionally not serialized; the reviewer-facing report contains only aggregate counts and unguarded violations.

