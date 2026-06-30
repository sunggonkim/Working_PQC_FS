# Crypto Plane Claim Guard

- Generated: `2026-06-29T08:31:43Z`
- Overall pass: `true`
- Parent D2 closed: `true`
- Unguarded D2 overclaims: `0`

## Checks

- trace_smoke_exists: `true`
- trace_smoke_overall_pass: `true`
- ordinary_io_has_aes_blocks: `true`
- ordinary_io_has_no_keyplane_batches: `true`
- forced_rekey_records_keyplane_work: `true`
- paper_crypto_plane_boundary_present: `true`
- no_unguarded_d2_overclaims: `true`

## Negative Claim Guard

Bulk file data must remain described as AES-GCM block data. ML-KEM/Kyber may be described only as key/session/envelope-plane work, and ordinary read/write claims must not put rekey on the critical path unless mounted trace evidence changes.

