# Crypto Plane Claim Guard

- Generated: `2026-07-01T13:04:43Z`
- Overall pass: `true`
- Parent D2 closed: `true`
- Unguarded D2 overclaims: `0`

## Checks

- trace_smoke_exists: `true`
- trace_smoke_overall_pass: `true`
- ordinary_io_has_aes_blocks: `true`
- ordinary_io_has_no_keyplane_batches: `true`
- ordinary_mount_skips_keyplane_startup: `true`
- ordinary_mount_skips_admission_startup: `true`
- ordinary_mount_skips_freshness_anchor: `true`
- forced_mount_starts_keyplane_worker: `true`
- forced_rekey_records_keyplane_work: `true`
- source_ordinary_writeback_cpu_first: `true`
- source_rekey_gpu_lane_reachable_by_default: `true`
- source_runtime_no_cuda_aes_prealloc: `true`
- source_runtime_disables_unused_admission: `true`
- source_anchor_backend_cached: `true`
- source_trace_byte_aggregation_gated: `true`
- paper_crypto_plane_boundary_present: `true`
- no_unguarded_d2_overclaims: `true`

## Negative Claim Guard

Bulk file data must remain described as AES-GCM block data. ML-KEM/Kyber may be described only as key/session/envelope-plane work, and ordinary read/write claims must not put rekey on the critical path or start key-plane machinery unless mounted trace evidence changes.

