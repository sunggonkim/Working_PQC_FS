# gocryptfs Frozen Workload Contract Run

- Overall pass: `true`
- Contract ID: `aegisq-fs-frozen-v2-2026-06-27`
- gocryptfs version: `gocryptfs 2.4.0; go-fuse 2.4.2; 2025-06-24 go1.22.2 linux/arm64`
- Scope: gocryptfs-only warm-cache execution of the frozen filesystem workload contract.  It can serve as the user-space encrypted filesystem baseline row once compared with other retained rows.
- File preparation valid: `true`
- Warm-cache valid repetitions: `5`
- Cold-cache status: `invalid_not_run`

## Warm-Cache Summary

- `throughput_mib_s` median `21.7125`, 95% CI [`20.9229`, `21.9221`]
- `latency_p50_us` median `38.656`, 95% CI [`37.632`, `40.704`]
- `latency_p95_us` median `50.944`, 95% CI [`49.92`, `51.456`]
- `latency_p99_us` median `60.672`, 95% CI [`59.648`, `60.672`]
- `latency_p99_9_us` median `183.296`, 95% CI [`118.272`, `272.384`]

## Retained Artifacts

- JSON summary: `artifacts/validation/frozen_gocryptfs_contract/frozen_gocryptfs_contract.json`
- CSV repetitions: `artifacts/validation/frozen_gocryptfs_contract/frozen_gocryptfs_repetitions.csv`
- File preparation: `artifacts/validation/frozen_gocryptfs_contract/file_preparation.json`
- Raw fio directory: `artifacts/validation/frozen_gocryptfs_contract/fio_raw`
- Mount logs: `artifacts/validation/frozen_gocryptfs_contract/mount_logs`
- Platform manifest: `artifacts/validation/frozen_gocryptfs_contract/platform_manifest.json`
- Thermal log: `artifacts/validation/frozen_gocryptfs_contract/thermal_tegrastats.log`

## Non-Claims

- This is not a complete plaintext/fscrypt/dm-crypt comparison matrix.
- The cold-cache row is not reported as a result unless privileged cache dropping is available.
