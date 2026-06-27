# gocryptfs Frozen Workload Contract Run

- Overall pass: `true`
- Contract ID: `aegisq-fs-frozen-v2-2026-06-27`
- gocryptfs version: `gocryptfs 2.4.0; go-fuse 2.4.2; 2025-06-24 go1.22.2 linux/arm64`
- Scope: gocryptfs-only warm-cache execution of the frozen filesystem workload contract.  It can serve as the user-space encrypted filesystem baseline row once compared with other retained rows.
- File preparation valid: `true`
- Warm-cache valid repetitions: `5`
- Cold-cache status: `invalid_not_run`

## Warm-Cache Summary

- `throughput_mib_s` median `21.7205`, 95% CI [`21.1897`, `21.8586`]
- `latency_p50_us` median `38.144`, 95% CI [`37.632`, `39.68`]
- `latency_p95_us` median `49.92`, 95% CI [`49.408`, `50.432`]
- `latency_p99_us` median `60.16`, 95% CI [`59.648`, `62.208`]
- `latency_p99_9_us` median `191.488`, 95% CI [`173.056`, `284.672`]

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
