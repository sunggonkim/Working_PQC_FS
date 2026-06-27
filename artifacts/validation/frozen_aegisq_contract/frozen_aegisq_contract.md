# AEGIS-Q Frozen Workload Contract Run

- Overall pass: `true`
- Contract ID: `aegisq-fs-frozen-v2-2026-06-27`
- Scope: AEGIS-Q-only warm-cache execution of the frozen filesystem workload contract.  Baseline modes and valid cold-cache rows are separate checklist items.
- File preparation valid: `true`
- Warm-cache valid repetitions: `5`
- Cold-cache status: `invalid_not_run`
- Comparison ready: `false`

## Warm-Cache Summary

- `throughput_mib_s` median `0.358321`, 95% CI [`0.285963`, `0.546995`]
- `latency_p50_us` median `10420.2`, 95% CI [`6782.98`, `13303.8`]
- `latency_p95_us` median `11075.6`, 95% CI [`7634.94`, `13697`]
- `latency_p99_us` median `11206.7`, 95% CI [`7700.48`, `13697`]
- `latency_p99_9_us` median `11730.9`, 95% CI [`7962.62`, `13959.2`]

## Retained Artifacts

- JSON summary: `artifacts/validation/frozen_aegisq_contract/frozen_aegisq_contract.json`
- CSV repetitions: `artifacts/validation/frozen_aegisq_contract/frozen_aegisq_repetitions.csv`
- File preparation: `artifacts/validation/frozen_aegisq_contract/file_preparation.json`
- Raw fio directory: `artifacts/validation/frozen_aegisq_contract/fio_raw`
- Mount logs: `artifacts/validation/frozen_aegisq_contract/mount_logs`
- Platform manifest: `artifacts/validation/frozen_aegisq_contract/platform_manifest.json`
- Thermal log: `artifacts/validation/frozen_aegisq_contract/thermal_tegrastats.log`

## Non-Claims

- This is not a gocryptfs/fscrypt/dm-crypt/plaintext comparison.
- The cold-cache row is not reported as a result unless privileged cache dropping is available.
