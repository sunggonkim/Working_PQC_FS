# Plaintext Frozen Workload Contract Run

- Overall pass: `true`
- Contract ID: `aegisq-fs-frozen-v2-2026-06-27`
- Scope: Plaintext lower-filesystem warm-cache execution of the frozen filesystem workload contract.  It separates raw lowerfs behavior from encrypted FUSE/kernel rows but is not a full matrix by itself.
- File preparation valid: `true`
- Warm-cache valid repetitions: `5`
- Cold-cache status: `invalid_not_run`

## Warm-Cache Summary

- `throughput_mib_s` median `32.8798`, 95% CI [`30.1556`, `36.9555`]
- `latency_p50_us` median `50.432`, 95% CI [`3.344`, `50.944`]
- `latency_p95_us` median `132.096`, 95% CI [`98.816`, `134.144`]
- `latency_p99_us` median `164.864`, 95% CI [`156.672`, `166.912`]
- `latency_p99_9_us` median `296.96`, 95% CI [`272.384`, `305.152`]

## Retained Artifacts

- JSON summary: `artifacts/validation/frozen_plaintext_contract/frozen_plaintext_contract.json`
- CSV repetitions: `artifacts/validation/frozen_plaintext_contract/frozen_plaintext_repetitions.csv`
- File preparation: `artifacts/validation/frozen_plaintext_contract/file_preparation.json`
- Raw fio directory: `artifacts/validation/frozen_plaintext_contract/fio_raw`
- Platform manifest: `artifacts/validation/frozen_plaintext_contract/platform_manifest.json`
- Thermal log: `artifacts/validation/frozen_plaintext_contract/thermal_tegrastats.log`

## Non-Claims

- This is not a complete fscrypt/dm-crypt comparison matrix.
- The cold-cache row is not reported as a result unless privileged cache dropping is available.
