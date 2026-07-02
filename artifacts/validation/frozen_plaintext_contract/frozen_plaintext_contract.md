# Plaintext Frozen Workload Contract Run

- Overall pass: `true`
- Contract ID: `aegisq-fs-frozen-v2-2026-06-27`
- Scope: Plaintext lower-filesystem warm-cache execution of the frozen filesystem workload contract.  It separates raw lowerfs behavior from encrypted FUSE/kernel rows but is not a full matrix by itself.
- File preparation valid: `true`
- Warm-cache valid repetitions: `5`
- Cold-cache status: `invalid_not_run`

## Warm-Cache Summary

- `throughput_mib_s` median `30.2377`, 95% CI [`22.6608`, `36.6312`]
- `latency_p50_us` median `49.92`, 95% CI [`5.472`, `63.232`]
- `latency_p95_us` median `56.064`, 95% CI [`51.968`, `99.84`]
- `latency_p99_us` median `79.36`, 95% CI [`55.04`, `144.384`]
- `latency_p99_9_us` median `209.92`, 95% CI [`134.144`, `399.36`]

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
