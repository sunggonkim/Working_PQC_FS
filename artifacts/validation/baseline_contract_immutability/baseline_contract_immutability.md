# Baseline Contract Immutability Verdict

- Gate: `B4`
- Submilestone: `B4-S0`
- Overall pass: `true`
- Contract ID: `aegisq-fs-frozen-v2-2026-06-27`
- Contract SHA-256: `72248aae3fd02b92b88f23ac0989e38960f3cc935a9799d361293ea5e50101d6`
- Workload profile: `fio_randrw_4k_70r30w_fdatasync_qd1`
- Expected repetitions: `5`
- Command template: `fio --name=frozen_randrw_4k_fdatasync --ioengine=psync --rw=randrw --rwmixread=70 --bs=4k --direct=0 --fdatasync=1 --iodepth=1 --numjobs=1 --size=1G --fallocate=none --allow_file_create=0 --overwrite=1 --time_based --runtime=60 --ramp_time=10 --directory=${BENCH_DIR} --output-format=json`

## Rows

### `plaintext_lowerfs`

- Status: `measured`
- Pass: `true`
- Artifact: `artifacts/validation/frozen_plaintext_contract/frozen_plaintext_contract.json`
- Paper status: `warm-cache measured row`
- Cache state: `warm`
- Warm valid repetitions: `5` / `5`
- Invalid-run reasons: `cold_cache:invalid_not_run, comparison_ready:false`
- Failed checks: `none`

### `gocryptfs`

- Status: `measured`
- Pass: `true`
- Artifact: `artifacts/validation/frozen_gocryptfs_contract/frozen_gocryptfs_contract.json`
- Paper status: `warm-cache measured row`
- Cache state: `warm`
- Warm valid repetitions: `5` / `5`
- Invalid-run reasons: `cold_cache:invalid_not_run, comparison_ready:false`
- Failed checks: `none`

### `dm_crypt_ext4`

- Status: `measured`
- Pass: `true`
- Artifact: `artifacts/validation/frozen_dmcrypt_contract/frozen_dmcrypt_contract.json`
- Paper status: `warm-cache measured row`
- Cache state: `warm`
- Warm valid repetitions: `5` / `5`
- Invalid-run reasons: `cold_cache:invalid_not_run, comparison_ready:false`
- Failed checks: `none`

### `aegis_q`

- Status: `measured`
- Pass: `true`
- Artifact: `artifacts/validation/frozen_aegisq_contract/frozen_aegisq_contract.json`
- Paper status: `warm-cache measured row`
- Cache state: `warm`
- Warm valid repetitions: `5` / `5`
- Invalid-run reasons: `cold_cache:invalid_not_run, comparison_ready:false`
- Failed checks: `none`

### `fscrypt`

- Status: `environment-blocked`
- Pass: `true`
- Artifact: `artifacts/validation/kernel_baseline_feasibility/paper_fscrypt_verdict.json`
- Paper status: `environment-blocked with proof`
- Cache state: `not run`
- Warm valid repetitions: `0` / `5`
- Blocking reasons: `noninteractive_sudo_unavailable, kernel_config_fs_encryption_disabled, root_ext4_encrypt_feature_not_enabled, fscrypt_status_nonzero`
- Invalid-run reasons: `noninteractive_sudo_unavailable, kernel_config_fs_encryption_disabled, root_ext4_encrypt_feature_not_enabled, fscrypt_status_nonzero`
- Failed checks: `none`

## Paper Guard

- Failed paper checks: `none`
- Bad historical lines: `0`
- Bad measured-kernel lines: `0`

## Top-Level Checks

- `master_contract_complete`: `true`
- `all_required_rows_accounted`: `true`
- `all_rows_have_valid_status`: `true`
- `measured_rows_tied_to_same_contract`: `true`
- `blocked_rows_have_proof`: `true`
- `paper_status_matches_rows`: `true`
