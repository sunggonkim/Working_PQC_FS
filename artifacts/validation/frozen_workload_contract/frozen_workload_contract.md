# Frozen Filesystem Workload Contract

- Contract ID: `aegisq-fs-frozen-v2-2026-06-27`
- Contract SHA-256: `72248aae3fd02b92b88f23ac0989e38960f3cc935a9799d361293ea5e50101d6`
- Overall pass: `true`
- Contract complete: `true`
- Current host execution ready: `false`
- Scope: this is a benchmark contract, not a benchmark result.

## Required workload

- Profile: `fio_randrw_4k_70r30w_fdatasync_qd1`
- Request size: `4096` bytes
- Read/write mix: `70`/`30`
- Sync mode: `psync` with fdatasync per write
- Queue depth: `1`
- Client count: `1`
- File size: `1073741824` bytes
- Repetitions: `5`
- Confidence interval: `nonparametric bootstrap` 95% with `10000` resamples
- File preparation: `Before each mode's warm-cache sequence, create ${BENCH_DIR}, create ${BENCH_DIR}/frozen_randrw_4k_fdatasync.0.0, and set its logical length to 1 GiB with posix_fallocate or an equivalent sparse truncate path recorded by the harness.  fio then runs with allow_file_create=0, overwrite=1, and fallocate=none so file-layout writes are not folded into the random read/write workload.`
- Fio command: `fio --name=frozen_randrw_4k_fdatasync --ioengine=psync --rw=randrw --rwmixread=70 --bs=4k --direct=0 --fdatasync=1 --iodepth=1 --numjobs=1 --size=1G --fallocate=none --allow_file_create=0 --overwrite=1 --time_based --runtime=60 --ramp_time=10 --directory=${BENCH_DIR} --output-format=json`

## Cache states

- `warm`: sparse-create the 1 GiB fio file by the file-preparation procedure, run one untimed fio pass with the same profile, then run the measured repetitions without dropping caches
- `cold`: sync, unmount the tested filesystem mode, drop page cache with /proc/sys/vm/drop_caches when privileged, remount the mode, and run the measured repetition; if this cannot be performed, mark the cold-cache row invalid instead of folding it into warm-cache results

## Required environment

- CPU governor: `performance`
- Thermal mode: fixed max-performance/power mode with active cooling; tegrastats must be retained and any thermal-throttle indication invalidates the run
- Storage device: `WD PC SN5000S SDEPNSJ-1T00 NVMe, /dev/nvme0n1`
- Lower filesystem: `ext4` on `/dev/nvme0n1p1` with `rw,relatime`

## Filesystem modes

- `plaintext_lowerfs`: raw lower-filesystem control; mount options `{"encryption": "none", "filesystem": "ext4", "options": "rw,relatime", "source": "/dev/nvme0n1p1"}`
- `aegis_q`: prototype secure-storage path; mount options `{"file_tier": "default encrypted data path; no latency-sensitive xattr", "fuse": "fuse3", "mount_command": "build/pqc_fuse ${AEGIS_BACKING} ${AEGIS_MOUNT} -f", "required_environment": {"PQC_FRESHNESS_ANCHOR_BACKEND": "file", "PQC_MASTER_PASSWORD": "set by harness secret file", "PQC_QOS_MODE": "disabled for this filesystem-throughput profile"}}`
- `gocryptfs`: user-space encrypted-filesystem baseline; mount options `{"cipher_directory": "${GOCRYPTFS_CIPHER}", "mount_command": "gocryptfs -fg -passfile ${PASSFILE} ${GOCRYPTFS_CIPHER} ${GOCRYPTFS_MOUNT}", "plaintext_names": false, "reverse_mode": false}`
- `fscrypt`: kernel filesystem-encryption baseline; mount options `{"filesystem": "ext4 with encrypt feature", "lower_device_policy": "same NVMe device class as plaintext and AEGIS-Q", "mount_command": "mount ${FSCRYPT_BLOCK_DEVICE} ${FSCRYPT_ROOT}", "policy": "fscrypt v2 policy with custom passphrase"}`
- `dm_crypt_ext4`: kernel block-encryption baseline; mount options `{"cryptsetup": "LUKS2, aes-xts-plain64, 512-bit key", "discard": "disabled", "filesystem": "ext4 inside /dev/mapper/${DMCRYPT_NAME}", "mount_command": "cryptsetup open ${DMCRYPT_BLOCK_DEVICE} ${DMCRYPT_NAME}; mount /dev/mapper/${DMCRYPT_NAME} ${DMCRYPT_MOUNT}"}`

## Validity rules

- Retain raw fio JSON, stdout/stderr, exact command lines, mount logs, and version strings for every mode.
- Do not compare a mode if its warm/cold cache state, queue depth, sync mode, file size, or repetition count deviates from the contract.
- Do not include fio file-layout writes in measured rows; the harness must precreate the 1 GiB file and run fio with allow_file_create=0, overwrite=1, and fallocate=none.
- Report five-repetition medians and bootstrap 95% confidence intervals; label any smaller run as smoke-only.
- Existing sequential fscrypt/dm-crypt fio reference outputs are not contract-compliant until rerun with this profile.

## Validation

- Missing fields: `0`
- Current-host warnings: `2`
- Warning: current host governor is not uniformly performance; set it before executing the contract
- Warning: jetson_clocks state was not captured successfully on this host
