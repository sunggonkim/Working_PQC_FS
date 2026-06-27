# Hardware Freshness Recovery Matrix

- Overall pass: `true`
- Scope: Recovery-case matrix for the current hardware-freshness scope.  Real TPM artifacts cover provisioned replay-after-advance, missing index, changed-PCR transient probe behavior, and wrong authorization.  New-disk/stale-TPM and interrupted-NV-update rows use final-binary command-path fault injection and do not model physical power loss.

| Case | Oracle verdict | Scope | Pass |
| --- | --- | --- | --- |
| `stale_disk_new_tpm` | `fail_closed` | real_tpm_replay_after_advance | `true` |
| `new_disk_stale_tpm` | `latest_committed` | final_binary_command_path_fault_injection_no_live_nv_mutation | `true` |
| `missing_index` | `fail_closed` | retained_final_binary_startup_probe | `true` |
| `changed_pcrs` | `transient_probe_rejects_drift` | transient_pcr_probe_not_persistent_filesystem_anchor | `true` |
| `authorization_failure` | `fail_closed` | real_tpm_non_destructive_wrong_auth_probe | `true` |
| `interrupted_nv_update` | `fail_closed_update_not_committed` | final_binary_command_path_fault_injection_not_physical_power_loss | `true` |
| `normal_replay_after_advance` | `fail_closed` | real_tpm_application_replay_after_advance | `true` |

## Raw Artifacts

### `stale_disk_new_tpm`

- Scenario: restore stale backing-directory snapshot after the TPM-backed anchor has advanced
- Evidence: restored snapshot mounted but payload read failed closed (rc=1): Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/usr/lib/python3.12/pathlib.py", line 1021, in read_bytes
    with self.open(mode='rb') as f:
         ^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/pathlib.py", line 1015, in open
    return io.open(self, mode, buffering, encoding, errors, newline)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
OSError: [Errno 5] Input/output error: '/tmp/tpm_mono_mnt_6z1mogbm/payload.bin'
- `artifacts/validation/tpm_monotonic_replay/tpm_monotonic_replay.json`
- `artifacts/validation/tpm_monotonic_replay/replay_mount/pqc_fuse.stdout.txt`
- `artifacts/validation/tpm_monotonic_replay/replay_mount/pqc_fuse.stderr.txt`
- `artifacts/validation/tpm_monotonic_replay/replay_mount/unmount.stdout.txt`
- `artifacts/validation/tpm_monotonic_replay/replay_mount/unmount.stderr.txt`

### `new_disk_stale_tpm`

- Scenario: local committed prefix is ahead of the TPM anchor fixture
- Evidence: anchor load accepts stored.global_sequence <= local sequence; stale TPM lag is not treated as disk rollback
- `artifacts/validation/hardware_freshness_recovery_matrix/fault_injection/new_disk_stale_tpm/new_disk_stale_tpm.json`
- `artifacts/validation/hardware_freshness_recovery_matrix/fault_injection/new_disk_stale_tpm/anchor_self_test.stdout.txt`
- `artifacts/validation/hardware_freshness_recovery_matrix/fault_injection/new_disk_stale_tpm/anchor_self_test.stderr.txt`
- `artifacts/validation/hardware_freshness_recovery_matrix/fault_injection/new_disk_stale_tpm/capture/nvwrite_input.bin`
- `artifacts/validation/hardware_freshness_recovery_matrix/fault_injection/new_disk_stale_tpm/capture/stale_tpm_anchor.bin`

### `missing_index`

- Scenario: start hardware-anchor mode when the configured NV index is absent
- Evidence: exit=1; stderr contains freshness-anchor ENODEV=True
- `artifacts/validation/tpm_unprovisioned.json`
- `artifacts/validation/tpm_unprovisioned.stdout`
- `artifacts/validation/tpm_unprovisioned.stderr`

### `changed_pcrs`

- Scenario: transient PCR-policy object unseals on current PCR digest and rejects a drifted PCR digest
- Evidence: good_unseal=True; drift_rejected=True
- `artifacts/validation/tpm_pcr_policy_probe/tpm_pcr_policy_probe.json`
- `artifacts/validation/tpm_pcr_policy_probe/unseal_drift.stderr.txt`

### `authorization_failure`

- Scenario: read the provisioned NV index with deliberate wrong owner authorization
- Evidence: TPM tool reports authorization failure without exposing NV data
- `artifacts/validation/hardware_freshness_recovery_matrix/probes/auth_failure.json`
- `artifacts/validation/hardware_freshness_recovery_matrix/probes/auth_failure.stdout.txt`
- `artifacts/validation/hardware_freshness_recovery_matrix/probes/auth_failure.stderr.txt`

### `interrupted_nv_update`

- Scenario: interrupt tpm2_nvwrite before the anchor update commits
- Evidence: final binary returns nonzero when the hardware-anchor update command is interrupted
- `artifacts/validation/hardware_freshness_recovery_matrix/fault_injection/interrupted_nv_update/interrupted_nv_update.json`
- `artifacts/validation/hardware_freshness_recovery_matrix/fault_injection/interrupted_nv_update/anchor_self_test.stdout.txt`
- `artifacts/validation/hardware_freshness_recovery_matrix/fault_injection/interrupted_nv_update/anchor_self_test.stderr.txt`

### `normal_replay_after_advance`

- Scenario: same-backing-store SQLite and dbm.dumb replay-after-advance campaigns
- Evidence: sqlite=stale snapshot mounted but SQLite read failed closed rc=1; dbm=stale dbm snapshot mounted but dbm read failed closed rc=1
- `artifacts/validation/combined_durability_bundle/combined_durability_bundle.json`
- `artifacts/validation/combined_durability_bundle/sqlite_replay_read.stdout.txt`
- `artifacts/validation/combined_durability_bundle/sqlite_replay_read.stderr.txt`
- `artifacts/validation/combined_durability_bundle/dbm_replay_read.stdout.txt`
- `artifacts/validation/combined_durability_bundle/dbm_replay_read.stderr.txt`
