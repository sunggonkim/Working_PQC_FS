# Mount-Key Lifecycle Manifest

- Overall pass: `true`
- Scope: Mount-key lifecycle scope artifact.  It supports authenticated storage-format correctness, clean remount, tamper rejection, and mounted open-file envelope refresh.  It does not claim hardware-backed credential release, deployed credential rotation, transactional rewrap recovery, or credential-only rollback resistance.

## Checks

- `all_required_decisions_present`: `true`
- `all_rows_have_evidence_and_scope`: `true`
- `source_checks_pass`: `true`
- `artifact_checks_pass`: `true`
- `paper_covers_all_lifecycle_decisions`: `true`

## Source Checks

- `mount_password_required`: `true`
- `pbkdf2_sha256_master_key`: `true`
- `random_dek_and_file_id_on_create`: `true`
- `metadata_store_wraps_under_mount_key`: `true`
- `metadata_store_hmac_authenticates_envelope`: `true`
- `metadata_load_hmac_before_unwrap`: `true`
- `metadata_load_rejects_bad_hmac`: `true`
- `rekey_worker_refreshes_open_file_secret`: `true`
- `rekey_worker_updates_runtime_epoch`: `true`
- `rekey_rewrap_not_transactional`: `true`
- `mounted_target_excludes_legacy_file_key_helper`: `true`
- `legacy_epoch_helper_is_not_mounted_path`: `true`

## Artifact Checks

- `clean_remount_passes`: `true`
- `tampered_envelope_rejected`: `true`
- `keyplane_workflow_passes`: `true`
- `keyplane_methodology_passes`: `true`
- `generation_matrix_passes`: `true`
- `file_anchor_old_state_is_negative_control`: `true`
- `tpm_old_state_fails_closed`: `true`
- `tpm_policy_scopes_no_credential_release`: `true`

## Lifecycle Rows

### password_derived_mount_key_boundary

- Current evidence: pqc_subsystem_init() requires PQC_MASTER_PASSWORD, derives g_master_key with PBKDF2-HMAC-SHA256, and pqc_create() generates a random 256-bit per-file DEK plus file identifier.
- Policy or scope: The mount key is the prototype root credential.  This closes only the storage-format correctness boundary, not deployed credential protection.
- Paper gate phrase: `mount key is password-derived and never hardware-released`

### hardware_backed_credential_release_plan

- Current evidence: The TPM policy manifest explicitly records no hardware-backed credential release, and the mounted FUSE source has no TPM/PCR release path for g_master_key.
- Policy or scope: Hardware-backed release remains out of scope; TPM evidence is limited to freshness-anchor behavior.
- Paper gate phrase: `no hardware-backed credential release path for the mount key`

### key_rotation

- Current evidence: The rekey worker batches open file descriptors, encapsulates fresh key material on the admitted executor, installs it in the open fd context, increments an in-memory key_epoch, and persists a new envelope.
- Policy or scope: This is open-file DEK refresh.  It is not deployed mount credential rotation, administrator key rollover, or a persistent KEM hierarchy.
- Paper gate phrase: `rekey is open-file DEK refresh rather than deployed credential rotation`

### envelope_rewrap

- Current evidence: metadata_store() masks the per-file DEK under the mount key and file id and HMAC-authenticates the envelope; metadata_load() verifies the HMAC before unwrapping.  The retained key-plane workflow refreshes 1,024 open files per mode, and the tamper regression rejects a corrupted metadata xattr with EKEYREJECTED.
- Policy or scope: Envelope rewrap evidence applies to mounted open files and authenticated open/remount behavior only.
- Paper gate phrase: `HMAC envelope rewrap for open files`

### epoch_counter_interaction

- Current evidence: The mounted path has an in-memory key_epoch incremented by the rekey worker.  The older pqc_file_key epoch/grace helper is present in source but is not part of the pqc_fuse CMake target.
- Policy or scope: Epochs are runtime bookkeeping for the evaluated path, not a persisted anti-rollback journal or stale-handle proof.
- Paper gate phrase: `runtime key epoch is in-memory bookkeeping, not a persistent anti-rollback journal`

### recovery_after_failed_rewrap

- Current evidence: No retained artifact exercises an interrupted envelope rewrap, and the rekey worker does not implement a transactional rewrap journal.  Existing retained evidence covers HMAC fail-closed behavior on later open and clean remount after normal writes.
- Policy or scope: The prototype makes no transactional recovery claim for failed rewrap; future work must add a cut-point campaign before making that claim.
- Paper gate phrase: `no transactional rewrap recovery claim`

### rollback_behavior_for_old_envelopes

- Current evidence: The generation fault matrix records file-anchor stale snapshot replay as an expected negative control and records the existing TPM stale snapshot artifact as fail closed.
- Policy or scope: A password-derived envelope by itself is replayable with a whole backing snapshot.  Rollback resistance is claimed only when generation/checkpoint and external-anchor evidence reject stale state.
- Paper gate phrase: `password-derived envelope alone is not rollback resistance`

## Non-Claims

- no hardware-backed credential release path for the mount key
- no deployed mount credential rotation
- no persistent KEM hierarchy
- no persistent epoch anti-rollback journal in the mounted path
- no transactional rewrap recovery claim
- no credential-only rollback resistance
- no recovery after a lost mount credential
