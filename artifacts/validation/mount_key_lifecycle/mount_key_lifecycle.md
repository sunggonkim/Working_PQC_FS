# Key-Management Lifecycle Manifest

- Overall pass: `true`
- Scope: Key-management lifecycle closeout for the mounted prototype. It covers the password-derived mount key, scrypt/PBKDF2 boundary, per-file envelope secret, AES-GCM data-block key, mount-lifetime ML-KEM keypair, freshness anchor material, TPM NV/PCR policy status, open-file rekey, recovery, rotation boundary, and failure modes. It does not claim hardware-backed credential release, deployed credential rotation, persistent PCR-bound key release, transactional rewrap recovery, or credential-only rollback resistance.

## Close Conditions

- `production_lifecycle_table_complete`: `true`
- `source_boundary_matches_lifecycle_table`: `true`
- `retained_d1_d2_c5_evidence_linked`: `true`
- `paper_summarizes_key_lifecycle_and_non_claims`: `true`

## lifecycle_table_checks

- `all_required_materials_present`: `true`
- `all_names_match_materials`: `true`
- `all_rows_have_owner_producer_storage_protector_recovery`: `true`
- `data_key_is_only_data_plane_critical`: `true`
- `tpm_pcr_policy_is_non_claim`: `true`
- `no_material_is_hardware_released`: `true`
- `self_test_validates_table`: `true`
- `self_test_wired_to_binary`: `true`
- `build_includes_lifecycle_source`: `true`

## source_checks

- `runtime_requires_mount_password`: `true`
- `new_roots_use_scrypt_metadata`: `true`
- `legacy_pbkdf2_is_compatibility_path`: `true`
- `envelope_wrapped_under_mount_key_and_file_id`: `true`
- `envelope_hmac_before_unwrap`: `true`
- `tampered_envelope_rejects`: `true`
- `data_plane_uses_aes_gcm_aad`: `true`
- `rekey_refreshes_open_file_envelope_not_data_key_rotation`: `true`
- `freshness_anchor_uses_file_or_tpm_nv_backend`: `true`
- `persistent_pcr_key_release_not_in_anchor_source`: `true`

## artifact_checks

- `d1_kdf_verdict_passes`: `true`
- `d2_crypto_plane_guard_passes`: `true`
- `c5_freshness_ladder_guard_passes`: `true`
- `tpm_policy_scopes_no_credential_release`: `true`
- `clean_remount_passes`: `true`
- `tampered_envelope_rejected`: `true`
- `keyplane_workflow_passes`: `true`
- `generation_matrix_passes`: `true`
- `file_anchor_old_state_is_negative_control`: `true`
- `tpm_old_state_fails_closed`: `true`

## paper_gates

- `scrypt_new_root`: `true`
- `mount_key_not_hardware_released`: `true`
- `no_hardware_credential_release`: `true`
- `rekey_boundary`: `true`
- `runtime_epoch_boundary`: `true`
- `no_transactional_rewrap`: `true`
- `envelope_not_rollback`: `true`
- `bulk_data_boundary`: `true`
- `persistent_pcr_nonclaim`: `true`

## Lifecycle Rows

### mount-key

- Material: `PQC_KEY_MATERIAL_MOUNT_KEY`
- Plane/status: `PQC_KEY_LIFECYCLE_PLANE_KEY` / `PQC_KEY_LIFECYCLE_STATUS_IMPLEMENTED`
- Owner: pqc_keyring.c
- Producer: pqc_keyring_derive_master_key
- Storage: process memory g_master_key; .pqc_kdf stores KDF metadata
- Protector: OpenSSL scrypt for new roots; PBKDF2 legacy compatibility only
- Rotation: cleared on runtime cleanup and rederived on mount
- Recovery: password plus canonical .pqc_kdf metadata
- Failure boundary: invalid KDF metadata rejects; no TPM/PCR key release

### per-file-envelope-secret

- Material: `PQC_KEY_MATERIAL_FILE_ENVELOPE_SECRET`
- Plane/status: `PQC_KEY_LIFECYCLE_PLANE_KEY` / `PQC_KEY_LIFECYCLE_STATUS_IMPLEMENTED`
- Owner: pqc_keyring.c,pqc_fd_context.c,pqc_rekey.c
- Producer: pqc_keyring_metadata_store/load and rekey worker
- Storage: user.pqc_metadata xattr plus fd-lifecycle memory
- Protector: HMAC-SHA256 metadata check and mount-key/file-id wrapping
- Rotation: explicit rekey worker refresh when configured
- Recovery: xattr unwrap under the current mount key and file id
- Failure boundary: missing or tampered xattr rejects authenticated open

### aes-gcm-data-block-key

- Material: `PQC_KEY_MATERIAL_DATA_BLOCK_KEY`
- Plane/status: `PQC_KEY_LIFECYCLE_PLANE_DATA` / `PQC_KEY_LIFECYCLE_STATUS_IMPLEMENTED`
- Owner: pqc_crypto.c,pqc_writeback.c,pqc_file_io.c
- Producer: fd-context per-file secret loaded from keyring metadata
- Storage: fd-lifecycle memory only
- Protector: AES-256-GCM nonce/AAD bound to file id, block, generation, length
- Rotation: inherits per-file envelope-secret refresh
- Recovery: reload per-file envelope secret before decrypting blocks
- Failure boundary: tag mismatch returns authentication failure

### mount-lifetime-kem-keypair

- Material: `PQC_KEY_MATERIAL_MOUNT_KEM_KEYPAIR`
- Plane/status: `PQC_KEY_LIFECYCLE_PLANE_KEY` / `PQC_KEY_LIFECYCLE_STATUS_IMPLEMENTED`
- Owner: pqc_runtime.c,pqc_rekey.c
- Producer: OQS_KEM_keypair at runtime init
- Storage: mount-lifetime process memory
- Protector: process isolation and explicit cleanup zeroization
- Rotation: new keypair on each mount
- Recovery: not persisted; remount generates a new keypair
- Failure boundary: keypair failure aborts runtime init

### committed-prefix-freshness-anchor

- Material: `PQC_KEY_MATERIAL_FRESHNESS_ANCHOR`
- Plane/status: `PQC_KEY_LIFECYCLE_PLANE_FRESHNESS` / `PQC_KEY_LIFECYCLE_STATUS_IMPLEMENTED`
- Owner: pqc_anchor.c,pqc_checkpoint.c,pqc_anchor_worker.c
- Producer: committed-prefix root over per-file generation state
- Storage: file backend or administrator-provisioned TPM NV backend
- Protector: prefix-anchor digest and fail-closed load checks
- Rotation: advanced by checkpoint/freshness-anchor publication
- Recovery: anchor load compares stored prefix against reconstructed state
- Failure boundary: file backend is replayable; TPM path depends on provisioning

### persistent-tpm-pcr-key-release-policy

- Material: `PQC_KEY_MATERIAL_TPM_PCR_POLICY`
- Plane/status: `PQC_KEY_LIFECYCLE_PLANE_FRESHNESS` / `PQC_KEY_LIFECYCLE_STATUS_NON_CLAIM`
- Owner: pqc_anchor.c
- Producer: not implemented in the mounted production path
- Storage: none in this revision
- Protector: none; persistent PCR-bound key release is a non-claim
- Rotation: not implemented
- Recovery: not implemented
- Failure boundary: must not be used to claim sealed-key or PCR-bound recovery

## Non-Claims

- no hardware-backed credential release path for the mount key
- no deployed mount credential rotation
- no persistent KEM hierarchy
- no persistent PCR-bound key release
- no TPM/PCR sealed-key recovery
- no persistent epoch anti-rollback journal in the mounted path
- no transactional rewrap recovery claim
- no credential-only rollback resistance
- no recovery after a lost mount credential
