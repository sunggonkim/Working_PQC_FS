# Integrity Comparison Manifest

- Overall pass: `true`
- Scope: Non-numeric integrity-oriented comparison.  The current Thor kernel does not expose fs-verity or dm-integrity, and their protection/update models are not matched throughput baselines for mutable AEGIS-Q FUSE records.

## Checks

- `kernel_config_records_fsverity_disabled`: `true`
- `kernel_config_records_dmintegrity_disabled`: `true`
- `aegisq_integrity_scope_audit_pass`: `true`
- `all_required_systems_present`: `true`
- `all_rows_explain_boundary_update_and_no_number`: `true`
- `paper_scope_gate_pass`: `true`

## Systems

### fs-verity

- Kernel config: `CONFIG_FS_VERITY` = `not_set`
- Runnable on current host: `false`
- Protection boundary: Per-file read-only authenticity for file contents using a Merkle tree whose root is supplied to the filesystem.
- Update model: Files become immutable after verity is enabled; mutable overwrite, journal/checkpoint publication, and SQLite-style updates are not the same workload.
- Why no throughput number: A fio overwrite/fdatasync number would measure an invalid update model: fs-verity protects sealed read-only files, while AEGIS-Q evaluates mutable encrypted records.

### dm-integrity

- Kernel config: `CONFIG_DM_INTEGRITY` = `not_set`
- Runnable on current host: `false`
- Protection boundary: Block-device integrity tags below the filesystem, optionally composed with dm-crypt/LUKS authenticated encryption.
- Update model: Block-level tag updates and dm-integrity journal or bitmap policy; no per-file envelope, checkpoint, or application oracle semantics.
- Why no throughput number: A number would depend on unavailable kernel support and on a chosen tag/journal mode, and would not isolate AEGIS-Q's per-file mutable FUSE publication protocol.

### AEGIS-Q

- Kernel config: `none` = `userspace FUSE prototype`
- Runnable on current host: `true`
- Protection boundary: Per-file encrypted record format with AEAD tags, HMAC-authenticated envelope/checkpoint metadata, and optional external freshness anchor.
- Update model: Mutable FUSE reads/writes with data-before-journal/checkpoint publication; selected recovery states use explicit application oracles.
- Why no throughput number: The existing AEGIS-Q throughput row is a mutable secure-storage result, not an integrity-only comparison against read-only fs-verity or block-tag dm-integrity.

## Non-Claims

- no fs-verity throughput result
- no dm-integrity throughput result
- no persisted per-file content Merkle tree in AEGIS-Q
- no replacement claim for kernel integrity mechanisms
