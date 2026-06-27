# TPM Freshness Policy Manifest

- Overall pass: `true`
- Scope: Policy-definition artifact only.  It records the intended TPM/freshness lifecycle and gates paper wording; it does not claim persistent PCR-bound freshness or deployed credential recovery.

## Checks

- `all_required_topics_present`: `true`
- `all_rows_have_evidence_and_policy`: `true`
- `paper_covers_all_policy_topics`: `true`
- `source_and_artifact_evidence_pass`: `true`

## Source Checks

- `nvreadpublic_succeeds`: `true`
- `nv_index_is_expected`: `true`
- `nv_ownerread_ownerwrite_recorded`: `true`
- `nv_size_compatible_with_anchor_record`: `true`
- `code_uses_owner_hierarchy_nv_ops`: `true`
- `code_requires_preprovisioned_index`: `true`
- `code_detects_stale_sequence`: `true`
- `transient_pcr_probe_passes`: `true`
- `monotonic_replay_fail_closed`: `true`
- `sqlite_tpm_stale_snapshot_fail_closed`: `true`
- `dbm_tpm_stale_snapshot_fail_closed`: `true`
- `unprovisioned_tpm_fails_closed`: `true`

## Policy Rows

### nv_authorization_model

- Current evidence: The retained provisioning probe observes NV index 0x01500010 with ownerread|ownerwrite attributes, and pqc_anchor.c uses the TPM owner hierarchy for nvread/nvwrite.
- Intended policy: The prototype treats the NV index as an administrator-owned freshness register.  It reads and writes only the authenticated prefix anchor record and does not claim a deployed TPM authorization policy.
- Paper gate phrase: `owner-authorized TPM NV index`

### pcr_binding_role

- Current evidence: The transient PCR probe unseals under the current PCR file and rejects a drifted PCR file; it does not provision the persistent filesystem NV index.
- Intended policy: PCR binding is a future release gate for stronger claims, not a property of the current persistent filesystem anchor.
- Paper gate phrase: `PCR binding is a transient probe rather than a persistent filesystem policy`

### index_lifecycle

- Current evidence: pqc_anchor.c checks tpm2_nvreadpublic and refuses to define an index implicitly; the unprovisioned TPM artifact records a nonzero hardware-anchor startup exit.
- Intended policy: Administrators create and rotate the NV index outside the daemon; the daemon fails closed when the configured index is absent.
- Paper gate phrase: `administrators create and rotate the index outside the daemon`

### reprovisioning

- Current evidence: The implementation has no automatic NV define/redefine path and no artifact showing safe automatic anchor migration.
- Intended policy: Reprovisioning is a deliberate administrative reset or migration step that must be paired with a compatible storage snapshot or a fresh anchor bootstrap.
- Paper gate phrase: `explicit reprovisioning/resealing`

### software_update

- Current evidence: PCR drift is rejected only in the transient PCR-policy probe; no persistent PCR-sealed filesystem anchor is installed.
- Intended policy: Software updates or PCR changes require explicit policy renewal before the paper can claim persistent PCR-bound freshness.
- Paper gate phrase: `software update or PCR change therefore requires explicit reprovisioning/resealing`

### backup_migration

- Current evidence: The monotonic replay and combined SQLite/dbm.dumb campaigns show that stale backing-store snapshots fail closed against the advanced TPM anchor.
- Intended policy: Backup and migration are not blind directory copies; a destination needs an authorized anchor bootstrap or the restored state should fail closed.
- Paper gate phrase: `Backup or migration cannot copy only the backing directory`

### lost_credentials

- Current evidence: The paper and code keep the mount password as the root credential; no hardware-backed credential release or recovery artifact exists.
- Intended policy: Lost mount credentials are unrecoverable in this prototype.
- Paper gate phrase: `Lost mount credentials are unrecoverable`

### expected_failure_behavior

- Current evidence: The source fails closed on missing NV index, TPM I/O failure, anchor digest mismatch, and stored sequence ahead of local state; retained replay artifacts expose fail-closed read/open behavior.
- Intended policy: Missing index, authorization failure, TPM I/O failure, digest mismatch, or a TPM sequence ahead of local state must fail closed instead of exposing stale data.
- Paper gate phrase: `Missing index, authorization failure, TPM I/O failure, digest mismatch, or a TPM sequence ahead of local state fail closed`

## Non-Claims

- no persistent PCR-bound filesystem freshness
- no hardware-backed credential release
- no automatic backup or migration recovery
- no recovery after lost mount credential
- no full power-loss or FUSE-daemon crash certification
