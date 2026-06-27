# PCR Anchor Decision Manifest

- Overall pass: `true`
- Decision: `no_persistent_pcr_bound_filesystem_anchor_in_this_revision`
- Scope: The persistent filesystem anchor is not PCR-bound in this revision. PCR evidence is retained only as a transient policy probe; persistent freshness evidence is limited to pre-provisioned TPM NV replay-after-advance fail-closed behavior.

## Checks

- `decision_rows_present`: `true`
- `source_checks_pass`: `true`
- `artifact_checks_pass`: `true`
- `paper_checks_pass`: `true`

## Source Checks

- `anchor_uses_preprovisioned_nv_index`: `true`
- `anchor_uses_owner_hierarchy_nv_ops`: `true`
- `anchor_source_has_no_persistent_pcr_commands`: `true`
- `pcr_probe_script_declares_transient_scope`: `true`
- `tpm_policy_manifest_declares_nonclaim`: `true`
- `tpm_policy_script_preserves_nonclaim`: `true`

## Artifact Checks

- `pcr_probe_passes_current_unseal`: `true`
- `pcr_probe_rejects_drift`: `true`
- `pcr_probe_note_is_transient`: `true`
- `provisioned_nv_index_recorded`: `true`
- `tpm_replay_after_advance_fail_closed`: `true`
- `tpm_policy_manifest_passes`: `true`

## Paper Checks

- `required_paper_gate_phrases_present`: `true`
- `all_high_risk_pcr_mentions_scoped`: `true`

## Decision Rows

### persistent_anchor_decision

- Decision: `no_persistent_pcr_bound_filesystem_anchor_in_this_revision`
- Evidence: pqc_anchor.c stores the persistent hardware anchor as an authenticated prefix record in a pre-provisioned TPM NV index using owner-hierarchy nvread/nvwrite commands.  It contains no PCR policy-session, sealed-object, or unseal path.
- Paper gate phrase: `PCR binding is a transient probe rather than a persistent filesystem policy`

### transient_pcr_probe_role

- Decision: `retain_pcr_only_as_probe`
- Evidence: The PCR probe artifact seals a transient object, unseals under the current PCR digest, and rejects a drifted PCR digest; its own note says it is not a filesystem freshness proof.
- Paper gate phrase: `not a claim that the persistent filesystem anchor itself is PCR-sealed`

### paper_claim_boundary

- Decision: `paper_must_negate_persistent_pcr_binding`
- Evidence: The main paper gates the claim in design, evaluation, security, abstract, and conclusion wording; all high-risk PCR mentions are in a negated, transient, or remaining-gap context.
- Paper gate phrase: `no persistent PCR-bound lifecycle claim`

## Non-Claims

- no persistent PCR-bound filesystem freshness
- no persistent PCR-sealed filesystem anchor
- no PCR-bound key release
- no software-update PCR migration protocol
- no PCR-bound backup or restore protocol
