# Freshness Ladder Claim Guard

- Generated: `2026-06-30T03:54:31Z`
- Overall pass: `true`
- Unguarded dangerous lines: `0`

## Ladder

- `L0_file_anchor_negative_control`: supported=`true`; allowed=file-backed witness is replayable; stale directory snapshot remains visible; forbidden=rollback resistance
- `L1_tpm_nv_replay_after_advance`: supported=`true`; allowed=pre-provisioned TPM NV replay-after-advance can fail closed under retained rows; forbidden=persistent PCR-bound freshness or full rollback resistance
- `L2_transient_pcr_policy_probe`: supported=`true`; allowed=transient PCR-policy seal/unseal probe rejects drift; forbidden=persistent filesystem PCR policy
- `L3_persistent_pcr_bound_policy`: supported=`false`; allowed=non-claim only in this revision; forbidden=persistent PCR-bound freshness
- `L4_sealed_key_release`: supported=`false`; allowed=non-claim only in this revision; forbidden=hardware-backed credential release or sealed-key release
- `L5_async_merkle_tpm_epoch_freshness`: supported=`false`; allowed=non-claim only while the bounded hardware epoch probe is environment-blocked and no TPM epoch is committed; forbidden=async Merkle + TPM epoch freshness
- `L6_full_rollback_resistance`: supported=`false`; allowed=non-claim only in this revision; forbidden=full/offline rollback resistance

## Unguarded Findings

- None.

## C6 TPM Epoch Guard

- TPM epoch probe verdict: `environment-blocked`
- Hardware epoch committed: `false`
- C6 unguarded dangerous lines: `0`
- None.

## Boundary

Freshness claims are ordered by evidence: file anchor replay is a negative control; TPM NV evidence supports only retained replay-after-advance fail-closed rows; PCR evidence is transient; the C6 hardware epoch probe is environment-blocked on this machine; persistent PCR-bound policy, sealed-key release, async Merkle TPM epoch freshness, TPM rollback resistance, full replay protection, and full rollback resistance remain non-claims.
