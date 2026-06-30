# Nonce/Generation Fault Verdict

- Generated: `2026-06-29T08:31:41Z`
- Overall pass: `true`
- Strict matrix: `artifacts/validation/generation_fault_matrix/generation_fault_matrix.json`
- Epoch matrix: `artifacts/validation/publication_protocol_fault_matrix/epoch_replay_fault_matrix.json`

## Checks

- `strict_matrix_overall`: `true` - strict matrix reports overall_pass, no generated nonce reuse, no silent corruption, and no unexpected liveness failures
  - Residual: process/final-binary fault model only
- `strict_partial_update_remount`: `true` - partial update/remount returns latest committed state with no generated duplicate journal pairs
- `strict_torn_record_tail`: `true` - torn strict journal tail is ignored and remount exposes the latest committed payload
- `strict_interrupted_generation_advance`: `true` - daemon kill after generation reservation/data fsync skips the unpublished generation on remount
  - Residual: not physical power loss
- `strict_replayed_metadata`: `true` - adversarial append of an older mapping creates a replay duplicate but does not supersede the newer mapping
- `strict_stale_checkpoint_ladder`: `true` - file-backed stale snapshot remains replayable, while retained TPM replay-after-advance artifact fails closed
  - Residual: not PCR-bound rollback resistance
- `strict_retained_raw_logs`: `true` - 6 retained strict journal files exist
- `epoch_matrix_overall`: `true` - epoch replay matrix reports all cases pass
  - Residual: committed-prefix replay only
- `epoch_torn_log_tail`: `true` - epoch committed-prefix replay ignores a torn log tail and recovers matching data
- `epoch_duplicate_append_fail_closed`: `true` - epoch replay rejects duplicate generation records instead of exposing data
- `epoch_stale_journal_repair`: `true` - epoch replay repairs a deliberately lost strict journal from the committed epoch prefix
- `epoch_normal_committed_prefix`: `true` - normal epoch remount compacts the committed prefix without duplicate generations
- `source_invariant_guard`: `true` - nonce/AAD bind file_id, block, generation; strict replay chooses highest committed generation; epoch replay rejects duplicate keys
- `paper_negative_claim_guard`: `true` - paper states duplicate-generation invariant and residual power-loss/PCR/crash-proof limits

## Claim Boundary

C3 proves nonce/generation safety only for the retained final-binary strict and epoch fault models. Strict evidence is per-file sidecar (block,generation) evidence; epoch replay additionally checks (file_id,block,generation) duplicate records. This does not prove physical power-loss, kernel-crash, drive-cache, PCR-bound rollback resistance, or complete POSIX crash certification.
