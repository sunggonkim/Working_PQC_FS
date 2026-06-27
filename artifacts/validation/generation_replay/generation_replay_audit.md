# Generation / nonce replay audit

Scope: this audit covers the current `pqc_fuse.c` generation path and the
retained `pqc_fuse --self-test` regression.  It does not claim arbitrary
power-loss crash certification.

Verified code path:

- Nonce derivation uses `(file_id, logical_block, generation)` in
  `derive_block_nonce()`.
- AEAD associated data uses `(file_id, logical_block, generation,
  plaintext_length)` in `build_block_aad()`.
- Each encrypted flush assigns fresh generations from
  `ctx->state->next_generation + 1 + block_index`.
- Ciphertext records are written and `fdatasync()`ed before journal mappings
  are appended.
- Journal records include the generation and a record digest.
- `journal_lookup_mapping()` selects the valid mapping with the highest
  generation for each logical block.
- `ctx->state->next_generation` advances only after the journal barrier and
  checkpoint update path succeed.

Retained regression:

- Command: `./build/pqc_fuse --self-test`
- Artifact summary:
  `artifacts/validation/generation_replay/generation_replay_self_test.json`
- Raw stderr:
  `artifacts/validation/generation_replay/pqc_fuse_self_test.stderr`
- Result: `generation_replay_self_test_pass=true`, `returncode=0`.

Regression semantics:

1. Append generation 1 and generation 2 mappings for the same logical block.
2. Append a replayed generation 1 mapping after generation 2.
3. Verify lookup still returns generation 2.
4. Encrypt under generation 2 and verify that decrypting the same
   ciphertext/tag under generation 1 fails.
5. Verify decrypting under generation 2 succeeds and recovers the original
   plaintext.

Open scope:

- The regression is not a complete torn-write, daemon-crash, or power-loss
  campaign.
- The current evidence supports generation-bound AEAD context and stale
  generation replay rejection at the journal/crypto primitive level.
