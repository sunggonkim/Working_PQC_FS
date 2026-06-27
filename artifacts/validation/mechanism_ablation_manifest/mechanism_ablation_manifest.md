# Mechanism Ablation Manifest

- Overall pass: `true`
- Scope: Mechanism attribution over existing retained evidence.  This is not a new broad filesystem comparison and does not close fscrypt/dm-crypt, cold-cache, PCR-bound freshness, or AI QoS gaps.

## Checks

- `plaintext_frozen_pass`: `true`
- `aegisq_frozen_pass`: `true`
- `qos_required_modes_pass`: `true`
- `keyplane_required_modes_pass`: `true`
- `anchor_required_modes_pass`: `true`
- `paper_scope_gate_pass`: `true`

## Entries

### fs_format_total_cost

- Mechanism: encrypted mounted format and journal/checkpoint path
- Variants: `plaintext_lowerfs, aegis_q`
- Artifacts: `artifacts/validation/frozen_plaintext_contract/frozen_plaintext_contract.json, artifacts/validation/frozen_aegisq_contract/frozen_aegisq_contract.json`
- Interpretation: The warm-cache frozen-contract gap attributes the fio overhead to the mounted encrypted format as a whole, including FUSE, AEAD, journal, checkpoint, and file-anchor work.  It does not isolate a plaintext-FUSE row or kernel encryption baselines.
- Scope boundary: No fscrypt/dm-crypt, cold-cache, or component-only filesystem decomposition.

- Throughput medians: plaintext `32.8798` MiB/s, AEGIS-Q `0.358321` MiB/s
- Conservative p99 medians: plaintext `0.164864` ms, AEGIS-Q `11.2067` ms

### qos_controller_variants

- Mechanism: SQLite foreground recovery under secure-storage pressure
- Variants: `app_only, unthrottled_storage, simple_controller, aegis_policy`
- Artifacts: `artifacts/validation/qos_sqlite_hero_bundle/qos_sqlite_hero_bundle.json`
- Interpretation: The controller rows attribute the SQLite recovery to throttling elastic mounted-FUSE writes.  The simple controller has slightly lower p99 in this run, while AEGIS-Q retains more background throughput and records daemon-side throttle evidence.
- Scope boundary: Single retained SQLite workflow; not foreground TensorRT/AI p99 recovery.

- SQLite p99 medians: unthrottled `13.8221` ms, simple `8.18509` ms, AEGIS-Q `8.75332` ms
- Background throughput: simple `2.26826` MB/s, AEGIS-Q `2.73609` MB/s

### keyplane_cpu_gpu_policy

- Mechanism: open-file envelope-refresh placement
- Variants: `cpu_only, gpu_batch, policy_fallback`
- Artifacts: `artifacts/validation/keyplane_rekey_methodology/keyplane_rekey_workflow.json`
- Interpretation: The optional batch lane improves this maintenance workflow when slack is present and falls back to CPU under zero slack/high pressure without changing the storage format.
- Scope boundary: Open-file envelope refresh only; not deployed credential lifecycle or data-plane acceleration.

- GPU-vs-CPU speedup median `1.18551`x (95% CI `1.14455`--`1.38135`)

### file_vs_tpm_anchor

- Mechanism: external freshness boundary
- Variants: `file_anchor_negative_control, tpm_replay_after_advance`
- Artifacts: `artifacts/validation/generation_fault_matrix/generation_fault_matrix.json, artifacts/validation/tpm_monotonic_replay/tpm_monotonic_replay.json`
- Interpretation: The file-backed anchor is replayable with the backing directory and is therefore a negative control.  The retained TPM replay-after-advance artifact fails closed against stale disk state.
- Scope boundary: No persistent PCR-bound filesystem freshness, NV authorization lifecycle, or power-loss certification.

- Oracle verdicts: file `previous_committed`, TPM `fail_closed`

## Non-Claims

- no fscrypt/dm-crypt frozen-contract rows
- no cold-cache filesystem result
- no plaintext-FUSE decomposition row
- no persistent PCR-bound freshness lifecycle
- no foreground TensorRT/AI p99 recovery
- no data-plane GPU acceleration claim
