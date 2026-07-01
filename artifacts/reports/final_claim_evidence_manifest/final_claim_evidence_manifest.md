# Final Claim-to-Evidence Manifest

- Generated: `2026-06-30T22:15:20Z`
- Overall pass: `True`
- Paper pages: `13`
- Numeric claim candidates: `21`
- Abstract/conclusion/security/recovery claims: `36`
- Uncovered numeric candidates: `0`
- Uncovered cross claims: `0`

## Evidence Rules

| Claim id | Category | Complete | Boundary |
| --- | --- | ---: | --- |
| `edge_runtime_claim_spine` | `claim-spine` | `True` | Top-level edge-runtime thesis; supporting correctness rows bound the claim rather than converting it into filesystem certification. |
| `mechanism_ablation_attribution` | `claim-attribution` | `True` | Mechanism attribution summary; not an independent throughput or recovery expansion. |
| `sqlite_hero_repeated_medians` | `numeric-performance` | `True` | SQLite foreground storage-visible control only; no non-storage QoS or broad filesystem superiority. |
| `frozen_filesystem_contract_rows` | `numeric-performance` | `True` | Warm-cache frozen-contract cost boundary; not high-throughput/general-purpose claim. |
| `kernel_baseline_unavailable_rows` | `baseline-boundary` | `True` | Kernel rows are measured or environment-blocked; no fscrypt/dm-crypt speedup claim. |
| `primitive_and_keyplane_placement` | `numeric-performance` | `True` | Placement asymmetry and mounted key-plane refresh; not bulk-data PQC encryption. |
| `kdf_parameters` | `security-claim` | `True` | Password-derived mount-key boundary; no hardware-backed credential release or offline-resistance claim. |
| `generation_and_tamper_safety` | `security-recovery-claim` | `True` | Strict/epoch nonce-generation evidence under stated model; no physical power-loss certification. |
| `tpm_freshness_boundary` | `security-recovery-claim` | `True` | Replay-after-advance and file-anchor negative control; no persistent PCR-bound rollback resistance. |
| `crash_and_recovery_boundary` | `security-recovery-claim` | `True` | Selected daemon/app cutpoints; no physical power-loss, kernel-crash, or drive-cache certification. |
| `second_macrobenchmark` | `numeric-performance` | `True` | Second scoped mounted workload; not a broad workload suite or non-storage QoS claim. |
| `epoch_mode_conditional_amortization` | `numeric-performance` | `True` | Epoch mode is conditional amortization with explicit p99/throughput wins and loss cases, not dominance across workloads. |
| `cache_manifest_workload` | `numeric-performance` | `True` | Third scoped mounted workload using closed-file rename, directory fsync, remount, and hash verification; not a broad workload suite. |
| `x6_strict_cost_reduction_model` | `numeric-performance` | `True` | Strict-path marker/checkpoint sync narrowing model; no throughput, kernel-upstreaming, or physical power-loss claim. |
| `foreground_inference_claim_removal` | `workload-boundary` | `True` | O1 closed by removing the inference/TensorRT/AI axis from the submitted paper claim. |
| `telemetry_and_sensitivity` | `numeric-performance` | `True` | Controller sensitivity and wiring evidence; not external application scheduling. |
| `posix_scope_boundary` | `correctness-boundary` | `True` | Narrow FUSE/POSIX envelope; no general-purpose POSIX filesystem claim. |
| `integrity_and_sidechannel_boundary` | `security-boundary` | `True` | Integrity parity and AES-GCM confidentiality only; no side-channel protection claim. |
| `platform_and_resource_context` | `numeric-platform` | `True` | Single tested Jetson stack; no portability claim. |
| `claim_firewalls` | `negative-claim` | `True` | Unsupported stronger claims remain absent, negated, or scoped. |
