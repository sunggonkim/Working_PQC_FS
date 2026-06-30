# Venue Gate

This Gate 0.3-S0 artifact separates the active SOSP/OSDI target from ATC/FAST fallback classifications. It does not close venue readiness; it records why the current evidence remains blocked or conditional.

- Generated: `2026-06-29T10:18:24Z`
- Review map: `artifacts/validation/refactor_inventory/review_objection_map.json`
- Worktree freeze: `artifacts/validation/refactor_inventory/worktree_freeze.json`
- Ready for SOSP/OSDI claim: `False`
- Parent checklist closed by this artifact: `False`

## Venue Verdicts

| Venue | Verdict | Blocker statuses | Classification rule |
| --- | --- | --- | --- |
| SOSP/OSDI | `blocked` | `artifact-required`=6, `code-required`=4, `negative-claim-only`=2, `paper-required`=2 | SOSP/OSDI is the active target. It cannot be claimed by scoped paper wording alone; every hard blocker must be closed by production-path code, retained artifact, paper text, and negative guard, or the corresponding venue-level claim must be removed. |
| ATC | `conditional-blocked` | `artifact-required`=5, `code-required`=2, `negative-claim-only`=1, `paper-required`=1 | ATC may become a fallback only if the user explicitly rescopes from SOSP/OSDI; it still needs clean implementation evidence, practical baselines, reproducibility, and explicit limitations. |
| FAST | `conditional-blocked` | `artifact-required`=5, `code-required`=1, `paper-required`=2 | FAST may become a fallback only if the paper is reframed as a storage publication/runtime mechanism; filesystem viability, baselines, crash semantics, and sync/fdatasync accounting remain required. |

## SOSP/OSDI Hard Blockers

- `O01` `code-required` `code/pqc_fuse.c`: No production-clean, modular-storage-runtime, or behavior-equivalent refactor claim before Gate 0.14 closes.
- `O02` `artifact-required` `code/pqc_file_io.c`: No high-throughput, competitive-with-gocryptfs, or general-purpose filesystem wording before viability breakdown and epoch evidence close.
- `O03` `artifact-required` `code/experiments/build_kernel_baseline_feasibility.py`: No fscrypt speedup or apples-to-apples implication without a measured fscrypt row.
- `O04` `artifact-required` `code/experiments/build_kernel_baseline_feasibility.py`: Historical dm-crypt diagnostics are not a paper comparison row.
- `O05` `artifact-required` `code/pqc_qos.c`: No SQLite p99 uniqueness claim before two kernel QoS baselines exist.
- `O06` `paper-required` `code/pqc_posix.c`: No general-purpose POSIX wording before shared mmap and POSIX envelope evidence close.
- `O07` `artifact-required` `code/pqc_test_hooks.c`: No power-loss safe, full crash certification, or crash-certified wording without physical or explicitly bounded evidence.
- `O08` `code-required` `code/pqc_anchor.c`: No TPM rollback resistance, persistent PCR-bound freshness, sealed-key release, or full rollback-resistance wording before Gate C6.
- `O09` `code-required` `code/pqc_keyring.c`: No offline-attack resistance or strong password-hardening claim while PBKDF2 remains unexplained.
- `O10` `negative-claim-only` `code/cuda_aead.cu`: No NVMe-to-UVM, GPUDirect, dma-buf zero-copy, CUDA isolation, or stream-priority QoS claim without production evidence.
- `O11` `code-required` `code/experiments/run_qos_sqlite_hero_bundle.py`: No broad workload-general or SOSP/OSDI-ready claim before a second macrobenchmark exists.
- `O12` `paper-required` `code/experiments/build_paper_spine_gate.py`: Dangerous phrases must be absent, explicitly negated, or tied to a closed gate.
- `O13` `artifact-required` `code/pqc_epoch_publish.c`: No eBPF passthrough, io_uring completion bypass, or async epoch fdatasync claim without production evidence.
- `O14` `negative-claim-only` `code/experiments/run_sidechannel_eval.py`: No side-channel protection, multi-tenant defense, compromised-kernel defense, deployment-ready, or ready-for-production wording.

## Non-Claim Boundary

- SOSP/OSDI: Do not say SOSP-ready, OSDI-ready, deployment-ready, general-purpose filesystem, broad POSIX support, or full rollback/crash/GPU-isolation defense until the listed blockers close.
- ATC: Do not use ATC fallback to keep SOSP/OSDI claims. If rescoped to ATC, remove or downgrade claims that depend on unclosed venue-level novelty, broad POSIX, rollback resistance, or second-workload generality.
- FAST: Do not claim a FAST-style storage mechanism until epoch publication, fdatasync amortization, recovery, and mode-aligned storage baselines are measured in the production mounted path.

## Next Cursor

`0.4-S0`: Refresh source ownership and module-decomposition inventory for the current code tree.
