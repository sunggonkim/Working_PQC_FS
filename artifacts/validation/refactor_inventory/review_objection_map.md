# Review Objection Map

This Gate 0.2-S0 artifact maps repeated review objections to the concrete code, retained or missing evidence, paper boundary, and negative guard that prevent another loop of paper-only fixes.

- Generated: `2026-06-30T03:54:31Z`
- Worktree freeze source: `artifacts/validation/refactor_inventory/worktree_freeze.json`
- Dirty-sidecar sync dedup patch: `None`
- Parent checklist closed by this artifact: `False`

## Status Counts

- `artifact-required`: 6
- `code-required`: 4
- `dropped-claim`: 0
- `negative-claim-only`: 2
- `paper-required`: 2

## Objections

| ID | Status | Primary module | Gates | Missing closure evidence | Negative guard |
| --- | --- | --- | --- | --- | --- |
| O01 | `code-required` | `code/frontend/pqc_fuse.c` | 0.1, 0.4, 0.5, 0.6, 0.7, 0.8, 0.14 | Gate 0.14 behavior-equivalence closeout after decomposition; paper negative guard for clean-module claims | No production-clean, modular-storage-runtime, or behavior-equivalent refactor claim before Gate 0.14 closes. |
| O02 | `artifact-required` | `code/fs/pqc_file_io.c` | A1, A2, A3, A4, A5, 0.9, 0.15, 0.16 | filesystem_viability_breakdown end-to-end component attribution; strict-versus-epoch lock-hold and p99/p99.9 closeout; paper text classifying the dominant bottleneck | No high-throughput, competitive-with-gocryptfs, or general-purpose filesystem wording before viability breakdown and epoch evidence close. |
| O03 | `artifact-required` | `code/experiments/build_kernel_baseline_feasibility.py` | B1, B4 | matched fscrypt fio row or environment-blocked proof | No fscrypt speedup or apples-to-apples implication without a measured fscrypt row. |
| O04 | `artifact-required` | `code/experiments/build_kernel_baseline_feasibility.py` | B2, B4 | LUKS/dmsetup setup logs and matched fio JSON for dm-crypt/ext4 | Historical dm-crypt diagnostics are not a paper comparison row. |
| O05 | `artifact-required` | `code/runtime/pqc_qos.c` | B3, E2 | artifacts/reports/qos_repeated/*.json; at least two cgroup/blk-throttle/ionice/BFQ baseline runs | No SQLite p99 uniqueness claim before two kernel QoS baselines exist. |
| O06 | `paper-required` | `code/fs/pqc_posix.c` | C1, C2, 0.14 | paper POSIX envelope table with support/formal rejection/paper limitation | No general-purpose POSIX wording before shared mmap and POSIX envelope evidence close. |
| O07 | `artifact-required` | `code/support/pqc_test_hooks.c` | C3, C4, 0.8, 0.9 | physical power-loss or explicit unavailable-environment boundary artifact | No power-loss safe, full crash certification, or crash-certified wording without physical or explicitly bounded evidence. |
| O08 | `code-required` | `code/storage/pqc_anchor.c` | C5, C6, D3 | production async Merkle or committed-prefix root plus TPM epoch evidence | No TPM rollback resistance, persistent PCR-bound freshness, sealed-key release, or full rollback-resistance wording before Gate C6. |
| O09 | `code-required` | `code/crypto/pqc_keyring.c` | D1, D2, D3 | Argon2id/scrypt implementation or PBKDF2 parameter/salt/offline-attack benchmark artifact | No offline-attack resistance or strong password-hardening claim while PBKDF2 remains unexplained. |
| O10 | `negative-claim-only` | `code/gpu/cuda_aead.cu` | 0.10, 0.11, 0.12, 0.18, A4, E5 | production mounted-path benefit traces before any NVIDIA term becomes a paper mechanism | No NVMe-to-UVM, GPUDirect, dma-buf zero-copy, CUDA isolation, or stream-priority QoS claim without production evidence. |
| O11 | `code-required` | `code/experiments/run_qos_sqlite_hero_bundle.py` | E1, E2, E3, E4, E5 | artifacts/validation/hero_result_contract/*.json; artifacts/validation/second_macrobenchmark/*.json; RocksDB/WAL, checkpointing, or secure inference-log macrobenchmark runner and raw logs | No broad workload-general or SOSP/OSDI-ready claim before a second macrobenchmark exists. |
| O12 | `paper-required` | `code/experiments/build_paper_spine_gate.py` | F1, F2, F3, F4, F5 | artifacts/validation/first_two_pages_thesis_audit/*.json; artifacts/validation/recurring_review_elimination_audit/*.json; paper text pass with dangerous phrases absent, negated, or evidence-backed | Dangerous phrases must be absent, explicitly negated, or tied to a closed gate. |
| O13 | `artifact-required` | `code/storage/pqc_epoch_publish.c` | A5, 0.9, 0.15, 0.16 | vfs_ebpf_fdatasync_storm context-switch/syscall/fdatasync/tail-latency breakdown | No eBPF passthrough, io_uring completion bypass, or async epoch fdatasync claim without production evidence. |
| O14 | `negative-claim-only` | `code/experiments/run_sidechannel_eval.py` | D4, F4, F5 | artifacts/validation/sidechannel_eval/*.json; artifacts/validation/tvla_leakage_eval/*.json; claim lint output proving out-of-scope language remains explicit | No side-channel protection, multi-tenant defense, compromised-kernel defense, deployment-ready, or ready-for-production wording. |

## Immediate Execution Consequence

The next row should refresh the venue gate. Broad checklist boxes remain unchecked because this artifact is an inventory/proof cursor, not code, paper text, and negative-claim closure for any parent gate.
