# AEGIS-Q Publication Protocol

- Generated: `2026-06-30T02:56:39Z`
- Scope: `Gate 0.9-S7 publication protocol architecture/evidence closeout.`
- Closeout complete: `true`
- Paper text status: `updated`
- Parent checklist closed: `true`

## Evidence Inputs

- `artifacts/validation/publication_protocol_fault_matrix/epoch_publication_comparison.json`
- `artifacts/validation/publication_protocol_fault_matrix/epoch_replay_fault_matrix.json`
- `artifacts/validation/parallel_commit_contract/parallel_commit_closure_audit.json`

## Production Mechanisms

- `strict_data_fsync`: `true`
- `strict_journal_fsync`: `true`
- `strict_checkpoint_stage`: `true`
- `epoch_mode_dispatch`: `true`
- `epoch_skips_strict_journal_fsync`: `true`
- `epoch_group_barrier`: `true`
- `epoch_group_config`: `true`
- `epoch_group_syncfs`: `true`
- `epoch_trace_group_size`: `true`
- `epoch_log_replay_parser`: `true`
- `epoch_checkpoint_compaction`: `true`
- `epoch_journal_repair`: `true`
- `epoch_duplicate_generation_reject`: `true`
- `fd_context_runs_epoch_compaction`: `true`

## Measured Publication Modes

| Mode | Workload | Clients | Syncs | Data fsync | Journal fsync | Epoch fsync | MiB/s | Client p99 ns | Max group | Sync primitive |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| strict | sequential_unique_file_fdatasync | 1 | 10 | 5 | 5 | 0 | 12.9597 | 1.35929e+06 | n/a | n/a |
| epoch-redo-log | sequential_unique_file_fdatasync | 1 | 10 | 5 | 0 | 5 | 10.8801 | 1.55278e+06 | n/a | n/a |
| strict grouped | concurrent_unique_file_fdatasync | 4 | 8 | 4 | 4 | 0 | 6.44612 | 2.66989e+06 | 0 | [] |
| epoch-redo-log grouped | concurrent_unique_file_fdatasync | 4 | 5 | 4 | 0 | 1 | 1.07649 | 5.31363e+07 | 4 | ['none', 'syncfs'] |

## Result Boundary

- Sequential epoch journal fsync removed: `true`
- Grouped sync amortized: `true`
- Grouped sync reduction percent: `37.5`
- Sequential throughput win: `false`
- Grouped throughput win: `false`

## Replay And Recovery Evidence

- Overall pass: `true`

- `replay_duplicate_generation`: pass=`true`, mutation=`duplicate_generation`, repair_max=`0`, torn_tail_max=`0`, duplicate_max=`1`
- `replay_journal_loss`: pass=`true`, mutation=`journal_loss`, repair_max=`4`, torn_tail_max=`0`, duplicate_max=`0`
- `replay_normal`: pass=`true`, mutation=`None`, repair_max=`0`, torn_tail_max=`0`, duplicate_max=`0`
- `replay_torn_tail`: pass=`true`, mutation=`torn_tail`, repair_max=`0`, torn_tail_max=`17`, duplicate_max=`0`

## Invariants

- Strict mode keeps data-sidecar fdatasync before journal fdatasync.
- Epoch mode writes ciphertext and data-sidecar fdatasync before appending committed epoch records.
- Epoch mode removes strict journal fdatasync from the foreground epoch path and repairs journal mappings from committed epoch prefixes at remount.
- Grouped epoch mode uses one filesystem-level syncfs barrier for joined foreground operations when PQC_EPOCH_GROUP_MAX is greater than one.
- Replay accepts only committed epoch prefixes, ignores torn tails, and rejects duplicate generation records.
- Reader-visible generation state advances only after publication dispatch returns successfully.

## Known Losses

- Sequential epoch-redo-log mode does not improve total sync count because each write still has one data fdatasync and one epoch-log barrier.
- Grouped epoch-redo-log reduces traced sync count in the retained concurrent workload, but the retained run does not show a throughput win.
- The grouped barrier uses syncfs, which can be broader and more expensive than per-file fdatasync.

## Non-Claims

- No direct NVMe-to-UVM, GPUDirect/RDMA, dma-buf zero-copy, eBPF/io_uring bypass, or kernel-bypass publication path is claimed.
- No physical power-loss certification or full crash certification is claimed.
- No persistent PCR-bound freshness, TPM rollback resistance, or side-channel protection is claimed by this gate.
- No general-purpose POSIX filesystem support or ready-for-deployment wording is supported by this closeout.

## Negative Claim Guard

Paper or README text may describe strict and epoch publication only within the retained artifact scope. It must not claim throughput improvement, full crash certification, kernel bypass, direct storage DMA, rollback resistance, side-channel defense, or general-purpose POSIX support from Gate 0.9 evidence.
