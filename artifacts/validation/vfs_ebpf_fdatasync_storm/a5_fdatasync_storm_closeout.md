# A5 VFS/FUSE and fdatasync Storm Closeout

- Overall pass: `True`
- Paper text status: `already_scoped_no_update`
- Parent checklist closed: `False`

## Path Classification

| Path | Classification | Evidence | Guard |
| --- | --- | --- | --- |
| ordinary mounted FUSE path | `accepted-measured` | `fuse_create_avg_ns=1644658`<br>`fuse_write_avg_ns=38287`<br>`fuse_fsync_avg_ns=6801778`<br>`daemon-side proxy, not kernel context-switch count` | Do not claim FUSE bypass or kernel scheduler context-switch evidence. |
| eBPF/io_uring completion bypass | `not-claimed` | `ebpf_audit_overall_pass=True`<br>`decision=scoped_out_no_mounted_ebpf_iouring_completion_bypass`<br>`mounted FUSE source has no io_uring/eBPF completion path` | eBPF/io_uring remains diagnostic or future work, not a mounted-path mechanism. |
| strict per-write data/journal fdatasync storm | `accepted-cost-boundary` | `a2_fdatasync=2140`<br>`a2_data_sidecar=1070`<br>`a2_journal_sidecar=1070`<br>`strict_sync_count_per_op=2.0`<br>`strict_journal_fsync_total=5` | Strict mode is a correctness boundary; do not present it as optimized throughput. |
| epoch foreground journal fdatasync | `eliminated-from-foreground-epoch-path` | `sequential_epoch_journal_fsync_removed=True`<br>`epoch_journal_fsync_total=0`<br>`epoch_log_fsync_total=5` | This removes strict journal fdatasync from foreground epoch publication, not all durability barriers. |
| grouped epoch syncfs barrier | `amortized-for-grouped-work` | `grouped_sync_amortized=True`<br>`grouped_sync_reduction_percent=37.5`<br>`a3_current_grouped_sync_count_reduced=True`<br>`strict_grouped_sync_total=8`<br>`epoch_grouped_sync_total=5` | Scope to grouped/concurrent mounted work; single-client frozen rows remain cost-boundary evidence. |
| kernel context-switch count | `unavailable-not-claimed` | `A4 records daemon-side FUSE operation latency only`<br>`A2 records fio-client syscall counts, not daemon scheduler switches` | Do not convert daemon-side latency or fio-client syscall counts into context-switch counts. |
| correctness/fault boundary for amortization | `measured` | `replay_fault_matrix_pass=True`<br>`a3_replay_fault_matrix_passes=True`<br>`a4_measured_classes=5` | Replay/fault evidence bounds epoch publication; it is not physical power-loss certification. |

## Proof Checks

- `a2_pass`: `True`
- `a3_pass`: `True`
- `a4_pass`: `True`
- `publication_closeout_complete`: `True`
- `publication_comparison_pass`: `True`
- `replay_fault_matrix_pass`: `True`
- `ebpf_iouring_audit_pass`: `True`
- `source_guards_present`: `True`
- `paper_scope_guards_present`: `True`
- `no_unscoped_dangerous_paper_hits`: `True`

## Non-Claims

- no mounted eBPF/io_uring completion bypass
- no kernel context-switch count
- no claim that epoch mode removes all fdatasync/syncfs barriers
- no general fast-filesystem throughput claim
- no physical power-loss certification
