# Gate 0.15 Concurrency Contract Smoke

- Overall pass: `true`
- Scope: production mounted-path lock timing smoke for current strict mode
- Lock hold events: `5906`
- Observed locks: `admission_state_lock, admission_trace_lock, anchor_epoch_record_lock, anchor_lifecycle_lock, anchor_worker_lock, commit_lock, committed_map_lock, fd_lock, file_anchor_commit_lock, file_state_table_lock, parallel_runtime_lock, qos_gpu_load_lock, rekey_lifecycle_lock, rekey_queue_lock, scheduler_lock, trace_sink_lock`
- Workload phases: `18`

## Workload Sweep

| Phase | Worker | Threads | Processes | Iterations/client | Timed out | Wall seconds | Worker errors |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| `same_file` | `writer_worker` | 1 | 0 | 4 | `false` | 0.015725 | 0 |
| `disjoint_files` | `writer_worker` | 1 | 0 | 4 | `false` | 0.011867 | 0 |
| `same_file_lifecycle` | `lifecycle_worker` | 1 | 0 | 4 | `false` | 0.014868 | 0 |
| `disjoint_lifecycle` | `lifecycle_worker` | 1 | 0 | 4 | `false` | 0.013083 | 0 |
| `same_file` | `writer_worker` | 2 | 0 | 4 | `false` | 0.021909 | 0 |
| `disjoint_files` | `writer_worker` | 2 | 0 | 4 | `false` | 0.012750 | 0 |
| `same_file_lifecycle` | `lifecycle_worker` | 2 | 0 | 4 | `false` | 0.025178 | 0 |
| `disjoint_lifecycle` | `lifecycle_worker` | 2 | 0 | 4 | `false` | 0.014375 | 0 |
| `same_file` | `writer_worker` | 4 | 0 | 4 | `false` | 0.041355 | 0 |
| `disjoint_files` | `writer_worker` | 4 | 0 | 4 | `false` | 0.063353 | 0 |
| `same_file_lifecycle` | `lifecycle_worker` | 4 | 0 | 4 | `false` | 0.049141 | 0 |
| `disjoint_lifecycle` | `lifecycle_worker` | 4 | 0 | 4 | `false` | 0.024104 | 0 |
| `same_file_process_lifecycle` | `lifecycle_process_worker` | 0 | 1 | 4 | `false` | 0.019270 | 0 |
| `disjoint_process_lifecycle` | `lifecycle_process_worker` | 0 | 1 | 4 | `false` | 0.018141 | 0 |
| `same_file_process_lifecycle` | `lifecycle_process_worker` | 0 | 2 | 4 | `false` | 0.028043 | 0 |
| `disjoint_process_lifecycle` | `lifecycle_process_worker` | 0 | 2 | 4 | `false` | 0.020035 | 0 |
| `same_file_process_lifecycle` | `lifecycle_process_worker` | 0 | 4 | 4 | `false` | 0.051194 | 0 |
| `disjoint_process_lifecycle` | `lifecycle_process_worker` | 0 | 4 | 4 | `false` | 0.029062 | 0 |

## Stress Coverage

- Max thread count: `4`
- Same-file lifecycle phases: `3`
- Disjoint lifecycle phases: `3`
- Same-file process lifecycle phases: `3`
- Disjoint process lifecycle phases: `3`
- Max process client count: `4`
- Timed-out phases: `0`
- Worker errors: `0`
- Blocking syscall profile pass: `true`

## Feasibility Decision

- Commit-lock refactor decision: `strict_prepare_crypto_outside_fd_and_commit_locks_but_gate_open`
- Reader visibility probe pass: `true`
- Reader observed old data during paused publish: `true`
- Reader observed new data after publish: `true`

## Lock Hold Summary

| Lock@site | Count | hold p50 ns | hold p95 ns | hold p99 ns | wait p99 ns |
| --- | ---: | ---: | ---: | ---: | ---: |
| `admission_state_lock@admission_init_policy` | 1 | 324 | 324 | 324 | 92 |
| `admission_state_lock@pqc_admission_shutdown` | 1 | 111 | 111 | 111 | 222 |
| `admission_trace_lock@admission_init_trace_fd` | 1 | 121 | 121 | 121 | 111 |
| `admission_trace_lock@admission_shutdown_trace` | 1 | 7213 | 7213 | 7213 | 222 |
| `admission_trace_lock@admission_trace_append` | 1 | 4028 | 4028 | 4028 | 64 |
| `anchor_epoch_record_lock@anchor_epoch_record_update` | 157 | 102 | 120 | 121 | 287 |
| `anchor_lifecycle_lock@pqc_anchor_worker_start_if_configured` | 1 | 31259 | 31259 | 31259 | 148 |
| `anchor_lifecycle_lock@pqc_anchor_worker_stop` | 3 | 83 | 16157 | 16157 | 241 |
| `anchor_worker_lock@anchor_worker_main` | 156 | 3787 | 16389 | 42981 | 0 |
| `anchor_worker_lock@anchor_worker_mark_committed` | 157 | 93 | 111 | 139 | 352 |
| `anchor_worker_lock@anchor_worker_take_dirty` | 172 | 1380 | 1713 | 3074 | 10019 |
| `anchor_worker_lock@pqc_anchor_worker_stage` | 168 | 1703 | 2325 | 4518 | 361 |
| `anchor_worker_lock@pqc_anchor_worker_start_if_configured` | 1 | 334 | 334 | 334 | 74 |
| `anchor_worker_lock@pqc_anchor_worker_stop` | 1 | 1694 | 1694 | 1694 | 84 |
| `commit_lock@fd_context_state_generation_snapshot` | 126 | 74 | 84 | 84 | 370 |
| `commit_lock@pqc_fd_context_set` | 126 | 75 | 507371 | 931426 | 389 |
| `commit_lock@pqc_file_state_mapping_cache_lookup` | 112 | 139 | 352 | 389 | 101 |
| `commit_lock@pqc_read` | 224 | 83 | 102 | 102 | 380 |
| `commit_lock@pqc_writeback_flush_locked` | 168 | 74 | 84 | 84 | 213 |
| `commit_lock@writeback_flush_authenticated_snapshot` | 504 | 83 | 667 | 824 | 426 |
| `committed_map_lock@build_prefix_anchor` | 157 | 84 | 176 | 213 | 84 |
| `committed_map_lock@pqc_anchor_load` | 90 | 74 | 84 | 84 | 268 |
| `committed_map_lock@pqc_anchor_record_file` | 673 | 84 | 176 | 232 | 416 |
| `fd_lock@pqc_fd_context_clear` | 126 | 556 | 1028 | 1084 | 185 |
| `fd_lock@pqc_fd_context_set` | 252 | 510 | 9898 | 16676 | 10398 |
| `fd_lock@pqc_flush` | 126 | 83 | 120 | 222 | 472 |
| `fd_lock@pqc_fsync` | 504 | 1268 | 4194 | 12278 | 472 |
| `fd_lock@pqc_read` | 224 | 5213 | 22453 | 40574 | 444 |
| `fd_lock@pqc_release` | 126 | 83 | 111 | 139 | 417 |
| `fd_lock@pqc_write` | 168 | 398 | 833 | 1398 | 509 |
| `fd_lock@restore_qos_class_for_fd` | 126 | 74 | 84 | 102 | 185 |
| `file_anchor_commit_lock@file_anchor_already_committed` | 157 | 186 | 334 | 555 | 425 |
| `file_anchor_commit_lock@file_anchor_note_committed` | 156 | 472 | 639 | 778 | 407 |
| `file_state_table_lock@pqc_file_state_acquire` | 126 | 83 | 870 | 2490 | 445 |
| `file_state_table_lock@pqc_file_state_release` | 126 | 74 | 2704 | 4473 | 333 |
| `parallel_runtime_lock@pqc_parallel_commit_runtime_shutdown` | 1 | 158 | 158 | 158 | 185 |
| `qos_gpu_load_lock@pqc_qos_gpu_load_ewma_read` | 168 | 111 | 130 | 148 | 379 |
| `rekey_lifecycle_lock@pqc_rekey_worker_start` | 1 | 73583 | 73583 | 73583 | 27352 |
| `rekey_lifecycle_lock@pqc_rekey_worker_stop` | 2 | 84 | 102 | 102 | 7546 |
| `rekey_queue_lock@pqc_rekey_queue_stats_snapshot` | 1 | 84 | 84 | 84 | 176 |
| `rekey_queue_lock@pqc_rekey_worker_start` | 1 | 1000 | 1000 | 1000 | 195 |
| `rekey_queue_lock@pqc_rekey_worker_stop` | 2 | 1963 | 7815 | 7815 | 296 |
| `rekey_queue_lock@rekey_worker_main` | 1 | 17630 | 17630 | 17630 | 0 |
| `scheduler_lock@scheduler_data_job_snapshot` | 168 | 111 | 167 | 213 | 361 |
| `scheduler_lock@scheduler_data_job_stats` | 168 | 102 | 111 | 112 | 84 |
| `scheduler_lock@scheduler_policy_snapshot` | 1 | 232 | 232 | 232 | 74 |
| `scheduler_lock@scheduler_record_data_bytes` | 168 | 112 | 176 | 213 | 352 |
| `scheduler_lock@scheduler_reload_publish` | 1 | 112 | 112 | 112 | 74 |
| `scheduler_lock@scheduler_reload_snapshot` | 1 | 102 | 102 | 102 | 74 |
| `scheduler_lock@scheduler_stats_snapshot` | 1 | 129 | 129 | 129 | 241 |
| `trace_sink_lock@trace_sink_close` | 3 | 138 | 139 | 139 | 9491 |

## Blocking Syscall Profile

| Syscall | Count | p50 ns | p95 ns | p99 ns | max ns |
| --- | ---: | ---: | ---: | ---: | ---: |
| `close` | 63 | 19000 | 25000 | 48000 | 53000 |
| `fdatasync` | 8 | 2315000 | 5674000 | 5674000 | 5674000 |
| `openat` | 80 | 25000 | 42000 | 1487000 | 3886000 |
| `pread64` | 8 | 84000 | 120000 | 120000 | 120000 |
| `pwrite64` | 8 | 50000 | 99000 | 99000 | 99000 |

## Non-Closure

- Gate 0.15 is not closed: this thread/process-count sweep is still narrow and lacks off-CPU profiling and long-duration client-count coverage.
- Strict full-tier writeback prepare, crypto, durable publication, authenticated-read recovery, truncate metadata publication, and fallocate metadata publication no longer run under fd_lock or commit_lock on their hot paths.
- Release now detaches fds, buffers, and file-state references under fd_lock and closes/frees/releases detached resources after unlocking.
- Same-file and disjoint-file lifecycle stress now exercises repeated open/write/fdatasync/read/close on the mounted path under bounded timeouts.
- External process lifecycle stress now exercises same-file and disjoint-file client phases up to the maximum configured client count.
- A strace blocking-syscall profile now records mounted-path client fdatasync/pwrite/pread syscall durations, but scheduler off-CPU sampling is still absent.
- Reader visibility is bounded by committed_generation and covered by one paused-publish probe, but lifecycle stress is still short-duration.
- Commit-lock shrinking remains incomplete until reserved-generation fault cases, concurrent reader visibility, and client-count sweeps are expanded.
- No paper scalability or parallel-commit claim is justified by this artifact.
