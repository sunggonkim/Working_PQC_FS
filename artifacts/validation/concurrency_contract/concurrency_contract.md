# Concurrency Contract

- Generated: `2026-06-30T02:56:54Z`
- Overall pass: `true`
- Observed hot locks: `16`
- Blocking profile pass: `true`
- Deadlock/livelock negative pass: `true`
- Forbidden claim hits: `0`

| Lock | Owner | Samples | max hold p99 ns | max wait p99 ns | Complete |
| --- | --- | ---: | ---: | ---: | --- |
| `admission_state_lock` | `admission` | 2 | 324 | 222 | `true` |
| `admission_trace_lock` | `metrics` | 3 | 7213 | 222 | `true` |
| `anchor_epoch_record_lock` | `anchor` | 157 | 121 | 287 | `true` |
| `anchor_lifecycle_lock` | `anchor` | 4 | 31259 | 241 | `true` |
| `anchor_worker_lock` | `anchor` | 655 | 42981 | 10019 | `true` |
| `commit_lock` | `publish` | 1260 | 931426 | 426 | `true` |
| `committed_map_lock` | `anchor` | 920 | 232 | 416 | `true` |
| `fd_lock` | `FUSE adapter` | 1652 | 40574 | 10398 | `true` |
| `file_anchor_commit_lock` | `anchor` | 313 | 778 | 425 | `true` |
| `file_state_table_lock` | `publish` | 252 | 4473 | 445 | `true` |
| `parallel_runtime_lock` | `publish` | 1 | 158 | 185 | `true` |
| `qos_gpu_load_lock` | `admission` | 168 | 148 | 379 | `true` |
| `rekey_lifecycle_lock` | `admission` | 3 | 73583 | 27352 | `true` |
| `rekey_queue_lock` | `admission` | 5 | 17630 | 296 | `true` |
| `scheduler_lock` | `admission` | 508 | 232 | 361 | `true` |
| `trace_sink_lock` | `metrics` | 3 | 139 | 9491 | `true` |
