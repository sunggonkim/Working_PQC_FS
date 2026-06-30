# Parallel Commit Contract

- Generated: `2026-06-30T02:56:32Z`
- Gate: `Gate 0.16 sharded queues and parallel commit contract.`
- Verdict: `not_closed`
- Current topology: `partial_parallel_or_epoch_source_present`
- Paper-eligible as parallel commit: `false`

## Current Evidence

- `strict_publish_commit_exists`: `True`
- `strict_publish_data_fdatasync`: `False`
- `strict_publish_journal_fdatasync`: `False`
- `strict_publish_checkpoint_stage`: `True`
- `writeback_has_publish_turn`: `True`
- `writeback_reserves_generation`: `True`
- `writeback_commits_generation_after_publish`: `True`
- `journal_append_is_per_mapping`: `False`
- `journal_append_function_exists`: `True`
- `journal_legacy_synced_append_exists`: `False`
- `file_state_has_single_publish_ticket`: `True`
- `file_state_has_current_publish_ticket`: `True`
- `fd_context_dirty_sidecar_dedup_exists`: `True`
- `flush_batch_can_cover_multiple_blocks`: `True`
- `parallel_commit_module_exists`: `True`
- `parallel_commit_has_shard_assignment`: `True`
- `parallel_commit_has_leader_follower_roles`: `True`
- `parallel_commit_has_bounded_wait`: `False`
- `parallel_commit_has_finish_wakeup`: `True`
- `parallel_commit_has_stats_snapshot`: `True`
- `parallel_commit_trace_has_wait_telemetry`: `True`
- `parallel_commit_trace_has_queue_depth_telemetry`: `True`
- `parallel_commit_telemetry_sweep_runner_exists`: `True`
- `parallel_commit_fairness_replay_runner_exists`: `True`
- `writeback_calls_parallel_commit`: `True`
- `epoch_publish_module_exists`: `True`
- `epoch_publish_sets_skip_journal_fsync`: `True`
- `epoch_publish_has_group_barrier`: `True`
- `epoch_publish_uses_syncfs_for_group_barrier`: `False`
- `epoch_log_replay_parser_exists`: `True`
- `epoch_log_checkpoint_compaction_exists`: `True`
- `epoch_log_journal_repair_exists`: `True`
- `epoch_log_duplicate_generation_reject_exists`: `True`
- `shard_or_queue_source_files`: `['pqc_epoch_log.c', 'pqc_epoch_log.h', 'pqc_epoch_publish.c', 'pqc_epoch_publish.h', 'pqc_parallel_commit.c', 'pqc_parallel_commit.h']`
- `mentions_group_commit`: `False`
- `mentions_sharded_commit`: `True`
- `mentions_leader_waiter`: `False`

## Mounted-Path Baseline

- Concurrency summary: `artifacts/validation/concurrency_contract/lock_profile_summary.json`
- Overall pass: `true`
- Thread counts: `[1, 2, 4]`
- Workload coverage: `{'disjoint_lifecycle_phases': 3, 'disjoint_process_lifecycle_phases': 3, 'disjoint_writer_phases': 3, 'max_process_client_count': 4, 'max_thread_count': 4, 'same_file_lifecycle_phases': 3, 'same_file_process_lifecycle_phases': 3, 'same_file_writer_phases': 3}`
- Blocking syscall profile pass: `true`

## Epoch-Gated Strict Mounted-Path Smoke

- Smoke artifact: `artifacts/validation/parallel_commit_contract/epoch_path_smoke.json`
- Overall pass: `true`
- Strict control pass: `true`
- Epoch-gated strict pass: `true`
- Begin events: `1`
- Finish events: `1`
- Roles: `['invalid', 'leader']`

## Parallel Commit Telemetry Sweep

- Sweep artifact: `artifacts/validation/parallel_commit_contract/parallel_commit_telemetry_sweep.json`
- Overall pass: `true`
- Coverage: `{'client_counts': [1, 2, 4], 'group_max_values': [1, 2], 'shard_counts': [1, 4], 'wait_ns_values': [0, 5000000]}`
- `single_client_s1_g1`: pass=`true`, clients=`1`, shards=`1`, group sizes=`{'1': 2}`, queue depths=`{'1': 2}`, trace=`artifacts/validation/parallel_commit_contract/telemetry_sweep/single_client_s1_g1/parallel_commit_trace.jsonl`
- `dual_client_s1_g2_wait`: pass=`true`, clients=`2`, shards=`1`, group sizes=`{'2': 4}`, queue depths=`{'2': 4}`, trace=`artifacts/validation/parallel_commit_contract/telemetry_sweep/dual_client_s1_g2_wait/parallel_commit_trace.jsonl`
- `quad_client_s4_g2_wait`: pass=`true`, clients=`4`, shards=`4`, group sizes=`{'1': 2, '2': 6}`, queue depths=`{'1': 2, '2': 6}`, trace=`artifacts/validation/parallel_commit_contract/telemetry_sweep/quad_client_s4_g2_wait/parallel_commit_trace.jsonl`

## Fairness and Trace Replay-Order Evidence

- Evidence artifact: `artifacts/validation/parallel_commit_contract/parallel_commit_fairness_replay.json`
- Overall pass: `true`
- Coverage: `{'case_names': ['starvation_same_shard', 'replay_order_multi_shard'], 'client_counts': [4, 8], 'group_max_values': [2, 4], 'shard_counts': [1, 4], 'wait_ns_values': [5000000, 10000000]}`
- `starvation_same_shard`: pass=`true`, starvation=`true`, replay=`true`, ops=`32/32`, replay_plan=`8`, replay_time_ns=`242667`, trace=`artifacts/validation/parallel_commit_contract/fairness_replay/starvation_same_shard/parallel_commit_trace.jsonl`
- `replay_order_multi_shard`: pass=`true`, starvation=`true`, replay=`true`, ops=`12/12`, replay_plan=`7`, replay_time_ns=`177121`, trace=`artifacts/validation/parallel_commit_contract/fairness_replay/replay_order_multi_shard/parallel_commit_trace.jsonl`

## Epoch Redo-Log Replay Fault Matrix

- Evidence artifact: `artifacts/validation/publication_protocol_fault_matrix/epoch_replay_fault_matrix.json`
- Overall pass: `true`
- Case pass: `{'replay_duplicate_generation': True, 'replay_journal_loss': True, 'replay_normal': True, 'replay_torn_tail': True}`
- Duplicate-generation rejection: `true`
- Torn-tail remount: `true`
- Journal-loss repair: `true`
- Journal repair records max: `4`

## Strict Versus Epoch Publication Measurement

- Evidence artifact: `artifacts/validation/publication_protocol_fault_matrix/epoch_publication_comparison.json`
- Overall pass: `true`
- Strict throughput MiB/s: `12.959676619366542`
- Epoch throughput MiB/s: `10.880134274435354`
- Strict sync count: `10`
- Epoch sync count: `10`
- Strict grouped sync count: `8`
- Epoch grouped sync count: `5`
- Epoch grouped max append group size: `4`
- Epoch grouped sync primitives: `['none', 'syncfs']`
- Strict journal fsync count: `5`
- Epoch journal fsync count: `0`
- Epoch log fsync count: `5`
- Verdict: Grouped epoch redo-log amortized the metadata barrier in the concurrent mounted workload: total traced sync count is lower than strict while journal fsync remains off the foreground epoch path.

## Candidate Topologies

### strict_per_file_publish_turn
- Status: `implemented-baseline`
- Shard assignment: one publication order per backing file state
- Leader/waiter protocol: condition-variable publish ticket, one active publisher
- Fairness policy: FIFO within one file only
- Recovery ordering rule: reader-visible generation advances only after data and journal barriers

### per_file_epoch_log
- Status: `candidate-not-implemented`
- Shard assignment: one epoch queue per backing file state
- Leader/waiter protocol: first waiter becomes epoch leader; followers join until size or age threshold
- Fairness policy: bounded epoch age plus foreground bypass for latency-sensitive fsync
- Recovery ordering rule: redo records replay in epoch order; checkpoint records only committed prefix

### hash_sharded_epoch_log
- Status: `candidate-not-implemented`
- Shard assignment: hash(file_id) modulo shard_count
- Leader/waiter protocol: one leader per shard epoch
- Fairness policy: per-shard FIFO plus starvation counter for cold shards
- Recovery ordering rule: replay is independent across file_id; global freshness anchor records committed shard vector

### global_epoch_log
- Status: `rejected-baseline-unless-proven`
- Shard assignment: single global queue
- Leader/waiter protocol: single epoch leader
- Fairness policy: global FIFO
- Recovery ordering rule: single total redo order

### per_directory_epoch_log
- Status: `deferred`
- Shard assignment: directory-local queue
- Leader/waiter protocol: leader per directory
- Fairness policy: directory FIFO
- Recovery ordering rule: requires stronger rename and directory-fsync semantics first

## Blocking Items


## Negative Claim Guard

No paper or README text may claim sharded queues, parallel commit, epoch fdatasync, group commit, or scalability from commit batching until this contract reports overall_pass=true with production mounted-path evidence.
