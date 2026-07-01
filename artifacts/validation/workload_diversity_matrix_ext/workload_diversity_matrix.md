# Workload Diversity Matrix

- Overall pass: `false`
- Scope: Scoped workload-diversity matrix over retained artifacts.  It separates synthetic fio evidence, SQLite WAL/FULL samples, SQLite DELETE/FULL QoS recovery, dbm.dumb stale-snapshot recovery, and TensorRT/YOLO co-run traces.  It does not create a broad filesystem baseline matrix or a foreground-AI recovery claim.

## Checks

- `required_workloads_present`: `true`
- `all_rows_have_filesystem_modes`: `true`
- `sqlite_qos_extra_row_pass`: `true`
- `paper_scope_gate_pass`: `false`
- `synthetic_fio_frozen_aegisq_pass`: `true`
- `sqlite_wal_full_retained_samples_pass`: `true`
- `sqlite_delete_full_qos_recovery_pass`: `true`
- `dbm_dumb_tpm_stale_snapshot_pass`: `true`
- `tensorrt_yolo_secure_io_corun_pass`: `true`

## Paper Scope Gate

- `mentions_workload_diversity_manifest`: `false`
- `separates_fio_and_application_evidence`: `false`
- `mentions_sqlite_dbm_application_scope`: `false`
- `keeps_tensorrt_trace_only`: `false`
- `keeps_no_ai_recovery_claim`: `false`

## Rows

### synthetic_fio_frozen_aegisq

- Pass: `true`
- Category: `synthetic_fio_style_filesystem_workload`
- Workload: fio 4 KiB 70/30 random read/write, psync, per-write fdatasync
- Evidence type: filesystem microbenchmark
- Claim scope: AEGIS-Q-only warm-cache frozen-contract row; not a cross-system superiority claim
- Artifacts: `artifacts/validation/frozen_aegisq_contract/frozen_aegisq_contract.json`
- Scripts: `code/experiments/run_frozen_aegisq_contract.py`

Filesystem modes:
- `fuse_cache_setting`: final pqc_fuse mount with no writeback_cache option in the runner
- `sync_fsync_behavior`: fio psync with --fdatasync=1, queue depth 1, one client
- `journal_or_wal_mode`: AEGIS-Q encrypted record plus journal/checkpoint path, not an application WAL
- `cache_state`: warm-cache row valid after one warmup pass; cold-cache row invalid when privileged cache dropping is unavailable

Row checks:

- `artifact_pass`: `true`
- `aegisq_mode`: `true`
- `warm_repetitions_5`: `true`
- `cold_cache_invalid_recorded`: `true`
- `fdatasync_contract`: `true`
- `direct_buffered_contract`: `true`
- `script_mounts_final_fuse`: `true`

### sqlite_wal_full_retained_samples

- Pass: `true`
- Category: `sqlite_wal_full`
- Workload: SQLite transactions and contention samples using WAL/FULL
- Evidence type: application-level WAL/FULL samples and recovery-oracle definition
- Claim scope: WAL/FULL workload evidence only; not the SQLite QoS recovery table and not crash certification
- Artifacts: `artifacts/validation/sqlite_recovery_oracle/sqlite_recovery_oracle.json, artifacts/results/motivation/sqlite_contention_latency.csv, artifacts/results/motivation/sqlite_latency.csv`
- Scripts: `code/experiments/run_motivation_bench.py, code/experiments/build_sqlite_recovery_oracle.py`

Filesystem modes:
- `fuse_cache_setting`: motivation runner does not rely on a SQLite mmap redirect; WAL may fall back to DELETE when the mounted stack rejects it
- `sync_fsync_behavior`: SQLite commit path uses PRAGMA synchronous=FULL and retained strace observes fsync/fdatasync boundaries
- `journal_or_wal_mode`: PRAGMA journal_mode=WAL, actual_mode=WAL, PRAGMA integrity_check=ok
- `cache_state`: no cold/warm filesystem claim; retained samples are application workload observations

Row checks:

- `oracle_exists`: `true`
- `wal_full_rows_present`: `true`
- `raw_sources_exist`: `true`
- `integrity_ok`: `true`
- `script_requests_wal_full`: `true`
- `script_records_writeback_cache_mode`: `true`
- `script_avoids_sqlite_mmap_redirect_env`: `true`

### sqlite_delete_full_qos_recovery

- Pass: `true`
- Category: `additional_sqlite_application_qos_row`
- Workload: SQLite foreground transactions under mounted secure-storage pressure
- Evidence type: application-level QoS recovery
- Claim scope: SQLite DELETE/FULL recovery under mounted AEGIS-Q pressure, not SQLite WAL/FULL and not TensorRT/AI recovery
- Artifacts: `artifacts/validation/qos_sqlite_hero_bundle/qos_sqlite_hero_bundle.json`
- Scripts: `code/experiments/run_qos_sqlite_hero_bundle.py`

Filesystem modes:
- `fuse_cache_setting`: final pqc_fuse mount; foreground file marked latency, background file marked elastic with user.pqc_qos_class
- `sync_fsync_behavior`: SQLite PRAGMA synchronous=FULL; background writer uses secure-storage writes and daemon throttle traces
- `journal_or_wal_mode`: PRAGMA journal_mode=DELETE for the QoS bundle
- `cache_state`: fresh temporary lower directory per mode; not a cold-cache filesystem baseline

Row checks:

- `artifact_pass`: `true`
- `required_modes_present`: `true`
- `aegis_mode_acceptable`: `true`
- `qos_script_delete_full`: `true`
- `qos_script_classifies_files`: `true`

### dbm_dumb_tpm_stale_snapshot

- Pass: `true`
- Category: `key_value_or_append_log_workload`
- Workload: Python dbm.dumb key-value store on TPM-backed AEGIS-Q FUSE backing store
- Evidence type: application-level key-value stale-snapshot recovery
- Claim scope: dbm.dumb stale-snapshot fail-closed evidence; not RocksDB, LMDB, or arbitrary append-log certification
- Artifacts: `artifacts/validation/combined_durability_bundle/combined_durability_bundle.json`
- Scripts: `code/experiments/run_combined_durability_bundle.py`

Filesystem modes:
- `fuse_cache_setting`: TPM-backed pqc_fuse mounted without writeback_cache option in the combined durability runner
- `sync_fsync_behavior`: dbm.dumb sync is invoked when available; replay verdict is oracle-based rather than a throughput result
- `journal_or_wal_mode`: dbm.dumb persistent .dat/.dir key-value files, no SQLite WAL
- `cache_state`: same-backing-store stale snapshot replay, not cold/warm cache benchmarking

Row checks:

- `artifact_exists`: `true`
- `dbm_baseline_one_row`: `true`
- `dbm_advanced_three_rows`: `true`
- `dbm_replay_fail_closed`: `true`
- `dbm_replay_acceptable`: `true`
- `script_uses_dbm_dumb`: `true`
- `script_mounts_hardware_anchor`: `true`

### tensorrt_yolo_secure_io_corun

- Pass: `true`
- Category: `foreground_inference_co_run_workload`
- Workload: TensorRT YOLOv8 foreground inference co-run with mounted AEGIS-Q secure writer
- Evidence type: foreground-inference interference trace
- Claim scope: co-run interference evidence only; not a validated closed-loop foreground AI p99 recovery claim
- Artifacts: `artifacts/reports/tensorrt_ci_report/tensorrt_ci_report.json, artifacts/results/motivation/tensorrt_interference.json`
- Scripts: `code/experiments/benchmark_tensorrt_interference.py`

Filesystem modes:
- `fuse_cache_setting`: benchmark_tensorrt_interference.py mounts final pqc_fuse without writeback_cache option
- `sync_fsync_behavior`: background secure_writer writes mounted stream.bin and calls os.fsync every sync cycle
- `journal_or_wal_mode`: secure-storage stream workload, no application WAL
- `cache_state`: temporary mount per trial; no cold/warm cache distinction or filesystem-comparison claim

Row checks:

- `report_exists`: `true`
- `raw_source_exists`: `true`
- `required_modes_present`: `true`
- `three_trials_each_mode`: `true`
- `script_mounts_final_fuse`: `true`
- `script_fsyncs_writer`: `true`
- `script_refuses_cpu_fallback`: `true`

## Non-Claims

- no fscrypt/dm-crypt frozen-contract execution
- no cold-cache filesystem result outside the invalid recorded row
- no broad application-workload generalization beyond SQLite/dbm.dumb and TensorRT trace scope
- no foreground TensorRT/AI p99 recovery claim
- no full crash or power-loss certification
