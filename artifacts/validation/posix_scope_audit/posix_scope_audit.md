# POSIX scope audit

Scope: final-binary audit for intentionally narrow POSIX semantics.

- Overall pass: `True`
- Command: `experiments/run_posix_scope_audit.py --out-dir artifacts/validation/posix_scope_audit`
- Required semantics covered: `True`

## Required semantic status

| Semantic | Status | Covered | Acceptable | Cases |
|---|---|---:|---:|---|
| `shared_mmap` | `formal rejection` | `True` | `True` | `default_shared_mmap_msync_on_encrypted_file` |
| `msync` | `formal rejection` | `True` | `True` | `default_shared_mmap_msync_on_encrypted_file` |
| `rename` | `supported_subset` | `True` | `True` | `closed_file_rename_supported_subset, closed_file_overwrite_rename_supported_subset, open_source_rename_supported_subset, open_target_rename_supported_subset, empty_directory_rename_supported_subset, directory_tree_rename_supported_subset` |
| `directory_fsync` | `supported_subset` | `True` | `True` | `directory_fsync_supported_subset` |
| `open_fd_truncate` | `support` | `True` | `True` | `truncate_fallocate_visibility` |
| `path_truncate` | `formal rejection` | `True` | `True` | `path_only_truncate_rejection` |
| `fallocate` | `support` | `True` | `True` | `truncate_fallocate_visibility` |
| `hard_link` | `supported_subset` | `True` | `True` | `hard_link_supported_subset` |
| `symlink` | `supported_subset` | `True` | `True` | `symlink_supported_subset` |
| `concurrent_disjoint_writes` | `support` | `True` | `True` | `concurrent_disjoint_writers` |
| `concurrent_same_block_writes` | `supported_subset` | `True` | `True` | `concurrent_same_block_full_overwrite_writers, concurrent_same_block_partial_overlap_writers` |
| `crash_time_visibility` | `paper limitation` | `True` | `True` | `crash_time_visibility_scope` |

## Audit rows

| Case | Status | Expected behavior | Acceptable | Scope |
|---|---|---|---:|---|
| `default_shared_mmap_msync_on_encrypted_file` | `formal rejection` | `rejected` | `True` | Default encrypted files use FUSE direct_io and do not claim shared mmap semantics. |
| `concurrent_disjoint_writers` | `support` | `serialized_by_daemon_lock_for_disjoint_write_fsync_path` | `True` | Narrow disjoint write/fsync path only; arbitrary conflicting app-level concurrency is not certified. |
| `concurrent_same_block_full_overwrite_writers` | `supported_subset` | `same_block_full_overwrites_linearize_without_mixed_plaintext` | `True` | Same-block full-overwrite write/fsync races are serialized to one complete block; arbitrary overlapping byte ranges, mmap coherence, and application-level transactions are not certified. |
| `concurrent_same_block_partial_overlap_writers` | `supported_subset` | `overlapping_same_block_partial_writes_match_one_serial_order` | `True` | One same-block overlapping partial-write/fsync race is serializable to a complete-block recovery order; arbitrary byte-range transactions, mmap coherence, and application-level locking semantics are not certified. |
| `truncate_fallocate_visibility` | `support` | `open_fd_ftruncate_and_fallocate_preserve_visible_size_and_zero_fill` | `True` | Open-file truncate/fallocate behavior only; path-only truncate, mmap coherence, and broad concurrent resize semantics are not certified. |
| `path_only_truncate_rejection` | `formal rejection` | `rejected_ENOTSUP_without_open_file_context` | `True` | Path-only truncate lacks an authenticated open-file context; open-fd ftruncate is tested separately. |
| `closed_file_rename_supported_subset` | `supported_subset` | `rename new closed regular file with sidecars` | `True` | Closed regular-file rename to a non-existing target is supported; open-file and directory cases are tested separately, and crash-atomic rename remains outside this row. |
| `closed_file_overwrite_rename_supported_subset` | `supported_subset` | `closed regular overwrite rename replaces closed target` | `True` | Closed regular-file overwrite rename is supported only when the target is not open; open-target and crash-atomic multi-file rename remain outside this row. |
| `open_source_rename_supported_subset` | `supported_subset` | `open source fd survives rename to new target` | `True` | Open-source regular-file rename to a non-existing target is supported after fd-context path retargeting; open-target overwrite is tested separately and crash-atomic rename remains outside this row. |
| `open_target_rename_supported_subset` | `supported_subset` | `rename over open target preserves stale target fd and publishes source at target path` | `True` | Open-target regular-file overwrite rename is supported through FUSE hidden-file retargeting for the stale target fd; crash-atomic multi-file rename and open-subtree retargeting remain outside this row. |
| `empty_directory_rename_supported_subset` | `supported_subset` | `empty directory rename to non-existing target succeeds` | `True` | Empty-directory rename to a non-existing target is supported only when no file contexts are open; open-subtree retargeting and crash-atomic namespace publication remain outside this row. |
| `directory_tree_rename_supported_subset` | `supported_subset` | `non-empty directory tree rename to non-existing target succeeds` | `True` | Directory-tree rename is supported only with no open file contexts and a non-existing target; open-subtree retargeting and crash-atomic multi-entry durability remain outside this row. |
| `directory_fsync_supported_subset` | `supported_subset` | `lower directory fsync succeeds` | `True` | Directory fsync is forwarded to the lower directory; this does not by itself certify crash-atomic multi-file rename. |
| `hard_link_supported_subset` | `supported_subset` | `no-open regular-file hard link shares marker and sidecar state` | `True` | Regular-file hard links are supported only with no open file contexts and linked marker/sidecar state; open hard-link creation, SQLite compatibility sidecars, crash-atomic multi-entry publication, and full link-count lifecycle certification remain outside this row. |
| `symlink_supported_subset` | `supported_subset` | `relative_symlink_readlink_and_read_through` | `True` | Relative symlink namespace objects are supported when they do not target internal sidecar names or escape through absolute/parent paths; this is not hard-link support or crash-atomic directory-tree certification. |
| `xattr_policy` | `support` | `user_xattr_roundtrip_internal_xattrs_hidden_invalid_tier_rejected` | `True` | User xattrs and validated tier control are supported; internal metadata/checkpoint xattrs are not user-visible controls. |
| `lower_filesystem_assumptions` | `paper limitation` | `recorded_not_certified` | `True` | The harness records lower-filesystem type and xattr availability; xattr atomicity and directory durability remain delegated, not certified. |
| `fuse_writeback_cache_and_direct_mmap_caps` | `formal rejection` | `capabilities_not_requested` | `True` | The daemon avoids FUSE writeback-cache and direct-IO mmap capabilities; no cache-coherence claim is made. |
| `crash_time_visibility_scope` | `paper limitation` | `bounded_to_existing_fault_models_not_full_posix_crash_visibility` | `True` | Crash-time visibility is delegated to C3/C4 fault models; C1 does not certify full POSIX crash visibility. |

The JSON file retains errno values, mount logs, source assertions, and lower-filesystem metadata.
