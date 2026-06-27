# POSIX scope audit

Scope: final-binary audit for intentionally narrow POSIX semantics.

- Overall pass: `True`
- Command: `experiments/run_posix_scope_audit.py --out-dir artifacts/validation/posix_scope_audit`

| Case | Expected behavior | Acceptable | Scope |
|---|---|---:|---|
| `default_mmap_on_encrypted_file` | `rejected` | `True` | Default encrypted files use FUSE direct_io and do not claim shared mmap semantics. |
| `concurrent_disjoint_writers` | `serialized_by_daemon_lock_for_disjoint_write_fsync_path` | `True` | Narrow disjoint write/fsync path only; arbitrary conflicting app-level concurrency is not certified. |
| `rename_rejection` | `rejected_ENOTSUP` | `True` | Rename would need atomic marker/data/journal/checkpoint/anchor transition; not implemented. |
| `directory_fsync_rejection` | `rejected_ENOTSUP` | `True` | Directory-entry durability is not a supported boundary in this prototype. |
| `xattr_policy` | `user_xattr_roundtrip_internal_xattrs_hidden_invalid_tier_rejected` | `True` | User xattrs and validated tier control are supported; internal metadata/checkpoint xattrs are not user-visible controls. |
| `lower_filesystem_assumptions` | `recorded_not_certified` | `True` | The harness records lower-filesystem type and xattr availability; xattr atomicity and directory durability remain delegated, not certified. |
| `fuse_writeback_cache_and_direct_mmap_caps` | `capabilities_not_requested` | `True` | The daemon avoids FUSE writeback-cache and direct-IO mmap capabilities; no cache-coherence claim is made. |

The JSON file retains errno values, mount logs, source assertions, and lower-filesystem metadata.
