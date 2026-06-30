# Shared mmap / shadow-paging verdict

- Gate: `C2`
- Submilestone: `C2-S0`
- Overall pass: `true`
- Decision: `formal_rejection_until_shadow_paging_is_implemented`

## Mounted probes

- `default_env_encrypted_file`: acceptable=`True`, errno=`ENODEV`, detail=`[Errno 19] No such device`
- `sqlite_mmap_env_encrypted_file`: acceptable=`True`, errno=`ENODEV`, detail=`[Errno 19] No such device`

## Shadow follow-up cases

- `concurrent_dirty_page`: `not_applicable_mapping_rejected`
- `mmap_truncate_interaction`: `not_applicable_mapping_rejected`
- `torn_write_after_msync`: `not_applicable_mapping_rejected`
- `remount_after_mmap_dirtying`: `not_applicable_mapping_rejected`
- `rollback_after_mmap_dirtying`: `not_applicable_mapping_rejected`

## Checks

- `mounted_probes_pass`: `true`
- `source_boundary_pass`: `true`
- `posix_audit_boundary_pass`: `true`
- `claim_guard_pass`: `true`
- `followup_cases_closed_by_rejection`: `true`
