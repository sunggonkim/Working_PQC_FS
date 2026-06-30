# Attacker-Model Claim Guard

- Overall pass: `true`
- Scope: D4 attacker-model and non-claim guard. It verifies the production trust-boundary table and rejects unguarded claims about compromised kernel/driver/FUSE defense, privileged local attackers, multi-tenant GPU isolation, GPU side-channel protection, deployment readiness, and general-purpose filesystem status.
- Candidate count: `62`
- Unguarded count: `0`

## Close Conditions

- `production_trust_boundary_table_complete`: `true`
- `paper_states_attacker_model_and_non_claims`: `true`
- `dangerous_claim_lint_passes`: `true`

## source_checks

- `all_required_subjects_present`: `true`
- `all_names_match_subjects`: `true`
- `all_rows_have_scope_fields`: `true`
- `required_subjects_are_non_claims`: `true`
- `required_non_claim_guards_present`: `true`
- `trust_self_test_checks_required_non_claims`: `true`
- `self_test_wired_to_binary`: `true`
- `build_includes_trust_boundary_source`: `true`
- `header_exposes_single_trust_boundary_api`: `true`

## paper_checks

- `trusts_kernel_fuse_driver_stack`: `true`
- `privileged_attacker_nonclaim`: `true`
- `gpu_side_channel_nonclaim`: `true`
- `limitations_privileged_sidechannel`: `true`
- `deployment_nonclaim`: `true`
- `conclusion_nonclaims`: `true`

## claim_checks

- `dangerous_claim_candidates_scanned`: `true`
- `unguarded_d4_claim_count_zero`: `true`

## Trust Boundary Rows

### daemon-process

- Subject: `PQC_TRUST_SUBJECT_DAEMON_PROCESS`
- Status: `PQC_TRUST_STATUS_IMPLEMENTED_BOUNDARY`
- Trusted component: pqc_fuse userspace daemon and OpenSSL/OQS libraries
- Excluded attacker: attacker that can read or modify daemon memory, ptrace the process,  or replace linked crypto libraries
- Failure boundary: daemon memory compromise can disclose mount and per-file secrets
- Non-claim guard: do not claim process compromise resistance or deployment hardening

### kernel-driver-fuse-stack

- Subject: `PQC_TRUST_SUBJECT_KERNEL_DRIVER_FUSE`
- Status: `PQC_TRUST_STATUS_NON_CLAIM`
- Trusted component: Linux kernel, VFS, FUSE driver, filesystem driver, and block stack
- Excluded attacker: compromised kernel, malicious FUSE driver, malicious filesystem  driver, or malicious block driver
- Failure boundary: kernel or driver compromise can observe, reorder, suppress, or forge I/O
- Non-claim guard: must not claim compromised-kernel, driver, or FUSE-stack defense

### backing-storage

- Subject: `PQC_TRUST_SUBJECT_BACKING_STORAGE`
- Status: `PQC_TRUST_STATUS_EXTERNAL_TCB`
- Trusted component: local lower filesystem and configured anchor backend
- Excluded attacker: administrator that can roll back the complete lower storage root and  all file-backed anchors without a hardware freshness source
- Failure boundary: file-backed anchors are replayable; TPM-backed freshness depends on provisioning
- Non-claim guard: must not claim full rollback resistance from file-backed anchors

### privileged-local-attacker

- Subject: `PQC_TRUST_SUBJECT_PRIVILEGED_LOCAL_ATTACKER`
- Status: `PQC_TRUST_STATUS_NON_CLAIM`
- Trusted component: host OS privilege separation
- Excluded attacker: root user, host administrator, or process with equivalent local privilege
- Failure boundary: privileged local access can read daemon memory, alter mounts, and modify storage
- Non-claim guard: must not claim privileged local attacker defense

### multi-tenant-gpu

- Subject: `PQC_TRUST_SUBJECT_MULTI_TENANT_GPU`
- Status: `PQC_TRUST_STATUS_NON_CLAIM`
- Trusted component: exclusive or administratively isolated GPU execution environment
- Excluded attacker: co-resident GPU tenant, hostile CUDA process, or shared-device observer
- Failure boundary: multi-tenant GPU deployment requires platform isolation outside AEGIS-Q
- Non-claim guard: must not claim multi-tenant GPU isolation or cross-tenant defense

### gpu-side-channel

- Subject: `PQC_TRUST_SUBJECT_GPU_SIDE_CHANNEL`
- Status: `PQC_TRUST_STATUS_NON_CLAIM`
- Trusted component: single-tenant GPU timing and resource domain assumption
- Excluded attacker: local observer measuring GPU timing, cache effects, memory traffic,  power, temperature, or scheduler occupancy
- Failure boundary: GPU side-channel resistance is out of scope for this implementation
- Non-claim guard: must not claim side-channel protection

### deployment-readiness

- Subject: `PQC_TRUST_SUBJECT_DEPLOYMENT_READINESS`
- Status: `PQC_TRUST_STATUS_NON_CLAIM`
- Trusted component: research prototype configuration and operator discipline
- Excluded attacker: production adversary model requiring complete POSIX coverage,  privileged-attacker defense, side-channel defense, or certified crash safety
- Failure boundary: the code is not a general-purpose deployment-ready secure filesystem
- Non-claim guard: must not claim ready for deployment or general-purpose filesystem status

## Non-Claims

- no compromised kernel/driver/FUSE-stack defense
- no privileged local attacker defense
- no multi-tenant GPU isolation or cross-tenant defense
- no GPU side-channel protection
- no deployment-ready or ready-for-production claim
- no general-purpose filesystem status
