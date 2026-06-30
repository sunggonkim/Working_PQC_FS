# Kernel Baseline Feasibility Audit

- Overall pass: `true`
- Contract ID: `aegisq-fs-frozen-v2-2026-06-27`
- Scope: Non-destructive feasibility audit only.  It does not execute fscrypt or dm-crypt benchmark rows and does not close the frozen kernel-baseline matrix item.
- fscrypt runnable without interactive root: `false`
- dm-crypt/ext4 runnable without interactive root: `false`
- fscrypt runnable with supplied sudo password: `false`
- dm-crypt/ext4 runnable with supplied sudo password: `false`

## Blocking Reasons

- fscrypt: `noninteractive_sudo_unavailable, kernel_config_fs_encryption_disabled, root_ext4_encrypt_feature_not_enabled, fscrypt_status_nonzero`
- dm-crypt/ext4: `noninteractive_sudo_unavailable`

## Next Required Action

fscrypt cannot be executed on this kernel while CONFIG_FS_ENCRYPTION is disabled.  A root-controlled LUKS2 dm-crypt/ext4 loop volume is probe-runnable with sudo, but the checklist item remains open until both kernel baseline rows are available under the frozen fio contract.  Do not use the historical sequential fio files as current comparison evidence.
