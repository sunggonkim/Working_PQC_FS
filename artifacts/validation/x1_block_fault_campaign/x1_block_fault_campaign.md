# X1 Block-Fault Campaign

- Overall pass: `true`
- Blocked: `false`
- Scope: Loopback ext4 lower filesystem on a device-mapper target.; Fault injection switches the lower block device from linear to error and back.; This is block-device interruption evidence, not physical power-loss, kernel-crash, or drive-cache certification.

## Rows

- `lower_block_error_during_latest_write`: verdict `previous_committed`, acceptable `True`, fault write rc `1`
- `lower_block_error_after_write_before_fsync`: verdict `previous_committed`, acceptable `True`, fault write rc `1`
- `lower_block_error_after_successful_fsync`: verdict `latest_committed`, acceptable `True`, fault write rc `0`

## Claim Boundary

This campaign supports lower-block interruption recovery on a loopback ext4/device-mapper stack. It includes a post-fsync accepted-state row, but still does not certify physical power loss, kernel crash, drive write-cache loss, arbitrary workloads, or full POSIX crash semantics.
