# Epoch Redo-Log Record Contract

- Generated: `2026-06-28T06:26:02Z`
- Overall pass: `true`
- Self-test pass: `true`
- Source pass: `true`
- Mounted path integration visible: `true`

## Covered Rejections

- checksum mismatch returns `-EBADMSG`
- version mismatch returns `-EPROTO`
- invalid plaintext length returns `-EINVAL`
- short record returns `-EMSGSIZE`

## Unsupported Durability Boundary

This codec artifact does not by itself prove append durability, group commit, checkpoint compaction, crash replay, fdatasync reduction, or throughput improvement. Those remain Gate 0.9-S2 through 0.9-S4 work.
