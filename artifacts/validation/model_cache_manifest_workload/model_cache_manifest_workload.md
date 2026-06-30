# Model-Cache Manifest Workload

- Overall pass: `true`
- Objects: `24`
- Payload bytes per object: `16384`
- Payload throughput MiB/s: `1.997`
- Publish p50/p99 us: `4223.1` / `28088.1`
- Manifest publish us: `7573.3`
- Mounted verify pass: `true`
- Remount verify pass: `true`

Scope: mounted model-cache object staging with closed-file rename and directory fsync; not a broad workload suite or full POSIX proof.
