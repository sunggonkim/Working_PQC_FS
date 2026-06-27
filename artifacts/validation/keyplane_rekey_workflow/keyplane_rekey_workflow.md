# Key-Plane Rekey Workflow

- Overall pass: `true`
- Files per mode: `1024`
- GPU-vs-CPU speedup: `1.158`

## Modes
- `cpu_only` acceptable=`true`, events=`1`, files_refreshed=`1024`, total_ms=`24.399`, throughput_files_s=`41969.6`, run_counts=`{'CPU': 1, 'GPU': 0}`, admission_targets=`{'CPU': 0, 'GPU': 0}`
- `gpu_batch` acceptable=`true`, events=`1`, files_refreshed=`1024`, total_ms=`21.070`, throughput_files_s=`48599.0`, run_counts=`{'CPU': 0, 'GPU': 1}`, admission_targets=`{'CPU': 0, 'GPU': 1}`
- `policy_fallback` acceptable=`true`, events=`1`, files_refreshed=`1024`, total_ms=`24.411`, throughput_files_s=`41947.6`, run_counts=`{'CPU': 1, 'GPU': 0}`, admission_targets=`{'CPU': 1, 'GPU': 0}`

## Scope
This artifact supports only a mounted open-file envelope-refresh workflow. It does not prove hardware-backed credential release, persistent PCR binding, or foreground QoS recovery.
