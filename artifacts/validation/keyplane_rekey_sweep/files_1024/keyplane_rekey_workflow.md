# Key-Plane Rekey Workflow

- Overall pass: `false`
- Files per mode: `1024`
- GPU-vs-CPU speedup: `0.968`
- Measured repetitions: `1`
- Warmup runs: `0`

## Mode Summaries
- `cpu_only` runs=`1`, acceptable_runs=`1`, median_total_ms=`24.902`, ci95_total_ms=`[24.902, 24.902]`, median_throughput_files_s=`41120.4`
- `gpu_batch` runs=`1`, acceptable_runs=`0`, median_total_ms=`25.735`, ci95_total_ms=`[25.735, 25.735]`, median_throughput_files_s=`39789.4`
- `policy_fallback` runs=`1`, acceptable_runs=`0`, median_total_ms=`24.807`, ci95_total_ms=`[24.807, 24.807]`, median_throughput_files_s=`41278.7`

## Representative Modes
- `cpu_only` acceptable=`true`, events=`512`, files_refreshed=`1024`, total_ms=`24.902`, throughput_files_s=`41120.4`, run_counts=`{'CPU': 512, 'GPU': 0}`, admission_targets=`{'CPU': 0, 'GPU': 0}`
- `gpu_batch` acceptable=`false`, events=`512`, files_refreshed=`1024`, total_ms=`25.735`, throughput_files_s=`39789.4`, run_counts=`{'CPU': 512, 'GPU': 0}`, admission_targets=`{'CPU': 0, 'GPU': 0}`
- `policy_fallback` acceptable=`false`, events=`512`, files_refreshed=`1024`, total_ms=`24.807`, throughput_files_s=`41278.7`, run_counts=`{'CPU': 512, 'GPU': 0}`, admission_targets=`{'CPU': 0, 'GPU': 0}`

## Scope
This artifact supports only a mounted open-file envelope-refresh workflow. It does not prove hardware-backed credential release, persistent PCR binding, or foreground QoS recovery.

## Methodology Metadata

- Run count meets headline minimum: `false`
- Full workload warmup retained: `false`
- CPU governor ready: `false`
- Thermal log nonempty: `true`
