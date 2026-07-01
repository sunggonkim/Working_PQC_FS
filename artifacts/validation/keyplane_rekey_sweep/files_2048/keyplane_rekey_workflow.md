# Key-Plane Rekey Workflow

- Overall pass: `false`
- Files per mode: `2048`
- GPU-vs-CPU speedup: `1.042`
- Measured repetitions: `1`
- Warmup runs: `0`

## Mode Summaries
- `cpu_only` runs=`1`, acceptable_runs=`1`, median_total_ms=`54.148`, ci95_total_ms=`[54.148, 54.148]`, median_throughput_files_s=`37822.3`
- `gpu_batch` runs=`1`, acceptable_runs=`0`, median_total_ms=`51.958`, ci95_total_ms=`[51.958, 51.958]`, median_throughput_files_s=`39416.1`
- `policy_fallback` runs=`1`, acceptable_runs=`0`, median_total_ms=`50.971`, ci95_total_ms=`[50.971, 50.971]`, median_throughput_files_s=`40179.9`

## Representative Modes
- `cpu_only` acceptable=`true`, events=`1024`, files_refreshed=`2048`, total_ms=`54.148`, throughput_files_s=`37822.3`, run_counts=`{'CPU': 1024, 'GPU': 0}`, admission_targets=`{'CPU': 0, 'GPU': 0}`
- `gpu_batch` acceptable=`false`, events=`1024`, files_refreshed=`2048`, total_ms=`51.958`, throughput_files_s=`39416.1`, run_counts=`{'CPU': 1024, 'GPU': 0}`, admission_targets=`{'CPU': 0, 'GPU': 0}`
- `policy_fallback` acceptable=`false`, events=`1024`, files_refreshed=`2048`, total_ms=`50.971`, throughput_files_s=`40179.9`, run_counts=`{'CPU': 1024, 'GPU': 0}`, admission_targets=`{'CPU': 0, 'GPU': 0}`

## Scope
This artifact supports only a mounted open-file envelope-refresh workflow. It does not prove hardware-backed credential release, persistent PCR binding, or foreground QoS recovery.

## Methodology Metadata

- Run count meets headline minimum: `false`
- Full workload warmup retained: `false`
- CPU governor ready: `false`
- Thermal log nonempty: `true`
