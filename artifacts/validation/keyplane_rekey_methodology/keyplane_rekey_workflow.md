# Key-Plane Rekey Workflow

- Overall pass: `true`
- Files per mode: `1024`
- GPU-vs-CPU speedup: `1.186`
- Measured repetitions: `5`
- Warmup runs: `1`

## Mode Summaries
- `cpu_only` runs=`5`, acceptable_runs=`5`, median_total_ms=`24.575`, ci95_total_ms=`[24.548, 24.717]`, median_throughput_files_s=`41669.0`
- `gpu_batch` runs=`5`, acceptable_runs=`5`, median_total_ms=`20.729`, ci95_total_ms=`[17.893, 21.447]`, median_throughput_files_s=`49398.9`
- `policy_fallback` runs=`5`, acceptable_runs=`5`, median_total_ms=`24.572`, ci95_total_ms=`[24.422, 24.898]`, median_throughput_files_s=`41673.4`

## Representative Modes
- `cpu_only` acceptable=`true`, events=`1`, files_refreshed=`1024`, total_ms=`24.548`, throughput_files_s=`41714.9`, run_counts=`{'CPU': 1, 'GPU': 0}`, admission_targets=`{'CPU': 0, 'GPU': 0}`
- `gpu_batch` acceptable=`true`, events=`1`, files_refreshed=`1024`, total_ms=`21.447`, throughput_files_s=`47744.7`, run_counts=`{'CPU': 0, 'GPU': 1}`, admission_targets=`{'CPU': 0, 'GPU': 1}`
- `policy_fallback` acceptable=`true`, events=`1`, files_refreshed=`1024`, total_ms=`24.898`, throughput_files_s=`41128.6`, run_counts=`{'CPU': 1, 'GPU': 0}`, admission_targets=`{'CPU': 1, 'GPU': 0}`

## Scope
This artifact supports only a mounted open-file envelope-refresh workflow. It does not prove hardware-backed credential release, persistent PCR binding, or foreground QoS recovery.

## Methodology Metadata

- Run count meets headline minimum: `true`
- Full workload warmup retained: `true`
- CPU governor ready: `false`
- Thermal log nonempty: `true`
