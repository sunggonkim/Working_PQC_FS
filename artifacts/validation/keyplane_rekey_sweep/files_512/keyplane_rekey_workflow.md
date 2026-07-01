# Key-Plane Rekey Workflow

- Overall pass: `false`
- Files per mode: `512`
- GPU-vs-CPU speedup: `1.014`
- Measured repetitions: `1`
- Warmup runs: `0`

## Mode Summaries
- `cpu_only` runs=`1`, acceptable_runs=`1`, median_total_ms=`12.710`, ci95_total_ms=`[12.710, 12.710]`, median_throughput_files_s=`40282.3`
- `gpu_batch` runs=`1`, acceptable_runs=`0`, median_total_ms=`12.536`, ci95_total_ms=`[12.536, 12.536]`, median_throughput_files_s=`40842.0`
- `policy_fallback` runs=`1`, acceptable_runs=`0`, median_total_ms=`12.989`, ci95_total_ms=`[12.989, 12.989]`, median_throughput_files_s=`39417.1`

## Representative Modes
- `cpu_only` acceptable=`true`, events=`256`, files_refreshed=`512`, total_ms=`12.710`, throughput_files_s=`40282.3`, run_counts=`{'CPU': 256, 'GPU': 0}`, admission_targets=`{'CPU': 0, 'GPU': 0}`
- `gpu_batch` acceptable=`false`, events=`256`, files_refreshed=`512`, total_ms=`12.536`, throughput_files_s=`40842.0`, run_counts=`{'CPU': 256, 'GPU': 0}`, admission_targets=`{'CPU': 0, 'GPU': 0}`
- `policy_fallback` acceptable=`false`, events=`256`, files_refreshed=`512`, total_ms=`12.989`, throughput_files_s=`39417.1`, run_counts=`{'CPU': 256, 'GPU': 0}`, admission_targets=`{'CPU': 0, 'GPU': 0}`

## Scope
This artifact supports only a mounted open-file envelope-refresh workflow. It does not prove hardware-backed credential release, persistent PCR binding, or foreground QoS recovery.

## Methodology Metadata

- Run count meets headline minimum: `false`
- Full workload warmup retained: `false`
- CPU governor ready: `false`
- Thermal log nonempty: `true`
