# Key-Plane Rekey Workflow

- Overall pass: `false`
- Files per mode: `128`
- GPU-vs-CPU speedup: `0.969`
- Measured repetitions: `1`
- Warmup runs: `0`

## Mode Summaries
- `cpu_only` runs=`1`, acceptable_runs=`1`, median_total_ms=`3.322`, ci95_total_ms=`[3.322, 3.322]`, median_throughput_files_s=`38528.7`
- `gpu_batch` runs=`1`, acceptable_runs=`0`, median_total_ms=`3.430`, ci95_total_ms=`[3.430, 3.430]`, median_throughput_files_s=`37322.1`
- `policy_fallback` runs=`1`, acceptable_runs=`0`, median_total_ms=`3.255`, ci95_total_ms=`[3.255, 3.255]`, median_throughput_files_s=`39322.9`

## Representative Modes
- `cpu_only` acceptable=`true`, events=`64`, files_refreshed=`128`, total_ms=`3.322`, throughput_files_s=`38528.7`, run_counts=`{'CPU': 64, 'GPU': 0}`, admission_targets=`{'CPU': 0, 'GPU': 0}`
- `gpu_batch` acceptable=`false`, events=`64`, files_refreshed=`128`, total_ms=`3.430`, throughput_files_s=`37322.1`, run_counts=`{'CPU': 64, 'GPU': 0}`, admission_targets=`{'CPU': 0, 'GPU': 0}`
- `policy_fallback` acceptable=`false`, events=`64`, files_refreshed=`128`, total_ms=`3.255`, throughput_files_s=`39322.9`, run_counts=`{'CPU': 64, 'GPU': 0}`, admission_targets=`{'CPU': 0, 'GPU': 0}`

## Scope
This artifact supports only a mounted open-file envelope-refresh workflow. It does not prove hardware-backed credential release, persistent PCR binding, or foreground QoS recovery.

## Methodology Metadata

- Run count meets headline minimum: `false`
- Full workload warmup retained: `false`
- CPU governor ready: `false`
- Thermal log nonempty: `true`
