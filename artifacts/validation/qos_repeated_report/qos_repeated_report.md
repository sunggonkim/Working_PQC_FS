# QoS repeated-run report

This bundle packages repeated telemetry-prototype runs only.

- Input directory: `/home/thor/skim/pqc_encrpyted_fs/artifacts/validation/qos_repeated_run`
- Runs found: `3`

## Runs

- latency_ms_median: `5827.11`
- latency_ms_range: `4428.28`--`5829.69`

- run 1: returncode=0, latency_ms=4428.278923034668, events={'hold': 47, 'enter': 1, 'exit': 1}, states=['open', 'throttled'], trace=artifacts/validation/qos_repeated_run/run_1_run_qos_gpu_trace.jsonl
- run 2: returncode=0, latency_ms=5829.685926437378, events={'hold': 48, 'enter': 1}, states=['open', 'throttled'], trace=artifacts/validation/qos_repeated_run/run_2_run_qos_gpu_trace.jsonl
- run 3: returncode=0, latency_ms=5827.10862159729, events={'hold': 48, 'enter': 1}, states=['open', 'throttled'], trace=artifacts/validation/qos_repeated_run/run_3_run_qos_gpu_trace.jsonl

This report does not claim PMU/CUPTI-backed stability.