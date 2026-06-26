# Telemetry trace report

This report summarizes the retained tegrastats JSONL traces.

## tegra_qos_daemon
- sample_count: 33
- throttle_high_fraction: 0.000
- throttle_transitions: 0
- hysteresis_sample_count: 33
- hysteresis_states: ['open']
- hysteresis_event_counts: {'hold': 33}
- avg_cpu_load_mean: 4.926
- avg_cpu_load_median: 4.929

## run_qos_gpu
- sample_count: 49
- throttle_high_fraction: 0.959
- throttle_transitions: 1
- hysteresis_sample_count: 49
- hysteresis_states: ['open', 'throttled']
- hysteresis_event_counts: {'hold': 48, 'enter': 1}
- gpu_power_mw_mean: 30246.531
- gpu_power_mw_median: 32432.000
