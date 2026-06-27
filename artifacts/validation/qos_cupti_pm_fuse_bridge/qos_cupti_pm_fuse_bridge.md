# QoS CUPTI PM FUSE bridge

This bundle writes through a real FUSE mount while a sudo CUPTI PM-sampling workload writes live PM-derived telemetry into the mounted daemon's telemetry file.

- Verified: `True`
- CUPTI return code: `0`
- CUPTI PM samples: `1741`
- Runtime throttle counts: `{'open': 14, 'throttled': 5}`
- Runtime throttle sleep total: `250000` us
- Writer stats: `{'chunks_written': 76, 'bytes_written': 9961472, 'throttle_sleeps': 0, 'sleep_time_s': 0.0, 'elapsed_s': 2.524104891344905, 'throughput_mb_s': 3.7637104672532713}`
- CUPTI trace: `artifacts/validation/qos_cupti_pm_fuse_bridge/cupti_pm_samples.jsonl`
- FUSE throttle trace: `artifacts/validation/qos_cupti_pm_fuse_bridge/runtime_fuse_throttle_trace.jsonl`

Interpretation: this closes a same-run PM/CUPTI-to-mounted-FUSE throttle wiring check. It does not prove foreground AI p99 recovery.
