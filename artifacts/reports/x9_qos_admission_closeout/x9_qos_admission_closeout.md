# X9 QoS/admission closeout

Overall pass: `True`

## Pareto result
- `app_only`: p99=7.253 ms, misses=0.0, background=0.000 MB/s
- `unthrottled_storage`: p99=9.621 ms, misses=0.0, background=6.984 MB/s
- `simple_controller`: p99=7.544 ms, misses=0.0, background=1.497 MB/s
- `aegis_policy`: p99=8.151 ms, misses=0.0, background=3.016 MB/s

AEGIS-Q improves p99 over unthrottled storage while spending background throughput. The simple controller is the lower-latency/lower-background point, so the paper must present AEGIS-Q as a selected Pareto point, not a free throughput improvement.

## Controller constants

- GPU min batch: `4096` bytes
- Producer slack budget: `1730000` ns
- Deadline margin: `100000` ns
- Producer slack stale: `250000000` ns
- Queue pressure threshold: `0.8`
- Storage pressure hysteresis: enter `0.7`, exit `0.6`, hold `2` samples

## Sensitivity cases

- `baseline`: p99=6.432 ms, misses=0, storage=3.832 MB/s
- `slow_sampling`: p99=6.971 ms, misses=0, storage=4.240 MB/s
- `high_threshold`: p99=8.832 ms, misses=0, storage=7.236 MB/s
- `queue_depth_2`: p99=12.311 ms, misses=1, storage=6.463 MB/s
- `background_128k`: p99=7.218 ms, misses=0, storage=4.997 MB/s
- `hysteresis_wave`: p99=6.952 ms, misses=0, storage=3.754 MB/s
- `low_pressure_no_throttle`: p99=7.372 ms, misses=0, storage=7.600 MB/s
- `no_slack_mounted`: p99=7.599 ms, misses=40, storage=3.696 MB/s

## Verdict

X9 is closed: the paper has enough retained evidence to answer the repeated QoS tradeoff review as a scoped Pareto/admission result, provided it keeps external application scheduling and free-throughput language out of claim scope.
