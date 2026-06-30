# Parallel Commit Telemetry Sweep

- Generated: `2026-06-28T05:54:39Z`
- Overall pass: `true`
- Case count: `3`

## single_client_s1_g1
- Pass: `true`
- Config: `{'name': 'single_client_s1_g1', 'shards': 1, 'group_max': 1, 'wait_ns': 0, 'clients': 1, 'ops_per_client': 2, 'payload_size': 4096}`
- Trace: `artifacts/validation/parallel_commit_contract/telemetry_sweep/single_client_s1_g1/parallel_commit_trace.jsonl`
- Begin/finish: `2` / `2`
- Group sizes: `{1: 2}`
- Queue depths: `{1: 2}`
- Wait ns summary: `{'count': 2, 'min': 297, 'max': 6740, 'mean': 3518.5, 'p50': 3518.5, 'p95': 6417.85, 'p99': 6675.570000000001}`

## dual_client_s1_g2_wait
- Pass: `true`
- Config: `{'name': 'dual_client_s1_g2_wait', 'shards': 1, 'group_max': 2, 'wait_ns': 5000000, 'clients': 2, 'ops_per_client': 2, 'payload_size': 4096}`
- Trace: `artifacts/validation/parallel_commit_contract/telemetry_sweep/dual_client_s1_g2_wait/parallel_commit_trace.jsonl`
- Begin/finish: `4` / `2`
- Group sizes: `{2: 4}`
- Queue depths: `{2: 4}`
- Wait ns summary: `{'count': 4, 'min': 415991, 'max': 1215352, 'mean': 822947.0, 'p50': 830222.5, 'p95': 1181228.3499999999, 'p99': 1208527.27}`

## quad_client_s4_g2_wait
- Pass: `true`
- Config: `{'name': 'quad_client_s4_g2_wait', 'shards': 4, 'group_max': 2, 'wait_ns': 5000000, 'clients': 4, 'ops_per_client': 2, 'payload_size': 4096}`
- Trace: `artifacts/validation/parallel_commit_contract/telemetry_sweep/quad_client_s4_g2_wait/parallel_commit_trace.jsonl`
- Begin/finish: `8` / `5`
- Group sizes: `{1: 2, 2: 6}`
- Queue depths: `{1: 2, 2: 6}`
- Wait ns summary: `{'count': 8, 'min': 429472, 'max': 5057602, 'mean': 2745885.5, 'p50': 2691467.5, 'p95': 5056727.0, 'p99': 5057427.0}`

## Negative Claim Guard

This sweep proves mounted-path telemetry coverage for group size, wait time, queue depth, shard count, and concurrent clients. It does not prove throughput improvement, fdatasync reduction, fairness, replay ordering, or Gate 0.16 closure.
