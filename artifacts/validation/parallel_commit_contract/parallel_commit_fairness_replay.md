# Parallel Commit Fairness and Replay-Order Evidence

- Generated: `2026-06-28T06:01:18Z`
- Overall pass: `true`

## starvation_same_shard
- Purpose: bounded starvation negative test under one shared commit shard
- Pass: `true`
- Starvation negative pass: `true`
- Replay-order pass: `true`
- Completed ops: `32` / `32`
- Op count distribution: `{4: 8}`
- Replay reconstruction time ns: `242667`
- Replay plan length: `8`
- Observed replay shards: `[0]`

## replay_order_multi_shard
- Purpose: per-shard epoch replay-order reconstruction
- Pass: `true`
- Starvation negative pass: `true`
- Replay-order pass: `true`
- Completed ops: `12` / `12`
- Op count distribution: `{3: 4}`
- Replay reconstruction time ns: `177121`
- Replay plan length: `7`
- Observed replay shards: `[0, 1, 2]`

## Negative Claim Guard

This artifact proves bounded mounted-path starvation-negative behavior and trace-level per-shard epoch replay-order reconstruction for the current coordinator. It does not prove redo-log recovery, crash replay, fdatasync reduction, throughput improvement, or Gate 0.16 closure.
