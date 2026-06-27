# Crash / recovery audit report

This report aggregates the retained crash and freshness evidence without claiming full app-level recovery.

## Retained evidence
- file replay negative control: 2 cut-point rows
- file replay final negative control: 2 cut-point rows
- post-keyfix replay summary: 2 cut-point rows
- E8 crash/replay regression: 8 summarized rows, max fail_closed_rate=1.00, min fail_closed_rate=1.00
- TPM unprovisioned path: exit=1
- hardware anchor round-trip: median 9.415056 ms, p95 28.107919 ms
- sqlite probe present: True
- sqlite fault campaign: 20 trials, unacceptable=0
- sqlite syscall crash TPM campaign: 3 trials, unacceptable=0

## Current gap
- app-level evidence includes selected-boundary SQLite replay and fdatasync-exact SQLite app-crash timing
- power-loss and FUSE-daemon crash timing remain open
