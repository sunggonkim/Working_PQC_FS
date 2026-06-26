# Crash / recovery audit report

This report aggregates the retained crash and freshness evidence without claiming full app-level recovery.

## Retained evidence
- file replay negative control: 2 cut-point rows
- file replay final negative control: 2 cut-point rows
- post-keyfix replay summary: 2 cut-point rows
- E8 crash/replay regression: 8 summarized rows, max success_rate=1.00, min success_rate=1.00
- TPM unprovisioned path: exit=1
- hardware anchor round-trip: median 9.415056 ms, p95 28.107919 ms
- sqlite probe present: True
- sqlite fault campaign: 20 trials, unacceptable=0

## Current gap
- app-level recovery remains open because the retained campaign is SQLite-only
- syscall-exact crash timing and a second application workload remain open
