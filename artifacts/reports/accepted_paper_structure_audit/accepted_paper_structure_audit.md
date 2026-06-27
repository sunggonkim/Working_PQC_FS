# Accepted-paper structure audit

- Overall pass: `True`
- Required source examples found: `5` / `5`
- AEGIS-Q page count at audit time: `12`

## Required local sources

| Source | Pages | Early result/table signal | Comparison/capability signal |
| --- | ---: | --- | --- |
| ScaleQsim | 28 | Fig. 1. Simulation time of ScaleQsim (Proposed), SOTA (cusvaer, HyQuas, Atlas), and Qsim (Existing), executed | simulations [19, 23, 29], and quantum machine learning [12, 17, 45, 69, 76], beyond the capabilities |
| AURORA-Q | 11 | computation with data movement through an async execution Fig. 1: Scalability of AURORA-Q (Proposed), SOTA (ScaleQsim, | not detected |
| CITADEL | 11 | Throughput (ops/s) | throttling, a capability absent in non-learning or stateless against PID control and Lyapunov optimization. PID control |
| AS2 | 18 | has a time complexity of O(N 2 ) at its worst, but performs sort- Figure 1: Performance of sequential sorting algorithms and AS2 | Table 1: Categories and comparison with previous studies. |
| previous paper | 11 | Abstract—This paper presents EdgeQuantum, an asyn- Latency comparison | achieves up to a 242.7× storage reduction through compression, TABLE I: Comparison with prior work across key capabilities |

## AEGIS-Q spine

- First-page pressure result: `True` (SQLite p99 (ms))
- Capability table: `True` (Table 1: Capability comparison that defines AEGIS-)
- Top-level order matches audit target: `True`
- Deployment takeaway present: `True`

## Mechanism-to-evaluation map

| Mechanism | Design present | Evaluation present | Mapped |
| --- | ---: | ---: | ---: |
| first-page pressure/gap | `True` | `True` | `True` |
| authenticated format and D/J/C publication | `True` | `True` | `True` |
| CPU-first data lane and elastic key lane | `True` | `True` | `True` |
| external freshness boundary | `True` | `True` | `True` |
| telemetry-to-storage QoS | `True` | `True` | `True` |
