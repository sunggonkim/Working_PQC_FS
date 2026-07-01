# Evaluation RQ audit

- Overall pass: `True`
- AEGIS-Q page count at audit time: `13`
- Defensive Evaluation hits: `0`

## Research-question shape

| RQ | Required category | Found title | Terms present | Evidence present | Pass |
| --- | --- | --- | ---: | ---: | ---: |
| RQ1 | CPU/GPU/PQC placement | CPU/GPU/PQC placement | `True` | `True` | `True` |
| RQ2 | mounted app QoS | Mounted QoS and edge-storage behavior | `True` | `True` | `True` |
| RQ3 | correctness | Correctness of authenticated publication | `True` | `True` | `True` |
| RQ4 | replay and recovery | Replay and recovery oracle | `True` | `True` | `True` |
| RQ5 | cost boundaries and sensitivity | Cost boundaries and sensitivity | `True` | `True` | `True` |

## Discussion boundary retention

| Boundary needle | Present |
| --- | ---: |
| `fscrypt is environment-blocked` | `True` |
| `physical power-loss` | `True` |
| `kernel-crash` | `True` |
| `full cross-SoC portability` | `True` |
| `GPU side channels` | `True` |
