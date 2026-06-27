# Paper spine gate

- Overall pass: `True`
- Page count: `12`
- Figure source: `artifacts/validation/qos_sqlite_hero_bundle/qos_sqlite_hero_bundle.json`

## First-page figure data

| Mode | p99 ms | Deadline misses | Background MB/s |
| --- | ---: | ---: | ---: |
| App | 6.436 | 0 | 0.000 |
| Pressure | 13.822 | 1 | 6.793 |
| Simple | 8.185 | 0 | 2.268 |
| AEGIS-Q | 8.753 | 0 | 2.736 |

## Gate checks

- First-page Figure 1 present: `True`
- First-page Figure 1 values present: `True`
- First-page Table 1 present: `True`
- Required Table 1 terms present: `True`
- Positive-before-defensive ordering: `True`

## Contribution map

| Contribution | Intro map | Design | Evaluation | Mapped |
| --- | ---: | ---: | ---: | ---: |
| C1 Secure-storage format and placement boundary | `True` | `True` | `True` | `True` |
| C2 Scoped placement and workflow evidence | `True` | `True` | `True` | `True` |
| C3 Recovery/freshness evidence | `True` | `True` | `True` | `True` |
| C4 Telemetry-to-storage QoS evidence | `True` | `True` | `True` | `True` |
