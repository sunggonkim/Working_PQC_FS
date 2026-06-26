# SQLite workload CI report

This report is derived from the checked-in SQLite raw samples.

- bootstrap trials: 10000
- statistic: median
- confidence level: 95%

## Workload
- full: WAL/WAL FULL median 0.406782 ms (95% CI 0.397671--0.434282), n=20, integrity=ok
- plain: WAL/WAL FULL median 0.399894 ms (95% CI 0.390467--0.441079), n=20, integrity=ok

## Contention
- full: WAL/WAL FULL median 0.595384 ms (95% CI 0.373065--0.817704), n=2, integrity=ok
- plain: WAL/WAL FULL median 0.563560 ms (95% CI 0.367351--0.759769), n=2, integrity=ok
