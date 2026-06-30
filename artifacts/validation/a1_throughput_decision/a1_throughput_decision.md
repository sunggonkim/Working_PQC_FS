# A1 Throughput Decision

- Overall pass: `True`
- Verdict: `cost-boundary-closeout`

## Metrics

- Frozen AEGIS-Q median throughput: `0.360 MiB/s`
- Frozen AEGIS-Q median p99: `11206.656 us`
- A2 short AEGIS-Q throughput: `1.858 MiB/s`
- A2 short gocryptfs throughput: `18.145 MiB/s`
- A2 AEGIS-Q/gocryptfs ratio: `0.102`

## Decision

The remaining strict path has data-sidecar durability before journal mapping durability, followed by marker/checkpoint publication. Replacing those barriers with one later sync would change crash-ordering obligations and requires a publication redesign/fault matrix, not a local optimization.

Use epoch/group commit only for batched or concurrent work where a shared barrier can amortize publication, or design a new strict-compatible compact publication format with fresh crash evidence.

## Proof Checks

- `frozen_aegisq_overall_pass`: `True`
- `a2_overall_pass`: `True`
- `strict_publication_boundaries_present`: `True`
- `paper_cost_boundary_present`: `True`
- `no_broad_high_throughput_claim`: `True`
