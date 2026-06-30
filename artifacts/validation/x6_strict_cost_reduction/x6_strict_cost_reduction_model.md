# X6 Strict Cost-Reduction Model

- Overall pass: `true`
- Marker metadata uses fsync: `true`
- Marker syncfs helper removed: `true`
- Retained A2 syncfs calls targeted: `1070`
- Modeled filesystem-wide syncfs calls before/after: `1070` / `0`
- Modeled blocking sync primitives before/after: `3210` / `3210`

Scope: production strict-path marker/checkpoint sync narrowing plus retained-count model; no throughput, kernel-upstreaming, or power-loss claim.
