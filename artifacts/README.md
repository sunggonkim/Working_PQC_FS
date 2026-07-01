# Artifact layout

This directory keeps retained evidence, not source code.  New experiments should
write into the category below instead of adding loose files at `artifacts/`.

## Top-level categories

- `results/`: derived experiment outputs that are still close to the raw run.
  Current subcategories are:
  - `results/baselines/`: plaintext, dm-crypt, gocryptfs, AEGIS-Q, and other
    matched-baseline outputs.  Large loopback images are intentionally
    excluded; scripts must recreate them.
  - `results/freshness/`: anchor-refresh and freshness-window measurements.
  - `results/microbench/`: standalone microbenchmark outputs such as
    zero-context latency breakdowns.
  - `results/motivation/`: historical motivation plots/data retained for
    context.
  - `results/placement/`: primitive placement and path-breakdown result files.
  - `results/qos/`: controller, QoS, admission, TensorRT interference, and
    foreground-pressure result files.
  - `results/recovery/`: crash/replay/freshness result files and low-level
    SQLite syscall traces.
- `models/`: ONNX and TensorRT model/engine files used by workload harnesses.
- `reports/`: derived JSON/Markdown reports and dashboards.  These summarize
  retained evidence but are not raw proof by themselves.  `reports/build_logs/`
  stores stdout/stderr from report-regeneration runs.
- `validation/`: raw validation bundles from concrete runs.
- `probes/`: standalone diagnostic probes, especially UMA/storage-DMA
  diagnostics.  `probes/evidence/` retains older low-level logs for
  auditability, and `probes/sidechannel/` keeps deprecated side-channel probe
  outputs out of the main result hierarchy.
- `repro_bundle/`: generated manifest and hashes for the current repository
  evidence state.

## Rules

- Do not place new loose files directly under `artifacts/`.
- Do not store generated loopback images or mount directories.
- A report is not a claim by itself; paper claims must point to raw logs or
  result files and the script that derived the figure/table.
- If an artifact path moves, update the scripts and README in the same change.
