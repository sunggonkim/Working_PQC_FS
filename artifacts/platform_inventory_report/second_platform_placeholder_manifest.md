# Second-platform placeholder manifest

- status: no second-platform raw outputs retained
- platform: none
- commands:
  - `python3 experiments/run_verified_microbench.py --runs 3 --out artifacts/validation/microbench`
  - `python3 experiments/run_m5_admission_sweep.py`
  - `python3 experiments/run_app_recovery_bundle.py`
- required output schema: platform manifest, per-run medians and observed range, queue-depth / slack traces, SQLite selected-boundary verdicts, crash-audit bundle
- required platform fields: device model, kernel, CUDA compiler / runtime, driver / firmware identifiers, accelerator notes
