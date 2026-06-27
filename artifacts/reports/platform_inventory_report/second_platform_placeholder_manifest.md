# Second-platform placeholder manifest

- status: no second-platform raw outputs retained
- platform: none
- commands:
  - `python3 experiments/run_verified_microbench.py --runs 3 --out artifacts/validation/microbench`
  - `python3 experiments/run_m5_admission_sweep.py`
  - `python3 experiments/run_app_recovery_bundle.py`
  - `PQC_SUDO_PASSWORD=<password> python3 experiments/run_combined_durability_bundle.py --out-dir artifacts/validation/combined_durability_bundle`
  - `PQC_SUDO_PASSWORD=<password> python3 experiments/run_sqlite_syscall_crash_tpm.py --out-dir artifacts/validation/sqlite_syscall_crash_tpm --when 1 2 3`
- required output schema: platform manifest, per-run medians and observed range, queue-depth / slack traces, SQLite selected-boundary verdicts, crash-audit bundle, combined SQLite/dbm.dumb replay verdicts, SQLite syscall-exact app-crash verdicts
- required platform fields: device model, kernel, CUDA compiler / runtime, driver / firmware identifiers, accelerator notes
