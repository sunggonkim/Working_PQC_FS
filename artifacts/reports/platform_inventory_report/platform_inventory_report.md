# Platform inventory report

This report records the current retained platform evidence.

- device model: NVIDIA Jetson AGX Thor Developer Kit
- kernel: Linux-6.8.12-tegra-aarch64-with-glibc2.39
- cuda: Build cuda_13.0.r13.0/compiler.36260728_0
- no second hardware / driver matrix is retained in the repository

## Second-platform contract
- minimum commands:
  - `python3 experiments/run_verified_microbench.py --runs 3 --out artifacts/validation/microbench`
  - `python3 experiments/run_m5_admission_sweep.py`
  - `python3 experiments/run_app_recovery_bundle.py`
  - `PQC_SUDO_PASSWORD=<password> python3 experiments/run_combined_durability_bundle.py --out-dir artifacts/validation/combined_durability_bundle`
  - `PQC_SUDO_PASSWORD=<password> python3 experiments/run_sqlite_syscall_crash_tpm.py --out-dir artifacts/validation/sqlite_syscall_crash_tpm --when 1 2 3`
- required output schema: platform manifest, per-run medians and observed range, queue-depth / slack traces, SQLite selected-boundary verdicts, crash-audit bundle, combined SQLite/dbm.dumb replay verdicts, and SQLite syscall-exact app-crash verdicts
- required platform fields: device model, kernel, CUDA compiler / runtime, driver / firmware identifiers, accelerator notes
- status: placeholder only; no second-platform raw outputs are retained

## Gap
- second hardware / driver matrix remains open
