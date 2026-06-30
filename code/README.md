# AEGIS-Q Code Ownership

`code/` is the only source-owned tree for the runtime, benchmarks, probes, and
CTest registrations. Root-level files may configure project dependencies or
delegate into this tree, but production sources, code-local experiments, tests,
and build ownership must not move back to root compatibility paths.

This directory is allowed to evolve while gates are being closed. Do not freeze
the layout to protect old artifacts or old runner assumptions. When a better
module boundary, build ownership split, benchmark location, or runner entry
point is needed for the current gate, move the code and update the build/test
wiring in the same change.

Paper, artifact, and checklist updates are normal integrated work when they
close the same gate as the verified code path, prevent stale claims from
surviving in the repository, or finish an already-verified gate. What is banned
is using broad paper/artifact work as a substitute for improving the runtime,
build graph, benchmark entry points, or mounted-path behavior.

## Build Entry Points

- `code/CMakeLists.txt` is the code-owned build entry point.
- `code/build_targets.cmake` owns production target creation for `pqc_fuse_core`
  and `pqc_fuse`.
- `code/production_linkage.cmake` owns production include and link policy.
- `code/cuda_backends.cmake` and `code/cuda_feature_wiring.cmake` own optional
  CUDA backend targets and feature definitions.
- `code/experiment_targets.cmake` owns benchmark, probe, and runner target
  registration under `code/experiments`.
- `code/tests.cmake` owns CTest registration through the split files in
  `code/tests/`.
- `code/summary.cmake` owns configure-time status output through the split files
  in `code/summary/`.

`build/pqc_fuse` remains the standard production-facing binary path. The target
is still defined under `code/`, but `code/build_targets.cmake` sets the runtime
output directory to the root build directory so runners do not accidentally use
stale binaries.

## Domain Directories

- `frontend/`: FUSE adapter, process entry point, and command-line behavior.
- `fs/`: file descriptor lifecycle, file I/O, namespace, file locking, and
  parallel commit coordination.
- `storage/`: journal, epoch publication, checkpoint, state, durability,
  anchor, recovery, trust-boundary persistence, and writeback.
- `crypto/`: data crypto, flush crypto, key lifecycle, keyring, and crypto-plane
  trust-boundary helpers.
- `runtime/`: runtime configuration, admission/QoS, workers, telemetry, lock
  profiling, rekey scheduling, test-hook dispatch, and process-level cleanup.
- `support/`: shared support utilities such as trace sinks.
- `common/`: common format and block-job definitions.
- `gpu/`: CUDA backend sources and headers.
- `experiments/`: production-facing benchmark, probe, and verifier scripts.
- `tests/`: CTest ownership files.
- `summary/`: configure summary ownership files.
- `experiment_targets/`: grouped CMake target registrations for experiments.

## Change Rules

- Put new production code in the owning domain directory and update that
  domain's `sources.cmake` or `includes.cmake`.
- Put new benchmark/probe scripts under `code/experiments`; do not recreate a
  root `experiments/` tree.
- Put new CTest registration in the matching file under `code/tests/`.
- Keep root `CMakeLists.txt` as a dependency-discovery and delegation file.
- Do not preserve a bad source layout because an old artifact or script expects
  it. Update the runner to the code-owned path instead.
- Paper or artifact updates are allowed when they close the same active gate as
  the code change, finish a verified gate, or prevent stale claims, but they
  must not replace missing production-path work.
- Code-layout cleanup is allowed as normal gate work when it removes root
  compatibility paths, moves sources into the owning domain, splits CMake/test
  ownership, or makes mounted performance/correctness review easier.
- Keep generated files out of this tree. Python caches, benchmark outputs,
  temporary mount roots, local traces, and build outputs belong under ignored
  paths, `/tmp`, `build/`, or a deliberately retained `artifacts/` gate output.
- Treat `code/` cleanup as legitimate engineering work when it removes stale
  compatibility paths, clarifies module ownership, or makes the mounted
  performance path easier to build, test, or benchmark.
