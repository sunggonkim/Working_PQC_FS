# SUBMISSION_CHECKLIST

This file is the single source of truth for the paper revival task.

Priority order:

1. Core correctness / memory / DMA / UVM evidence
2. eBPF / tracepoint / QoS evidence
3. Benchmark definitions and fairness
4. fscrypt / dm-crypt / TPM / crash evidence
5. Writing quality, related work, reproducibility, and artifact packaging

Rules:

- Mark an item `[x]` only when a concrete artifact, log, benchmark output, or code path directly supports it.
- If evidence is missing, keep the item unchecked and state the exact gap.
- Do not claim more in the paper than the repo can currently support.
- Keep `Paper/main.pdf` at exactly 12 pages.
- Prefer evidence-first wording over aspirational wording.

Operating correction:

- Do not treat a broad unchecked claim as blocked before trying the smaller evidence steps inside it.
- Before marking a device path unavailable, rerun the exact command with the intended privilege level, including `sudo -S` when root access is needed.
- Prefer strengthening existing code paths and scripts before adding new experimental concepts or directories.
- Split every remaining broad claim into subclaims that can be independently verified, implemented, scoped out, or left open.
- A parent claim remains unchecked until every required subclaim has direct retained evidence, but verified subclaims should still be checked.

Decision rubric:

- `verified`: concrete artifact or code path exists and the paper can state the claim narrowly.
- `needs evidence`: the claim is plausible but not yet backed by a retained artifact or rerunnable command.
- `scoped out`: the claim is intentionally removed from the paper until stronger evidence exists.

Current feasibility classification:

- Locally actionable on the current Thor box:
  - UMA/storage-DMA: continue probing the existing `io_uring_uvm` / Nsight path for migration, coherence, or documented non-exposure of the relevant counter.
  - QoS: add a PMU/CUPTI/Nsight-derived pressure adapter that feeds the existing `pqc_admission_update_telemetry()` / scheduler-smoke path, then retain the trace and report bundle.
  - TPM: rerun provisioning and policy probes under the intended sudo/TCTI setting before treating permission errors as blockers.
  - App recovery: define SQLite durable cut points from existing WAL/strace evidence, add a recovery oracle, and rerun fault injection over those cut points.
  - Durability campaign: the TPM-only, app-only, and combined single-command bundles already run successfully; what remains is a unified hardware-backed study that connects TPM freshness validation with app-level crash recovery on the same backing store.
- Requires evidence outside the current Thor-only environment:
  - Second-platform portability result. The local work can freeze the benchmark contract and placeholder manifest, but the checked claim requires real raw outputs from another hardware / driver matrix.
- Paper integration rule:
  - Every closed subclaim must update the manuscript or appendix with the exact narrow wording supported by the retained artifact.
  - Parent claims stay unchecked until all required subclaims have artifacts and the paper claim has been adjusted to match those artifacts.

## Acceptance-critical verified evidence

### Build, smoke, and document gates

- [x] `cmake --build build --parallel 2` passes.
- [x] `ctest --test-dir build --output-on-failure` passes 2/2 tests.
- [x] `FUSE` self-test, clean mount/remount, envelope tamper rejection, UVM smoke, and scheduler smoke were executed (`artifacts/validation/fuse_self_test.*`, `artifacts/validation/fuse_mount.stdout`, `artifacts/validation/fuse_mount.stderr`, `artifacts/validation/fuse_roundtrip.json`, `artifacts/validation/fuse_tamper_rejection.json`, `artifacts/validation/um_smoke.json`, `artifacts/validation/scheduler_smoke.stdout`, `artifacts/validation/scheduler_smoke.stderr`).
- [x] `pdflatex → bibtex → pdflatex ×2` builds `Paper/main.pdf` to exactly 12 pages.
- [x] `Paper/main.pdf` is still readable after the latest figure/text layout pass.

### Core correctness / memory / DMA / UVM evidence

- [x] `cudaMallocManaged` / prefetch / stream-sync managed-memory smoke is verified by `artifacts/validation/um_smoke.json` and `artifacts/motivation/um_counters.json`.
- [x] `cudaHostRegister` on CPU memory succeeds, and the auxiliary `io_uring` direct-read smoke completes with a pinned buffer (`artifacts/evidence/repro_malloc_register.out`, `artifacts/evidence/io_uring_uvm.out`).
- [x] A root-run raw-NVMe `O_DIRECT` read smoke completes with a pinned buffer, and the GPU reads the same storage-filled mapped buffer with a matching checksum (`artifacts/evidence/io_uring_uvm_nvme_sudo.out`, `artifacts/validation/uma_storage_dma_same_buffer/`, `artifacts/validation/uma_storage_dma_profile_combined_report/uma_storage_dma_profile_report.md`).
- [x] Verify the retained UMA / storage-DMA evidence chain, limited to same-buffer visibility, managed-buffer diagnostics, and counter non-exposure rather than full NVMe-to-UVM DMA semantics:
  - [x] Thor UVM support check: confirm the platform exposes the UVM/full-coherence capability expected by the CUDA for Tegra docs.
  - [x] storage-visible buffer proof: show the raw NVMe `O_DIRECT` path plus the same-buffer GPU checksum evidence on the same retained buffer, without yet claiming full NVMe-to-UVM DMA semantics.
  - [x] UVM/coherence counter non-exposure evidence: document that the platform/tooling exposes no corresponding transfer/page-fault/migration/coherence counter for the retained storage-visible buffer.
  - [x] Runtime pointer-attribute diagnostics on the same retained buffer: retain `cudaPointerGetAttributes` / `cudaHostGetFlags` output showing that the storage-visible probe is a registered host allocation with a mapped device alias, and that UVM range-location queries reject that buffer as a managed-memory target (`artifacts/validation/uma_storage_dma_profile_combined_report/uma_storage_dma_profile_report.md`).
  - [x] Managed storage-buffer diagnostic: retain a storage-filled managed buffer whose same-buffer CPU/GPU checksum matches and whose `cudaMemRangeGetAttribute` last-prefetch location flips device → host on the same buffer after explicit prefetches (`artifacts/validation/uma_storage_dma_profile_combined_report/uma_storage_dma_profile_report.md`).
  - Subclaim breakdown:
    - [x] Build and retain the pinning binary path (`build/repro_malloc_register`, `build/io_uring_uvm`).
    - [x] Verify `cudaHostRegister` and device-pointer mapping on CPU memory.
    - [x] Verify raw NVMe `O_DIRECT` read under root privileges.
    - [x] Profile the raw-read path with Nsight and retain `raw_probe.nsys-rep`, SQLite export, and stats CSVs.
    - [x] Strengthen the existing `pqc_fuse --um-smoke` path so it allocates managed memory, prefetches it, launches a GPU kernel that touches it, and validates host-visible data.
    - [x] Profile the strengthened managed-memory smoke path with Nsight and retain CUDA API / OS runtime summaries.
    - [x] Prove that the storage buffer itself is the GPU-visible buffer, rather than a separate raw-read buffer plus separate managed-memory smoke.
    - [x] Retain Nsight Compute memory-workload counters that directly correspond to the same storage-visible checksum kernel (`artifacts/validation/uma_storage_dma_profile_combined/ncu/`, `artifacts/validation/uma_storage_dma_profile_combined_report/uma_storage_dma_profile_report.md`).
    - [x] Retain UVM migration/coherence/suppression counter evidence for that same storage-visible buffer, or document that the platform/tooling exposes no such counter (`artifacts/validation/uma_counter_availability/` documents that the retained Nsight Systems reports expose no UM transfer/page-fault event rows for the raw probe or managed smoke, and `ncu --query-metrics` exposes no matching UVM/migration/coherence metric names).
    - [x] Audit paper/README language after the storage-buffer and counter probes; keep the claim at same-buffer pinned-memory visibility, managed-buffer diagnostics, and counter non-exposure, not full storage-DMA or migration suppression.
  - Current status: the Thor UVM/full-coherence capability is documented, the repository retains host-pinning plus raw-NVMe `O_DIRECT` smoke paths, and the current combined Nsight bundle in `artifacts/validation/uma_storage_dma_profile_combined/` now contains three complementary probes: (1) the raw `O_DIRECT` pinned-host same-buffer checksum path, (2) the existing `pqc_fuse --um-smoke` managed-memory path, and (3) a new root-run managed-storage probe where `io_uring_uvm --managed-buffer /dev/nvme0n1` performs a buffered `pread` into a `cudaMallocManaged` allocation, records `cudaPointerGetAttributes(type=managed)` on the storage-filled buffer, shows `last_prefetch_location_type` changing to `device` and then back to `host` on that same buffer, and matches the CPU/GPU checksum over it. The counter-availability audit in `artifacts/validation/uma_counter_availability/` records that Nsight Systems still exposes no UM transfer/page-fault rows for the raw probe, managed smoke, or managed-storage probe, and `ncu --query-metrics` still exposes no matching UVM/migration/coherence metric names.
  - Paper impact: the paper may now state the narrow managed-storage-buffer diagnostic in addition to the raw pinned-host same-buffer result, but must still not claim full NVMe-to-UVM DMA semantics or migration suppression.
  - [x] `experiments/run_uma_storage_dma_probe.py` packages the existing pinning / raw-read smoke commands and can optionally wrap an external profiler command.
  - [x] `experiments/run_uma_storage_dma_repeated.py` repeats the probe and keeps a conservative per-run bundle; this is repetition scaffolding, not DMA proof.
  - [x] `experiments/build_uma_storage_dma_report.py` packages the repeated probe outputs into a derived JSON/MD report; this is report packaging, not DMA proof.
  - [x] `experiments/profile_uma_storage_dma.py` can profile the probe path with `nsys` when available or fall back to a probe-only command record; this is profiler scaffolding, not DMA proof.
  - [x] `experiments/build_uma_storage_dma_profile_report.py` packages the profiler-backed raw-read and managed-smoke runs into a retained JSON/MD report; this is report packaging, not DMA proof.
  - [x] `artifacts/uma_storage_dma_nsys_report/uma_storage_dma_profile_report.{json,md}` retains the profiler-backed raw-read bundle and shows `cudaHostRegister` plus no recorded CUDA Unified Memory transfer rows, but it is still not a verified NVMe-to-UVM DMA proof.
  - [x] `artifacts/validation/uma_storage_dma_profile_root/` captures the rerun-able root+Nsight bundle and `artifacts/validation/uma_storage_dma_profile_root_report/` packages the retained Nsight summaries; it records `cudaHostRegister` plus no recorded CUDA Unified Memory transfer rows, but it is still not a verified NVMe-to-UVM DMA proof.
  - [x] Add a reproducible profiler invocation for the storage workload itself and emit one report bundle that links the workload command, profiler output, and the relevant storage/UVM counters.
  - Current status: the repo has a conservative probe wrapper, fresh reruns exist in `artifacts/uma_storage_dma_probe/uma_storage_dma_probe.json`, the rebuilt `artifacts/uma_storage_dma_probe/{repro_malloc_register.*,io_uring_uvm_nvme.*}` outputs, the sudo replay bundle in `artifacts/uma_storage_dma_probe_sudo/`, the same-buffer proof bundle in `artifacts/validation/uma_storage_dma_same_buffer/`, and the combined profiler-backed bundle in `artifacts/validation/uma_storage_dma_profile_combined/` plus the retained summary in `artifacts/validation/uma_storage_dma_profile_combined_report/`; the bundle now proves both the raw pinned-host same-buffer path and a separate storage-filled managed-buffer path, while still documenting counter non-exposure rather than migration suppression.
  - Paper impact: do not upgrade the claim beyond what the artifact bundle can reproduce.

### eBPF / tracepoint / QoS evidence

- [x] `bpftrace` tracepoint evidence and the raw `nvme_complete_rq` histogram artifact were captured on the device (`artifacts/evidence/trace_nvme_lat.out`).
- [x] `TensorRT` / `YOLO` co-run interference artifacts and p99 / throughput measurements are preserved (`artifacts/e3_hero_results.json`, `artifacts/e3_hero_summary.csv`, `artifacts/m3_qos_results.*`, and `artifacts/m6_tradeoff_yolov8_*`).
- [x] Deterministic admission-controller skeleton and JSONL scheduler trace logging are implemented in `pqc_admission.c/h` and `pqc_block_job.h`; this is a code-backed prototype, not a validated closed-loop controller.
- [x] `experiments/tegra_qos_daemon.py` and `experiments/run_qos_gpu.py` parse `tegrastats` and toggle a shared throttle flag in code; this is a telemetry/throttle prototype, not a validated closed-loop controller.
- [x] `artifacts/validation/admission_sweep.*` and `artifacts/m5_admission_sweep.*` retain a deterministic controller-unit-test sweep over AI slack values; route counts change with budget, but this is still not validated end-to-end QoS.
- [x] The prototype telemetry inputs, sampling policy, hysteresis state, controller response curve, and throttle decisions are defined and logged in `experiments/tegra_qos_daemon.py`, `experiments/run_qos_gpu.py`, `pqc_admission.c/h`, `artifacts/validation/scheduler_smoke.stderr`, `artifacts/validation/tegra_qos_daemon_trace.jsonl`, and `artifacts/validation/run_qos_gpu_trace.jsonl`; this is still not a validated closed loop.
  - Evidence: the traces record raw `tegrastats` lines, parsed telemetry, hysteresis enter/hold/exit fields, and throttle transitions on the device.
- [x] `artifacts/validation/telemetry_trace_report/` summarizes the retained tegrastats traces into throttle-transition and load/power statistics; this is a derived audit report, not PMU-backed QoS.
  - Evidence: the report records sample counts, throttle fractions, transitions, and load/power summaries from the traced device runs.
- [x] `artifacts/motivation/uvm_proxy_report/` summarizes the existing managed-memory smoke and emulated semantic-gap projection; this is proxy evidence, not hardware-counter proof.
  - Evidence: the report packages `artifacts/motivation/um_counters.json` and `artifacts/motivation/semantic_gap.json` into a compact audit artifact.
- [ ] Verify the QoS control loop with measured workload pressure: PMU-counter integration, hysteresis state, and repeated runs that show stable admission / throttle behavior under load.
  - Current status: `pqc_admission.c/h` already implements the software proxy path (`pqc_admission_record_uma_event`, `pqc_admission_update_telemetry`) and the prototype telemetry traces show throttle decisions plus measured latency / throughput under load. The Python telemetry prototypes now use hysteresis rather than one-shot threshold flips, and `artifacts/validation/qos_repeated_run/` retains per-run traces for a fixed heavy-GPU pressure point. A new Nsight-derived adapter bundle (`artifacts/validation/qos_measured_pressure_adapter/`) maps retained NCU metrics into `PQC_TELEMETRY_MEM_BANDWIDTH` and `PQC_TELEMETRY_TENSOR_CORE`, invokes the existing `pqc_admission_update_telemetry()` / `pqc_admit()` path through `pqc_fuse --admission-telemetry-smoke`, and retains JSONL traces with the measured telemetry fields and route decision. The live bridge in `artifacts/validation/qos_live_telemetry_admission/` samples `tegrastats` while GPU pressure is active and feeds each sample into the same admission path; it records 6/6 CPU fallback decisions under high GPU-power-derived pressure. A newer mounted-workload bridge in `artifacts/validation/qos_fuse_live_bridge/` writes through a real FUSE mount while sampling live `tegrastats`; each sample is fed into `pqc_fuse --admission-telemetry-smoke`, and the same sample drives a harness-level hysteresis throttle that pauses the mounted writer in the same execution.
  - Missing evidence: a validated end-to-end workload controller path where measured PMU/CUPTI/Nsight inputs are sampled during the workload and directly throttle/admit real FUSE work from inside the controller/runtime rather than through a harness-level `tegrastats` bridge.
  - Paper impact: keep QoS language prototype-only unless the controller includes the missing PMU/CUPTI inputs and an end-to-end validated workload path.
  - To close: one retained end-to-end run bundle with PMU/CUPTI/Nsight inputs sampled during the workload and feeding the controller, plus the already-retained hysteresis transitions and repeated traces at the same pressure point.
  - Subclaim breakdown:
    - [x] Retain prototype tegrastats traces with throttle decisions.
    - [x] Retain deterministic admission sweeps over AI slack.
    - [x] Retain queue-depth sensitivity sweeps.
    - [x] Retain repeated-run scaffolding for the telemetry prototype.
    - [x] Add hysteresis state to the existing admission / telemetry path instead of reporting one-shot threshold flips only.
    - [x] Record hysteresis enter/hold/exit transitions in JSONL traces.
    - [x] Add a PMU/CUPTI or Nsight-derived measurement adapter that feeds real measured pressure into the existing controller path (`experiments/run_qos_measured_pressure_adapter.py`, `pqc_fuse --admission-telemetry-smoke`, `artifacts/validation/qos_measured_pressure_adapter/`).
    - [x] Add a live telemetry-to-admission bridge that samples `tegrastats` during GPU pressure and feeds each sample into the existing admission path (`experiments/run_qos_live_telemetry_admission.py`, `artifacts/validation/qos_live_telemetry_admission/`; path evidence only, GR3D percent was not exposed in the retained tegrastats format).
    - [x] Rerun at one fixed pressure point for multiple trials and retain per-run traces plus a stability summary (`artifacts/validation/qos_repeated_run/`, `artifacts/validation/qos_repeated_report/`).
    - [x] Keep paper language prototype-only until workload-sampled controller evidence exists; the current revision remains prototype-only because the end-to-end workload path is still missing.
    - [x] Add a same-run mounted-FUSE bridge where live `tegrastats` samples both reach the admission path and drive a harness-level throttle on real FUSE writes (`experiments/run_qos_fuse_live_bridge.py`, `artifacts/validation/qos_fuse_live_bridge/`).
    - [ ] Sample PMU/CUPTI/Nsight inputs during the same workload execution that is used to drive the controller.
    - [ ] Feed those PMU/CUPTI/Nsight inputs into a retained run where real FUSE work is admitted or throttled in the same execution without relying on a harness-level `tegrastats` throttle.
  - [x] The repository already has a split between the telemetry loop (`experiments/tegra_qos_daemon.py`, `experiments/run_qos_gpu.py`) and the profiling pass (`experiments/profile_e3_nsys.py`, `experiments/run_motivation_profiling.py`); this is a code path split, not a validated PMU-backed loop.
  - [x] `experiments/run_qos_repeated.py` repeats the telemetry prototype and retains a conservative bundle of per-run traces and summaries; this is repetition scaffolding, not PMU-backed stability proof.
  - [x] `experiments/build_qos_repeated_report.py` packages the repeated-run outputs into a derived JSON/MD report; this is report packaging, not a PMU-backed stability claim.

### Benchmark definitions and fairness

- [x] Benchmark metric definitions are fixed in the scripts and artifact schema, and `LLM_TTFT_ms` is documented as a derived proxy in `README.md`.
- [x] Latest three runs were used to regenerate the microbenchmark figure.
- [x] `artifacts/e1_results.json` retains the primitive latency baseline data that underlies the placement narrative.
- [x] `artifacts/motivation/baseline_write_latency.png`, `artifacts/motivation/contention_latency.*`, `artifacts/motivation/inference_latency.*`, and `artifacts/motivation/spillover_latency.*` retain the motivation-figure traces behind the README's baseline/contention/inference/spill-over narrative.
- [x] `artifacts/e3_hero_results.json` and `artifacts/e3_hero_summary.csv` retain the raw hero runs behind the interference figure.
- [x] `artifacts/m3_qos_results.json` and `artifacts/m3_qos_results.csv` retain the raw QoS comparison rows used for the paper narrative.
- [x] `artifacts/m5_fastlane_stress.json` and `artifacts/m5_fastlane_stress.csv` retain a synthetic concurrent-pressure fast-lane tail probe; it is a pressure benchmark, not an end-to-end AI workload.
- [x] `artifacts/e6_breakdown.json` retains the latency breakdown that separates kernel, staging, journal, and anchor cost on the placement path.
- [x] `artifacts/m6_tradeoff_yolov8_elasticcontend/` and `artifacts/m6_tradeoff_yolov8_adaptive_tight/` retain the TensorRT interference traces that define the adaptive vs. elastic comparison.

### fscrypt / dm-crypt / TPM / crash evidence

- [x] `fscrypt` / `dm-crypt` sequential-write baselines are preserved with matched `bs`, `size`, `numjobs`, and `iodepth`.
- [x] Clean remount and envelope-tamper checks are retained in `artifacts/validation/fuse_roundtrip.json` and `artifacts/validation/fuse_tamper_rejection.json`; they validate the repaired remount path and the authenticated envelope rejection path, but they are still regression checks rather than a full crash campaign.
- [x] File-backed replay negative control, unprovisioned TPM fail-closed (`artifacts/validation/tpm_unprovisioned.json`, `.stdout`, `.stderr`), file-anchor latency (`artifacts/motivation/anchor_latency.*`), and anchor round-trip artifacts (`artifacts/anchor_refresh/hardware_anchor_latency.*`) are retained; the replay family also includes the final `rollback_visible` matrix, the post-key-fix `rollback_reject` matrix, and the deterministic E8 crash/replay regression (`artifacts/crash_replay_e8_test_matrix.*`, `artifacts/crash_replay_e8_test_summary.*`, `artifacts/e8_crash_replay.json`).
- [x] `pqc_anchor.c` and `experiments/run_crash_replay_e8.py` fail closed on unprovisioned / unavailable TPM hardware instead of synthesizing a result (`artifacts/validation/tpm_unprovisioned.json`, `artifacts/validation/tpm_unprovisioned.stdout`, `artifacts/validation/tpm_unprovisioned.stderr`, `experiments/run_crash_replay_e8.py`).
- [x] `artifacts/anchor_refresh/anchor_latency.json` records file-anchor latency and explicitly skips the hardware backend when the anchor self-test fails, rather than inventing a hardware latency number.
- [x] `artifacts/anchor_refresh/hardware_anchor_latency.json` records an actual hardware-backend round-trip latency bundle on a usable TCTI session (median 9.415 ms, p95 28.108 ms), not a synthetic placeholder.
- [x] `pqc_anchor.c` already routes the file anchor through file and hardware NV backends and fails closed on unprovisioned TPM hardware (`tpm2_nvreadpublic`, `tpm2_nvread`, `tpm2_nvwrite`).
- [x] The freshness-window tradeoff artifact under `artifacts/m4_freshness/` is documented as a derived model, not a hardware freshness proof.
- [x] Epoch-based file-key access control and grace-period rejection are implemented in `pqc_file_key.c/h`; this is a code-backed rotation mechanism, not TPM/PCR-backed freshness.
- [x] SQLite WAL commit and contention artifacts are retained in `artifacts/motivation/sqlite_latency.*` and `artifacts/motivation/sqlite_contention_latency.*`; `artifacts/sqlite_strace.log` shows the real journal/WAL/fdatasync path; they exercise WAL-first commit paths and `PRAGMA integrity_check=ok`, but they are still short of a full app-level crash study.
- [x] `artifacts/motivation/crash_replay_matrix.json` and `artifacts/motivation/crash_replay_summary.json` are retained and documented as fail-closed negative controls, not full crash certification.
- [x] Verify TPM provisioning, authorization state, transient PCR-policy rejection, and the hardware-backed freshness recovery flow.
  - Current status: epoch-based file-key access control exists in code; `pqc_anchor.c` routes the file anchor through file and hardware NV backends with `tpm2_nvreadpublic` fail-closed gating plus `tpm2_nvread`/`tpm2_nvwrite` calls; `artifacts/validation/tpm_provisioning_probe_sudo/` now records the live NV index `0x01500010` with `ownerwrite|ownerread` attributes and the corrected 88-byte size expected by the current anchor record; `artifacts/validation/tpm_pcr_policy_probe/` records a transient PCR-bound seal/unseal probe where current-PCR unseal succeeds and a drifted PCR digest is rejected; `experiments/run_tpm_monotonic_replay.py` now reliably detects mounts, writes/reads through the root-owned FUSE mount, and retains a replay-after-advance result in `artifacts/validation/tpm_monotonic_replay/`; the current retained run is `fail_closed`; and `artifacts/validation/tpm_recovery_verdict/` plus `artifacts/validation/tpm_freshness_report/` package the retained hardware-backed recovery evidence.
  - Code changes that made this closeable: `checkpoint_store()` now propagates hardware-anchor update failures instead of silently ignoring them, and `ctx_set()` now propagates checkpoint/anchor load failures instead of opening the file anyway.
  - Paper impact: the paper may now claim a retained hardware-backed replay-after-advance fail-closed result, but it should still avoid claiming that the persistent filesystem anchor itself is PCR-sealed because the retained PCR evidence is a transient policy probe.
  - Subclaim breakdown:
    - [x] Retain fail-closed behavior when TPM hardware or provisioning is unavailable.
    - [x] Retain hardware NV round-trip latency on a usable TCTI session.
    - [x] Record tpm2-tools availability and TCTI/device state with `experiments/check_tpm_provisioning.py`.
    - [x] Rerun TPM provisioning probe under the intended privilege/TCTI setting before treating permission errors as blockers (`artifacts/validation/tpm_provisioning_probe_sudo/`).
    - [x] Record the exact NV index, attributes, owner/authorization policy, and whether the index already exists (`0x01500010`, `ownerwrite|ownerread`, size 88 after reprovisioning to match the 88-byte anchor record, sha256 name/hash; this records owner authorization attributes, not a persistent PCR policy).
    - [x] Add a PCR policy probe that seals or binds a test value and records failure after PCR drift (`experiments/run_tpm_pcr_policy_probe.py`, `artifacts/validation/tpm_pcr_policy_probe/`; this is a transient object probe and does not provision the persistent filesystem NV anchor).
    - [x] Retain the replay-after-advance harness and the retained hardware-backed artifact (`artifacts/validation/tpm_monotonic_replay/`; current mode `fail_closed`).
    - [x] Retain a fail-closed restored-snapshot artifact on the hardware-backed replay path.
    - [x] Add a recovery verdict artifact that shows mount/recovery behavior with the hardware-backed freshness state (`experiments/build_tpm_recovery_verdict.py`, `artifacts/validation/tpm_recovery_verdict/`; conservative recovery verdict only, not monotonic freshness proof).
    - [x] Keep rollback-resistance language narrower than the retained evidence; the paper and README may claim replay-after-advance fail-closed behavior, but not persistent PCR sealing of the filesystem anchor.
  - [x] `experiments/run_tpm_freshness_bundle.py` groups the current anchor-latency, power-fail, monotonic-replay, and crash-replay harnesses into one conservative bundle; this is bundle scaffolding, not hardware-backed freshness proof.
  - [x] `experiments/build_tpm_freshness_report.py` packages the retained TPM freshness-related outputs into a derived JSON/MD report; this is report packaging, not freshness proof.
  - [x] `experiments/check_tpm_provisioning.py` records the configured TCTI/NV index and any available tpm2-tools responses as a provisioning-state probe; this is provisioning scaffolding, not freshness proof.
  - [x] `artifacts/validation/tpm_provisioning_probe/tpm_provisioning_probe.json` captures a fresh rerun of the provisioning probe, including tpm2-tools availability and the current TCTI/device permission failures.
  - [x] `artifacts/validation/tpm_provisioning_probe_sudo/tpm_provisioning_probe.{json,md}` captures the sudo/TCTI rerun and the current NV public attributes; this is provisioning-state evidence, not PCR sealing or hardware-backed freshness proof.
- [x] Verify app-level recovery and fault injection across durable boundaries on a real workload such as SQLite/RocksDB-class traces, scoped to the retained SQLite selected-boundary campaign.
  - Current status: remount, replay, negative controls, deterministic E8 crash/replay regression, SQLite WAL/commit/strace artifacts, the derived `artifacts/crash_audit_report/` evidence map, the refreshed `artifacts/validation/app_recovery_bundle/app_recovery_bundle.json` bundle, the SQLite durable-boundary/oracle definition in `artifacts/validation/sqlite_recovery_oracle/`, and the executable SQLite selected-boundary campaign in `artifacts/validation/sqlite_fault_campaign/` exist. The campaign records 20 SQLite-only trials over four selected durable-boundary states with zero unacceptable oracle verdicts.
  - Missing evidence: a broader second application workload and syscall-exact crash timing inside SQLite/FUSE rather than deterministic file-state mutation.
  - Paper impact: present the SQLite selected-boundary campaign as SQLite-only evidence, not as complete application-level certification.
  - To close: either add one second workload with the same workload/cut-point/replay/oracle schema, or explicitly scope the manuscript claim to SQLite-only selected-boundary recovery.
  - Subclaim breakdown:
    - [x] Retain clean remount and envelope-tamper regression checks.
    - [x] Retain SQLite WAL / commit / strace artifacts as partial application evidence.
    - [x] Retain deterministic E8 crash/replay regression artifacts.
    - [x] Retain crash-audit and app-recovery packaging bundles.
    - [x] Define the exact durable-boundary cut points for the current SQLite path using existing journal/WAL/fsync observations (`artifacts/validation/sqlite_recovery_oracle/` defines four observed SQLite cut points from `artifacts/sqlite_strace.log`).
    - [x] Add a SQLite recovery oracle that records post-replay `PRAGMA integrity_check`, expected row counts, and content digest (`artifacts/validation/sqlite_recovery_oracle/` defines the oracle contract and records retained sample digests; it does not execute per-cut replay).
    - [x] Rerun SQLite fault injection across the selected cut points and retain per-cut oracle verdicts (`artifacts/validation/sqlite_fault_campaign/` records 20 trials, previous/latest/fail-closed oracle classification, and zero silent-corruption verdicts; this is SQLite-only file-state mutation, not syscall-exact crash timing).
    - [x] Explicitly scope the manuscript claim down to SQLite-only selected-boundary recovery instead of promising a second workload family that is not retained locally.
    - [x] Update paper wording from oracle-definition-only to SQLite-only selected-boundary campaign evidence while keeping full app-level certification out of scope.

### Remaining evidence to collect, in priority order

| Priority | Remaining claim | Minimum closing artifact |
|---|---|---|
| 1 | QoS control loop under workload pressure | Adapter wiring exists; remaining close requires (a) PMU/CUPTI/Nsight inputs sampled during the workload and (b) those measured inputs driving real FUSE admission/throttle decisions in the same retained run bundle |
| 2 | Combined durability campaign | TPM freshness recovery is now retained as `fail_closed`; the remaining gap is a unified same-backing-store durability study that combines the closed TPM path with the app-level recovery campaign |
| 3 | SQLite-only selected-boundary recovery | Retained SQLite campaign is verified; broader second-workload coverage remains future work |
| 4 | Second-platform portability | Frozen benchmark contract can be closed locally; actual second-platform result requires raw outputs from another matrix |

### Next execution order

1. Keep the QoS claim prototype-only unless both workload-sampled inputs and same-run FUSE admission/throttle decisions are retained.
2. Keep the TPM/durability claim scoped to the retained fail-closed replay-after-advance result unless the app-level recovery study is unified onto the same backing store.
3. Keep the app-recovery story scoped to the retained SQLite campaign unless a second workload is actually added later.

### Writing quality, related work, reproducibility, and artifact packaging

- [x] `README.md` and FUSE source headers were corrected so old Orin / eBPF / io_uring / TPM / QoS wording is no longer presented as current evidence.
- [x] Paper structure now mirrors the evidence-first flow: introduction, background, related work, design, implementation, evaluation, discussion, conclusion, appendix.
- [x] Implementation details are consolidated in `Paper/7_Implementation_Details.tex`.
- [x] Artifact bundle and reproducibility entry points are documented in `README.md` and the appendix.
- [x] The intro and related-work sections now contain explicit roadmap / boundary language.
- [x] The appendix now includes a recovery-state oracle and interruption-point taxonomy for future fault-campaigns (`Paper/11_Appendix.tex`).
- [x] `artifacts/crash_audit_report/` conservatively aggregates the retained crash/recovery/freshness evidence and records the still-open app-level gap.
- [x] `artifacts/platform_inventory_report/` conservatively records the current Thor platform manifest and makes the missing second hardware / driver matrix explicit.
- [x] `artifacts/remaining_gap_map/` consolidates every still-open checklist item with the retained evidence and the exact missing proof obligation.
- [x] `experiments/build_evidence_dashboard.py` packages the retained evidence maps and the still-open gaps into a bookkeeping dashboard; this is status packaging, not proof.
- [x] `experiments/run_app_recovery_bundle.py` groups the retained crash audit and dashboard outputs into a conservative bundle; this is bundle scaffolding, not a recovery oracle.
- [x] `experiments/build_app_recovery_report.py` packages the app-recovery bundle into a derived JSON/MD report; this is report packaging, not a recovery oracle.
  - [x] `artifacts/validation/app_recovery_bundle/app_recovery_bundle.json` captures the refreshed bundle rerun; it confirms packaging and trace retention, not full application-recovery certification.
- [x] `experiments/build_status_register.py` records which packaging artifacts are present and which open items remain open; this is bookkeeping only.
- [x] `experiments/build_index_report.py` points at the retained package/report artifacts and records whether they exist; this is an index only.
- [x] `experiments/build_all_reports.py` refreshes the retained bookkeeping/report generators in one pass and stores their logs; this is orchestration only, not proof.

## Scoped out from claims

- Side-channel / constant-time claims stay out of scope unless physical evidence is produced.
  - Current status: no power, EM, or constant-time campaign is present.
  - Missing evidence: no device-side leakage study and no formal leakage bound.
  - Paper impact: do not upgrade this into a security claim.

- Development-only prototype scripts such as `experiments/bench_io_uring_ebpf.cu`, `experiments/bench_iouring_poll.c`, and `experiments/verify_gup_pinning.py` are not evidence artifacts.
  - Current status: these files now carry explicit prototype / simulation warnings.
  - Paper impact: do not cite them as measured results, even if they emit numeric placeholders.

- `experiments/trace_gup.sh` only emits a trace helper; it is not a measurement artifact.
  - Current status: it is a methodology reminder, not a captured trace.
  - Paper impact: do not cite the helper as evidence of pinned storage-DMA behavior.

## Best paper stretch goals

- These are not acceptance gates. They are the highest-leverage additions for turning the current revision into a stronger best-paper contender.

- [x] Add one fair external baseline on the same harness for at least one adjacent systems alternative that is actually supportable in this repository.
  - Evidence: `experiments/run_baselines.sh` recreates matched `fscrypt` and `dm-crypt` loop-backed runs with identical `bs`, `size`, `numjobs`, and `iodepth`, and the retained outputs under `artifacts/baselines/fscrypt_fio.json` and `artifacts/baselines/dm_crypt_fio.json` preserve the raw fio results.
  - Why it matters: it gives the revision a kernel-integrated baseline on the same reporting rules rather than only a self-comparison.

- [x] Add one more workload family or longer-horizon recovery campaign with exact execution settings and confidence intervals beyond the current SQLite report.
  - Evidence: `experiments/build_tensorrt_ci_report.py` derives bootstrap confidence intervals from the retained per-trial TensorRT/YOLO traces under `artifacts/motivation/tensorrt_interference.json`, and `artifacts/tensorrt_ci_report/tensorrt_ci_report.md` records the exact engine path, run duration, trial count, and median CI for each retained mode.
  - Why it matters: it broadens the evidence base beyond the SQLite report with a second workload family and a reproducible CI summary.

- [x] Add one end-to-end application workload with exact execution settings and confidence intervals, using the retained SQLite samples as the workload report base (`experiments/build_sqlite_ci_report.py`, `artifacts/sqlite_ci_report/sqlite_ci_report.json`, `artifacts/sqlite_ci_report/sqlite_ci_report.md`).
  - Evidence: the report preserves the exact journal/sync settings and bootstrap 95% median intervals for the retained SQLite workload and contention traces.
  - Why it matters: it turns the current prototype story into a more review-resistant application story.

- [x] Add a one-command reproduction bundle for the current paper sources and artifact set, including artifact manifests and hashes (`experiments/build_repro_bundle.py`, `artifacts/repro_bundle/manifest.json`, `artifacts/repro_bundle/sha256sums.txt`, `artifacts/repro_bundle/paper_pages.txt`).
  - Evidence: the bundle command hashes the current source/artifact inventory and rechecks that `Paper/main.pdf` remains 12 pages.
  - Why it matters: it lowers reviewer friction and makes the submission easier to trust and rerun.

- [x] Add a reviewer-facing claim-to-evidence table that maps every main-paper claim to the exact code path, artifact, or retained negative control supporting it.
  - Evidence: `Paper/11_Appendix.tex` now includes the `API-to-claim audit` table (`Table~\ref{tab:api_audit}`), which maps the major mechanism claims to the corresponding source-level behavior and explicitly labels what is and is not claimed.
  - Why it matters: this makes the submission easier to audit and reduces the chance that a strong claim survives without a traceable artifact.

- [x] Add a sensitivity sweep for the scheduler / admission thresholds across AI slack values so the robustness claim is backed by an explicit stability curve rather than a single working point.
  - Evidence: the repository retains the budget sweep (`artifacts/validation/admission_sweep.*`, `artifacts/m5_admission_sweep.*`) and the rendered sensitivity figure path (`Paper/Figures/fig_telemetry_sensitivity.pdf`), which show admission changes across AI slack values.
  - Why it matters: best-paper reviewers usually look for evidence that the policy is stable across a meaningful operating range, not just at the tuned point.

- [x] Add a queue-depth sensitivity sweep for the scheduler / admission controller so robustness is also evaluated under varying queue-pressure settings, not only varying AI slack.
  - Evidence: `experiments/run_m5_admission_sweep.py` now sweeps `PQC_SCHED_SMOKE_CPU_QUEUE_DEPTH` and `PQC_SCHED_SMOKE_GPU_QUEUE_DEPTH`, and the retained artifact set `artifacts/m5_admission_sweep_qdepth.json` / `.csv` varies queue depth while holding the rest of the controller input fixed.
  - Why it matters: queue-pressure sensitivity is the remaining dimension needed to justify a broader robustness curve.

- [x] Add a portability inventory for the current Thor-only stack so the manuscript can state exactly which JetPack / kernel / driver combination was measured.
  - Evidence: `artifacts/platform_inventory_report/` records the retained Thor platform manifest, including device model, kernel, CUDA stack, and the fact that all retained artifacts come from the same Thor platform.
  - Why it matters: portability claims need a concrete inventory before the paper can explain what was and was not measured.

- [ ] Add one more portability result beyond the current Thor-only stack, ideally a second JetPack / kernel / driver matrix or an equivalent hardware variant that can run the same benchmark harness.
  - Current status: the Thor inventory exists, but it is still a single-platform measurement set.
  - Missing evidence: retained outputs from a second hardware / driver matrix running the same benchmark harness.
  - Why it matters: portability evidence is one of the clearest ways to move from a good prototype paper to a convincing systems paper.
  - To close: one second-platform run bundle with the frozen command line, device path, input sizes, queue depths, output schema, and platform manifest.
  - Subclaim breakdown:
    - [x] Retain current Thor platform manifest.
    - [x] Retain current Thor artifact inventory.
    - [x] Freeze the minimum benchmark contract for a second platform using existing commands only (`artifacts/platform_inventory_report/platform_inventory_report.{json,md}` and `artifacts/platform_inventory_report/second_platform_placeholder_manifest.{json,md}`).
    - [x] Record the expected output schema and required platform fields for the second run (`artifacts/platform_inventory_report/platform_inventory_report.{json,md}` and `artifacts/platform_inventory_report/second_platform_placeholder_manifest.{json,md}`).
    - [x] Add a placeholder manifest that clearly says no second-platform raw outputs are present yet (`artifacts/platform_inventory_report/second_platform_placeholder_manifest.{json,md}`).
    - [ ] Copy in or generate raw outputs from a real second hardware / driver matrix.
    - [x] Update the paper to distinguish current Thor-only evidence from any future second-platform result.

- [ ] Add a hardware-backed durability campaign that combines TPM freshness validation with app-level crash recovery on SQLite/RocksDB-class traces.
  - Current status: the repository has TPM fail-closed and round-trip artifacts plus WAL / replay / negative-control evidence, and the three bundle entry points already run successfully as orchestration; however, that still is not a unified hardware-backed durability study.
  - Missing evidence: PCR sealing, provisioning, authorization, monotonic update semantics, recovery after interruption, exact workload traces, injected cut points, and an external recovered-state oracle.
  - Why it matters: this is the clearest way to turn the current durability story into a review-resistant systems result.
  - To close: three rerunnable bundles (TPM-only, app-only, combined) with retained artifacts and a single-command entry point for each.
  - Subclaim breakdown:
    - [x] Retain TPM fail-closed and hardware round-trip artifacts.
    - [x] Retain app-level partial evidence from SQLite WAL / strace and clean remount.
    - [x] Retain current TPM freshness and app-recovery packaging bundles.
    - [x] Define a TPM-only single-command bundle that packages the retained TPM freshness harnesses behind one command; this is orchestration only, not a proof of monotonic update or recovery verdict (`experiments/run_tpm_only_bundle.py`, `artifacts/validation/tpm_only_bundle/`).
    - [x] Define an app-only single-command bundle that exercises durable-boundary fault injection and an external recovery oracle (`experiments/run_app_recovery_bundle.py` now runs the crash audit, SQLite oracle definition, SQLite fault campaign, and dashboard packaging).
    - [x] Define a combined single-command bundle that packages the TPM-only and app-only evidence behind one command; this is orchestration only, not a single shared backing-store proof (`experiments/run_combined_durability_bundle.py`, `artifacts/validation/combined_durability_bundle/`).
    - [x] Retain artifacts for all three bundles.
    - [x] Confirm that the TPM-only, app-only, and combined bundle orchestrations all returncode 0 on the retained Thor runs; this is orchestration evidence, not a unified hardware-backed durability study.
    - [x] Keep the combined hardware-backed durability claim open until all three bundles pass; the retained bundles all returncode-0, but they still do not constitute a unified hardware-backed durability study.

## Evidence index

- [main.pdf](/home/thor/skim/pqc_encrpyted_fs/Paper/main.pdf)
- [tpm_provisioning_probe.json](/home/thor/skim/pqc_encrpyted_fs/artifacts/validation/tpm_provisioning_probe/tpm_provisioning_probe.json)
- [tpm_provisioning_probe_sudo.md](/home/thor/skim/pqc_encrpyted_fs/artifacts/validation/tpm_provisioning_probe_sudo/tpm_provisioning_probe.md)
- [app_recovery_bundle.json](/home/thor/skim/pqc_encrpyted_fs/artifacts/validation/app_recovery_bundle/app_recovery_bundle.json)
- [uma_storage_dma_probe.json](/home/thor/skim/pqc_encrpyted_fs/artifacts/uma_storage_dma_probe/uma_storage_dma_probe.json)
- [uma_storage_dma_probe_sudo.json](/home/thor/skim/pqc_encrpyted_fs/artifacts/uma_storage_dma_probe_sudo/uma_storage_dma_probe.json)
- [uma_storage_dma_profile_report.md](/home/thor/skim/pqc_encrpyted_fs/artifacts/uma_storage_dma_nsys_report/uma_storage_dma_profile_report.md)
- [uma_storage_dma_profile_root_report.md](/home/thor/skim/pqc_encrpyted_fs/artifacts/validation/uma_storage_dma_profile_root_report/uma_storage_dma_profile_report.md)
- [uma_storage_dma_profile_combined_report.md](/home/thor/skim/pqc_encrpyted_fs/artifacts/validation/uma_storage_dma_profile_combined_report/uma_storage_dma_profile_report.md)
- [uma_counter_availability.md](/home/thor/skim/pqc_encrpyted_fs/artifacts/validation/uma_counter_availability/uma_counter_availability.md)
- [um_smoke_nsys stats](</home/thor/skim/pqc_encrpyted_fs/artifacts/validation/uma_storage_dma_profile_combined/um_smoke/stats_cuda_api_sum.csv>)
- [repro_malloc_register.stdout.txt](/home/thor/skim/pqc_encrpyted_fs/artifacts/uma_storage_dma_probe/repro_malloc_register.stdout.txt)
- [io_uring_uvm_nvme.stdout.txt](/home/thor/skim/pqc_encrpyted_fs/artifacts/uma_storage_dma_probe/io_uring_uvm_nvme.stdout.txt)
- [run_fuse_tamper_rejection.py](/home/thor/skim/pqc_encrpyted_fs/experiments/run_fuse_tamper_rejection.py)
- [summary.json](/home/thor/skim/pqc_encrpyted_fs/artifacts/validation/microbench/summary.json)
- [fig_verified_microbench.pdf](/home/thor/skim/pqc_encrpyted_fs/Paper/Figures/fig_verified_microbench.pdf)
- [tegra_qos_daemon_trace.jsonl](/home/thor/skim/pqc_encrpyted_fs/artifacts/validation/tegra_qos_daemon_trace.jsonl)
- [tegra_qos_daemon_summary.json](/home/thor/skim/pqc_encrpyted_fs/artifacts/validation/tegra_qos_daemon_summary.json)
- [run_qos_gpu_trace.jsonl](/home/thor/skim/pqc_encrpyted_fs/artifacts/validation/run_qos_gpu_trace.jsonl)
- [run_qos_gpu_summary.json](/home/thor/skim/pqc_encrpyted_fs/artifacts/validation/run_qos_gpu_summary.json)
- [telemetry_trace_report.md](/home/thor/skim/pqc_encrpyted_fs/artifacts/validation/telemetry_trace_report/telemetry_trace_report.md)
- [qos_measured_pressure_adapter.md](/home/thor/skim/pqc_encrpyted_fs/artifacts/validation/qos_measured_pressure_adapter/qos_measured_pressure_adapter.md)
- [io_uring_uvm_nvme_sudo.out](/home/thor/skim/pqc_encrpyted_fs/artifacts/evidence/io_uring_uvm_nvme_sudo.out)
- [sqlite_ci_report.md](/home/thor/skim/pqc_encrpyted_fs/artifacts/sqlite_ci_report/sqlite_ci_report.md)
- [sqlite_recovery_oracle.md](/home/thor/skim/pqc_encrpyted_fs/artifacts/validation/sqlite_recovery_oracle/sqlite_recovery_oracle.md)
- [m3_qos_ci_report.md](/home/thor/skim/pqc_encrpyted_fs/artifacts/m3_qos_ci_report/m3_qos_ci_report.md)
- [tensorrt_ci_report.md](/home/thor/skim/pqc_encrpyted_fs/artifacts/tensorrt_ci_report/tensorrt_ci_report.md)
- [crash_audit_report.md](/home/thor/skim/pqc_encrpyted_fs/artifacts/crash_audit_report/crash_audit_report.md)
- [platform_inventory_report.md](/home/thor/skim/pqc_encrpyted_fs/artifacts/platform_inventory_report/platform_inventory_report.md)
