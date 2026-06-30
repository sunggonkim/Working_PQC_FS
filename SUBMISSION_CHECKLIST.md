# SUBMISSION_CHECKLIST

This is the only live checklist for AEGIS-Q.  It intentionally stays compact:
closed work is summarized only when it prevents a repeated review objection from
returning.

## Direction Lock

`EDGE_FILE_ENCRYPTION_CPU_GPU_UMA_RUNTIME`

AEGIS-Q is an edge secure file-encryption runtime for UMA SoCs, not a general
filesystem-conformance paper.

The thesis is:

- AES-GCM authenticated file publication stays CPU-first.
- ML-KEM/cuPQC is a key-plane/maintenance lane, admitted only for independent,
  batched, slack-tolerant work.
- CUDA managed memory is executor-local locality and ordering machinery, not a
  storage-DMA or NVMe-to-UVM correctness claim.
- The mounted runtime exposes storage-visible QoS, bounded recovery, and
  replay-after-advance checks for edge workloads.

Use the previous papers in `Paper/Previous paper` only as a logic template:
problem pressure, why static/naive approaches fail, system insight, concrete
mechanisms, and evaluation questions that close exactly those claims.  Do not
copy their prose or broaden AEGIS-Q into a full filesystem, full POSIX,
physical-crash, or portability paper.

## Current Submission State

The latest review is a weak-reject breadth/impact signal, not a contradiction
of the current claim.  Under the scoped edge-runtime paper:

- O1 is closed by claim removal: no foreground non-storage QoS motivation,
  workload, evaluation axis, or rendered-bibliography cue remains in the
  submitted paper.
- O2 is closed by claim removal on this host: `fscrypt` is environment-blocked
  with kernel/filesystem proof and is not reported as measured throughput or
  speedup.
- O3 is closed by hybrid-barrier evidence: strict single-client publication is
  the D/J/C cost boundary; epoch/group publication is the implemented path for
  concurrent or batched writes that can share barriers.
- O4 is closed as observation, not efficiency: retained tegrastats rows report
  same-run power/thermal context without claiming energy superiority.
- X13 is closed: direct measured baselines remain plaintext/lowerfs, gocryptfs,
  dm-crypt, and AEGIS-Q; GPU-storage/out-of-core/TEE/integrity/PQC-primitive
  systems are related-work boundaries unless the paper changes claim.

`review_response_strategy.py` should report:

```text
submission_defense_ready=True
osdi_strengthening_complete=True
p0_next=none
p1_next=none
```

This means no current P0 contradiction remains.  It does not mean guaranteed
SOSP/OSDI acceptance; breadth preferences can still appear.

## Latest Review Diagnosis

The recurring review accepts much of the technical story:

- CPU-first AES-GCM data path under UMA contention.
- Elastic GPU/PQC maintenance lane for batched ML-KEM envelope refresh.
- One durable authenticated format independent of executor choice.
- Explicit D/J/C publication ordering and remount/fault oracles.
- Storage-visible QoS improving SQLite p99 under secure-storage pressure.
- Honest claim boundaries.

The remaining complaints are venue-breadth pressure:

| Concern | Current answer | Do not overreact by |
| --- | --- | --- |
| Strict FUSE path is slow | Present strict as the cost boundary; use epoch/group publication only when admission finds shareable barriers. | Claiming general high-throughput filesystem performance. |
| POSIX envelope is narrow | Treat POSIX support as a mounted-workload guardrail. | Chasing shared `mmap`, full rename/link lifecycle, or full directory crash semantics as the main thesis. |
| `fscrypt` missing | Keep fscrypt unavailable with kernel/filesystem proof; do not imply measured speedup. | Deleting baselines or inventing unsupported rows. |
| No foreground non-storage QoS | O1 is closed by removing that axis from the claim and rendered bibliography. | Reintroducing a non-storage foreground workload without a real same-run row. |
| No physical power-loss | State the selected crash model: daemon cutpoints, D/J/C matrices, remount oracles, and lower-block interruption. | Calling it physical power-loss, kernel-crash, reboot, or drive-cache certification. |
| Single Jetson | Jetson-class UMA is the representative edge setting for this scoped claim. | Opening cross-platform work unless portability is claimed. |
| Modest GPU workflow speedup | Explain the X11 break-even: primitive speedup is large, mounted envelope/FUSE/admission overhead shifts workflow gain. | Claiming PQC accelerates ordinary file writes. |
| Energy/thermal | Report retained same-run observations only. | Claiming energy efficiency or power superiority. |
| Kernel integration path | Discuss only as future kernel assistance: D/J/C barrier coalescing or fscrypt-adjacent publication hooks that preserve the same format and recovery oracle. | Reframing the current paper as a kernel filesystem implementation. |
| Maintenance-lane extension | Merkle-prefix recomputation or integrity sealing may enter only if independent, batch-shaped, slack-tolerant, and format-compatible. | Claiming arbitrary GPU storage acceleration or broad integrity-service support. |

## Baseline And Competitor Strategy

Baseline deletion verdict: no.  Do not delete baselines to avoid criticism.
That reads as hiding the cost boundary and will invite a stronger review
objection.  The paper should instead make the comparison contract explicit:
what is measured directly, what is unavailable on this host, and what is a
related-work boundary because it solves a different systems problem.

Internet-checked verdict on 2026-07-01: the current baseline set is defensible
only for the scoped edge-runtime claim.  Existing systems split into three
families:

- Storage-encryption baselines: `fscrypt`, `dm-crypt`, and `gocryptfs`.
  These are the right direct comparison family because they encrypt storage
  under a mounted or deployable Linux boundary.
- GPU cryptographic-storage competitors: FAST'19 speculative GPU CFS and
  GPUstore.  These are the closest "why not GPU-encrypt the data path?"
  related-work competitors, so the paper must explicitly say that AEGIS-Q's
  contribution is not raw GPU encryption throughput; it is CPU-first AES-GCM
  publication plus slack-gated PQC maintenance and QoS under UMA pressure.
- GPU file/storage stacks: GPUfs, GeminiFS, GPU4FS, and GPUDirect Storage.
  These prove that GPU-side file/storage access is a real research area, but
  they require different APIs, kernel/driver paths, or direct storage-GPU data
  movement and do not provide AEGIS-Q's authenticated mounted format.

The current evidence is sufficient for the scoped claim only if the paper says
the following clearly: AEGIS-Q is not replacing fscrypt, dm-crypt, TEEs,
GPUDirect-style storage stacks, or GPU-resident filesystems.  It studies one
mounted edge file-encryption runtime where authenticated publication,
storage-visible QoS, scoped recovery/replay checks, and slack-gated PQC
maintenance are visible at the same boundary.  If the paper claims broader
filesystem maturity, cross-platform generality, physical durability, or kernel
baseline completeness, then the current matrix is not sufficient.

| Bucket | Systems | Paper treatment |
| --- | --- | --- |
| Direct measured baselines | Plaintext/lowerfs, gocryptfs, dm-crypt, AEGIS-Q | Keep the frozen contract rows.  They show publication/QoS cost, not throughput superiority. |
| Required unavailable baseline | fscrypt | Keep the kernel/filesystem proof and say no measured fscrypt speedup/throughput is claimed. |
| Related-work competitors | FAST'19 speculative GPU CFS, GPUstore, GPUfs, GeminiFS, GPU4FS, GPUDirect Storage, PPPQEFS/PQC cost models, FUSE performance work | Cite and separate by boundary: data-path GPU encryption, GPU-resident storage APIs, direct storage-GPU movement, PQC primitive cost, and FUSE overhead are not AEGIS-Q's mounted edge-runtime contract. |

Current baseline sufficiency decision:

- Enough for: "AEGIS-Q exposes storage-visible QoS and bounded publication at a
  mounted edge encryption boundary, while using the GPU only for elastic PQC
  maintenance under UMA pressure."
- Not enough for: "AEGIS-Q is faster than mature kernel encryption,"
  "AEGIS-Q is a general POSIX filesystem," "AEGIS-Q is a GPU filesystem," or
  "AEGIS-Q gives physical power-loss / rollback-resistance certification."
- Rebuttal posture: keep measured baselines visible, call fscrypt unsupported on
  this host, and move GPU-storage/PQC-only systems into related work with an
  explicit boundary sentence.  Do not remove baselines to make the table look
  cleaner.

The rebuttal line is: AEGIS-Q is compared directly against deployable
encryption boundaries where the local host can run a matched storage contract;
GPU-storage and PQC-only systems are not deleted but treated as boundary
comparisons because they do not provide this paper's combination of one
authenticated FUSE format, D/J/C publication, storage-visible QoS, external
replay-after-advance checks, and slack-gated ML-KEM maintenance.

### Competitor Boundary Audit

Use this when writing related work, evaluation caveats, or rebuttal text:

| Competitor family | What it proves | Why it is not a same-contract baseline |
| --- | --- | --- |
| fscrypt | Mature in-kernel filesystem encryption with per-file/directory policy. | This host lacks the required kernel/filesystem support; no measured fscrypt row or speedup is claimed. |
| dm-crypt | Mature block-device encryption and an important measured kernel boundary. | It does not see file-level publication, xattrs, QoS class, per-file envelope refresh, or replay-after-advance state. |
| gocryptfs / FUSE work | Closest user-space encryption and known FUSE cost boundary. | It does not provide AEGIS-Q's D/J/C authenticated publication, external replay-after-advance checks, or CPU/GPU/PQC admission policy. |
| FAST'19 speculative GPU CFS | GPU can help cryptographic file systems when masks/contexts are speculative and throughput-oriented. | It optimizes a different CFS data-encryption strategy; it does not evaluate AEGIS-Q's AES-GCM authenticated publication, PQC maintenance lane, UMA QoS gate, or recovery/replay contract. |
| GPUstore | GPU can accelerate selected kernel storage subsystems, including encryption, inside a different OS/kernel integration model. | It is a kernel framework for storage subsystems, not a FUSE mounted edge runtime with external replay checks and one CPU/GPU fallback-safe format. |
| GPUfs / GeminiFS / GPU4FS | GPU-side or kernel/GPU storage integration can expose storage services to GPU programs or storage subsystems. | Their target is GPU storage access/acceleration, not an authenticated mounted edge-encryption runtime with one fallback-safe persistent format. |
| FlashNeuron / Fastensor / SnuQS / GPUDirect Storage | Direct or staged storage movement can help large GPU workloads and out-of-core capacity. | They motivate UMA/storage pressure but do not provide file-encryption semantics, D/J/C recovery, replay-after-advance checks, or PQC key-plane policy. |
| PPPQEFS / PQC cost models | PQC primitive and encrypted-file-system cost modeling matters. | They are not mounted runtime measurements with FUSE publication, QoS, managed-memory executor locality, or crash/replay evidence. |
| OP-TEE / SGX-style secure storage | Stronger isolation/replay boundaries are possible under a different trust model. | AEGIS-Q deliberately does not claim enclave/TEE isolation or full rollback resistance. |

Paper rule: every competitor paragraph must end with the boundary that separates
it from AEGIS-Q.  Otherwise the reader will treat it as a missing baseline.

## Paper Spine

Keep every section aligned to the same argument.

| Section | Must emphasize | Must not become |
| --- | --- | --- |
| Abstract | UMA pressure, CPU-first data path, elastic GPU/PQC maintenance, SQLite p99 result, scoped non-claim boundary. | Feature ledger or FS/POSIX caveat dump. |
| Introduction | Secure file encryption as a placement/QoS/runtime problem. | Non-storage workload scheduling, full filesystem, or pure crash-consistency motivation. |
| Background | Existing encryption layers hide placement/QoS policy; UVM is executor-local, not storage DMA. | GPUDirect/UVM correctness claim. |
| Design | CPU-first AES-GCM publication, slack/telemetry-gated ML-KEM maintenance, QoS/replay/recovery boundary. | Implementation inventory before the insight. |
| Evaluation | RQ2/RQ3 close the thesis; correctness/crash rows support the boundary. | Broad artifact ledger or workload zoo. |
| Discussion | Explicitly label future expansions: measured fscrypt, physical power loss, portability, persistent PCR lifecycle, broad POSIX, kernel assistance. | Treating future rows as current blockers. |
| Related Work | Direct baselines vs boundary comparisons. | Deleting baselines or adding non-storage workload signals to dodge comparison criticism. |

## Closed Gate Register

| ID | Status | Closure target |
| --- | --- | --- |
| O1 | DONE | Submitted paper and rendered bibliography no longer present foreground non-storage QoS as motivation, workload, or evaluation. Reopen only with a real same-run foreground workload. |
| O2 | DONE | `artifacts/validation/kernel_baseline_feasibility/paper_fscrypt_verdict.json` passes; paper keeps fscrypt unavailable and forbids measured fscrypt speedup/throughput claims. |
| O3 | DONE | `artifacts/validation/strict_path_practicality/strict_path_practicality.json` passes; strict is the D/J/C cost boundary, epoch/group publication is the hybrid barrier path. |
| O4 | DONE | `artifacts/reports/o4_energy_thermal_result/o4_energy_thermal_result.json` records same-run tegrastats observations without efficiency claims. |
| X1C | DONE | Daemon cutpoints, D/J/C fault matrices, remount oracles, and lower-block interruption support previous/latest committed recovery within the selected model. |
| X2 | DONE | Bounded mounted POSIX guardrails cover the evaluated workloads; broad POSIX remains out of scope. |
| X3 | DONE | Cache-manifest object publish workload adds scoped mounted workload diversity without claiming broad workload coverage. |
| X6 | DONE | Marker/checkpoint durability narrowed from `syncfs` to marker-file `fsync`; no throughput dominance claim. |
| X8 | DONE | Paper spine foregrounds edge UMA placement/QoS over filesystem-conformance objections. |
| X9 | DONE | SQLite QoS Pareto framing: p99 improves by spending elastic background throughput; kernel controls and sensitivity are reported. |
| X10 | DONE | Generation robustness ties FileID/block/generation/length AAD, publish-ticket reservation, near-wrap refusal, and bounded writer stress to the claim. |
| X11 | DONE | ML-KEM/cuPQC break-even model explains why primitive speedup becomes modest mounted workflow speedup. |
| X12 | DONE | Related work separates direct encryption baselines from GPU-storage, TEE, integrity, logging, and QoS boundary comparisons. |
| X13 | DONE | Preserve measured baselines; keep fscrypt environment-blocked until a supported-host row exists. |
| X14 | DONE | Internet-checked competitor map separates storage-encryption baselines, GPU cryptographic-storage competitors, GPU file/storage stacks, GPUDirect-style movement, and PQC cost models. |

## Active Queue

No P0/P1 work is open under the current claim boundary.

Future work opens only if the paper deliberately expands its claim:

| ID | Status | Open only if |
| --- | --- | --- |
| F1 measured fscrypt | FUTURE | A supported kernel/filesystem environment is available and the paper wants a measured fscrypt row. |
| F2 physical power-loss | FUTURE | The paper claims physical durability certification, drive-cache behavior, reboot/kernel-crash behavior, or power-fail semantics. |
| F3 cross-platform UMA | FUTURE | The paper claims portability beyond Jetson-class representative edge UMA. |
| F4 persistent PCR lifecycle | FUTURE | The paper claims PCR-bound freshness, sealed mount-key release, or rollback resistance. |
| F5 foreground non-storage QoS | FUTURE | The paper reintroduces a non-storage foreground workload as motivation. |
| F6 broad POSIX | FUTURE | A thesis-aligned mounted workload requires shared `mmap`, full link lifecycle, broad rename semantics, or arbitrary byte-range transactions. |
| F7 kernel assistance | FUTURE | The paper claims a kernel-integrated publication path, fscrypt hook, or reduced-barrier deployment path. |

## Artifact Rule

Before creating any retained artifact, answer in one sentence:

```text
Which repeated reviewer objection becomes harder to repeat after this artifact exists?
```

If the answer is only "more evidence exists," do not create it.

Allowed now:

- Small paper edits that keep the edge-runtime spine crisp.
- Narrow script updates when a current guard is stale.
- A supported-host fscrypt run only if the environment actually supports it.

Not allowed now:

- More POSIX, strict-path, ML-KEM, or crash artifacts for already-closed rows.
- Broad verifier runs while a concrete paper-positioning issue is known.
- Artifacts whose only purpose is to make a checklist row look done.
- Real power-cut, reboot, kernel-crash, drive-cache, or cross-platform campaigns
  unless the paper explicitly expands to those claims.

## Required Verification

Run only the checks needed for the active change.  For paper/claim edits, use:

```bash
(cd Paper && \
  pdflatex -interaction=nonstopmode -halt-on-error main.tex && \
  bibtex main && \
  pdflatex -interaction=nonstopmode -halt-on-error main.tex && \
  pdflatex -interaction=nonstopmode -halt-on-error main.tex)
python3 code/experiments/build_review_response_strategy.py
python3 code/experiments/build_review_acceptance_structure_audit.py
python3 code/experiments/build_recurring_review_elimination_audit.py
python3 code/experiments/build_related_work_applicability_audit.py
python3 code/experiments/build_dangerous_claim_lint.py
python3 code/experiments/build_architecture_claim_firewall.py
python3 code/experiments/build_paper_spine_gate.py
python3 code/experiments/build_first_two_pages_thesis_audit.py
git diff --check --
```

For LaTeX submission readiness, run the full sequence:

```bash
(cd Paper && \
  pdflatex -interaction=nonstopmode -halt-on-error main.tex && \
  bibtex main && \
  pdflatex -interaction=nonstopmode -halt-on-error main.tex && \
  pdflatex -interaction=nonstopmode -halt-on-error main.tex)
pdfinfo Paper/main.pdf | rg '^Pages:'
rg -n "undefined|Citation .*undefined|Reference .*undefined|There were undefined|Rerun to get cross-references" Paper/main.log
```

Expected state: 12 pages and no undefined citation/reference hits.

## Goal Prompt

Use this prompt for the next Codex goal:

```text
Use /home/thor/skim/pqc_encrpyted_fs/SUBMISSION_CHECKLIST.md as the only live
checklist. Preserve the direction lock:
EDGE_FILE_ENCRYPTION_CPU_GPU_UMA_RUNTIME.

Improve AEGIS-Q as an edge secure file-encryption runtime, not as a
general-purpose filesystem-conformance project. Keep the paper centered on
dynamic CPU/GPU placement under UMA contention: CPU-first AES-GCM authenticated
file publication, slack-gated ML-KEM/cuPQC key-plane maintenance,
executor-local managed-memory locality, storage-visible QoS, and scoped
recovery/replay checks.

Do not reintroduce foreground non-storage QoS, broad POSIX, physical
power-loss, cross-platform portability, persistent PCR-bound freshness, or
fscrypt speedup claims unless a real matching experiment and paper claim
expansion are explicitly requested. Do not create new artifacts unless a named
repeated review objection becomes harder to repeat because of that artifact.

For paper edits, verify with the review-response strategy, acceptance structure
audit, recurring-review elimination audit, related-work applicability audit,
dangerous-claim lint, architecture claim firewall, paper spine gate,
first-two-pages thesis audit, LaTeX build, and git diff check.
```

## Reference Basis

- NVIDIA CUDA for Tegra: https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/
- NVIDIA GPUDirect Storage: https://docs.nvidia.com/gpudirect-storage/
- NVIDIA Jetson power/thermal tools: https://docs.nvidia.com/jetson/archives/r38.2/DeveloperGuide/SD/PlatformPowerAndPerformance.html
- NIST FIPS 203 ML-KEM: https://csrc.nist.gov/pubs/fips/203/final
- Linux kernel `fscrypt`: https://docs.kernel.org/filesystems/fscrypt.html
- Linux kernel `dm-crypt`: https://docs.kernel.org/admin-guide/device-mapper/dm-crypt.html
- gocryptfs: https://github.com/rfjakob/gocryptfs
- To FUSE or Not to FUSE FAST'17: https://www.usenix.org/conference/fast17/technical-sessions/presentation/vangoor
- Speculative Encryption on GPU Applied to Cryptographic File Systems FAST'19: https://www.usenix.org/conference/fast19/presentation/eduardo
- GPUfs ASPLOS'13: https://dl.acm.org/doi/10.1145/2499368.2451169
- GeminiFS FAST'25: https://www.usenix.org/conference/fast25/presentation/qiu
- GPU4FS: https://os.itec.kit.edu/21_3957.php
- FlashNeuron FAST'21: https://www.usenix.org/conference/fast21/presentation/choi
- Fastensor: https://dl.acm.org/doi/10.1145/3582016.3582018
- SnuQS: https://dl.acm.org/doi/10.1145/3721146
- Predicting performance for post-quantum encrypted-file systems: https://cr.yp.to/papers/pppqefs-20240327.pdf
