# SUBMISSION_CHECKLIST

This is the only live checklist for AEGIS-Q.  Keep it short: closed work is
summarized only when it prevents a repeated review objection from returning.

## Direction Lock

`EDGE_FILE_ENCRYPTION_CPU_GPU_UMA_RUNTIME`

AEGIS-Q is an edge secure file-encryption runtime for Jetson-class UMA SoCs,
not a general filesystem-conformance, power-loss-certification, or portability
paper.

Core thesis:

- Foreground file publication is CPU-first AES-GCM plus D/J/C ordering.
- GPU/cuPQC is an elastic key-plane maintenance lane for independent, batched,
  slack-tolerant ML-KEM work.
- CUDA managed memory is executor-local locality/order machinery, not storage
  DMA, GPUDirect, or NVMe-to-UVM correctness.
- The mounted runtime combines authenticated publication, storage-visible QoS,
  scoped recovery/replay checks, and fail-closed placement decisions.

Use `Paper/Previous paper` only as a structure template: pressure, failed naive
approaches, core insight, concrete mechanisms, and evaluation questions that
close exactly those claims.

## Paper Length And Design Shape

Target shape: 13 pages of argument-bearing main text.  `Paper/main.pdf` is
currently 13 pages.  Do not compress the paper back into a short prototype
report unless a venue page limit forces it.  The extra space must buy argument
quality, not artifact inventory.

The 13-page budget is intentional.  AEGIS-Q needs room to argue why the runtime
exists before it explains how it is implemented:

- Background/Motivation must show the pressure: Jetson-class UMA contention,
  4 KiB page/record granularity, CPU AES-GCM versus GPU AES-GCM through 1 MiB,
  and cuPQC ML-KEM as the separate elastic win.
- Design must start from the overall architecture/procedure figure, then name
  the components in that figure and explain their contracts.
- Evaluation must close the same claims: dynamic placement, QoS Pareto behavior,
  and scoped recovery/security boundaries.

Do not spend the extra pages on flag descriptions, retained artifact lists,
corner-case POSIX detail, or review-by-review patch history.  Those belong in
Implementation, Limitations, or this checklist when they are needed at all.

Previous-paper design pattern to preserve:

1. Motivation pressure first: show why static CPU-only, GPU-everything, and
   storage-agnostic acceleration fail on Jetson-class UMA edge workloads.
2. Overall architecture/procedure figure at the beginning of Design, before
   mechanism details.  This is mandatory: a reviewer should see the mounted
   path, GPU lane, QoS lane, and recovery boundary before any algorithm.
3. Name each figure component in a component-contract table: component name,
   owned boundary, why it matters to the edge-runtime claim, and which
   evaluation row closes it.  The table is the design outline.
4. Expand only the important named boxes into subsections.  Subsection titles
   must be component or algorithm names that a reader can find in the figure,
   not vague implementation buckets, flags, scripts, or review patches.
5. Each design subsection must answer three questions: what boundary it owns,
   why it is necessary for the UMA edge thesis, and which evaluation row closes
   it.  A subsection that cannot point back to the architecture figure and
   forward to an evaluation row belongs in Implementation, Limitations, or the
   checklist, not Design.

Design section page budget:

- First design page: overall architecture figure plus the top-down procedure
  narrative.  A reviewer should understand the mounted path, GPU lane, QoS
  lane, and recovery exposure boundary before seeing algorithms.
- Next half page: component-contract table.  Every row must be a named box from
  the figure, not an implementation file or environment variable.
- Remaining design pages: only the important named boxes become subsections.
  Each subsection should be readable as a mechanism: inputs, decision rule or
  invariant, failure/fallback behavior, and evaluation closure.

Current Design subsection contract:

| Subsection | Component/algorithm | Evaluation closure |
| --- | --- | --- |
| Overall procedure and component contracts | Full mounted runtime and claim firewall | Design/evaluation isomorphism gate |
| Foreground CPU data plane and D/J/C publisher | 4 KiB AES-GCM records, generation/AAD, D/J/C exposure | strict/epoch, remount, generation, lower-block matrices |
| Elastic admission controller and GPU lane | slack-gated ML-KEM/cuPQC maintenance with CPU fallback | AES-GCM negative control, ML-KEM break-even, mounted refresh |
| Storage-visible QoS controller | FUSE-visible elastic throttling by class and telemetry | SQLite p99/background Pareto result |
| Recovery oracle and external anchor | remount exposure, tamper/replay-after-advance boundary | tamper/remount/anchor negative-control rows |

Do not add subsections for incidental flags, benchmark scripts, one-off
artifacts, unsupported POSIX corner cases, or review-by-review defensive
patches.  If a new subsection cannot be named from Figure 4 and cannot be
closed by RQ1--RQ5, keep it in Implementation or Limitations instead.

Design expansion rule for the next paper-editing goal:

```text
Do not start with implementation details.  First explain the overall mounted
procedure from Figure 4, then use the component-contract table as the design
outline.  Expand only these mechanism names:
1. Foreground CPU data plane and D/J/C publisher
2. Elastic admission controller and GPU lane
3. Storage-visible QoS controller
4. Recovery oracle and external anchor
Everything else is support code, evaluation method, limitation, or future work.
```

## Current State

Paper:

- `Paper/main.pdf` is a 13-page architecture-first draft.
- Section 3 starts with the overall architecture figure and `Overall procedure
  and component contracts`, names the figure components in a
  component-contract table, then
  expands those names as mechanism subsections: foreground CPU data plane and
  D/J/C publisher, elastic admission controller and GPU lane, storage-visible
  QoS controller, and recovery oracle/external anchor.  The FUSE adapter is a
  boundary component, not an independent research mechanism.
- Background includes the 4 KiB--1 MiB AES-GCM CPU/GPU negative-control graph
  and explains the 4 KiB page/record boundary.
- Evaluation closes placement/QoS first; correctness/crash rows support the
  scoped boundary rather than becoming the main thesis.

Runtime default path:

- Ordinary mounts use CPU AES-GCM writeback and D/J/C publication.
- No default ML-KEM object, KEM keypair, background rekey worker, CUDA AES
  executor preallocation, elastic admission controller, scheduler-policy reload,
  QoS monitor startup, or external freshness anchor.
- These lanes activate only when rekey, QoS, telemetry, admission tracing, or
  explicit GPU diagnostic batches are configured.
- `crypto_plane_claim_guard` verifies both sides: ordinary mount keeps the
  foreground path clean; forced key-plane mount still starts admission/KEM/worker.
- The lower-filesystem contract is D/J sidecar `fdatasync` ordering plus
  checkpoint xattr visibility through the same lower filesystem.  Reordering
  after acknowledged sync, lost xattrs, device-cache behavior, kernel crash, and
  physical power loss remain outside the selected crash model.

Expected review-response state:

```text
submission_defense_ready=True
osdi_strengthening_complete=True
p0_next=none
p1_next=none
```

This means no known contradiction remains under the scoped edge-runtime claim.
It does not mean guaranteed SOSP/OSDI acceptance; breadth preferences can still
produce weak rejects.

## Latest Review Diagnosis

The repeated weak-reject review is mostly venue-breadth pressure, not a thesis
contradiction.

| Concern | Current answer | Do not do |
| --- | --- | --- |
| Strict FUSE path is slow | Treat as the main engineering target: reduce hot-path overhead while preserving D/J/C. | Hide the cost or weaken publication. |
| CPU/GPU placement overhead | CPU fast path for foreground AES-GCM; cheap admission only for GPU-eligible maintenance. | Remove GPU entirely or route every write through admission. |
| Modest GPU workflow gain | Explain X11: primitive cuPQC speedup is large, mounted envelope/FUSE/admission overhead bounds workflow gain. | Claim PQC accelerates ordinary writes. |
| Noisy/stale telemetry | Missing/stale slack, pressure, and deadline failures return CPU; sensitivity rows quantify slow sampling, high threshold, hysteresis, and two-writer fragility. | Present telemetry as an app scheduler or learning policy. |
| Block/coalescing sensitivity | 4 KiB records are the format; bounded coalescing changes amortization only. | Pretend tuning removes D/J/C barrier cost. |
| POSIX envelope is narrow | Treat POSIX as mounted-workload guardrails. | Chase shared mmap/full rename/link semantics as the main thesis. |
| No physical power-loss | Claim daemon cutpoints, D/J/C matrices, remount oracles, and lower-block interruption only. | Call it power-loss/kernel-crash/drive-cache certification. |
| Single Jetson | Jetson-class accelerated UMA is the representative edge target. | Add cross-platform work unless the claim changes. |
| fscrypt missing | Keep environment proof; no measured fscrypt speedup is claimed. | Delete baselines or invent rows. |
| Baselines | Keep plaintext/lowerfs, gocryptfs, dm-crypt, AEGIS-Q; treat GPU-storage/PQC-only systems as boundary related work. | Remove uncomfortable baselines. |

## Closed Gate Register

Keep these closed unless the paper deliberately expands its claim:

| ID | Closure |
| --- | --- |
| O1 | No foreground non-storage QoS motivation/workload/evaluation remains. |
| O2 | fscrypt is environment-blocked and not reported as measured throughput. |
| O3 | Strict path is the D/J/C cost boundary; epoch/group publication is conditional. |
| O4 | Power/thermal rows are observations, not energy-efficiency claims. |
| X1C | Selected daemon/D/J/C/remount/lower-block crash model is closed without physical power-loss claims. |
| X2 | Mounted POSIX guardrails cover evaluated workloads; broad POSIX is future work. |
| X8 | Paper spine foregrounds edge UMA placement/QoS over filesystem conformance. |
| X9 | SQLite QoS is a Pareto tradeoff: foreground p99 improves by spending elastic background throughput. |
| X10 | Generation robustness is tied to FileID/block/generation/length AAD, reservation, near-wrap refusal, and bounded writer stress. |
| X11 | ML-KEM/cuPQC break-even explains bounded mounted workflow gain. |
| X13 | Measured baselines stay visible; fscrypt remains unsupported until a real host row exists. |
| X14 | Related work separates storage encryption, GPU-storage, PQC cost models, TEEs, and FUSE overhead by boundary. |

## Baseline And Competitor Strategy

Baseline deletion verdict: no.  Do not delete baselines; hiding the cost boundary is worse than showing where AEGIS-Q loses.

Direct measured baselines: Plaintext/lowerfs, gocryptfs, dm-crypt, AEGIS-Q.
Required unavailable baseline: fscrypt; no measured fscrypt speedup/throughput is claimed.

Related-work competitors: FAST'19 speculative GPU CFS, fscrypt-GPU, GPUstore,
GPUfs, GeminiFS, GPU4FS, GPUDirect Storage, FlashNeuron, Fastensor, SnuQS,
SPDK/cuFile-style storage bypass.

Current baseline sufficiency decision:

- Enough for: scoped edge-runtime claim with authenticated publication,
  placement/QoS, and replay boundary.
- Not enough for: kernel replacement, broad filesystem deployment, direct
  storage/GPU path, or fscrypt superiority.
- Do not remove baselines; reviewers should see where AEGIS-Q loses as well as
  where it adds a different boundary.

## Active Queue

No P0/P1 work is open under the current claim boundary.  Remaining useful work
is P1 engineering on placement/strict-path overhead:

- Reduce strict/publication/writeback overhead without changing AES-GCM records,
  D/J/C ordering, checkpoint/recovery, or replay semantics.
- Preserve the CPU/GPU thesis: foreground AES-GCM stays CPU; GPU/cuPQC stays
  explicit, batched, slack-gated maintenance.
- Do not remove GPU; make admission cheaper and narrower.
- Prefer code changes that remove default-path config parsing, scheduler work,
  tracing, telemetry, allocation, or redundant sidecar work.
- Re-measure only when a code change can plausibly move strict-path or QoS
  numbers.  Do not create artifacts merely to make a checklist row look done.

Current default-path cleanup summary:

- Default FUSE callbacks bypass latency-trace wrappers unless
  `PQC_FUSE_TRACE_PATH` is set.
- Durability timing counters are off unless evidence runs enable them.
- Disabled QoS/admission/scheduler/key-plane/GPU machinery stays out of the
  ordinary mount.
- Open-path QoS xattr inheritance is loaded only when mounted write throttling
  is enabled; ordinary opens keep the default elastic class without the QoS
  sidecar lookup.
- Strict/default reads skip epoch fallback setup unless the fd can actually use
  epoch fallback.
- Default strict readable opens allocate only the journal read cache; the epoch
  lookup cache is allocated only when a committed epoch prefix or writable epoch
  log makes fallback possible.
- Ordinary release skips hidden-unlink sidecar cleanup path copying; marker
  cleanup is prepared only for `.fuse_hidden*` marker names.
- Writes snapshot rekey policy once and avoid rekey checks when the trigger is
  disabled.
- Foreground writes now defer rekey-policy snapshotting until after a successful
  encrypted-buffer write and skip the key-plane decision entirely for plaintext
  tier files.
- Ordinary encrypted writes first check the cached rekey trigger bit; when
  rekey is disabled they do not enter the force/interval policy snapshot path.
- Admission defaults keep the elastic GPU lane conservative: foreground 4 KiB
  AES-GCM records stay CPU-only, the default GPU byte gate is 128 KiB, and
  key-plane GPU use still requires envelope-count/slack/telemetry gates.
- Rekey defaults are aligned with that byte gate: the worker can now collect up
  to 128 envelopes with a 1 ms default elastic collect window, and it skips
  admission-context setup, load sampling, and admission work entirely until the
  candidate ML-KEM batch reaches the configured GPU byte threshold.  Setting
  `PQC_REKEY_BATCH_COLLECT_MS=0` disables that wait for latency-oriented
  diagnostics.
- Successful rekey batches no longer emit per-batch stderr/file logs by
  default; `PQC_REKEY_VERBOSE=1` is reserved for diagnostics, smoke tests, and
  retained key-plane workflow artifact runs that parse rekey events.
- Admission-enabled paths use atomic route counters/telemetry snapshots and a
  cached trace-enabled gate before taking the trace lock.
- Ordinary mounts skip data-plane scheduler byte accounting; the atomic
  scheduler counters are enabled only when admission/scheduler policy is active.
- Crash/fault cutpoints stay compiled in for selected-boundary evidence, but
  default mounts hit only an initialized atomic fast gate unless a cutpoint is
  configured.
- Plane-trace-disabled batch AES-GCM paths skip trace-only byte aggregation, so
  ordinary writeback does not rescan block descriptors just to emit disabled
  counters.
- Explicit GPU AES-GCM batch success keeps the tags produced by the batch crypto
  layer; the flush wrapper no longer recomputes per-block tags on CPU or retries
  CPU fallback outside the batch crypto layer.
- Anchor backend selection is cached; default no-anchor mounts must not re-read
  anchor configuration inside each checkpoint/publication turn.
- Review-acceptance structure gate now requires Jetson-style accelerated UMA to
  be framed as the scoped representative target, not cross-SoC portability.
- Writeback, journal, epoch-log, crypto, path-copy, and flush-batch code have
  been cleaned to avoid repeated full-buffer clearing, duplicate metadata work,
  and unnecessary context reconstruction on ordinary paths.
- Encrypted reads now take the visible-size/generation snapshot once per read
  before scratch acquisition; the old EOF precheck path no longer takes a
  second `commit_lock` on every normal read.
- Encrypted reads now check the visible-size EOF boundary before acquiring read
  scratch, so out-of-range probes do not contend on read scratch buffers.
- Default strict reads no longer copy the marker path; that 4 KiB path copy is
  done only when epoch fallback may need path-based recovery lookup.
- Successful writeback now reuses the logical size already established by the
  publish turn; the flush wrapper no longer reacquires `commit_lock` just to
  copy the same value back into the fd context.
- Writeback snapshots now copy DEK secret material only for authenticated
  encrypted flushes and copy the epoch-log path only when the fd actually has an
  epoch log; ordinary strict snapshots avoid those conditional-plane copies.
- QoS-enabled writeback uses a single throttle gate: the flush path checks the
  cached runtime throttle bit once, then calls the enabled-only throttle helper
  instead of rechecking the same atomic inside the helper.
- No-anchor mounts now skip freshness-window and windowed-file-anchor policy
  parsing; anchor worker startup returns before rereading anchor path when the
  cached backend is disabled.
- Runtime startup now skips freshness-anchor probe and worker start entirely
  when the cached anchor backend is disabled.

Remaining P1 target: publication/barrier cost and measured QoS Pareto behavior.
Do not reopen diagnostic-overhead cleanup unless a profile shows it is material.

Future claim expansions stay closed unless deliberately requested:

| ID | Open only if |
| --- | --- |
| F2 physical power-loss | The paper claims physical durability, reboot/kernel-crash, or drive-cache certification. |
| F3 cross-platform UMA | The paper claims portability beyond Jetson-class representative UMA. |
| F4 persistent PCR lifecycle | The paper claims sealed mount-key release, persistent PCR policy, or rollback resistance. |

## Paper Spine

| Section | Must emphasize | Must not become |
| --- | --- | --- |
| Abstract | UMA pressure, CPU-first data path, elastic GPU/PQC maintenance, SQLite p99 result, scoped non-claims. | Feature ledger. |
| Introduction | Secure file encryption as a placement/QoS/runtime problem. | Full filesystem or pure crash-consistency paper. |
| Background | Existing encryption layers hide placement/QoS; UVM is executor-local; size sweep motivates CPU-first AES-GCM. | GPUDirect/UVM correctness claim. |
| Design | Overall architecture first, component contracts second, then named mechanism subsections that mirror the figure. | Implementation inventory before insight. |
| Evaluation | Placement and QoS close the thesis; correctness rows bound the system. | Broad workload zoo. |
| Discussion | Future fscrypt/power-loss/PCR/POSIX/kernel work is clearly labeled. | Future work presented as current blocker. |
| Related Work | Direct baselines vs boundary comparisons. | Baseline deletion. |

## Artifact Rule

Before creating any retained artifact, answer:

```text
Which repeated reviewer objection becomes harder to repeat after this artifact exists?
```

If the answer is only "more evidence exists," do not create it.

Allowed now:

- Small paper edits that keep the edge-runtime spine crisp.
- Narrow guard/script updates when a current guard is stale.
- Code changes that reduce ordinary-path overhead without changing claims.

Not allowed now:

- More POSIX, strict-path, ML-KEM, or crash artifacts for already-closed rows.
- Real power-cut, reboot, kernel-crash, drive-cache, or cross-platform campaigns
  unless the paper explicitly expands to those claims.
- Compatibility flag churn for corner cases unrelated to the thesis.

## Verification

For code changes:

```bash
cmake --build build -j$(nproc)
./build/pqc_fuse --self-test
ctest --test-dir build --output-on-failure
python3 code/experiments/run_crypto_plane_trace_smoke.py
python3 code/experiments/build_crypto_plane_claim_guard.py
python3 code/experiments/run_a4_overhead_trace_smoke.py
git diff --check --
```

For paper/claim changes:

```bash
(cd Paper && \
  pdflatex -interaction=nonstopmode -halt-on-error main.tex && \
  bibtex main && \
  pdflatex -interaction=nonstopmode -halt-on-error main.tex && \
  pdflatex -interaction=nonstopmode -halt-on-error main.tex)
python3 code/experiments/build_paper_spine_gate.py
python3 code/experiments/build_recurring_review_elimination_audit.py
python3 code/experiments/build_review_acceptance_structure_audit.py
python3 code/experiments/build_design_eval_isomorphism_audit.py
python3 code/experiments/build_review_response_strategy.py
git diff --check --
```

Expected paper state: at most 13 pages and no undefined citation/reference hits.
The design gate must preserve the previous-paper structure: motivation pressure,
overall architecture/procedure, named mechanism components, and evaluation rows
that close the same mechanisms.

## Goal Prompt

```text
Use /home/thor/skim/pqc_encrpyted_fs/SUBMISSION_CHECKLIST.md as the only live
checklist. Preserve EDGE_FILE_ENCRYPTION_CPU_GPU_UMA_RUNTIME.

Improve AEGIS-Q as an edge secure file-encryption runtime, not a filesystem-
conformance project. Keep the paper centered on Jetson-class UMA placement:
CPU-first AES-GCM authenticated publication, slack-gated ML-KEM/cuPQC
maintenance, executor-local managed memory, storage-visible QoS, and scoped
recovery/replay checks.

Keep the paper at the current 13-page architecture-first shape unless a venue
limit requires cuts.  Use Paper/Previous paper as the structural model:
motivation pressure, overall architecture/procedure figure, short component
contract table, then mechanism subsections named after the important boxes or
algorithms.  The first Design page should let a reviewer map the figure boxes to
the rest of the section without reading implementation details.  Do not let
Design become an implementation inventory.

Prefer code changes that reduce strict publication, writeback, QoS, admission,
or default-mount overhead while preserving D/J/C publication, AES-GCM record
format, CPU-first foreground writes, and slack-gated GPU maintenance. Reflect
code changes in the paper only when the implementation story changes materially.

Do not reintroduce foreground non-storage QoS, broad POSIX, physical power-loss,
cross-platform portability, persistent PCR-bound freshness, or fscrypt speedup
claims unless explicitly requested with matching evidence. Do not create new
artifacts unless they close a named repeated reviewer objection.
```
