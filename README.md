# AEGIS-Q

AEGIS-Q is a UMA-aware secure edge file-encryption runtime.  The current paper
direction is locked to:

```text
EDGE_FILE_ENCRYPTION_CPU_GPU_UMA_RUNTIME
```

The submission claim is narrow and deliberate:

- AES-GCM authenticated file publication stays on the CPU data path.
- ML-KEM/cuPQC is an elastic key-plane maintenance lane for independent,
  batched, slack-tolerant work.
- CUDA managed memory is executor-local locality and ordering machinery, not a
  storage-DMA or direct NVMe-to-UVM claim.
- The mounted runtime exposes storage-visible QoS, bounded recovery, and
  replay-after-advance checks for edge workloads.

`SUBMISSION_CHECKLIST.md` is the only live review-response checklist.  Older
reports, traces, and historical scripts are retained for provenance only; they
do not override the checklist or the paper.

## Current Paper Scope

The paper is an edge secure file-encryption runtime paper, not a general
filesystem-conformance paper.

Claimed:

- CPU-first AES-GCM file records with one durable authenticated format.
- Slack/telemetry-gated ML-KEM/cuPQC envelope-refresh maintenance.
- Storage-visible QoS under secure-storage pressure.
- SQLite p99 recovery under the mounted secure-storage workload.
- Scoped remount, tamper, generation, daemon cutpoint, lower-block interruption,
  and TPM replay-after-advance evidence.
- Mode-aligned measured rows for plaintext/lowerfs, gocryptfs, dm-crypt, and
  AEGIS-Q; fscrypt is explicitly unavailable on this host with
  kernel/filesystem proof.

Not claimed:

- Foreground non-storage QoS recovery.
- Full POSIX, shared `mmap`, full hard-link lifecycle, or arbitrary rename and
  directory-fsync crash semantics.
- Physical power-loss, kernel-crash, reboot, or drive-cache certification.
- No persistent PCR-bound freshness, sealed mount-key release, or rollback
  resistance claim.
- No direct NVMe-to-UVM DMA, GPUDirect/RDMA, eBPF/io_uring completion bypass, or
  kernel-bypass publication claim.
- No side-channel protection or GPU constant-time behavior claim.
- Portability beyond the tested Jetson-class CUDA/TPM stack.

## Review Position

The latest weak-reject review is treated as venue-breadth pressure, not as a
contradiction in the scoped claim.  The repeated issues map as follows:

| Review concern | Current response |
| --- | --- |
| Slow strict FUSE path | Strict mode is the D/J/C cost boundary; epoch/group publication is the hybrid barrier path for concurrent or batched writes that can share barriers. |
| Missing fscrypt | fscrypt is unavailable with kernel/filesystem proof and is not reported as measured throughput or speedup. |
| No foreground non-storage QoS | Removed from the paper claim; do not reintroduce it without a real same-run foreground workload. |
| No physical power-loss | The claim is the selected crash model: daemon cutpoints, D/J/C matrices, remount oracles, and lower-block interruption. |
| Narrow POSIX envelope | POSIX support is a mounted-workload guardrail, not the research contribution. |
| Single Jetson platform | Jetson-class UMA is the representative edge setting unless portability is explicitly claimed. |
| Modest GPU workflow speedup | X11 explains the break-even: primitive cuPQC speedup is large, but mounted envelope/FUSE/admission overhead shifts workflow gains. |

## Important Paths

- Paper source: `Paper/main.tex` and `Paper/*.tex`
- Built paper: `Paper/main.pdf`
- Live checklist: `SUBMISSION_CHECKLIST.md`
- Strategy gate: `code/experiments/build_review_response_strategy.py`
- Acceptance audit: `code/experiments/build_review_acceptance_structure_audit.py`
- Recurring-review audit: `code/experiments/build_recurring_review_elimination_audit.py`
- Related-work/applicability audit:
  `code/experiments/build_related_work_applicability_audit.py`
- Claim firewall: `code/experiments/build_architecture_claim_firewall.py`
- Dangerous-claim lint: `code/experiments/build_dangerous_claim_lint.py`

## Verification

For paper or claim-boundary edits, run:

```bash
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

For LaTeX readiness:

```bash
(cd Paper && \
  pdflatex -interaction=nonstopmode -halt-on-error main.tex && \
  bibtex main && \
  pdflatex -interaction=nonstopmode -halt-on-error main.tex && \
  pdflatex -interaction=nonstopmode -halt-on-error main.tex)
pdfinfo Paper/main.pdf | rg '^Pages:'
rg -n "undefined|Citation .*undefined|Reference .*undefined|There were undefined|Rerun to get cross-references" Paper/main.log
```

Expected paper state: 12 pages and no undefined citation/reference hits.

## Artifact Discipline

Do not create new retained artifacts unless a named repeated reviewer objection
becomes harder to repeat because of that artifact.

Allowed now:

- Small paper edits that keep the edge-runtime spine crisp.
- Narrow script updates when a guard is stale.
- A supported-host fscrypt run only if the environment actually supports it and
  the paper deliberately wants that measured row.

Not useful under the current claim:

- More POSIX, strict-path, ML-KEM, or crash artifacts for already-closed rows.
- New foreground non-storage QoS traces unless the paper deliberately reopens
  that claim.
- Real power-cut, reboot, kernel-crash, drive-cache, or cross-platform
  campaigns unless the paper explicitly expands to those claims.

## Current Gate Expectations

`build_review_response_strategy.py` should report:

```text
submission_defense_ready=True
osdi_strengthening_complete=True
p0_next=none
p1_next=none
```

This does not guarantee SOSP/OSDI acceptance.  It means the repeated review no
longer maps to an unsupported paper claim under the current scope.
