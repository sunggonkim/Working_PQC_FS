# SUBMISSION_CHECKLIST

## Priority 1: Mode-Aligned Baselines

- [ ] Resolve the fscrypt frozen-contract baseline gap.
- [ ] Produce or explicitly scope the dm-crypt frozen-contract baseline row.

## Priority 2: Related Work and Applicability

- [ ] Strengthen related work with mode-aligned comparisons.
- [ ] Add an applicability envelope section/table.
- [ ] Add GPU elastic-lane risk guidance without claiming side-channel defense.

## Priority 3: Traceability and Reproducibility

- [ ] Rebuild all paper figures from retained data.
- [ ] Create a claim-to-evidence manifest for the submission.
- [ ] Re-run the final artifact taxonomy audit before submission.
- [ ] Add a lightweight claim-lint gate.

## Priority 4: Final Paper Gates

- [ ] Reduce LaTeX layout warnings while keeping the paper exactly 12 pages.
- [ ] Run final build and test gates.
- [ ] Run final document gates.

## Best-Paper Stretch Goals

- [ ] Produce a portability result on a second platform or driver matrix.
- [ ] Add energy/thermal analysis for the hero experiment.
- [ ] Add a broader application workload set.

## Final Gate Before Any Resubmission

- [ ] `cmake --build build --parallel 2` completes cleanly.
- [ ] `ctest --test-dir build --output-on-failure` passes.
- [ ] LaTeX rebuild completes without dangling references.
- [ ] `pdfinfo Paper/main.pdf` reports exactly `Pages: 12`.
- [ ] Every main-paper numeric claim points to a retained artifact.
- [ ] Claim-to-evidence manifest covers every abstract/conclusion number and every security/recovery claim.
- [ ] Statistical/thermal metadata is retained for every reported performance result.
- [ ] Accepted-paper structure, paper-spine, hero-result, design-evaluation isomorphism, novelty-isolation, case-study, figure/table obligation, evaluation-completeness, and recurring-objection gates are all closed.
- [ ] No appendix or artifact ledger is included in the submitted paper body.
- [ ] The paper does not claim direct NVMe-to-UVM DMA, eBPF/io_uring completion bypass, persistent PCR-bound freshness, foreground AI QoS recovery, full crash certification, side-channel protection, or portability unless the corresponding checklist item is closed with real evidence.
