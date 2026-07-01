# Paper Structure Alignment (2026-07-01)

## Source corpus unpacked
- Paper/Previous paper/src_icdcs26 (from ICDCS26 zip)
- Paper/Previous paper/src_sigmetrics26 (from SIGMETRICS26 zip)

## Structural template extracted from prior papers
- Introduction opens with 1 figure + 1 table.
- Design is mechanism-first with named subsections and frequent local figures.
- Evaluation is explicitly staged:
  - Evaluation Setup
  - Performance section(s)
  - Time Analysis
  - Sensitivity Analysis
- Related Work and Conclusion are concise terminal sections.

## Current-paper changes applied for alignment
1. Evaluation headings were rewritten to match prior-paper flow:
   - Evaluation Setup
   - Performance with Baselines and Placement Boundary
   - Reliability and Recovery Boundary
   - Time Analysis and Sensitivity
2. Time-analysis evidence was made explicit by switching the diagnostic figure to latency breakdown.
3. Design now includes an executable-style publication mini-spec for D/J/C/C-marker state transitions.
4. Security now states minimal lower-filesystem properties (P1-P3) used by the selected crash model.

## Why this matters
- Reduces “disclaimer-first” reading friction by restoring a predictable systems-paper evaluation arc.
- Makes checkpoint/journal/marker terminology auditable with state semantics.
- Converts implicit lowerfs assumptions into explicit model requirements.

## Remaining delta vs prior-paper style
- Current manuscript still has stronger boundary disclaimers than prior HPC papers due to security-scope requirements.
- Evaluation breadth is intentionally narrower (single-platform edge runtime) than exascale multi-node scalability papers.

## Repro note
- Rebuild command used after edits:
  - cd Paper
  - pdflatex -interaction=nonstopmode -halt-on-error main.tex
  - bibtex main
  - pdflatex -interaction=nonstopmode -halt-on-error main.tex
  - pdflatex -interaction=nonstopmode -halt-on-error main.tex
