# X11 ML-KEM/cuPQC break-even closeout

Overall pass: `True`

## Verdict

X11 is closed: cuPQC is not oversold as ordinary-write acceleration; the retained data show a primitive GPU crossover at batch 16, a modest 1.186x mounted 1024-file workflow gain under slack, CPU-equivalent fallback under pressure, and a modeled mounted break-even near 760--812 files.

## Mounted workflow

- CPU median: `24.575` ms
- GPU-with-slack median: `20.729` ms
- Policy fallback median: `24.572` ms
- Speedup median: `1.186x`

## Modeled break-even

- Base range: `764`--`812` files
- +5 ms pressure range: `1.09`--`1.10`K files

## Checks

- `ml_kem_keygen_required_batches_present`: `True`
- `ml_kem_encaps_required_batches_present`: `True`
- `ml_kem_decaps_required_batches_present`: `True`
- `ml_kem_keygen_batch1_cpu_wins`: `True`
- `ml_kem_encaps_batch1_cpu_wins`: `True`
- `ml_kem_decaps_batch1_cpu_wins`: `True`
- `ml_kem_keygen_batch16_gpu_wins`: `True`
- `ml_kem_encaps_batch16_gpu_wins`: `True`
- `ml_kem_decaps_batch16_gpu_wins`: `True`
- `workflow_batch_1024`: `True`
- `methodology_runs_5`: `True`
- `workflow_speedup_positive_but_modest`: `True`
- `fallback_cpu_equivalent`: `True`
- `modes_all_acceptable`: `True`
- `base_break_even_below_1024`: `True`
- `plus5_pressure_break_even_above_1024`: `True`
- `paper_complete`: `True`
- `checklist_complete`: `True`

## Residual risk

This is a retained-data model on one Jetson-class UMA platform, not a cross-platform portability proof or a claim that ML-KEM accelerates bulk file writes.
