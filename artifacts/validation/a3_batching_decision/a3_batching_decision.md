# A3 Batching Decision

- Overall pass: `True`
- Verdict: `batching-boundary-closeout`

## Decision

- Helps: Concurrent grouped mounted writes can reduce traced publication sync count and can improve throughput or tail latency in the runs where the grouped barrier is actually formed.
- Loss/non-claim: Single-client per-write fdatasync and frozen strict-path rows remain the authenticated-publication cost boundary. Evidence also contains grouped runs where throughput or p99 does not improve, so the paper must not claim a general fast-path or filesystem-wide throughput win.
- Next valid code work: Only change publication performance further if the production protocol changes its crash ordering or grouping semantics and the same change is accompanied by replay/fault proof.

## Comparison Summary

### current_100us_viability

- Overall pass: `True`
- Strict grouped sync total: `8`
- Epoch grouped sync total: `6`
- Strict grouped throughput: `9.052 MiB/s`
- Epoch grouped throughput: `9.721 MiB/s`
- Strict grouped p99: `47612792.88999999` ns
- Epoch grouped p99: `40138994.73` ns
- Grouped sync reduced: `True`
- Grouped throughput improved: `True`
- Grouped p99 improved: `True`

### wait1ms_viability

- Overall pass: `True`
- Strict grouped sync total: `8`
- Epoch grouped sync total: `6`
- Strict grouped throughput: `9.218 MiB/s`
- Epoch grouped throughput: `9.597 MiB/s`
- Strict grouped p99: `45141936.230000004` ns
- Epoch grouped p99: `40910277.78` ns
- Grouped sync reduced: `True`
- Grouped throughput improved: `True`
- Grouped p99 improved: `True`

### fault_matrix_publication

- Overall pass: `True`
- Strict grouped sync total: `8`
- Epoch grouped sync total: `5`
- Strict grouped throughput: `6.446 MiB/s`
- Epoch grouped throughput: `1.076 MiB/s`
- Strict grouped p99: `2669886.0199999996` ns
- Epoch grouped p99: `53136288.72` ns
- Grouped sync reduced: `True`
- Grouped throughput improved: `False`
- Grouped p99 improved: `False`

## Proof Checks

- `source_epoch_group_commit_present`: `True`
- `current_grouped_sync_count_reduced`: `True`
- `grouped_evidence_has_scoped_win_and_loss`: `True`
- `single_or_frozen_path_not_general_win`: `True`
- `replay_fault_matrix_passes`: `True`
- `parallel_commit_closure_passes`: `True`
- `paper_scopes_batching_claim`: `True`
- `negative_claim_guard_present`: `True`
