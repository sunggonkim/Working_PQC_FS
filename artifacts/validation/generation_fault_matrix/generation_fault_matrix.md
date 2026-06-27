# Generation fault matrix

Scope: final-binary FUSE generation/nonce regression matrix.  This is not full power-loss or daemon-crash certification.

- Command: `experiments/run_generation_fault_matrix.py --out-dir artifacts/validation/generation_fault_matrix`
- Overall pass: `True`
- No generated nonce-pair reuse: `True`
- No silent corruption verdicts: `True`

| Case | Verdict | Acceptable | Scope |
|---|---:|---:|---|
| `self_test_older_generation_regression` | `latest_committed` | `True` |  |
| `partial_update_and_remount` | `latest_committed` | `True` |  |
| `torn_journal_write` | `latest_committed` | `True` |  |
| `older_generation_append_after_newer_mapping` | `latest_committed` | `True` |  |
| `stale_snapshot_replay_file_anchor_negative_control` | `previous_committed` | `True` | file-backed anchor is replayable with the backing directory; previous_committed is expected negative-control behavior, not rollback protection. |
| `stale_snapshot_replay_tpm_anchor_existing_artifact` | `fail_closed` | `True` | existing hardware-backed TPM replay-after-advance artifact; not rerun by this generation matrix script. |

Raw JSON retains mount logs, journal summaries, and SHA-256 digests.
