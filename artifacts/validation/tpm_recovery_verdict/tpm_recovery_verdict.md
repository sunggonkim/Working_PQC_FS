# TPM recovery verdict

This artifact packages the retained hardware-backend crash/replay rows together with the hardware anchor round-trip measurement.

- crash replay summary: `/home/thor/skim/pqc_encrpyted_fs/artifacts/crash_replay_e8_test_summary.json`
- crash replay matrix: `/home/thor/skim/pqc_encrpyted_fs/artifacts/crash_replay_e8_test_matrix.json`
- hardware anchor latency: `/home/thor/skim/pqc_encrpyted_fs/artifacts/anchor_refresh/hardware_anchor_latency.json`
- hardware backend rows present: `True`
- hardware backend fail-closed count: `40`
- hardware backend rollback-accept count: `0`
- hardware backend fail-closed-rate range: `1.0` .. `1.0`

This artifact should be read together with the separate monotonic replay result; it still does not close the combined durability claim.
