# TPM freshness report

This bundle packages the retained PCR-policy, anchor, crash-replay, and analytical freshness-window helpers.

- Input directory: `/home/thor/skim/pqc_encrpyted_fs/artifacts/validation/tpm_freshness_bundle`
- Checks: `6`
- Bundle present: `True`

- PCR policy probe present: `True`
- PCR drift rejected: `True`
- Monotonic replay present: `True`
- Monotonic replay mode: `fail_closed`
- Recovery verdict present: `True`
- Hardware backend rows present: `True`
- Hardware backend fail-closed count: `40`

This report does not claim persistent filesystem PCR sealing, hardware-backed freshness, physical power-loss safety, kernel-crash safety, or drive-cache safety.