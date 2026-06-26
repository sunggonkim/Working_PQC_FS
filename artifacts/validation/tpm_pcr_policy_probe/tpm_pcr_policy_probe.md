# TPM PCR policy probe

This artifact records a non-destructive PCR-policy seal/unseal probe.
It does not provision AEGIS-Q's persistent NV index and does not prove the full hardware-backed freshness flow.

## Results

- TCTI: `device:/dev/tpmrm0`
- PCR list: `sha256:0,1,2,3`
- Current PCR digest file SHA-256: `8e2f9e17c3352c54386536c5c94166f5a1ac6614358ef204261a3af177c1f49d`
- Drifted PCR digest file SHA-256: `94bbba136c1029b9e9d4f2032fa014ad32f4c5d40d6c8547f12a3f942cabfb05`
- Good unseal return code: `0`
- Good unseal matches secret: `True`
- Drift policy return code: `1`
- Drift unseal return code: `1`
- Drift rejected: `True`

## Conservative interpretation

- This closes a transient PCR-policy probe: current PCR policy authorizes unseal, while a drifted PCR digest is rejected.
- This is not a persistent freshness proof for the filesystem anchor.
- Full closure still requires monotonic update semantics and mount/recovery behavior tied to the hardware-backed anchor state.
