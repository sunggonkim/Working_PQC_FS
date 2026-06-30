# Crypto Plane Separation Smoke

- Generated: `2026-06-28T23:30:28Z`
- Overall pass: `true`
- Ordinary read/write AES-GCM only: `true`
- Forced rekey key-plane only: `true`

## Ordinary Mounted I/O

- Trace: `artifacts/validation/crypto_plane_separation/ordinary.plane_trace.json`
- AES-GCM encrypt blocks: `4`
- AES-GCM decrypt blocks: `4`
- Key-plane batches: `0`
- Key-plane refreshed files: `0`

## Forced Key-Plane Workflow

- Trace: `artifacts/validation/crypto_plane_separation/forced_keyplane.plane_trace.json`
- Rekey log seen: `true`
- AES-GCM encrypt blocks: `5`
- AES-GCM decrypt blocks: `5`
- Key-plane batches: `1`
- Key-plane refreshed files: `1`
- Key-plane work bytes: `1120`

## Scope

This smoke proves the mounted implementation separates AES-GCM block I/O counters from the ML-KEM envelope-refresh workflow. It does not claim that bulk file data is encrypted directly by PQC primitives.
