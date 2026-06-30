# KDF Current-State Verdict

- Overall pass: `true`
- Verdict: `scrypt_metadata_paper_guard_complete`
- Parent D1 closed: `true`

## Current Production KDF

- Algorithm: `scrypt-with-PBKDF2-legacy-compat`
- Iterations: `600000`
- Salt scope: `per_filesystem_metadata`
- Salt length: `32`
- Output length: `32`

## Toolchain

- Argon2 header compile: `false`
- OpenSSL scrypt compile: `true`

## Paper-Facing KDF Text

- Paper text pass: `true`
- Failed paper checks: `none`

## Claim Guard

- Guard candidates: `7`
- Unguarded candidates: `0`

## Next Production Step

- Move to D2-S0: add mounted-path crypto-plane route evidence so data-plane AES-GCM and key-plane PQC remain visibly separated.

## Boundary

This verdict allows only the implemented OpenSSL scrypt new-root path, the explicit PBKDF2 legacy compatibility boundary, and the password/offline-guessing limitation.  It does not claim Argon2id deployment, hardware-backed credential release, full offline-attack resistance, or broader security readiness.
