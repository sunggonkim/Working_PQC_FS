# KDF Scrypt Mounted Smoke

- Overall pass: `true`
- KDF metadata valid: `true`
- Correct remount read matches: `true`
- Wrong password fails closed: `true`

## Checks

- `initial_mount_started`: `true`
- `kdf_metadata_exists`: `true`
- `kdf_metadata_valid`: `true`
- `kdf_metadata_uses_scrypt`: `true`
- `kdf_salt_nonzero`: `true`
- `kdf_hidden_initial`: `true`
- `kdf_hidden_remount`: `true`
- `correct_remount_read_matches`: `true`
- `wrong_password_fails_closed`: `true`
- `source_checks_pass`: `true`

## Boundary

D1-S1 mounted smoke for production scrypt KDF metadata.  It does not close the full D1 paper/security gate.
