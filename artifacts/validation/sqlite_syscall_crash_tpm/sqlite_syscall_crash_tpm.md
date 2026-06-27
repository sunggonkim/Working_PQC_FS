# SQLite syscall-exact TPM crash campaign

Each trial runs SQLite on TPM-backed FUSE and kills the advancing transaction at an exact `fdatasync` call using `strace --inject`.

- Trials: `3`
- Acceptable trials: `3`
- Unacceptable trials: `0`

## Per-trial verdicts

- fdatasync when=1: verdict `previous_committed`, acceptable `True`, writer rc `-9`, strace files `1`
- fdatasync when=2: verdict `previous_committed`, acceptable `True`, writer rc `-9`, strace files `1`
- fdatasync when=3: verdict `previous_committed`, acceptable `True`, writer rc `-9`, strace files `1`

Conservative interpretation: this is syscall-exact app-crash timing for SQLite on the TPM-backed mounted FUSE path. It does not model power loss of the FUSE daemon or arbitrary kernel-level interruption.
