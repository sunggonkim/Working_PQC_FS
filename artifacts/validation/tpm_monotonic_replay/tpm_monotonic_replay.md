# TPM monotonic replay

This run snapshots the file-backed storage directory, advances the hardware anchor, restores the stale snapshot, and then attempts a remount.

- Replay mode: `fail_closed`
- Detail: `restored snapshot mounted but payload read failed closed (rc=1): Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/usr/lib/python3.12/pathlib.py", line 1021, in read_bytes
    with self.open(mode='rb') as f:
         ^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/pathlib.py", line 1015, in open
    return io.open(self, mode, buffering, encoding, errors, newline)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
OSError: [Errno 5] Input/output error: '/tmp/tpm_mono_mnt_nh82in01/payload.bin'`
- Return code: `None`

- Live mount logs: `/home/thor/skim/pqc_encrpyted_fs/artifacts/validation/tpm_monotonic_replay/live_mount/pqc_fuse.stderr.txt`
- Replay mount logs: `/home/thor/skim/pqc_encrpyted_fs/artifacts/validation/tpm_monotonic_replay/replay_mount/pqc_fuse.stderr.txt`

This artifact is conservative: it only supports the exact replay classification captured in the logs.
