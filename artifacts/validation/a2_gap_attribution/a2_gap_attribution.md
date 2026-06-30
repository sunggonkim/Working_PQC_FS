# A2 Gap Attribution

This is a short same-profile attribution probe derived from the frozen filesystem contract.
It is not a replacement for the retained full frozen-contract rows.

## Result

- AEGIS-Q throughput: `1.858 MiB/s`
- gocryptfs throughput: `18.145 MiB/s`
- AEGIS-Q/gocryptfs ratio: `0.102`

## AEGIS-Q Durability Boundary

- fdatasync calls: `2140`
- syncfs calls: `1070`
- data sidecar publications: `1070`
- journal sidecar publications: `1070`
- marker metadata publications: `1070`

## Same-Workload Client Syscalls

- AEGIS-Q fio-client syscall count: `7778`
- gocryptfs fio-client syscall count: `24467`
- AEGIS-Q fio-client fsync-family calls: `948`
- gocryptfs fio-client fsync-family calls: `9292`

## Interpretation

- The gap is not presented as a surprise throughput bug or a GPU/PQC failure.
- The current strict path pays an authenticated-publication boundary: data sidecar, journal sidecar, and marker/checkpoint publication.
- Collapsing these barriers would require a different crash-ordering proof, not just replacing fdatasync calls with a later syncfs.
- Daemon-level strace is not used by this probe because ptraced fusermount can be rejected by the host.
