# Technique Transfer Matrix

- Generated: `2026-06-30T02:43:57Z`
- Overall pass: `true`
- Complete rows: `9/9`

| Prior mechanism | Source | AEGIS-Q analogue | Status | Complete | Boundary |
| --- | --- | --- | --- | --- | --- |
| Append-only logging and roll-forward recovery | F2FS FAST'15 | Strict journal plus opt-in epoch redo-log with replay of committed prefixes. | `implemented-scoped-analogue` | `true` | Do not call the epoch log an F2FS implementation or a flash-translation/filesystem replacement. |
| Operation logs and decentralized filesystem metadata | ScaleFS SOSP'17 | Operation-style epoch records and a sharded leader/follower commit coordinator. | `implemented-scoped-analogue` | `true` | No ScaleFS-equivalent scalability or decentralized metadata claim is allowed. |
| Compact metadata logging and selective flushing | FastCommit ATC'24 | Batched strict journal append, epoch commit records, checkpoint compaction, and opt-in windowed anchor flushing. | `implemented-scoped-analogue` | `true` | Do not imply ext4 FastCommit semantics, broad selective flushing safety, or physical power-loss certification. |
| Group commit and parallel journaling | FAST/SOSP/OSDI storage systems | Opt-in epoch leader/follower group barrier with sharded runtime telemetry and replay ordering. | `implemented-scoped-analogue` | `true` | No broad scalability, multicore, or concurrent-client claim before Gate 0.15/A3 close. |
| Kernel QoS and I/O scheduling controls | Linux ionice/cgroup/BFQ-style controls | Storage-visible QoS classification and throttle path compared against two kernel-level controls. | `measured-comparison` | `true` | No SQLite uniqueness or general kernel-QoS impossibility claim. |
| Kernel-native encryption baselines | fscrypt and dm-crypt | Mode-aligned baseline verdicts with measured, environment-blocked, or unavailable rows. | `baseline-boundary` | `true` | No apples-to-apples fscrypt/dm-crypt speedup without matched measured rows. |
| TPM/TEE freshness and rollback boundaries | TPM/TEE systems | TPM NV replay-after-advance checks plus freshness ladder, without sealed-key release or PCR-bound rollback resistance. | `implemented-negative-boundary` | `true` | No TPM rollback-resistance or PCR-bound freshness claim before Gate C6 closes. |
| FUSE passthrough, eBPF, io_uring, and kernel-bypass completion | Modern kernel bypass and tracing paths | Measured non-claim and ordinary FUSE path; no production bypass is part of the verified system. | `rejected-nonclaim` | `true` | No eBPF/io_uring bypass claim unless production mounted-path evidence exists. |
| GPU/UMA scheduling and storage staging | GPUfs, GPUDirect, FlashNeuron/Fastensor-style staging | CPU-first AES-GCM data path, optional GPU key-plane/maintenance lane, and explicit Jetson memory non-claims. | `implemented-scoped-placement` | `true` | No NVIDIA mechanism is a paper mechanism without production mounted-path benefit evidence. |
