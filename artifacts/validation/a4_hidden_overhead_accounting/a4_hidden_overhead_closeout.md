# A4 Hidden Overhead Closeout

- Overall pass: `True`
- Paper text status: `already_scoped_no_update`
- Parent checklist closed: `False`

## Overhead Classification

| Overhead | Classification | Evidence | Guard |
| --- | --- | --- | --- |
| FUSE daemon operation latency | `measured` | `create_avg_ns=1644658`<br>`write_avg_ns=38287`<br>`fsync_avg_ns=6801778`<br>`read_avg_ns=227037`<br>`daemon-side proxy only` | Do not call this a kernel context-switch count or eBPF/io_uring bypass proof. |
| durability syscalls and publication barriers | `measured` | `fdatasync=2`<br>`syncfs=1`<br>`data_sidecar=1`<br>`journal_sidecar=1`<br>`marker_metadata=1`<br>`publication_sync_count_total=10`<br>`publication_elapsed_ns_total=6072629` | Report strict publication cost as a boundary, not a general filesystem ranking. |
| journal/checkpoint update | `measured` | `journal_sidecar=1`<br>`marker_metadata=1`<br>`publication_count=5` | Keep journal/checkpoint wording tied to D/J/C publication evidence. |
| freshness anchor refresh | `measured` | `freshness_anchor_events=1`<br>`freshness_anchor_successes=1`<br>`freshness_anchor_file_backend=1`<br>`freshness_anchor_hardware_backend=0` | Do not upgrade file-backed or replay-after-advance evidence to PCR-bound rollback resistance. |
| AES-GCM data-plane routing | `measured` | `encrypt_blocks=16`<br>`decrypt_blocks=16`<br>`cpu_blocks=32`<br>`gpu_blocks=0` | Do not imply bulk file data is encrypted directly by PQC primitives. |
| CUDA launch and stream synchronization | `diagnostic-only` | `production_kernel_launches=11`<br>`cuda_qos_paper_eligible=False`<br>`profiler_trace_retained=False` | CUDA launch evidence is not a paper mechanism until same-run mounted-path traces show benefit. |
| GPU initialization | `diagnostic-only` | `device_count=1`<br>`selected_device=0`<br>`A4 smoke does not time initialization on the mounted path` | Do not report GPU initialization as hidden mounted-path overhead until a production trace times it. |
| staging copy / UVM / pinned memory | `diagnostic-only` | `managed_memory_class=local_probe_pass`<br>`ladder_paper_mechanism_eligible_count=0`<br>`no production mounted-path benefit retained` | Keep UVM, pinned, and registered-memory wording diagnostic unless production mounted-path benefit exists. |
| dma-buf, GPUDirect/RDMA, direct NVMe-to-UVM DMA | `not-claimed` | `non_claim_terms=['CUPTI', 'dma-buf zero-copy', 'GPUDirect/RDMA', 'direct NVMe-to-UVM DMA']`<br>`no mounted-path peer-DMA proof` | These terms remain non-claims. |

## Proof Checks

- `a4_smoke_pass`: `True`
- `jetson_memory_pass`: `True`
- `cuda_qos_pass`: `True`
- `optimization_ladder_pass`: `True`
- `measured_fuse_latency`: `True`
- `measured_publication_barriers`: `True`
- `paper_has_scope_guards`: `True`
- `no_unscoped_dangerous_positive_hits`: `True`

## Non-Claims

- not a kernel context-switch count
- not proof of eBPF/io_uring bypass
- not a CUDA optimization paper mechanism
- not a direct NVMe-to-UVM, GPUDirect/RDMA, or dma-buf zero-copy claim
- not a TPM rollback-resistance claim
