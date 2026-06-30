# Jetson Memory Contract

- Generated: `2026-06-29T08:33:10Z`
- Overall pass: `true`
- Probe executable: `build/jetson_memory_contract_probe`
- Raw stdout: `artifacts/validation/jetson_memory_contract/jetson_memory_probe.stdout.json`
- Raw stderr: `artifacts/validation/jetson_memory_contract/jetson_memory_probe.stderr.txt`

## Official Basis

- [NVIDIA CUDA for Tegra memory/coherency guidance](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html)
- [NVIDIA GPUDirect RDMA caveats](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html)

## Device Summary

- Device count: `1`
- CUDA runtime version: `13000`
- CUDA driver version: `13000`

## Claim Contract

| Claim | Evidence level | Paper mechanism eligible | Evidence |
| --- | --- | --- | --- |
| cudaDeviceProp/cudaDeviceGetAttribute inventory | `local_probe` | `true` | device_count=1; selected_device_name=NVIDIA Thor; compute_capability=11.0 |
| pinned host allocation via cudaHostAlloc | `local_probe_pass` | `true` | cudaHostAlloc=pass; mapped_kernel=pass |
| registered host memory via cudaHostRegister | `local_probe_pass` | `true` | hostRegisterSupported=1; cudaHostRegister=pass |
| managed memory / UVM allocation | `local_probe_pass` | `true` | managedMemory=1; concurrentManagedAccess=1; cudaMallocManaged=pass; cudaMemPrefetchAsyncManaged=pass |
| pageable memory access | `local_probe_pass` | `true` | pageableMemoryAccess=1; pageableMemoryAccessUsesHostPageTables=1; pageableMemoryPrefetch=pass |
| dma-buf export/import | `non_claim` | `false` | /dev/dma_heap_present=False; no CUDA dma-buf export/import production probe is implemented |
| GPUDirect/RDMA applicability | `non_claim` | `false` | cudaDevAttrGPUDirectRDMASupported=0; nvidia_peermem_or_nv_peer_mem=False; infiniband_devices=[] |
| NVMe-to-UVM direct DMA | `non_claim` | `false` | no NVMe peer-DMA proof; no UVM direct-DMA proof; no production mounted-path trace |

## Negative Claim Guard

No paper or README text may claim direct NVMe-to-UVM DMA, GPUDirect/RDMA, dma-buf zero-copy, pageable-memory coherency, UVM production benefit, or pinned/registered host-memory benefit unless this contract reports paper_mechanism_eligible=true for that mechanism and a later production mounted-path artifact uses it.
