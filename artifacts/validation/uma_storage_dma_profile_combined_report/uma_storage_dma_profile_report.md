# UMA / storage-DMA profile report

This bundle packages the profiler-backed raw-read run and the retained Nsight summaries.

- Input directory: `artifacts/validation/uma_storage_dma_profile_combined`
- Probe JSON: `artifacts/validation/uma_storage_dma_profile_combined/probe/uma_storage_dma_probe.json`

## Raw probe

- Command: `['sudo', '-S', '/usr/local/cuda-13.0/bin/nsys', 'profile', '--trace=cuda,osrt,tegra-accelerators', '--output', 'artifacts/validation/uma_storage_dma_profile_combined/probe/raw_probe', 'python3', 'experiments/run_uma_storage_dma_probe.py', '--nvme-device', '/dev/nvme0n1', '--out-dir', 'artifacts/validation/uma_storage_dma_profile_combined/probe']`
- Return code: `0`
- Nsight report: `artifacts/validation/uma_storage_dma_profile_combined/probe/raw_probe.nsys-rep`
- SQLite export: `artifacts/validation/uma_storage_dma_profile_combined/probe/raw_probe.sqlite`
- Same storage-filled buffer observed by GPU checksum: `True`

### Raw Probe same-buffer evidence

- `[+] Memory pinned and mapped to device pointer: 0xffff8a510000`
- `[+] POINTER_ATTR host_ptr type=host device=0 host_ptr=0xffff8a510000 device_ptr=0xffff8a510000`
- `[+] HOST_FLAGS host_ptr raw=0xa portable=0 mapped=1 writecombined=0`
- `[!] RANGE_ATTR host_ptr preferred_location_type error=invalid argument`
- `[!] RANGE_ATTR host_ptr preferred_location_id error=invalid argument`
- `[!] RANGE_ATTR host_ptr last_prefetch_location_type error=invalid argument`
- `[!] RANGE_ATTR host_ptr last_prefetch_location_id error=invalid argument`
- `[+] POINTER_ATTR device_alias type=host device=0 host_ptr=0xffff8a510000 device_ptr=0xffff8a510000`
- `[+] Async read completed: 4194304 bytes`
- `[+] CPU checksum over storage buffer: 0x0000000cd9141bd5`
- `[+] GPU mapped-buffer checksum:      0x0000000cd9141bd5`
- `[+] CHECKSUM_MATCH: GPU observed the same O_DIRECT storage-filled pinned buffer`

### Raw Probe CUDA API summary

- cudaHostRegister: calls=2, total_ns=214071156
- cudaLaunchKernel: calls=1, total_ns=225166
- cudaDeviceSynchronize: calls=1, total_ns=174102
- cudaFree: calls=1, total_ns=132973
- cudaMalloc: calls=1, total_ns=105481

### Raw Probe Unified Memory summary

- No CUDA Unified Memory transfer rows were recorded for the raw probe.

## Managed-memory smoke

- Command: `['sudo', '-S', '/usr/local/cuda-13.0/bin/nsys', 'profile', '--trace=cuda,osrt,tegra-accelerators', '--output', 'artifacts/validation/uma_storage_dma_profile_combined/um_smoke/um_smoke', '/home/thor/skim/pqc_encrpyted_fs/build/pqc_fuse', '--um-smoke']`
- Return code: `0`
- Nsight report: `artifacts/validation/uma_storage_dma_profile_combined/um_smoke/um_smoke.nsys-rep`
- SQLite export: `artifacts/validation/uma_storage_dma_profile_combined/um_smoke/um_smoke.sqlite`

### Unified Memory summary

- No CUDA Unified Memory transfer rows were recorded in this profiling run.

### CUDA API summary

- cudaMallocManaged: calls=1, total_ns=101974572
- cudaFree: calls=1, total_ns=1227824
- cudaDeviceSynchronize: calls=2, total_ns=495315
- cudaLaunchKernel: calls=1, total_ns=362509
- cuLibraryLoadData: calls=1, total_ns=121148

### OS runtime summary

- poll: calls=13, total_ns=150408692
- ioctl: calls=449, total_ns=111977902
- mmap: calls=23, total_ns=1482254
- sem_timedwait: calls=10, total_ns=690630
- mmap64: calls=15, total_ns=465695
- pthread_create: calls=3, total_ns=170250
- open64: calls=36, total_ns=164387
- fopen: calls=26, total_ns=126600
- fgets: calls=8, total_ns=104807
- open: calls=24, total_ns=93359

## Managed storage buffer probe

- Command: `['sudo', '-S', '/usr/local/cuda-13.0/bin/nsys', 'profile', '--trace=cuda,osrt,tegra-accelerators', '--output', 'artifacts/validation/uma_storage_dma_profile_combined/managed_storage/managed_storage', '/home/thor/skim/pqc_encrpyted_fs/build/io_uring_uvm', '--managed-buffer', '/dev/nvme0n1']`
- Return code: `0`
- Same storage-filled managed buffer observed by GPU checksum: `True`
- Managed pointer attribute observed: `True`
- Device-prefetch location observed: `True`
- Host-prefetch location observed: `True`

### Managed storage diagnostics

- `[+] Managed buffer allocated at: 0xfffd00000000`
- `[+] POINTER_ATTR managed_ptr_before_read type=managed device=0 host_ptr=0xfffd00000000 device_ptr=0xfffd00000000`
- `[+] HOST_FLAGS managed_ptr_before_read raw=0x2 portable=0 mapped=1 writecombined=0`
- `[+] RANGE_ATTR managed_ptr_before_read preferred_location_type=0(invalid)`
- `[+] RANGE_ATTR managed_ptr_before_read preferred_location_id=4294967294`
- `[+] RANGE_ATTR managed_ptr_before_read last_prefetch_location_type=0(invalid)`
- `[+] RANGE_ATTR managed_ptr_before_read last_prefetch_location_id=4294967294`
- `[+] Buffered pread completed: 4194304 bytes`
- `[+] POINTER_ATTR managed_ptr_after_read type=managed device=0 host_ptr=0xfffd00000000 device_ptr=0xfffd00000000`
- `[+] HOST_FLAGS managed_ptr_after_read raw=0x2 portable=0 mapped=1 writecombined=0`
- `[+] RANGE_ATTR managed_ptr_after_read preferred_location_type=0(invalid)`
- `[+] RANGE_ATTR managed_ptr_after_read preferred_location_id=4294967294`
- `[+] RANGE_ATTR managed_ptr_after_read last_prefetch_location_type=0(invalid)`
- `[+] RANGE_ATTR managed_ptr_after_read last_prefetch_location_id=4294967294`
- `[+] POINTER_ATTR managed_ptr_after_prefetch_device type=managed device=0 host_ptr=0xfffd00000000 device_ptr=0xfffd00000000`
- `[+] HOST_FLAGS managed_ptr_after_prefetch_device raw=0x2 portable=0 mapped=1 writecombined=0`
- `[+] RANGE_ATTR managed_ptr_after_prefetch_device preferred_location_type=0(invalid)`
- `[+] RANGE_ATTR managed_ptr_after_prefetch_device preferred_location_id=4294967294`
- `[+] RANGE_ATTR managed_ptr_after_prefetch_device last_prefetch_location_type=1(device)`
- `[+] RANGE_ATTR managed_ptr_after_prefetch_device last_prefetch_location_id=0`
- `[+] POINTER_ATTR managed_ptr_after_prefetch_host type=managed device=0 host_ptr=0xfffd00000000 device_ptr=0xfffd00000000`
- `[+] HOST_FLAGS managed_ptr_after_prefetch_host raw=0x2 portable=0 mapped=1 writecombined=0`
- `[+] RANGE_ATTR managed_ptr_after_prefetch_host preferred_location_type=0(invalid)`
- `[+] RANGE_ATTR managed_ptr_after_prefetch_host preferred_location_id=4294967294`
- `[+] RANGE_ATTR managed_ptr_after_prefetch_host last_prefetch_location_type=2(host)`
- `[+] RANGE_ATTR managed_ptr_after_prefetch_host last_prefetch_location_id=4294967295`
- `[+] CPU checksum over storage buffer: 0x0000000cd9141bd5`
- `[+] GPU managed-buffer checksum:      0x0000000cd9141bd5`
- `[+] CHECKSUM_MATCH: GPU observed the same storage-filled managed buffer`

## Nsight Compute storage-visible buffer counters

- Command: `['sudo', '-S', '/usr/local/cuda-13.0/bin/ncu', '--set', 'detailed', '--kernel-name', 'regex:checksum_kernel', '--target-processes', 'all', '--export', 'artifacts/validation/uma_storage_dma_profile_combined/ncu/io_uring_uvm_checksum', '--force-overwrite', '/home/thor/skim/pqc_encrpyted_fs/build/io_uring_uvm', '/dev/nvme0n1']`
- Return code: `0`
- NCU report: `artifacts/validation/uma_storage_dma_profile_combined/ncu/io_uring_uvm_checksum.ncu-rep`
- CSV export: `artifacts/validation/uma_storage_dma_profile_combined/ncu/io_uring_uvm_checksum.csv`
- Same-buffer checksum match in profiled run: `True`
- Selected metric rows: `34`

### Selected NCU metrics

- GPU Speed Of Light Throughput / Elapsed Cycles: 220574 cycle
- GPU Speed Of Light Throughput / Memory Throughput: 4.46 %
- GPU Speed Of Light Throughput / Duration: 140.10 us
- GPU Speed Of Light Throughput / L1/TEX Cache Throughput: 6.49 %
- GPU Speed Of Light Throughput / L2 Cache Throughput: 3.74 %
- GPU Speed Of Light Throughput / SM Active Cycles: 151604.75 cycle
- GPU Speed Of Light Throughput / Compute (SM) Throughput: 33.22 %
- Memory Workload Analysis / Local Memory Spilling Requests: 0 
- Memory Workload Analysis / Local Memory Spilling Request Overhead: 0 %
- Memory Workload Analysis / L1/TEX Hit Rate: 0.03 %
- Memory Workload Analysis / L2 Persisting Size: 6.29 Mbyte
- Memory Workload Analysis / L2 Compression Success Rate: 0 %
- Memory Workload Analysis / L2 Compression Ratio: 0 
- Memory Workload Analysis / L2 Compression Input Sectors: 0 sector
- Memory Workload Analysis / L2 Hit Rate: 19.06 %
- Launch Statistics / Function Cache Configuration: CachePreferNone 
- Launch Statistics / Shared Memory Configuration Size: 65.54 Kbyte
- Launch Statistics / Driver Shared Memory Per Block: 1.02 Kbyte/block
- Launch Statistics / Dynamic Shared Memory Per Block: 0 byte/block
- Launch Statistics / Static Shared Memory Per Block: 2.05 Kbyte/block
- Occupancy / Overall GPU Occupancy: 0 %
- Occupancy / Cluster Occupancy: 0 %
- Occupancy / Theoretical Occupancy: 100 %
- Occupancy / Achieved Occupancy: 82.61 %
- GPU and Memory Workload Distribution / Average L1 Active Cycles: 151604.75 cycle
- GPU and Memory Workload Distribution / Total L1 Elapsed Cycles: 4411462 cycle
- GPU and Memory Workload Distribution / Average L2 Active Cycles: 164329.50 cycle
- GPU and Memory Workload Distribution / Total L2 Elapsed Cycles: 3529152 cycle
- GPU and Memory Workload Distribution / Average MC Channel Active Cycles: nan cycle
- GPU and Memory Workload Distribution / Total MC Channel Elapsed Cycles: nan cycle
- GPU and Memory Workload Distribution / Average SM Active Cycles: 151604.75 cycle
- GPU and Memory Workload Distribution / Total SM Elapsed Cycles: 4411462 cycle
- GPU and Memory Workload Distribution / Average SMSP Active Cycles: 148002.79 cycle
- GPU and Memory Workload Distribution / Total SMSP Elapsed Cycles: 17645848 cycle

This report does not claim verified NVMe-to-UVM DMA semantics or migration suppression.
