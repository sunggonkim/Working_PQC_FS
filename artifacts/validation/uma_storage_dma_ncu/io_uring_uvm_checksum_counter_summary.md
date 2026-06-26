# UMA storage-visible buffer Nsight Compute counter summary

Nsight Compute counters for checksum_kernel over the same O_DIRECT-filled cudaHostRegister mapped buffer. This is storage-visible GPU memory-access evidence, not CUDA Unified Memory migration proof.

- Return code: `0`
- Checksum match in profiled run: `True`
- NCU report: `artifacts/validation/uma_storage_dma_ncu/io_uring_uvm_checksum_detailed.ncu-rep`
- CSV export: `artifacts/validation/uma_storage_dma_ncu/io_uring_uvm_checksum_detailed.csv`

## Selected metrics
- GPU Speed Of Light Throughput / Elapsed Cycles: 207110 cycle
- GPU Speed Of Light Throughput / Memory Throughput: 4.75 %
- GPU Speed Of Light Throughput / Duration: 131.55 us
- GPU Speed Of Light Throughput / L1/TEX Cache Throughput: 6.40 %
- GPU Speed Of Light Throughput / L2 Cache Throughput: 3.99 %
- GPU Speed Of Light Throughput / SM Active Cycles: 153873.35 cycle
- GPU Speed Of Light Throughput / Compute (SM) Throughput: 35.38 %
- Memory Workload Analysis / Local Memory Spilling Requests: 0 
- Memory Workload Analysis / Local Memory Spilling Request Overhead: 0 %
- Memory Workload Analysis / L1/TEX Hit Rate: 0.03 %
- Memory Workload Analysis / L2 Persisting Size: 6.29 Mbyte
- Memory Workload Analysis / L2 Compression Success Rate: 0 %
- Memory Workload Analysis / L2 Compression Ratio: 0 
- Memory Workload Analysis / L2 Compression Input Sectors: 0 sector
- Memory Workload Analysis / L2 Hit Rate: 19.84 %
- Launch Statistics / Function Cache Configuration: CachePreferNone 
- Launch Statistics / Shared Memory Configuration Size: 65.54 Kbyte
- Launch Statistics / Driver Shared Memory Per Block: 1.02 Kbyte/block
- Launch Statistics / Dynamic Shared Memory Per Block: 0 byte/block
- Launch Statistics / Static Shared Memory Per Block: 2.05 Kbyte/block
- Occupancy / Overall GPU Occupancy: 0 %
- Occupancy / Cluster Occupancy: 0 %
- Occupancy / Theoretical Occupancy: 100 %
- Occupancy / Achieved Occupancy: 84.05 %
- GPU and Memory Workload Distribution / Average L1 Active Cycles: 153873.35 cycle
- GPU and Memory Workload Distribution / Total L1 Elapsed Cycles: 4142182 cycle
- GPU and Memory Workload Distribution / Average L2 Active Cycles: 162481.19 cycle
- GPU and Memory Workload Distribution / Total L2 Elapsed Cycles: 3313696 cycle
- GPU and Memory Workload Distribution / Average MC Channel Active Cycles: nan cycle
- GPU and Memory Workload Distribution / Total MC Channel Elapsed Cycles: nan cycle
- GPU and Memory Workload Distribution / Average SM Active Cycles: 153873.35 cycle
- GPU and Memory Workload Distribution / Total SM Elapsed Cycles: 4142182 cycle
- GPU and Memory Workload Distribution / Average SMSP Active Cycles: 145826.86 cycle
- GPU and Memory Workload Distribution / Total SMSP Elapsed Cycles: 16568728 cycle
