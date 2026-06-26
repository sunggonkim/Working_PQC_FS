# UMA / storage-DMA profile report

This bundle packages the profiler-backed raw-read run and the retained Nsight summaries.

- Nsight report: `artifacts/uma_storage_dma_nsys/io_uring_uvm.nsys-rep`
- SQLite export: `artifacts/uma_storage_dma_nsys/io_uring_uvm.sqlite`
- Input directory: `artifacts/uma_storage_dma_nsys`

## Unified Memory summary

- No CUDA Unified Memory transfer rows were recorded in this profiling run.

## CUDA API summary

- cudaHostRegister: calls=1, total_ns=106859530
- cudaHostUnregister: calls=1, total_ns=40796
- cuModuleGetLoadingMode: calls=1, total_ns=574

## OS runtime summary

- poll: calls=13, total_ns=151848821
- ioctl: calls=441, total_ns=112647753
- sem_timedwait: calls=10, total_ns=4766380
- mmap64: calls=15, total_ns=466861
- mmap: calls=20, total_ns=307268
- pthread_create: calls=3, total_ns=181510
- open64: calls=36, total_ns=158667
- fopen: calls=27, total_ns=122213
- fgets: calls=8, total_ns=102443
- open: calls=24, total_ns=87201

This report does not claim verified NVMe-to-UVM DMA semantics or migration suppression.
