# UMA / storage-DMA profile report

This bundle packages the profiler-backed raw-read run and the retained Nsight summaries.

- Nsight report: `artifacts/validation/uma_storage_dma_profile_root/io_uring_uvm.nsys-rep`
- SQLite export: `artifacts/validation/uma_storage_dma_profile_root/io_uring_uvm.sqlite`
- Input directory: `artifacts/validation/uma_storage_dma_profile_root`

## Unified Memory summary

- No CUDA Unified Memory transfer rows were recorded in this profiling run.

## CUDA API summary

- cudaHostRegister: calls=2, total_ns=229879263
- cudaHostUnregister: calls=1, total_ns=35880
- cuModuleGetLoadingMode: calls=2, total_ns=760

## OS runtime summary

- poll: calls=30, total_ns=732939561
- ioctl: calls=882, total_ns=242588004
- waitpid: calls=2, total_ns=78661453
- sem_timedwait: calls=20, total_ns=1457149
- mmap64: calls=36, total_ns=955981
- read: calls=84, total_ns=850754
- open64: calls=138, total_ns=694839
- mmap: calls=40, total_ns=591121
- stat64: calls=221, total_ns=432622
- pthread_create: calls=6, total_ns=378517

This report does not claim verified NVMe-to-UVM DMA semantics or migration suppression.
