# UMA counter availability audit

This artifact records whether retained Nsight outputs expose UVM/page-fault/migration/coherence counters for the same-buffer storage-visible probe.
It does not prove NVMe-to-UVM DMA semantics or migration suppression.

## Nsight Systems reports

### raw_probe

- Report file: `artifacts/validation/uma_storage_dma_profile_combined/probe/raw_probe.nsys-rep`
- `um_sum`: rc=0, data_lines=0, skipped=`SKIPPED: /home/thor/skim/pqc_encrpyted_fs/artifacts/validation/uma_storage_dma_profile_combined/probe/raw_probe.sqlite does not contain CUDA Unified Memory CPU page faults data.`
- `um_total_sum`: rc=0, data_lines=0, skipped=`SKIPPED: /home/thor/skim/pqc_encrpyted_fs/artifacts/validation/uma_storage_dma_profile_combined/probe/raw_probe.sqlite does not contain CUDA Unified Memory CPU page faults data.`
- `um_cpu_page_faults_sum`: rc=0, data_lines=0, skipped=`SKIPPED: /home/thor/skim/pqc_encrpyted_fs/artifacts/validation/uma_storage_dma_profile_combined/probe/raw_probe.sqlite does not contain CUDA Unified Memory CPU page faults data.`
- `cuda_api_sum`: rc=0, data_lines=13
- SQLite matched event tables: `[]`
- SQLite matched enum tables: `['ENUM_CUDA_DEV_MEM_EVENT_OPER', 'ENUM_CUDA_FUNC_CACHE_CONFIG', 'ENUM_CUDA_KERNEL_LAUNCH_TYPE', 'ENUM_CUDA_MEMCPY_OPER', 'ENUM_CUDA_MEMPOOL_OPER', 'ENUM_CUDA_MEMPOOL_TYPE', 'ENUM_CUDA_MEM_KIND', 'ENUM_CUDA_SHARED_MEM_LIMIT_CONFIG', 'ENUM_CUDA_UNIF_MEM_ACCESS_TYPE', 'ENUM_CUDA_UNIF_MEM_MIGRATION', 'ENUM_CUPTI_STREAM_TYPE', 'ENUM_CUPTI_SYNC_TYPE', 'ENUM_DIAGNOSTIC_SEVERITY_LEVEL', 'ENUM_DIAGNOSTIC_SOURCE_TYPE', 'ENUM_DIAGNOSTIC_TIMESTAMP_SOURCE', 'ENUM_NSYS_EVENT_CLASS', 'ENUM_NSYS_EVENT_TYPE', 'ENUM_NSYS_GENERIC_EVENT_FIELD_ETW_FLAGS', 'ENUM_NSYS_GENERIC_EVENT_FIELD_ETW_PROPERTY', 'ENUM_NSYS_GENERIC_EVENT_FIELD_ETW_TYPE', 'ENUM_NSYS_GENERIC_EVENT_FIELD_TYPE', 'ENUM_NSYS_GENERIC_EVENT_GROUP', 'ENUM_NSYS_GENERIC_EVENT_SOURCE', 'ENUM_SAMPLING_THREAD_STATE', 'ENUM_SCHEDULING_THREAD_BLOCK', 'ENUM_STACK_UNWIND_METHOD']`

### um_smoke

- Report file: `artifacts/validation/uma_storage_dma_profile_combined/um_smoke/um_smoke.nsys-rep`
- `um_sum`: rc=0, data_lines=0, skipped=`SKIPPED: /home/thor/skim/pqc_encrpyted_fs/artifacts/validation/uma_storage_dma_profile_combined/um_smoke/um_smoke.sqlite does not contain CUDA memory transfers data.`
- `um_total_sum`: rc=0, data_lines=0, skipped=`SKIPPED: /home/thor/skim/pqc_encrpyted_fs/artifacts/validation/uma_storage_dma_profile_combined/um_smoke/um_smoke.sqlite does not contain CUDA memory transfers data.`
- `um_cpu_page_faults_sum`: rc=0, data_lines=0, skipped=`SKIPPED: /home/thor/skim/pqc_encrpyted_fs/artifacts/validation/uma_storage_dma_profile_combined/um_smoke/um_smoke.sqlite does not contain CUDA Unified Memory CPU page faults data.`
- `cuda_api_sum`: rc=0, data_lines=10
- SQLite matched event tables: `[]`
- SQLite matched enum tables: `['ENUM_CUDA_DEV_MEM_EVENT_OPER', 'ENUM_CUDA_FUNC_CACHE_CONFIG', 'ENUM_CUDA_KERNEL_LAUNCH_TYPE', 'ENUM_CUDA_MEMCPY_OPER', 'ENUM_CUDA_MEMPOOL_OPER', 'ENUM_CUDA_MEMPOOL_TYPE', 'ENUM_CUDA_MEM_KIND', 'ENUM_CUDA_SHARED_MEM_LIMIT_CONFIG', 'ENUM_CUDA_UNIF_MEM_ACCESS_TYPE', 'ENUM_CUDA_UNIF_MEM_MIGRATION', 'ENUM_CUPTI_STREAM_TYPE', 'ENUM_CUPTI_SYNC_TYPE', 'ENUM_DIAGNOSTIC_SEVERITY_LEVEL', 'ENUM_DIAGNOSTIC_SOURCE_TYPE', 'ENUM_DIAGNOSTIC_TIMESTAMP_SOURCE', 'ENUM_NSYS_EVENT_CLASS', 'ENUM_NSYS_EVENT_TYPE', 'ENUM_NSYS_GENERIC_EVENT_FIELD_ETW_FLAGS', 'ENUM_NSYS_GENERIC_EVENT_FIELD_ETW_PROPERTY', 'ENUM_NSYS_GENERIC_EVENT_FIELD_ETW_TYPE', 'ENUM_NSYS_GENERIC_EVENT_FIELD_TYPE', 'ENUM_NSYS_GENERIC_EVENT_GROUP', 'ENUM_NSYS_GENERIC_EVENT_SOURCE', 'ENUM_SAMPLING_THREAD_STATE', 'ENUM_SCHEDULING_THREAD_BLOCK', 'ENUM_STACK_UNWIND_METHOD']`

### managed_storage_probe

- Report file: `artifacts/validation/uma_storage_dma_profile_combined/managed_storage/managed_storage.nsys-rep`
- `um_sum`: rc=0, data_lines=0, skipped=`SKIPPED: /home/thor/skim/pqc_encrpyted_fs/artifacts/validation/uma_storage_dma_profile_combined/managed_storage/managed_storage.sqlite does not contain CUDA Unified Memory CPU page faults data.`
- `um_total_sum`: rc=0, data_lines=0, skipped=`SKIPPED: /home/thor/skim/pqc_encrpyted_fs/artifacts/validation/uma_storage_dma_profile_combined/managed_storage/managed_storage.sqlite does not contain CUDA Unified Memory CPU page faults data.`
- `um_cpu_page_faults_sum`: rc=0, data_lines=0, skipped=`SKIPPED: /home/thor/skim/pqc_encrpyted_fs/artifacts/validation/uma_storage_dma_profile_combined/managed_storage/managed_storage.sqlite does not contain CUDA Unified Memory CPU page faults data.`
- `cuda_api_sum`: rc=0, data_lines=13
- SQLite matched event tables: `[]`
- SQLite matched enum tables: `['ENUM_CUDA_DEV_MEM_EVENT_OPER', 'ENUM_CUDA_FUNC_CACHE_CONFIG', 'ENUM_CUDA_KERNEL_LAUNCH_TYPE', 'ENUM_CUDA_MEMCPY_OPER', 'ENUM_CUDA_MEMPOOL_OPER', 'ENUM_CUDA_MEMPOOL_TYPE', 'ENUM_CUDA_MEM_KIND', 'ENUM_CUDA_SHARED_MEM_LIMIT_CONFIG', 'ENUM_CUDA_UNIF_MEM_ACCESS_TYPE', 'ENUM_CUDA_UNIF_MEM_MIGRATION', 'ENUM_CUPTI_STREAM_TYPE', 'ENUM_CUPTI_SYNC_TYPE', 'ENUM_DIAGNOSTIC_SEVERITY_LEVEL', 'ENUM_DIAGNOSTIC_SOURCE_TYPE', 'ENUM_DIAGNOSTIC_TIMESTAMP_SOURCE', 'ENUM_NSYS_EVENT_CLASS', 'ENUM_NSYS_EVENT_TYPE', 'ENUM_NSYS_GENERIC_EVENT_FIELD_ETW_FLAGS', 'ENUM_NSYS_GENERIC_EVENT_FIELD_ETW_PROPERTY', 'ENUM_NSYS_GENERIC_EVENT_FIELD_ETW_TYPE', 'ENUM_NSYS_GENERIC_EVENT_FIELD_TYPE', 'ENUM_NSYS_GENERIC_EVENT_GROUP', 'ENUM_NSYS_GENERIC_EVENT_SOURCE', 'ENUM_SAMPLING_THREAD_STATE', 'ENUM_SCHEDULING_THREAD_BLOCK', 'ENUM_STACK_UNWIND_METHOD']`

## Nsight Compute metric query

- Command: `['/usr/local/cuda-13.0/bin/ncu', '--query-metrics']`
- Return code: `0`
- Matching metric lines: `0`

## Conservative interpretation

- The retained raw probe and managed-storage probe have same-buffer GPU checksum evidence, but the Nsight Systems UM reports do not expose UVM transfer/page-fault rows for those runs.
- If the UM reports are skipped or empty, the paper may document counter non-exposure for this bundle, not claim migration suppression.
