# UMA / Storage-DMA Scope Audit

- Overall pass: `true`
- Decision: `accept_critical_path_direct_nvme_to_uvm_dma_scoped_out`
- Paper scope pass: `true`
- Profile diagnostic pass: `true`
- Counter diagnostic pass: `true`

## High-Risk Paper Mentions
- `Paper/10_Discussion_and_Limitations.tex:24` `storage_dma_path` scoped=`true`: on CUDA managed memory and cuPQC. It uses \texttt{cudaMemPrefetchAsync} and stream synchronization as locality/ordering operations within the executor, not as storage-DMA registration or permissions. Porting the policy to another API requires a separate proof of that API's allocation, synchronization, and device-I/O semantics.
- `Paper/1_Introduction.tex:39` `direct_nvme_to_uvm_dma` scoped=`true`: ng short of a foreground p99 recovery claim. \end{enumerate} The paper is intentionally conservative about unsupported mechanisms. It does not claim verified \texttt{O\_DIRECT} NVMe-to-UVM DMA, an \texttt{io\_uring}/eBPF completion bypass, GPU constant-time execution, persistent PCR-bound freshness, true power-loss crash certification, or portability
- `Paper/1_Introduction.tex:19` `storage_dma_path` scoped=`true`: & Independent ML-KEM batches obtain a large GPU throughput advantage.\\ Unified memory & Treat accessibility as DMA proof & Managed memory and prefetch are not storage-DMA registration or mutual exclusion.\\ Freshness & Store all witnesses in the backing directory & Whole-directory replay restores file-backed witnesses and remain
- `Paper/1_Introduction.tex:27` `gpudirect_storage` scoped=`true`: -anchor path that can fail closed when replayed state no longer matches protected freshness state. This is a narrower and more defensible thesis than claiming GPUDirect-like storage, eBPF completion bypass, or physical side-channel resistance. It is also stronger than a microbenchmark-only story because it ties placement decisions
- `Paper/2_Background.tex:39` `storage_dma_path` scoped=`true`: ite{jetson}. CUDA managed memory makes an allocation accessible to CPU and GPU code, and prefetch is a performance hint; neither API is a permission bit nor a storage-DMA registration protocol. The present implementation allocates managed buffers only inside the CUDA executor, prefetches them around a GPU operation, and synchro
- `Paper/2_Background.tex:57` `storage_dma_path` scoped=`true`: ing and checkpoint determine exposure. \item \textbf{No optimistic accelerator semantics.} CUDA managed memory and prefetch are local executor mechanisms until storage-DMA behavior is separately proven. \item \textbf{Fail-closed freshness.} A file-backed witness is an integrity check, not a freshness root; external anchors must b
- `Paper/3_Design.tex:120` `storage_dma_path` scoped=`true`: efetch complete.\\ \texttt{posix\_memalign} + \texttt{cudaHostRegister} & Prototype-only & Host-memory pinning and executor-local access checks; not a verified storage-DMA path.\\ \texttt{pread}/\texttt{pwrite} sidecars & Verified & Backing ciphertext and journal use ordinary POSIX I/O.\\ eBPF tracepoint notification & Not claime
- `Paper/4_Evaluation.tex:34` `storage_dma_path` scoped=`true`: nt, and queue depth & Broader mode-aligned filesystem comparison\\ Managed-memory smoke & Managed allocation and host/device prefetch calls complete & Verified storage-DMA or O\_DIRECT-to-UVM semantics\\ eBPF Tracepoint Latency & Raw \texttt{nvme\_complete\_rq} histogram artifact is retained & No final eBPF notification path is c
- `Paper/4_Evaluation.tex:34` `storage_dma_path` scoped=`true`: epth & Broader mode-aligned filesystem comparison\\ Managed-memory smoke & Managed allocation and host/device prefetch calls complete & Verified storage-DMA or O\_DIRECT-to-UVM semantics\\ eBPF Tracepoint Latency & Raw \texttt{nvme\_complete\_rq} histogram artifact is retained & No final eBPF notification path is claimed \\ Hardware Q
- `Paper/4_Evaluation.tex:117` `gpudirect_storage` scoped=`true`: esult. \subsection{What this evaluation leaves open} \label{sec:eval_stability} We do not provide a fair gocryptfs, fs-verity, dm-integrity, OP-TEE, SPDK, or GPUDirect Storage baseline; physical side-channel measurement; or a second UMA architecture. The repository retains matched fscrypt and dm-crypt sequential-write fio runs, but
- `Paper/4_Evaluation.tex:71` `migration_or_coherence_suppression` scoped=`true`: ped. The retained host-pinning and managed-storage probes show same-buffer CPU/GPU checksum agreement plus CUDA pointer diagnostics, but they do not prove UVM migration suppression or final FUSE data-path DMA. The file-backed anchor produces the deliberately negative result: across four cut points and five trials per cut point, restoring
- `Paper/4_Evaluation.tex:71` `final_fuse_dma_path` scoped=`true`: inning and managed-storage probes show same-buffer CPU/GPU checksum agreement plus CUDA pointer diagnostics, but they do not prove UVM migration suppression or final FUSE data-path DMA. The file-backed anchor produces the deliberately negative result: across four cut points and five trials per cut point, restoring an entire backing-directory
- `Paper/5_Related_Works.tex:13` `gpudirect_storage` scoped=`true`: nuqs_storage, infscaler}. SnuQS alone is a storage-capacity example, not a validation of the present filesystem format~\cite{snuqs_storage}. GPUfs, SPDK, and GPUDirect Storage target APIs and hardware paths that must be assessed separately from an ordinary FUSE \texttt{pread}/\texttt{pwrite} path. The initial AEGIS-Q draft incorrect
- `Paper/6_Conclusion.tex:6` `direct_nvme_to_uvm_dma` scoped=`true`: sured AES-GCM data path, while GPU wins for large-batch ML-KEM key generation. Equally important, this revision removes unsupported claims. It does not claim direct NVMe-to-UVM DMA, eBPF/io\_uring completion bypass, GPU constant-time behavior, PCR-bound persistent freshness, power-loss crash certification, foreground AI p99 recovery, or p
- `Paper/7_Implementation_Details.tex:15` `storage_dma_path` scoped=`true`: mentation} & \textbf{Boundary}\\ \midrule CUDA storage path & Managed allocation with explicit prefetch and stream synchronization & Not a verified NVMe-to-UVM storage-DMA path\\ CUDA failure policy & CUDA executor falls back to CPU on failure & Does not change the on-disk record format\\ Anchor backend & File witness plus option
- `Paper/main.tex:89` `direct_nvme_to_uvm_dma` scoped=`true`: } stale-snapshot recovery evidence, SQLite syscall-exact app-crash timing, and CUPTI/tegrastats mounted-throttle wiring. The paper deliberately does not claim direct NVMe-to-UVM DMA, \texttt{io\_uring}/eBPF completion bypass, physical side-channel resistance, PCR-bound persistent freshness, power-loss certification, foreground AI-p99 resto

## Required Scope Phrases
- `does not claim direct NVMe-to-UVM DMA` found=`true`
- `does not claim verified \texttt{O\_DIRECT} NVMe-to-UVM DMA` found=`true`
- `not a verified NVMe-to-UVM storage-DMA path` found=`true`
- `do not prove UVM migration suppression or final FUSE data-path DMA` found=`true`
- `ordinary \texttt{pread}/\texttt{pwrite}` found=`true`

## Profile Checks
- `profile_report_exists` passed=`true`
- `report_is_scoped_not_dma_proof` passed=`true`
- `raw_probe_same_buffer_checksum` passed=`true`
- `managed_storage_same_buffer_checksum` passed=`true`
- `managed_storage_pointer_attr` passed=`true`
- `managed_storage_prefetch_diagnostics` passed=`true`
- `ncu_same_buffer_checksum` passed=`true`

## Counter Checks
- `counter_report_exists` passed=`true`
- `counter_report_is_not_suppression_proof` passed=`true`
- `nsys_runs_present` passed=`true`
- `um_reports_empty_or_skipped` passed=`true`
- `ncu_query_has_no_uvm_migration_metric_matches` passed=`true`
