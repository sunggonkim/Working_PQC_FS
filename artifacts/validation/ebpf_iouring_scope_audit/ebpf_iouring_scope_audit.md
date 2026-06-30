# eBPF/io_uring Scope Audit

- Overall pass: `true`
- Paper scope pass: `true`
- FUSE source pass: `true`
- Trace artifact pass: `true`
- io_uring probe pass: `true`

## Paper Mentions
- `Paper/1_Introduction.tex:52` `ebpf` scoped=`true`: page_qos}. \end{enumerate} The paper is conservative about unsupported mechanisms. It does not claim verified \texttt{O\_DIRECT} NVMe-to-UVM DMA, an \texttt{io\_uring}/eBPF completion bypass, GPU constant-time execution, persistent PCR-bound freshness, true power-loss crash certification, or portability beyond the tested stack. The rest of
- `Paper/1_Introduction.tex:52` `io_uring` scoped=`true`: {fig:first_page_qos}. \end{enumerate} The paper is conservative about unsupported mechanisms. It does not claim verified \texttt{O\_DIRECT} NVMe-to-UVM DMA, an \texttt{io\_uring}/eBPF completion bypass, GPU constant-time execution, persistent PCR-bound freshness, true power-loss crash certification, or portability beyond the tested stack. The r
- `Paper/1_Introduction.tex:52` `completion_bypass` scoped=`true`: qos}. \end{enumerate} The paper is conservative about unsupported mechanisms. It does not claim verified \texttt{O\_DIRECT} NVMe-to-UVM DMA, an \texttt{io\_uring}/eBPF completion bypass, GPU constant-time execution, persistent PCR-bound freshness, true power-loss crash certification, or portability beyond the tested stack. The rest of the paper follows
- `Paper/3_Design.tex:35` `ebpf` scoped=`true`: ests; orange dashed arrows denote optional accelerator or externally provisioned hardware paths. This revision does not claim \texttt{O\_DIRECT} UMA pinning, a completed eBPF completion path, or a closed-loop hardware-QoS controller.} \Description{Architecture diagram of the AEGIS-Q FUSE data path, key envelope, checkpoint, optional CUDA ex
- `Paper/3_Design.tex:122` `ebpf` scoped=`true`: nd executor-local access checks; not a storage-DMA path.\\ \texttt{pread}/\texttt{pwrite} sidecars & Exercised & Backing ciphertext and journal use ordinary POSIX I/O.\\ eBPF tracepoint notification & Not claimed & No eBPF notification path is part of the mounted implementation.\\ QoS telemetry & SQLite recovery only & Workload/CUPTI traces c
- `Paper/3_Design.tex:122` `ebpf` scoped=`true`: MA path.\\ \texttt{pread}/\texttt{pwrite} sidecars & Exercised & Backing ciphertext and journal use ordinary POSIX I/O.\\ eBPF tracepoint notification & Not claimed & No eBPF notification path is part of the mounted implementation.\\ QoS telemetry & SQLite recovery only & Workload/CUPTI traces can drive mounted-FUSE throttling; no foreground
- `Paper/3_Design.tex:131` `completion_bypass` scoped=`true`: 2~ms SQLite p99 and 6.98~MB/s background writes; AEGIS-Q reports 8.15~ms p99 and 3.02~MB/s by throttling elastic writes, not by claiming foreground-inference recovery or completion bypass. The producer-facing interface is conservative. A job enters the elastic lane only when its size and batch shape make GPU execution plausible and the producer supplies
- `Paper/3_Design.tex:122` `tracepoint_notification` scoped=`true`: ecutor-local access checks; not a storage-DMA path.\\ \texttt{pread}/\texttt{pwrite} sidecars & Exercised & Backing ciphertext and journal use ordinary POSIX I/O.\\ eBPF tracepoint notification & Not claimed & No eBPF notification path is part of the mounted implementation.\\ QoS telemetry & SQLite recovery only & Workload/CUPTI traces can drive mounted-FUSE th
- `Paper/4_Evaluation.tex:37` `ebpf` scoped=`true`: n throughput rows and cold-cache rows remain unavailable\\ RQ2/RQ5 UMA & Managed allocation and host/device prefetch complete & Storage-DMA or O\_DIRECT-to-UVM\\ RQ2/RQ5 eBPF & \texttt{nvme\_complete\_rq} diagnostic histogram & Mounted eBPF/io\_uring path\\ RQ3 mounted telemetry & Telemetry reaches the FUSE write-throttle path & End-to-end AI
- `Paper/4_Evaluation.tex:37` `ebpf` scoped=`true`: UMA & Managed allocation and host/device prefetch complete & Storage-DMA or O\_DIRECT-to-UVM\\ RQ2/RQ5 eBPF & \texttt{nvme\_complete\_rq} diagnostic histogram & Mounted eBPF/io\_uring path\\ RQ3 mounted telemetry & Telemetry reaches the FUSE write-throttle path & End-to-end AI p99 controller\\ RQ3 stress & 4-writer tail-latency probe under p
- `Paper/4_Evaluation.tex:37` `io_uring` scoped=`true`: & Managed allocation and host/device prefetch complete & Storage-DMA or O\_DIRECT-to-UVM\\ RQ2/RQ5 eBPF & \texttt{nvme\_complete\_rq} diagnostic histogram & Mounted eBPF/io\_uring path\\ RQ3 mounted telemetry & Telemetry reaches the FUSE write-throttle path & End-to-end AI p99 controller\\ RQ3 stress & 4-writer tail-latency probe under pressure &
- `Paper/4_Evaluation.tex:106` `completion_bypass` scoped=`true`: m pressure reaching the mounted daemon: they record 8 and 5 daemon-throttled flushes, respectively. This remains wiring evidence rather than TensorRT/AI p99 recovery or fast-notification bypass. The app-level result uses SQLite transaction latency. Table~\ref{tab:qos_sqlite_recovery} reports five-run hero medians and one-run kernel QoS controls. AEGIS-Q repo
- `Paper/6_Conclusion.tex:4` `ebpf` scoped=`true`: Thor, AEGIS-Q reports 8.15~ms SQLite p99, zero median misses, and 3.02~MB/s background progress under secure-storage pressure. It does not claim direct NVMe-to-UVM DMA, eBPF/io\_uring completion bypass, GPU constant-time behavior, persistent PCR-bound freshness, power-loss certification, foreground AI p99 recovery, or portability. The lesso
- `Paper/6_Conclusion.tex:4` `io_uring` scoped=`true`: AEGIS-Q reports 8.15~ms SQLite p99, zero median misses, and 3.02~MB/s background progress under secure-storage pressure. It does not claim direct NVMe-to-UVM DMA, eBPF/io\_uring completion bypass, GPU constant-time behavior, persistent PCR-bound freshness, power-loss certification, foreground AI p99 recovery, or portability. The lesson is that
- `Paper/6_Conclusion.tex:4` `completion_bypass` scoped=`true`: eports 8.15~ms SQLite p99, zero median misses, and 3.02~MB/s background progress under secure-storage pressure. It does not claim direct NVMe-to-UVM DMA, eBPF/io\_uring completion bypass, GPU constant-time behavior, persistent PCR-bound freshness, power-loss certification, foreground AI p99 recovery, or portability. The lesson is that edge secure storag

## Required Scope Phrases
- `does not claim verified \texttt{O\_DIRECT} NVMe-to-UVM DMA, an \texttt{io\_uring}/eBPF completion bypass` found=`true`
- `Mounted eBPF/io\_uring path` found=`true`
- `No eBPF notification path is part of the mounted implementation` found=`true`
- `eBPF/io\_uring completion bypass` found=`true`
- `does not claim direct NVMe-to-UVM DMA, eBPF/io\_uring completion bypass` found=`true`

## Artifact Scope
- Trace output: `artifacts/probes/evidence/trace_nvme_lat.out`
- Trace scope: Standalone nvme_complete_rq latency histogram; not a mounted FUSE notification path.
- Probe scope: Standalone prototype/simulation source; not a paper result.
