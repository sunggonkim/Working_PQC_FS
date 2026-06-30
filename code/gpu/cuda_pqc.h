/*
 * cuda_pqc.h — GPU Elastic Lane: PQC Key-Plane Batch Interface
 *
 * AEGIS-Q routes KEY_PLANE jobs (ML-KEM-768 keygen/encaps/decaps) to the GPU
 * elastic lane when the AI QoS budget permits.  This header defines the C ABI
 * that the admission controller calls.
 *
 * Design note (system paper scope):
 *   This is a systems paper, not a cryptography paper.  The NTT polynomial
 *   arithmetic in ML-KEM-768 is not implemented from scratch here.  Instead
 *   this interface wraps an existing open-source CUDA PQC kernel (liboqs CUDA
 *   port or cuPQC skeleton), so that the paper's contribution is the scheduler
 *   integration—specifically, when to batch, how to respect the AI QoS budget,
 *   and how to measure GPU SM occupancy with CUPTI during batch execution.
 *
 * Why GPU for ML-KEM?
 *   CPU baseline (workload_map.csv): ~64.8 K keygen/s, near-constant across
 *   batch sizes 1–4096, indicating that the single-threaded CPU baseline is saturated.
 *   ML-KEM-768's NTT polynomial multiply is data-parallel across independent file operations
 *   and structurally suits GPU warp-level parallelism. A GPU batch executor
 *   should show a sharp throughput increase as fixed overheads are amortized,
 *   which the CPU cannot achieve.
 *
 * Measurement target:
 *   Fill the "unsupported" GPU rows in artifacts/workload_map.csv.
 *   Demonstrate SM saturation → YOLOv8 p99 degradation → scheduler recovery.
 *
 * Copyright 2025 AEGIS-Q Authors.  See LICENSE.
 */

#ifndef AEGISQ_CUDA_PQC_H
#define AEGISQ_CUDA_PQC_H

#include <stddef.h>
#include <stdint.h>

/*
 * ML-KEM-768 parameter sizes (FIPS 203).
 *   Public key  : 1184 bytes
 *   Secret key  : 2400 bytes
 *   Ciphertext  : 1088 bytes
 *   Shared secret: 32  bytes
 */
#define MLKEM768_PK_BYTES   1184u
#define MLKEM768_SK_BYTES   2400u
#define MLKEM768_CT_BYTES   1088u
#define MLKEM768_SS_BYTES     32u
#define MLKEM768_KEYGEN_SEED_BYTES 64u
#define MLKEM768_ENCAPS_SEED_BYTES 32u

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Availability probe
 * -------------------------------------------------------------------------
 * Returns 1 if a GPU PQC executor is compiled in and a CUDA device is present.
 * Returns 0 if the GPU path is unsupported (workload_map.csv: "unsupported").
 * The caller must fall back to the CPU path (liboqs) when this returns 0.
 */
int skim_cuda_pqc_available(void);

/* -------------------------------------------------------------------------
 * ML-KEM-768 Batch Operations
 * -------------------------------------------------------------------------
 *
 * All batch functions process `count` independent ML-KEM-768 operations in
 * parallel across GPU warps.  Buffers follow SoA (structure of arrays) layout
 * for coalesced memory access.
 *
 * Memory ownership:
 *   All input/output pointers must be accessible from both CPU and GPU.
 *   On UMA platforms (Jetson) they may be cudaMallocManaged regions;
 *   on discrete GPU they must be cudaMallocHost pinned or pre-staged.
 *   The implementation records UVM fault/migration bytes via skim_cuda_um_stats.
 *
 * Return value: 0 on success, -1 on error or if GPU is unavailable.
 */

/*
 * skim_cuda_mlkem768_keygen_batch()
 *   Generate `count` independent ML-KEM-768 key pairs in parallel.
 *
 *   pk[i*MLKEM768_PK_BYTES .. (i+1)*MLKEM768_PK_BYTES]  <- public key i
 *   sk[i*MLKEM768_SK_BYTES .. (i+1)*MLKEM768_SK_BYTES]  <- secret key i
 *   seeds[i*MLKEM768_KEYGEN_SEED_BYTES ..] <- random seed for key pair i
 *                                             (caller provides from a DRBG)
 *
 * Workload-map target: fill "ml_kem_keygen, gpu, <batch>" rows.
 */
int skim_cuda_mlkem768_keygen_batch(const uint8_t *seeds,
                                    uint8_t *pk,
                                    uint8_t *sk,
                                    size_t count);

/*
 * skim_cuda_mlkem768_encaps_batch()
 *   Encapsulate a shared secret for each of `count` public keys in parallel.
 *
 *   pk[i*MLKEM768_PK_BYTES ..]  <- recipient public key i
 *   ct[i*MLKEM768_CT_BYTES ..]  <- output ciphertext i
 *   ss[i*MLKEM768_SS_BYTES ..]  <- output shared secret i
 *   seeds[i*MLKEM768_ENCAPS_SEED_BYTES ..] <- random coins for encaps i
 *
 * Workload-map target: fill "ml_kem_encaps, gpu, <batch>" rows.
 */
int skim_cuda_mlkem768_encaps_batch(const uint8_t *pk,
                                    const uint8_t *seeds,
                                    uint8_t *ct,
                                    uint8_t *ss,
                                    size_t count);

/*
 * skim_cuda_mlkem768_decaps_batch()
 *   Decapsulate `count` ciphertexts in parallel.
 *
 *   sk[i*MLKEM768_SK_BYTES ..]  <- secret key i
 *   ct[i*MLKEM768_CT_BYTES ..]  <- ciphertext i
 *   ss[i*MLKEM768_SS_BYTES ..]  <- output shared secret i
 *
 * Workload-map target: fill "ml_kem_decaps, gpu, <batch>" rows.
 */
int skim_cuda_mlkem768_decaps_batch(const uint8_t *sk,
                                    const uint8_t *ct,
                                    uint8_t *ss,
                                    size_t count);

/* -------------------------------------------------------------------------
 * Benchmark harness for workload_map.csv
 * -------------------------------------------------------------------------
 * Runs `count` ML-KEM-768 operations and returns wall-clock nanoseconds.
 * Used by workload_map_bench.cpp to fill the "unsupported" GPU rows.
 * Does NOT use CUPTI; call from a non-profiled process for latency numbers.
 */
typedef struct {
    uint64_t wall_ns;      /* wall-clock elapsed                       */
    uint64_t ops;          /* number of operations completed           */
    double   ops_per_sec;  /* throughput                               */
    int      gpu_used;     /* 1 if GPU path was taken, 0 if CPU fallback */
} pqc_bench_result_t;

pqc_bench_result_t skim_cuda_mlkem768_bench_keygen(size_t count);
pqc_bench_result_t skim_cuda_mlkem768_bench_encaps(size_t count);
pqc_bench_result_t skim_cuda_mlkem768_bench_decaps(size_t count);

/* -------------------------------------------------------------------------
 * SM Occupancy Telemetry (for E3/E4 interference experiments)
 * -------------------------------------------------------------------------
 * When CUPTI is available, these functions record SM occupancy samples during
 * a batch operation so that the scheduler can read the AI QoS budget.
 * NOT called from the latency benchmark path — profiling overhead would
 * distort latency numbers.  Called separately in the interference experiments.
 *
 * Usage:
 *   1. Start TensorRT YOLOv8 inference stream.
 *   2. Call skim_cuda_pqc_sm_monitor_start() on a separate thread.
 *   3. Submit ML-KEM batch via skim_cuda_mlkem768_keygen_batch().
 *   4. Call skim_cuda_pqc_sm_monitor_stop() and read occupancy.
 *   5. Compare YOLOv8 p99 with and without scheduler admission.
 */
typedef struct {
    double   mean_sm_occupancy;   /* 0.0–1.0 fraction of SMs active     */
    double   peak_sm_occupancy;
    uint64_t uvm_fault_count;
    uint64_t uvm_migration_bytes;
    uint64_t uvm_stall_ns;
} pqc_sm_telemetry_t;

/* Returns -1 if CUPTI is not available (compile without -DAEGISQ_CUPTI). */
int skim_cuda_pqc_sm_monitor_start(void);
pqc_sm_telemetry_t skim_cuda_pqc_sm_monitor_stop(void);

#ifdef __cplusplus
}
#endif

#endif /* AEGISQ_CUDA_PQC_H */
