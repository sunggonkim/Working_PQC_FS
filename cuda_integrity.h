/*
 * cuda_integrity.h — GPU Elastic Lane: Integrity-Plane Batch Interface
 *
 * AEGIS-Q routes INTEGRITY_PLANE jobs (Merkle tree / hash-tree maintenance)
 * to the GPU elastic lane when the AI QoS budget permits.  This header defines
 * the C ABI for batch leaf hashing and tree-level reduce.
 *
 * SHA-256 vs BLAKE3 design note:
 *   SHA-256 is inherently sequential within a single hash (64 rounds, strict
 *   dependency chain) — GPU parallelism comes only from batching independent
 *   leaf hashes, not from within-hash acceleration.
 *
 *   BLAKE3 is designed from the ground up for parallel tree hashing.  Its
 *   internal SIMD/AVX2 structure maps naturally to GPU warp-level parallelism,
 *   and its binary Merkle structure allows the GPU to compute the full tree
 *   in O(log n) rounds rather than reducing serially.  For edge Merkle trees
 *   with ≥256 leaves, BLAKE3 is the preferred integrity algorithm.
 *
 *   CPU baseline (workload_map.csv): SHA-256 leaf throughput ~2.28 M leaves/s
 *   at 1 M batch, near-constant across sizes (single-thread baseline plateau).
 *   GPU target: sharp throughput increase with BLAKE3 or batched SHA-256 as fixed overheads are amortized.
 *
 * System paper scope:
 *   The hash computation itself uses an existing high-quality implementation
 *   (BLAKE3 reference CUDA or cuBLAS-style SHA-256 batch).  The paper's
 *   contribution is the scheduler integration: when to schedule a tree rebuild,
 *   how to overlap it with inference idle cycles, and how to measure the
 *   resulting SM occupancy and UVM pressure.
 *
 * Copyright 2025 AEGIS-Q Authors.  See LICENSE.
 */

#ifndef AEGISQ_CUDA_INTEGRITY_H
#define AEGISQ_CUDA_INTEGRITY_H

#include <stddef.h>
#include <stdint.h>

#define AEGISQ_SHA256_DIGEST_BYTES  32u
#define AEGISQ_BLAKE3_DIGEST_BYTES  32u

/*
 * Hash algorithm selector.
 * SHA256_BATCH: independent leaf SHA-256 hashes, GPU parallelism = batch count.
 * BLAKE3_TREE:  full parallel BLAKE3 Merkle tree, exploits intra-tree parallelism.
 *               Preferred for large batches (≥256 leaves) on GPU.
 */
typedef enum {
    AEGISQ_HASH_SHA256_BATCH = 0,
    AEGISQ_HASH_BLAKE3_TREE  = 1,
} aegisq_hash_algo_t;

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Availability
 * -------------------------------------------------------------------------
 */
int skim_cuda_integrity_available(void);

/* -------------------------------------------------------------------------
 * Batch Leaf Hash
 * -------------------------------------------------------------------------
 * Compute `count` independent hash digests in parallel on the GPU.
 * Each leaf[i] is `leaf_bytes` bytes long (homogeneous batch).
 *
 *   leaves[i * leaf_bytes .. (i+1) * leaf_bytes]  <- input leaf i
 *   digests[i * 32 .. (i+1) * 32]                 <- output digest i
 *
 * algo selects SHA-256 batch or BLAKE3 tree mode.
 *
 * Workload-map target: fill "sha256_leaf, gpu, <batch>" rows.
 * Expected result: high batch throughput vs CPU single-thread plateau ~2.28 M/s.
 *
 * Return: 0 on success, -1 on failure.
 */
int skim_cuda_integrity_leaf_batch(const uint8_t *leaves,
                                   size_t leaf_bytes,
                                   uint8_t *digests,
                                   size_t count,
                                   aegisq_hash_algo_t algo);

/* -------------------------------------------------------------------------
 * Full Merkle Tree Reduce (BLAKE3 tree mode)
 * -------------------------------------------------------------------------
 * Given `leaf_count` pre-hashed leaves, compute the Merkle root.
 * leaf_count must be a power of two (pad with zero-leaves if needed).
 *
 *   leaf_digests[i * 32 ..]  <- leaf hashes (from leaf_batch or pre-computed)
 *   root_out[32]             <- output Merkle root
 *
 * This operation is O(log leaf_count) GPU rounds.
 * For the AEGIS-Q integrity plane, this covers the committed-prefix root
 * that the TPM freshness anchor certifies.
 *
 * Return: 0 on success, -1 on failure or if leaf_count is not a power of two.
 */
int skim_cuda_integrity_merkle_root(const uint8_t *leaf_digests,
                                    size_t leaf_count,
                                    uint8_t *root_out);

/* -------------------------------------------------------------------------
 * Benchmark harness for workload_map.csv
 * -------------------------------------------------------------------------
 * Fills the "sha256_leaf, gpu, <batch>" and "blake3_leaf, gpu, <batch>" rows.
 * Does NOT use CUPTI; call without profiler for latency-accurate measurements.
 */
typedef struct {
    uint64_t wall_ns;
    uint64_t leaf_count;
    double   leaves_per_sec;
    int      gpu_used;
    aegisq_hash_algo_t algo;
} integrity_bench_result_t;

integrity_bench_result_t skim_cuda_integrity_bench_leaf(size_t leaf_count,
                                                        size_t leaf_bytes,
                                                        aegisq_hash_algo_t algo);

/* -------------------------------------------------------------------------
 * SM Occupancy Telemetry (interference experiments)
 * -------------------------------------------------------------------------
 * Same pattern as cuda_pqc.h: call start/stop around a batch operation to
 * capture SM occupancy and UVM counters for the E3/E4 figures.
 * Separate from the latency benchmark path to avoid profiling overhead.
 *
 * Hero Figure scenario:
 *   1. Baseline:     YOLOv8 alone           → p99 ~1.73 ms
 *   2. Naïve static: GPU BLAKE3 always on   → p99 degrades (SM contention)
 *   3. AEGIS-Q:      Elastic lane admits     → p99 restored, tree still built
 */
typedef struct {
    double   mean_sm_occupancy;
    double   peak_sm_occupancy;
    uint64_t uvm_fault_count;
    uint64_t uvm_migration_bytes;
    uint64_t uvm_stall_ns;
} integrity_sm_telemetry_t;

int skim_cuda_integrity_sm_monitor_start(void);
integrity_sm_telemetry_t skim_cuda_integrity_sm_monitor_stop(void);

#ifdef __cplusplus
}
#endif

#endif /* AEGISQ_CUDA_INTEGRITY_H */
