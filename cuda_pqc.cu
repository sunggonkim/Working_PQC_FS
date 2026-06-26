/*
 * cuda_pqc.cu — GPU Elastic Lane: ML-KEM-768 Batch Executor
 *
 * Implements the cuda_pqc.h interface using NVIDIA cuPQC SDK.
 * One block per operation, cuPQC handles the NTT polynomial arithmetic
 * with warp-level SIMD parallelism internally.
 *
 * Design (system paper):
 *   The NTT is not implemented here; cuPQC provides it.
 *   AEGIS-Q's contribution is: when to admit a batch (6-rule scheduler),
 *   how to measure SM occupancy alongside inference (CUPTI), and how to
 *   size batches for the Elastic Lane without starving YOLOv8.
 *
 * Build:
 *   nvcc -dlto -arch=native -std=c++17 -O3 -I/usr/local/include \
 *        -L/usr/local/lib -lcupqc-pk -c cuda_pqc.cu -o cuda_pqc.o
 *
 * Copyright 2025 AEGIS-Q Authors. Apache-2.0.
 */

#include "cuda_pqc.h"
#include <cuda_runtime.h>
#include <cupqc/pk.hpp>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <mutex>

using namespace cupqc;

/* -------------------------------------------------------------------------
 * ML-KEM-768 type aliases via cuPQC operator-composition DSL
 * -------------------------------------------------------------------------
 * cuPQC uses a template-composition DSL:
 *   ML_KEM_768() + Function<function::Keygen>() + Block()  → keygen operator
 *   ML_KEM_768() + Function<function::Encaps>() + Block()  → encaps operator
 *   ML_KEM_768() + Function<function::Decaps>() + Block()  → decaps operator
 */
using MLKEM768Key    = decltype(ML_KEM_768() + Function<function::Keygen>() + Block());
using MLKEM768Encaps = decltype(ML_KEM_768() + Function<function::Encaps>() + Block());
using MLKEM768Decaps = decltype(ML_KEM_768() + Function<function::Decaps>() + Block());

/* -------------------------------------------------------------------------
 * GPU kernels — one CUDA block per ML-KEM-768 operation
 * -------------------------------------------------------------------------
 * cuPQC's BlockDim defines the required number of threads per block.
 * Shared memory is pre-allocated per-block using ::shared_memory_size.
 */
__global__ static void keygen_kernel(uint8_t *pk, uint8_t *sk,
                                     uint8_t *workspace, uint8_t *randombytes)
{
    __shared__ uint8_t smem[MLKEM768Key::shared_memory_size];
    int blk = blockIdx.x;
    uint8_t *opk  = pk          + blk * MLKEM768Key::public_key_size;
    uint8_t *osk  = sk          + blk * MLKEM768Key::secret_key_size;
    uint8_t *ork  = randombytes + blk * MLKEM768Key::entropy_size;
    uint8_t *owrk = workspace   + blk * MLKEM768Key::workspace_size;
    MLKEM768Key().execute(opk, osk, ork, owrk, smem);
}

__global__ static void encaps_kernel(uint8_t *ct, uint8_t *ss,
                                     const uint8_t *pk,
                                     uint8_t *workspace, uint8_t *randombytes)
{
    __shared__ uint8_t smem[MLKEM768Encaps::shared_memory_size];
    int blk = blockIdx.x;
    uint8_t *oct  = ct          + blk * MLKEM768Encaps::ciphertext_size;
    uint8_t *oss  = ss          + blk * MLKEM768Encaps::shared_secret_size;
    const uint8_t *opk = pk     + blk * MLKEM768Encaps::public_key_size;
    uint8_t *ork  = randombytes + blk * MLKEM768Encaps::entropy_size;
    uint8_t *owrk = workspace   + blk * MLKEM768Encaps::workspace_size;
    MLKEM768Encaps().execute(oct, oss, opk, ork, owrk, smem);
}

__global__ static void decaps_kernel(uint8_t *ss,
                                     const uint8_t *ct, const uint8_t *sk,
                                     uint8_t *workspace)
{
    __shared__ uint8_t smem[MLKEM768Decaps::shared_memory_size];
    int blk = blockIdx.x;
    uint8_t *oss  = ss       + blk * MLKEM768Decaps::shared_secret_size;
    const uint8_t *oct = ct  + blk * MLKEM768Decaps::ciphertext_size;
    const uint8_t *osk = sk  + blk * MLKEM768Decaps::secret_key_size;
    uint8_t *owrk = workspace + blk * MLKEM768Decaps::workspace_size;
    MLKEM768Decaps().execute(oss, oct, osk, owrk, smem);
}

/* -------------------------------------------------------------------------
 * C interface implementation
 * -------------------------------------------------------------------------
 */

extern "C" int skim_cuda_pqc_available(void)
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0 ? 1 : 0;
}

extern "C" int skim_cuda_mlkem768_keygen_batch(const uint8_t *seeds,
                                               uint8_t *pk,
                                               uint8_t *sk,
                                               size_t count)
{
    if (!skim_cuda_pqc_available() || count == 0 || !seeds || !pk || !sk)
        return -1;

    const size_t pk_total  = MLKEM768Key::public_key_size  * count;
    const size_t sk_total  = MLKEM768Key::secret_key_size  * count;

    uint8_t *d_pk = nullptr, *d_sk = nullptr;
    uint8_t *workspace   = make_workspace<MLKEM768Key>((unsigned)count);
    uint8_t *randombytes = get_entropy<MLKEM768Key>((unsigned)count);

    cudaError_t rc = cudaMalloc(&d_pk, pk_total);
    if (rc != cudaSuccess) goto fail;
    rc = cudaMalloc(&d_sk, sk_total);
    if (rc != cudaSuccess) goto fail;

    /* cuPQC get_entropy() fills device-side random; caller-provided seeds
     * are an additional domain-separation input staged to device. */
    (void)seeds; /* TODO: XOR caller seeds into randombytes for deterministic mode */

    keygen_kernel<<<(unsigned)count, MLKEM768Key::BlockDim>>>(
        d_pk, d_sk, workspace, randombytes);
    rc = cudaGetLastError();
    if (rc != cudaSuccess) goto fail;
    rc = cudaDeviceSynchronize();
    if (rc != cudaSuccess) goto fail;

    cudaMemcpy(pk, d_pk, pk_total, cudaMemcpyDeviceToHost);
    cudaMemcpy(sk, d_sk, sk_total, cudaMemcpyDeviceToHost);

    cudaFree(d_pk); cudaFree(d_sk);
    destroy_workspace(workspace);
    release_entropy(randombytes);
    return 0;

fail:
    if (d_pk) cudaFree(d_pk);
    if (d_sk) cudaFree(d_sk);
    if (workspace)   destroy_workspace(workspace);
    if (randombytes) release_entropy(randombytes);
    return -1;
}

extern "C" int skim_cuda_mlkem768_encaps_batch(const uint8_t *pk,
                                               const uint8_t *seeds,
                                               uint8_t *ct,
                                               uint8_t *ss,
                                               size_t count)
{
    if (!skim_cuda_pqc_available() || count == 0 || !pk || !ct || !ss)
        return -1;

    const size_t pk_total = MLKEM768Encaps::public_key_size  * count;
    const size_t ct_total = MLKEM768Encaps::ciphertext_size  * count;
    const size_t ss_total = MLKEM768Encaps::shared_secret_size * count;

    uint8_t *d_pk = nullptr, *d_ct = nullptr, *d_ss = nullptr;
    uint8_t *workspace   = make_workspace<MLKEM768Encaps>((unsigned)count);
    uint8_t *randombytes = get_entropy<MLKEM768Encaps>((unsigned)count);
    (void)seeds;

    cudaError_t rc = cudaMalloc(&d_pk, pk_total); if (rc != cudaSuccess) goto fail;
    rc = cudaMalloc(&d_ct, ct_total);              if (rc != cudaSuccess) goto fail;
    rc = cudaMalloc(&d_ss, ss_total);              if (rc != cudaSuccess) goto fail;

    cudaMemcpy(d_pk, pk, pk_total, cudaMemcpyHostToDevice);

    encaps_kernel<<<(unsigned)count, MLKEM768Encaps::BlockDim>>>(
        d_ct, d_ss, d_pk, workspace, randombytes);
    rc = cudaGetLastError(); if (rc != cudaSuccess) goto fail;
    rc = cudaDeviceSynchronize(); if (rc != cudaSuccess) goto fail;

    cudaMemcpy(ct, d_ct, ct_total, cudaMemcpyDeviceToHost);
    cudaMemcpy(ss, d_ss, ss_total, cudaMemcpyDeviceToHost);

    cudaFree(d_pk); cudaFree(d_ct); cudaFree(d_ss);
    destroy_workspace(workspace);
    release_entropy(randombytes);
    return 0;

fail:
    if (d_pk) cudaFree(d_pk);
    if (d_ct) cudaFree(d_ct);
    if (d_ss) cudaFree(d_ss);
    if (workspace)   destroy_workspace(workspace);
    if (randombytes) release_entropy(randombytes);
    return -1;
}

extern "C" int skim_cuda_mlkem768_decaps_batch(const uint8_t *sk,
                                               const uint8_t *ct,
                                               uint8_t *ss,
                                               size_t count)
{
    if (!skim_cuda_pqc_available() || count == 0 || !sk || !ct || !ss)
        return -1;

    const size_t sk_total = MLKEM768Decaps::secret_key_size   * count;
    const size_t ct_total = MLKEM768Decaps::ciphertext_size   * count;
    const size_t ss_total = MLKEM768Decaps::shared_secret_size * count;

    uint8_t *d_sk = nullptr, *d_ct = nullptr, *d_ss = nullptr;
    uint8_t *workspace = make_workspace<MLKEM768Decaps>((unsigned)count);

    cudaError_t rc = cudaMalloc(&d_sk, sk_total); if (rc != cudaSuccess) goto fail;
    rc = cudaMalloc(&d_ct, ct_total);              if (rc != cudaSuccess) goto fail;
    rc = cudaMalloc(&d_ss, ss_total);              if (rc != cudaSuccess) goto fail;

    cudaMemcpy(d_sk, sk, sk_total, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ct, ct, ct_total, cudaMemcpyHostToDevice);

    decaps_kernel<<<(unsigned)count, MLKEM768Decaps::BlockDim>>>(
        d_ss, d_ct, d_sk, workspace);
    rc = cudaGetLastError(); if (rc != cudaSuccess) goto fail;
    rc = cudaDeviceSynchronize(); if (rc != cudaSuccess) goto fail;

    cudaMemcpy(ss, d_ss, ss_total, cudaMemcpyDeviceToHost);

    cudaFree(d_sk); cudaFree(d_ct); cudaFree(d_ss);
    destroy_workspace(workspace);
    return 0;

fail:
    if (d_sk) cudaFree(d_sk);
    if (d_ct) cudaFree(d_ct);
    if (d_ss) cudaFree(d_ss);
    if (workspace) destroy_workspace(workspace);
    return -1;
}

/* -------------------------------------------------------------------------
 * Benchmark harness — fills workload_map.csv GPU rows
 * -------------------------------------------------------------------------
 */
static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

extern "C" pqc_bench_result_t skim_cuda_mlkem768_bench_keygen(size_t count)
{
    pqc_bench_result_t r = {0, count, 0.0, 0};
    if (!skim_cuda_pqc_available() || count == 0) return r;

    const size_t pk_total = MLKEM768Key::public_key_size * count;
    const size_t sk_total = MLKEM768Key::secret_key_size * count;

    uint8_t *pk = (uint8_t *)malloc(pk_total);
    uint8_t *sk = (uint8_t *)malloc(sk_total);
    uint8_t *seeds = (uint8_t *)calloc(64 * count, 1); /* bench: zero seeds */
    if (!pk || !sk || !seeds) { free(pk); free(sk); free(seeds); return r; }

    /* Warm up — first CUDA launch pays JIT / driver init cost */
    skim_cuda_mlkem768_keygen_batch(seeds, pk, sk, count < 4 ? count : 4);

    uint64_t t0 = now_ns();
    int rc = skim_cuda_mlkem768_keygen_batch(seeds, pk, sk, count);
    uint64_t t1 = now_ns();

    free(pk); free(sk); free(seeds);

    if (rc == 0) {
        r.wall_ns    = t1 - t0;
        r.ops        = count;
        r.ops_per_sec = (double)count / ((double)(t1 - t0) * 1e-9);
        r.gpu_used   = 1;
    }
    return r;
}

extern "C" pqc_bench_result_t skim_cuda_mlkem768_bench_encaps(size_t count)
{
    pqc_bench_result_t r = {0, count, 0.0, 0};
    if (!skim_cuda_pqc_available() || count == 0) return r;

    const size_t pk_total = MLKEM768Encaps::public_key_size   * count;
    const size_t ct_total = MLKEM768Encaps::ciphertext_size   * count;
    const size_t ss_total = MLKEM768Encaps::shared_secret_size * count;
    const size_t sk_total = MLKEM768Key::secret_key_size * count;

    /* Generate keys first */
    uint8_t *pk    = (uint8_t *)malloc(pk_total);
    uint8_t *sk    = (uint8_t *)malloc(sk_total);
    uint8_t *seeds = (uint8_t *)calloc(64 * count, 1);
    uint8_t *ct    = (uint8_t *)malloc(ct_total);
    uint8_t *ss    = (uint8_t *)malloc(ss_total);
    if (!pk || !sk || !ct || !ss || !seeds) goto free_enc;

    if (skim_cuda_mlkem768_keygen_batch(seeds, pk, sk, count) != 0) goto free_enc;

    /* Warm up */
    skim_cuda_mlkem768_encaps_batch(pk, seeds, ct, ss, count < 4 ? count : 4);

    {
        uint64_t t0 = now_ns();
        int rc = skim_cuda_mlkem768_encaps_batch(pk, seeds, ct, ss, count);
        uint64_t t1 = now_ns();
        if (rc == 0) {
            r.wall_ns     = t1 - t0;
            r.ops         = count;
            r.ops_per_sec = (double)count / ((double)(t1 - t0) * 1e-9);
            r.gpu_used    = 1;
        }
    }

free_enc:
    free(pk); free(sk); free(seeds); free(ct); free(ss);
    return r;
}

extern "C" pqc_bench_result_t skim_cuda_mlkem768_bench_decaps(size_t count)
{
    pqc_bench_result_t r = {0, count, 0.0, 0};
    if (!skim_cuda_pqc_available() || count == 0) return r;

    const size_t pk_total = MLKEM768Key::public_key_size      * count;
    const size_t sk_total = MLKEM768Key::secret_key_size      * count;
    const size_t ct_total = MLKEM768Encaps::ciphertext_size   * count;
    const size_t ss_total = MLKEM768Encaps::shared_secret_size * count;
    uint8_t *seeds = (uint8_t *)calloc(64 * count, 1);
    uint8_t *pk = (uint8_t *)malloc(pk_total);
    uint8_t *sk = (uint8_t *)malloc(sk_total);
    uint8_t *ct = (uint8_t *)malloc(ct_total);
    uint8_t *ss = (uint8_t *)malloc(ss_total);
    if (!pk || !sk || !ct || !ss || !seeds) goto free_dec;

    if (skim_cuda_mlkem768_keygen_batch(seeds, pk, sk, count) != 0) goto free_dec;
    if (skim_cuda_mlkem768_encaps_batch(pk, seeds, ct, ss, count) != 0) goto free_dec;

    /* Warm up */
    skim_cuda_mlkem768_decaps_batch(sk, ct, ss, count < 4 ? count : 4);

    {
        uint64_t t0 = now_ns();
        int rc = skim_cuda_mlkem768_decaps_batch(sk, ct, ss, count);
        uint64_t t1 = now_ns();
        if (rc == 0) {
            r.wall_ns     = t1 - t0;
            r.ops         = count;
            r.ops_per_sec = (double)count / ((double)(t1 - t0) * 1e-9);
            r.gpu_used    = 1;
        }
    }

free_dec:
    free(pk); free(sk); free(ct); free(ss); free(seeds);
    return r;
}

/* SM monitor stubs — full CUPTI implementation is in a separate analysis build */
extern "C" int skim_cuda_pqc_sm_monitor_start(void) { return -1; /* CUPTI not linked */ }
extern "C" pqc_sm_telemetry_t skim_cuda_pqc_sm_monitor_stop(void)
{
    pqc_sm_telemetry_t t = {0};
    return t;
}
