/*
 * code/experiments/bench_gpu_pqc.cu — ML-KEM-768 GPU Batch Benchmark
 *
 * Fills the "unsupported" GPU rows in artifacts/results/placement/workload_map.csv:
 *   ml_kem_keygen, gpu, <batch>, <ops_per_sec>
 *   ml_kem_encaps, gpu, <batch>, <ops_per_sec>
 *   ml_kem_decaps, gpu, <batch>, <ops_per_sec>
 *
 * Batch sizes: 1, 4, 16, 64, 256, 1024, 4096
 *
 * Run WITHOUT profiler for accurate latency numbers (per pro-tip):
 *   ./bench_gpu_pqc | tee artifacts/gpu_pqc_bench.txt
 *
 * Run WITH Nsight for SM occupancy analysis (separate pass):
 *   nsys profile --stats=true ./bench_gpu_pqc
 *
 * Build:
 *   nvcc -dlto -arch=native -std=c++17 -O3 -I/usr/local/include \
 *        -L/usr/local/lib -lcupqc-pk \
 *        code/experiments/bench_gpu_pqc.cu -o bench_gpu_pqc
 *
 * Copyright 2025 AEGIS-Q Authors. Apache-2.0.
 */

#include <cupqc/pk.hpp>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

using namespace cupqc;

/* ---- Type aliases ---- */
using MLKEM768Key    = decltype(ML_KEM_768() + Function<function::Keygen>() + Block());
using MLKEM768Encaps = decltype(ML_KEM_768() + Function<function::Encaps>() + Block());
using MLKEM768Decaps = decltype(ML_KEM_768() + Function<function::Decaps>() + Block());

/* ---- Kernels ---- */
__global__ static void keygen_kernel(uint8_t *pk, uint8_t *sk,
                                     uint8_t *ws, uint8_t *rnd)
{
    __shared__ uint8_t smem[MLKEM768Key::shared_memory_size];
    int b = blockIdx.x;
    MLKEM768Key().execute(pk + b * MLKEM768Key::public_key_size,
                          sk + b * MLKEM768Key::secret_key_size,
                          rnd + b * MLKEM768Key::entropy_size,
                          ws  + b * MLKEM768Key::workspace_size, smem);
}

__global__ static void encaps_kernel(uint8_t *ct, uint8_t *ss,
                                     const uint8_t *pk,
                                     uint8_t *ws, uint8_t *rnd)
{
    __shared__ uint8_t smem[MLKEM768Encaps::shared_memory_size];
    int b = blockIdx.x;
    MLKEM768Encaps().execute(ct  + b * MLKEM768Encaps::ciphertext_size,
                             ss  + b * MLKEM768Encaps::shared_secret_size,
                             pk  + b * MLKEM768Encaps::public_key_size,
                             rnd + b * MLKEM768Encaps::entropy_size,
                             ws  + b * MLKEM768Encaps::workspace_size, smem);
}

__global__ static void decaps_kernel(uint8_t *ss,
                                     const uint8_t *ct, const uint8_t *sk,
                                     uint8_t *ws)
{
    __shared__ uint8_t smem[MLKEM768Decaps::shared_memory_size];
    int b = blockIdx.x;
    MLKEM768Decaps().execute(ss + b * MLKEM768Decaps::shared_secret_size,
                             ct + b * MLKEM768Decaps::ciphertext_size,
                             sk + b * MLKEM768Decaps::secret_key_size,
                             ws + b * MLKEM768Decaps::workspace_size, smem);
}

/* ---- Timing helper ---- */
static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

/* ---- Benchmark one operation at a given batch size ---- */
static void bench_keygen(unsigned batch)
{
    auto *ws  = make_workspace<MLKEM768Key>(batch);
    auto *rnd = get_entropy<MLKEM768Key>(batch);

    uint8_t *d_pk, *d_sk;
    cudaMalloc(&d_pk, MLKEM768Key::public_key_size  * batch);
    cudaMalloc(&d_sk, MLKEM768Key::secret_key_size  * batch);

    /* Warm up */
    keygen_kernel<<<batch, MLKEM768Key::BlockDim>>>(d_pk, d_sk, ws, rnd);
    cudaDeviceSynchronize();

    uint64_t t0 = now_ns();
    keygen_kernel<<<batch, MLKEM768Key::BlockDim>>>(d_pk, d_sk, ws, rnd);
    cudaDeviceSynchronize();
    uint64_t t1 = now_ns();

    double secs = (t1 - t0) * 1e-9;
    printf("ml_kem_keygen,gpu,%u,%.1f,%.3f\n",
           batch, (double)batch / secs, secs * 1e3);

    cudaFree(d_pk); cudaFree(d_sk);
    destroy_workspace(ws); release_entropy(rnd);
}

static void bench_encaps(unsigned batch)
{
    /* Need a batch of public keys first */
    auto *ks_ws  = make_workspace<MLKEM768Key>(batch);
    auto *ks_rnd = get_entropy<MLKEM768Key>(batch);
    uint8_t *d_pk, *d_sk_dummy;
    cudaMalloc(&d_pk,       MLKEM768Key::public_key_size * batch);
    cudaMalloc(&d_sk_dummy, MLKEM768Key::secret_key_size * batch);
    keygen_kernel<<<batch, MLKEM768Key::BlockDim>>>(d_pk, d_sk_dummy, ks_ws, ks_rnd);
    cudaDeviceSynchronize();
    cudaFree(d_sk_dummy);
    destroy_workspace(ks_ws); release_entropy(ks_rnd);

    auto *ws  = make_workspace<MLKEM768Encaps>(batch);
    auto *rnd = get_entropy<MLKEM768Encaps>(batch);
    uint8_t *d_ct, *d_ss;
    cudaMalloc(&d_ct, MLKEM768Encaps::ciphertext_size   * batch);
    cudaMalloc(&d_ss, MLKEM768Encaps::shared_secret_size * batch);

    /* Warm up */
    encaps_kernel<<<batch, MLKEM768Encaps::BlockDim>>>(d_ct, d_ss, d_pk, ws, rnd);
    cudaDeviceSynchronize();

    uint64_t t0 = now_ns();
    encaps_kernel<<<batch, MLKEM768Encaps::BlockDim>>>(d_ct, d_ss, d_pk, ws, rnd);
    cudaDeviceSynchronize();
    uint64_t t1 = now_ns();

    double secs = (t1 - t0) * 1e-9;
    printf("ml_kem_encaps,gpu,%u,%.1f,%.3f\n",
           batch, (double)batch / secs, secs * 1e3);

    cudaFree(d_pk); cudaFree(d_ct); cudaFree(d_ss);
    destroy_workspace(ws); release_entropy(rnd);
}

static void bench_decaps(unsigned batch)
{
    /* Generate key pairs + ciphertexts */
    auto *ks_ws  = make_workspace<MLKEM768Key>(batch);
    auto *ks_rnd = get_entropy<MLKEM768Key>(batch);
    uint8_t *d_pk, *d_sk;
    cudaMalloc(&d_pk, MLKEM768Key::public_key_size * batch);
    cudaMalloc(&d_sk, MLKEM768Key::secret_key_size * batch);
    keygen_kernel<<<batch, MLKEM768Key::BlockDim>>>(d_pk, d_sk, ks_ws, ks_rnd);
    cudaDeviceSynchronize();
    destroy_workspace(ks_ws); release_entropy(ks_rnd);

    auto *en_ws  = make_workspace<MLKEM768Encaps>(batch);
    auto *en_rnd = get_entropy<MLKEM768Encaps>(batch);
    uint8_t *d_ct, *d_ss;
    cudaMalloc(&d_ct, MLKEM768Encaps::ciphertext_size   * batch);
    cudaMalloc(&d_ss, MLKEM768Encaps::shared_secret_size * batch);
    encaps_kernel<<<batch, MLKEM768Encaps::BlockDim>>>(d_ct, d_ss, d_pk, en_ws, en_rnd);
    cudaDeviceSynchronize();
    cudaFree(d_pk); cudaFree(d_ss);
    destroy_workspace(en_ws); release_entropy(en_rnd);

    auto *ws = make_workspace<MLKEM768Decaps>(batch);
    uint8_t *d_ss2;
    cudaMalloc(&d_ss2, MLKEM768Decaps::shared_secret_size * batch);

    /* Warm up */
    decaps_kernel<<<batch, MLKEM768Decaps::BlockDim>>>(d_ss2, d_ct, d_sk, ws);
    cudaDeviceSynchronize();

    uint64_t t0 = now_ns();
    decaps_kernel<<<batch, MLKEM768Decaps::BlockDim>>>(d_ss2, d_ct, d_sk, ws);
    cudaDeviceSynchronize();
    uint64_t t1 = now_ns();

    double secs = (t1 - t0) * 1e-9;
    printf("ml_kem_decaps,gpu,%u,%.1f,%.3f\n",
           batch, (double)batch / secs, secs * 1e3);

    cudaFree(d_sk); cudaFree(d_ct); cudaFree(d_ss2);
    destroy_workspace(ws);
}

int main(void)
{
    int dev; cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    printf("# GPU: %s  SM%d.%d  CUDA %d\n",
           prop.name, prop.major, prop.minor,
           CUDART_VERSION);
    printf("# MLKEM768Key BlockDim=%d  shared_memory=%zu bytes\n",
           (int)MLKEM768Key::BlockDim.x,
           (size_t)MLKEM768Key::shared_memory_size);
    printf("#\n");
    printf("# op,executor,batch,ops_per_sec,wall_ms\n");
    printf("#\n");

    unsigned batches[] = {1, 4, 16, 64, 256, 1024, 4096};
    int nb = sizeof(batches) / sizeof(batches[0]);

    for (int i = 0; i < nb; i++) bench_keygen(batches[i]);
    printf("\n");
    for (int i = 0; i < nb; i++) bench_encaps(batches[i]);
    printf("\n");
    for (int i = 0; i < nb; i++) bench_decaps(batches[i]);

    return 0;
}
