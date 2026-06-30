/*
 * cuda_integrity.cu — GPU Elastic Lane: Integrity-Plane Batch Executor
 *
 * Implements cuda_integrity.h utilizing native CUDA SHA-256 device implementation.
 * Uses a bottom-up O(log N) parallel reduction scheme for Merkle root computation.
 *
 * Features Leaf & Internal node domain separation for Merkle tree security:
 *   Leaf:     SHA-256(0x00 || 16-byte leaf entry)  (17 bytes input)
 *   Internal: SHA-256(0x01 || left_digest || right_digest) (65 bytes input)
 *
 * Copyright 2025 AEGIS-Q Authors. Apache-2.0.
 */

#include "cuda_integrity.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

// SHA-256 constants
__constant__ static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define Ch(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define Maj(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define Sigma0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define Sigma1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define sigma0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define sigma1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))

__device__ static void sha256_compress(uint32_t state[8], const uint32_t block[16]) {
    uint32_t W[64];
    for (int i = 0; i < 16; i++) W[i] = block[i];
    for (int i = 16; i < 64; i++) {
        W[i] = sigma1(W[i - 2]) + W[i - 7] + sigma0(W[i - 15]) + W[i - 16];
    }

    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    for (int i = 0; i < 64; i++) {
        uint32_t T1 = h + Sigma1(e) + Ch(e, f, g) + K[i] + W[i];
        uint32_t T2 = Sigma0(a) + Maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + T1;
        d = c;
        c = b;
        b = a;
        a = T1 + T2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

/* Generalized block hashing helper supporting custom lengths */
__device__ static void sha256_hash_block(const uint8_t *in, size_t in_len, uint8_t out[32]) {
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    uint32_t block[16];
    size_t offset = 0;
    
    // Process full 64-byte chunks
    while (in_len - offset >= 64) {
        for (int i = 0; i < 16; i++) {
            size_t idx = offset + i * 4;
            block[i] = ((uint32_t)in[idx] << 24) | ((uint32_t)in[idx+1] << 16) |
                        ((uint32_t)in[idx+2] << 8) | (uint32_t)in[idx+3];
        }
        sha256_compress(state, block);
        offset += 64;
    }

    // Process remaining chunk with padding
    size_t rem = in_len - offset;
    uint8_t temp[128] = {0};
    for (size_t i = 0; i < rem; i++) {
        temp[i] = in[offset + i];
    }
    temp[rem] = 0x80; // Pad bit

    size_t total_blocks = (rem < 56) ? 1 : 2;
    uint64_t total_bits = (uint64_t)in_len * 8;

    // Serialization of total bits
    size_t len_offset = total_blocks * 64 - 8;
    for (int i = 0; i < 8; i++) {
        temp[len_offset + i] = (uint8_t)(total_bits >> (8 * (7 - i)));
    }

    for (size_t b = 0; b < total_blocks; b++) {
        for (int i = 0; i < 16; i++) {
            size_t idx = b * 64 + i * 4;
            block[i] = ((uint32_t)temp[idx] << 24) | ((uint32_t)temp[idx+1] << 16) |
                        ((uint32_t)temp[idx+2] << 8) | (uint32_t)temp[idx+3];
        }
        sha256_compress(state, block);
    }

    // Write state back to digests
    for (int i = 0; i < 8; i++) {
        out[i*4]   = (uint8_t)(state[i] >> 24);
        out[i*4+1] = (uint8_t)(state[i] >> 16);
        out[i*4+2] = (uint8_t)(state[i] >> 8);
        out[i*4+3] = (uint8_t)(state[i]);
    }
}

/* -------------------------------------------------------------------------
 * CUDA Kernels
 * -------------------------------------------------------------------------
 */

__global__ void sha256_batch_kernel(const uint8_t *in, size_t in_bytes,
                                    uint8_t *out, size_t count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    // Apply domain separation prefix 0x00 for leaf hashing
    uint8_t leaf_buf[128]; // Max leaf entry fits easily
    size_t hash_len = in_bytes + 1;
    leaf_buf[0] = 0x00;
    for (size_t i = 0; i < in_bytes; i++) {
        leaf_buf[1 + i] = in[tid * in_bytes + i];
    }
    sha256_hash_block(leaf_buf, hash_len, out + tid * 32);
}

__global__ void merkle_reduce_kernel(const uint8_t *in, uint8_t *out, size_t count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    // Apply domain separation prefix 0x01 for internal node reduction
    uint8_t node_buf[65];
    node_buf[0] = 0x01;
    for (int i = 0; i < 64; i++) {
        node_buf[1 + i] = in[tid * 64 + i];
    }
    sha256_hash_block(node_buf, 65, out + tid * 32);
}

/* -------------------------------------------------------------------------
 * C ABI Implementation
 * -------------------------------------------------------------------------
 */

extern "C" int skim_cuda_integrity_available(void)
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0 ? 1 : 0;
}

extern "C" int skim_cuda_integrity_leaf_batch(const uint8_t *leaves,
                                              size_t leaf_bytes,
                                              uint8_t *digests,
                                              size_t count,
                                              aegisq_hash_algo_t algo)
{
    if (!skim_cuda_integrity_available() || count == 0 || !leaves || !digests)
        return -1;

    (void)algo;

    const size_t in_total = leaf_bytes * count;
    const size_t out_total = 32 * count;

    uint8_t *d_in = nullptr, *d_out = nullptr;
    cudaError_t rc = cudaMalloc(&d_in, in_total);
    if (rc != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_in failed: %s (%d)\n", cudaGetErrorString(rc), (int)rc);
        goto fail;
    }
    rc = cudaMalloc(&d_out, out_total);
    if (rc != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_out failed: %s (%d)\n", cudaGetErrorString(rc), (int)rc);
        goto fail;
    }

    rc = cudaMemcpy(d_in, leaves, in_total, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy H2D failed: %s (%d)\n", cudaGetErrorString(rc), (int)rc);
        goto fail;
    }

    {
        unsigned threads = 256;
        unsigned blocks = (count + threads - 1) / threads;
        sha256_batch_kernel<<<blocks, threads>>>(d_in, leaf_bytes, d_out, count);
    }

    rc = cudaGetLastError();
    if (rc != cudaSuccess) {
        fprintf(stderr, "Kernel launch error check failed: %s (%d)\n", cudaGetErrorString(rc), (int)rc);
        goto fail;
    }
    rc = cudaDeviceSynchronize();
    if (rc != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s (%d)\n", cudaGetErrorString(rc), (int)rc);
        goto fail;
    }

    rc = cudaMemcpy(digests, d_out, out_total, cudaMemcpyDeviceToHost);
    if (rc != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2H failed: %s (%d)\n", cudaGetErrorString(rc), (int)rc);
        goto fail;
    }

    cudaFree(d_in); cudaFree(d_out);
    return 0;

fail:
    if (d_in) cudaFree(d_in);
    if (d_out) cudaFree(d_out);
    return -1;
}

extern "C" int skim_cuda_integrity_merkle_root(const uint8_t *leaf_digests,
                                               size_t leaf_count,
                                               uint8_t *root_out)
{
    if (!skim_cuda_integrity_available() || leaf_count == 0 || !leaf_digests || !root_out)
        return -1;

    // Check power of two
    if ((leaf_count & (leaf_count - 1)) != 0)
        return -1;

    uint8_t *d_buf1 = nullptr, *d_buf2 = nullptr;
    cudaError_t rc = cudaMalloc(&d_buf1, leaf_count * 32);
    if (rc != cudaSuccess) goto fail;
    rc = cudaMalloc(&d_buf2, (leaf_count / 2) * 32);
    if (rc != cudaSuccess) goto fail;

    rc = cudaMemcpy(d_buf1, leaf_digests, leaf_count * 32, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) goto fail;

    {
        size_t current_count = leaf_count;
        uint8_t *d_in = d_buf1;
        uint8_t *d_out = d_buf2;

        while (current_count > 1) {
            size_t next_count = current_count / 2;
            unsigned threads = 256;
            unsigned blocks = (next_count + threads - 1) / threads;

            merkle_reduce_kernel<<<blocks, threads>>>(d_in, d_out, next_count);

            rc = cudaGetLastError();
            if (rc != cudaSuccess) goto fail;
            rc = cudaDeviceSynchronize();
            if (rc != cudaSuccess) goto fail;

            current_count = next_count;
            // Swap buffers
            uint8_t *temp = d_in;
            d_in = d_out;
            d_out = temp;
        }

        rc = cudaMemcpy(root_out, d_in, 32, cudaMemcpyDeviceToHost);
        if (rc != cudaSuccess) goto fail;
    }

    cudaFree(d_buf1); cudaFree(d_buf2);
    return 0;

fail:
    fprintf(stderr, "GPU integrity Merkle root failed: CUDA error = %s (%d)\n", cudaGetErrorString(rc), (int)rc);
    if (d_buf1) cudaFree(d_buf1);
    if (d_buf2) cudaFree(d_buf2);
    return -1;
}

/* -------------------------------------------------------------------------
 * Benchmark Harness
 * -------------------------------------------------------------------------
 */
static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

extern "C" integrity_bench_result_t skim_cuda_integrity_bench_leaf(size_t leaf_count,
                                                                    size_t leaf_bytes,
                                                                    aegisq_hash_algo_t algo)
{
    integrity_bench_result_t r = {0, leaf_count, 0.0, 0, algo};
    if (!skim_cuda_integrity_available() || leaf_count == 0) return r;

    uint8_t *leaves = (uint8_t *)calloc(leaf_bytes * leaf_count, 1);
    uint8_t *digests = (uint8_t *)malloc(32 * leaf_count);
    if (!leaves || !digests) { free(leaves); free(digests); return r; }

    // Warm up
    skim_cuda_integrity_leaf_batch(leaves, leaf_bytes, digests, leaf_count < 4 ? leaf_count : 4, algo);

    uint64_t t0 = now_ns();
    int rc = skim_cuda_integrity_leaf_batch(leaves, leaf_bytes, digests, leaf_count, algo);
    uint64_t t1 = now_ns();

    free(leaves); free(digests);

    if (rc == 0) {
        r.wall_ns = t1 - t0;
        r.leaf_count = leaf_count;
        r.leaves_per_sec = (double)leaf_count / ((double)(t1 - t0) * 1e-9);
        r.gpu_used = 1;
    }
    return r;
}

/* Telemetry placeholders */
extern "C" int skim_cuda_integrity_sm_monitor_start(void) { return -1; }
extern "C" integrity_sm_telemetry_t skim_cuda_integrity_sm_monitor_stop(void)
{
    integrity_sm_telemetry_t t = {0};
    return t;
}
