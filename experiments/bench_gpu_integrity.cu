/*
 * experiments/bench_gpu_integrity.cu — Correctness Verification and Fine-Grained Benchmark
 *
 * Verifies correctness and measures execution breakdowns of leaf hashing and Merkle root.
 *
 * Copyright 2025 AEGIS-Q Authors. Apache-2.0.
 */

#include "cuda_integrity.h"
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <thread>
#include <numeric>
#include <cmath>
#include <ctime>
#include <mutex>
#include <condition_variable>

#ifndef _WIN32
#include <pthread.h>
#endif

// Extern declarations of GPU kernels for fine-grained profiling
extern __global__ void sha256_batch_kernel(const uint8_t *in, size_t in_bytes, uint8_t *out, size_t count);
extern __global__ void merkle_reduce_kernel(const uint8_t *in, uint8_t *out, size_t count);

// Synchronization Barrier for clean multi-threaded CPU benchmarking without launch overhead
struct Barrier {
    std::mutex mutex;
    std::condition_variable cv;
    size_t count;
    size_t initial;
    size_t generation;

    Barrier(size_t count) : count(count), initial(count), generation(0) {}

    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        size_t gen = generation;
        if (--count == 0) {
            generation++;
            count = initial;
            cv.notify_all();
        } else {
            cv.wait(lock, [this, gen] { return gen != generation; });
        }
    }
};

static void pin_thread_to_core(std::thread &th, int core_id) {
#ifndef _WIN32
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_t thread_handle = th.native_handle();
    pthread_setaffinity_np(thread_handle, sizeof(cpu_set_t), &cpuset);
#endif
}

// In-memory entry format matching pqc_anchor.c
typedef struct {
    uint64_t file_id;
    uint64_t max_generation;
} committed_entry_t;

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

// CPU helper to compute SHA-256 of buffer
static int cpu_sha256(const uint8_t *in, size_t len, uint8_t out[32]) {
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    if (!ctx) return -1;
    int ok = EVP_DigestInit_ex(ctx, EVP_sha256(), NULL) == 1 &&
             EVP_DigestUpdate(ctx, in, len) == 1 &&
             EVP_DigestFinal_ex(ctx, out, NULL) == 1;
    EVP_MD_CTX_free(ctx);
    return ok ? 0 : -1;
}

// CPU bottom-up Merkle root helper with domain separation (0x00 leaf, 0x01 node)
static int cpu_merkle_root(const committed_entry_t *entries, size_t count, uint8_t out_root[32]) {
    if (count == 0) {
        return cpu_sha256(NULL, 0, out_root);
    }
    size_t n = 1;
    while (n < count) { n <<= 1; }

    uint8_t *leaf_digests = (uint8_t *)calloc(n, 32);
    if (!leaf_digests) return -1;

    for (size_t i = 0; i < count; i++) {
        uint8_t leaf_buf[17];
        leaf_buf[0] = 0x00; // leaf domain sep
        uint64_t fid = entries[i].file_id;
        uint64_t gen = entries[i].max_generation;
        for (int b = 0; b < 8; ++b) { leaf_buf[1 + b]   = (uint8_t)(fid >> (8 * b)); }
        for (int b = 0; b < 8; ++b) { leaf_buf[1 + 8 + b] = (uint8_t)(gen >> (8 * b)); }

        cpu_sha256(leaf_buf, 17, leaf_digests + i * 32);
    }

    size_t current_count = n;
    uint8_t *buf = (uint8_t *)malloc(n * 32);
    if (!buf) { free(leaf_digests); return -1; }
    memcpy(buf, leaf_digests, n * 32);
    free(leaf_digests);

    while (current_count > 1) {
        size_t next_count = current_count / 2;
        for (size_t i = 0; i < next_count; i++) {
            uint8_t node_buf[65];
            node_buf[0] = 0x01; // internal node domain sep
            memcpy(node_buf + 1, buf + i * 64, 64);
            cpu_sha256(node_buf, 65, buf + i * 32);
        }
        current_count = next_count;
    }

    memcpy(out_root, buf, 32);
    free(buf);
    return 0;
}

// Correctness Verification Functions (Return true on pass, false on fail)

static bool run_leaf_parity_test_size(size_t leaf_size) {
    std::vector<uint8_t> test_input(leaf_size);
    for (size_t i = 0; i < leaf_size; i++) test_input[i] = (uint8_t)(0xAB + i);

    uint8_t cpu_digest[32];
    std::vector<uint8_t> cpu_buf(leaf_size + 1);
    cpu_buf[0] = 0x00; // Domain separation
    memcpy(cpu_buf.data() + 1, test_input.data(), leaf_size);
    cpu_sha256(cpu_buf.data(), leaf_size + 1, cpu_digest);

    uint8_t gpu_digest[32];
    int rc = skim_cuda_integrity_leaf_batch(test_input.data(), leaf_size, gpu_digest, 1, AEGISQ_HASH_SHA256_BATCH);
    if (rc != 0) {
        printf("  [FAIL] GPU Leaf parity (size %zu): execution failed.\n", leaf_size);
        return false;
    }

    if (memcmp(cpu_digest, gpu_digest, 32) == 0) {
        printf("  [PASS] GPU Leaf parity matches OpenSSL (size %zu, with 0x00 prefix)\n", leaf_size);
        return true;
    } else {
        printf("  [FAIL] GPU Leaf parity mismatch (size %zu)!\n", leaf_size);
        return false;
    }
}

static bool run_merkle_parity_test(size_t count) {
    std::vector<committed_entry_t> entries(count);
    for (size_t i = 0; i < count; i++) {
        entries[i].file_id = i + 100;
        entries[i].max_generation = i * 2 + 5;
    }

    uint8_t cpu_root[32];
    cpu_merkle_root(entries.data(), count, cpu_root);

    size_t n = 1;
    while (n < count) { n <<= 1; }
    
    std::vector<uint8_t> leaves_buf(count * 16);
    for (size_t i = 0; i < count; i++) {
        uint64_t fid = entries[i].file_id;
        uint64_t gen = entries[i].max_generation;
        for (int b = 0; b < 8; ++b) { leaves_buf[i * 16 + b]   = (uint8_t)(fid >> (8 * b)); }
        for (int b = 0; b < 8; ++b) { leaves_buf[i * 16 + 8 + b] = (uint8_t)(gen >> (8 * b)); }
    }

    std::vector<uint8_t> leaf_digests(n * 32, 0); // padding leaves are 0
    int rc = skim_cuda_integrity_leaf_batch(leaves_buf.data(), 16, leaf_digests.data(), count, AEGISQ_HASH_SHA256_BATCH);
    if (rc != 0) {
        printf("  [FAIL] Merkle parity count %zu: Leaf batch execution failed.\n", count);
        return false;
    }

    uint8_t gpu_root[32];
    rc = skim_cuda_integrity_merkle_root(leaf_digests.data(), n, gpu_root);
    if (rc != 0) {
        printf("  [FAIL] Merkle parity count %zu: Root reduction failed.\n", count);
        return false;
    }

    if (memcmp(cpu_root, gpu_root, 32) == 0) {
        printf("  [PASS] Merkle parity matches for count %zu (Leaves: %zu, Tree: %zu)\n", count, count, n);
        return true;
    } else {
        printf("  [FAIL] Merkle parity mismatch for count %zu!\n", count);
        return false;
    }
}

static bool run_empty_tree_test() {
    uint8_t root[32];
    int rc = cpu_merkle_root(nullptr, 0, root);
    if (rc != 0) {
        printf("  [FAIL] Empty tree test: CPU hash failed\n");
        return false;
    }

    // Expected SHA-256 of empty string: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    const uint8_t expected[32] = {
        0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9, 0x24,
        0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52, 0xb8, 0x55
    };

    if (memcmp(root, expected, 32) == 0) {
        printf("  [PASS] Empty tree (count=0) matches standard empty SHA-256 string hash\n");
        return true;
    } else {
        printf("  [FAIL] Empty tree mismatch!\n");
        return false;
    }
}

// Pure CPU SHA-256 Implementation (to avoid OpenSSL EVP global lock contention)
static const uint32_t cpu_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

#define CPU_ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CPU_Ch(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define CPU_Maj(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define CPU_Sigma0(x) (CPU_ROTR(x, 2) ^ CPU_ROTR(x, 13) ^ CPU_ROTR(x, 22))
#define CPU_Sigma1(x) (CPU_ROTR(x, 6) ^ CPU_ROTR(x, 11) ^ CPU_ROTR(x, 25))
#define CPU_sigma0(x) (CPU_ROTR(x, 7) ^ CPU_ROTR(x, 18) ^ ((x) >> 3))
#define CPU_sigma1(x) (CPU_ROTR(x, 17) ^ CPU_ROTR(x, 19) ^ ((x) >> 10))

static void cpu_sha256_compress_pure(uint32_t state[8], const uint32_t block[16]) {
    uint32_t W[64];
    for (int i = 0; i < 16; i++) W[i] = block[i];
    for (int i = 16; i < 64; i++) {
        W[i] = CPU_sigma1(W[i - 2]) + W[i - 7] + CPU_sigma0(W[i - 15]) + W[i - 16];
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
        uint32_t T1 = h + CPU_Sigma1(e) + CPU_Ch(e, f, g) + cpu_K[i] + W[i];
        uint32_t T2 = CPU_Sigma0(a) + CPU_Maj(a, b, c);
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

static void cpu_sha256_pure(const uint8_t *in, size_t in_len, uint8_t out[32]) {
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    uint32_t block[16];
    size_t offset = 0;
    
    while (in_len - offset >= 64) {
        for (int i = 0; i < 16; i++) {
            size_t idx = offset + i * 4;
            block[i] = ((uint32_t)in[idx] << 24) | ((uint32_t)in[idx+1] << 16) |
                        ((uint32_t)in[idx+2] << 8) | (uint32_t)in[idx+3];
        }
        cpu_sha256_compress_pure(state, block);
        offset += 64;
    }

    size_t rem = in_len - offset;
    uint8_t temp[128] = {0};
    for (size_t i = 0; i < rem; i++) {
        temp[i] = in[offset + i];
    }
    temp[rem] = 0x80;

    size_t total_blocks = (rem < 56) ? 1 : 2;
    uint64_t total_bits = (uint64_t)in_len * 8;

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
        cpu_sha256_compress_pure(state, block);
    }

    for (int i = 0; i < 8; i++) {
        out[i*4]   = (uint8_t)(state[i] >> 24);
        out[i*4+1] = (uint8_t)(state[i] >> 16);
        out[i*4+2] = (uint8_t)(state[i] >> 8);
        out[i*4+3] = (uint8_t)(state[i]);
    }
}

// Work-partitioned CPU thread worker (fixed: stride-based work distribution)
static void cpu_thread_worker_partitioned(
    int thread_id, int num_threads, size_t batch, int iterations,
    Barrier *start_barrier, Barrier *end_barrier)
{
    uint8_t leaf_buf[17];
    leaf_buf[0] = 0x00;
    memset(leaf_buf + 1, 0xAA, 16);
    uint8_t digest[32];

    start_barrier->wait();

    // Stride-based distribution: each thread processes every num_threads-th item
    // This guarantees no thread is starved even for batch=1
    for (int it = 0; it < iterations; it++) {
        for (size_t i = thread_id; i < batch; i += num_threads) {
            cpu_sha256_pure(leaf_buf, 17, digest);
        }
    }

    end_barrier->wait();
}

// Single-thread CPU worker
static void cpu_thread_worker_single(size_t count, int iterations) {
    uint8_t leaf_buf[17];
    leaf_buf[0] = 0x00;
    memset(leaf_buf + 1, 0xAA, 16);
    uint8_t digest[32];
    for (int it = 0; it < iterations; it++) {
        for (size_t i = 0; i < count; i++) {
            cpu_sha256_pure(leaf_buf, 17, digest);
        }
    }
}

// CPU full-tree computation: leaf hashing + sequential reduction
static double cpu_compute_full_tree_blocking(
    const uint8_t *leaves, size_t leaf_count, size_t leaf_bytes, uint8_t out_root[32])
{
    uint64_t t0 = now_ns();
    
    // 1. Leaf hashing
    size_t n = 1;
    while (n < leaf_count) { n <<= 1; }
    uint8_t *digests = (uint8_t *)malloc(n * 32);
    for (size_t i = 0; i < leaf_count; i++) {
        uint8_t leaf_buf[257];
        leaf_buf[0] = 0x00;
        memcpy(leaf_buf + 1, leaves + i * leaf_bytes, leaf_bytes);
        cpu_sha256_pure(leaf_buf, leaf_bytes + 1, digests + i * 32);
    }
    memset(digests + leaf_count * 32, 0, (n - leaf_count) * 32);
    
    // 2. Merkle reduction
    size_t current_count = n;
    uint8_t *buf = digests;
    while (current_count > 1) {
        size_t next_count = current_count / 2;
        for (size_t i = 0; i < next_count; i++) {
            uint8_t node_buf[65];
            node_buf[0] = 0x01;
            memcpy(node_buf + 1, buf + i * 64, 64);
            cpu_sha256_pure(node_buf, 65, buf + i * 32);
        }
        current_count = next_count;
    }
    
    memcpy(out_root, buf, 32);
    free(digests);
    
    uint64_t t1 = now_ns();
    return (t1 - t0) * 1e-6; // ms
}

// Parallel full-tree: each thread hashes leaf_count/num_threads leaves, then single-threaded reduction
static double cpu_compute_full_tree_parallel(
    const uint8_t *leaves, size_t leaf_count, size_t leaf_bytes, int num_threads, 
    uint8_t out_root[32])
{
    size_t n = 1;
    while (n < leaf_count) { n <<= 1; }
    uint8_t *digests = (uint8_t *)malloc(n * 32);
    
    Barrier leaf_barrier(num_threads + 1);
    
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    
    // Launch leaf-hashing threads
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([t, num_threads, leaf_count, leaf_bytes, leaves, digests, &leaf_barrier]() {
            for (size_t i = t; i < leaf_count; i += num_threads) {
                uint8_t leaf_buf[257];
                leaf_buf[0] = 0x00;
                memcpy(leaf_buf + 1, leaves + i * leaf_bytes, leaf_bytes);
                cpu_sha256_pure(leaf_buf, leaf_bytes + 1, digests + i * 32);
            }
            leaf_barrier.wait();
        });
        pin_thread_to_core(threads.back(), t % std::thread::hardware_concurrency());
    }
    
    uint64_t t0 = now_ns();
    leaf_barrier.wait();  // Wait for all threads to finish leaf hashing
    
    // Single-threaded reduction (sequential, main thread only)
    memset(digests + leaf_count * 32, 0, (n - leaf_count) * 32);
    size_t current_count = n;
    uint8_t *buf = digests;
    while (current_count > 1) {
        size_t next_count = current_count / 2;
        for (size_t i = 0; i < next_count; i++) {
            uint8_t node_buf[65];
            node_buf[0] = 0x01;
            memcpy(node_buf + 1, buf + i * 64, 64);
            cpu_sha256_pure(node_buf, 65, buf + i * 32);
        }
        current_count = next_count;
    }
    
    memcpy(out_root, buf, 32);
    
    uint64_t t1 = now_ns();
    
    // Wait for all threads to finish
    for (auto &t : threads) t.join();
    
    free(digests);
    
    return (t1 - t0) * 1e-6; // ms
}

// Performance benchmark with full breakdowns and correct thread-pooling proxy
static void run_performance_bench(size_t batch) {
    size_t leaf_bytes = 16;
    size_t total_bytes = batch * leaf_bytes;

    uint8_t *leaves = (uint8_t *)malloc(total_bytes);
    uint8_t *digests = (uint8_t *)malloc(32 * batch);
    memset(leaves, 0xAB, total_bytes);

    // Warm up GPU
    skim_cuda_integrity_leaf_batch(leaves, leaf_bytes, digests, batch, AEGISQ_HASH_SHA256_BATCH);
    cudaDeviceSynchronize();

    const int iterations = 100;

    // CUDA Event based fine-grained profiling
    cudaEvent_t start_ev, stop_ev;
    cudaEventCreate(&start_ev);
    cudaEventCreate(&stop_ev);

    auto measure_cuda_time = [&](auto&& fn) -> double {
        cudaEventRecord(start_ev);
        fn();
        cudaEventRecord(stop_ev);
        cudaEventSynchronize(stop_ev);
        float ms = 0;
        cudaEventElapsedTime(&ms, start_ev, stop_ev);
        return (double)ms;
    };

    // 1. Measure Allocation (Cold)
    double alloc_ms = 0;
    {
        uint8_t *d_temp_in = nullptr;
        uint8_t *d_temp_out = nullptr;
        uint64_t t0 = now_ns();
        cudaMalloc(&d_temp_in, total_bytes);
        cudaMalloc(&d_temp_out, 32 * batch);
        uint64_t t1 = now_ns();
        alloc_ms = (t1 - t0) * 1e-6;
        cudaFree(d_temp_in);
        cudaFree(d_temp_out);
    }

    // Allocate persistent buffers
    uint8_t *d_in = nullptr;
    uint8_t *d_out = nullptr;
    cudaMalloc(&d_in, total_bytes);
    cudaMalloc(&d_out, 32 * batch);

    size_t n = 1;
    while (n < batch) { n <<= 1; }

    uint8_t *d_red_in = nullptr;
    uint8_t *d_red_out = nullptr;
    if (n > 1) {
        cudaMalloc(&d_red_in, n * 32);
        cudaMalloc(&d_red_out, (n / 2) * 32);
    }

    // 2. Measure H2D Staging
    double h2d_ms = measure_cuda_time([&]() {
        cudaMemcpy(d_in, leaves, total_bytes, cudaMemcpyHostToDevice);
    });

    // 3. Measure Leaf Hashing Kernel
    unsigned int threads_per_block = 256;
    unsigned int blocks = (batch + threads_per_block - 1) / threads_per_block;
    double leaf_kernel_ms = measure_cuda_time([&]() {
        sha256_batch_kernel<<<blocks, threads_per_block>>>(d_in, leaf_bytes, d_out, batch);
    });

    // 4. Measure D2H Staging
    double d2h_ms = measure_cuda_time([&]() {
        cudaMemcpy(digests, d_out, 32 * batch, cudaMemcpyDeviceToHost);
    });

    // 5. Measure Reduction Kernel (from pre-hashed leaves)
    double reduction_kernel_ms = 0;
    if (n > 1) {
        cudaMemcpy(d_red_in, digests, batch * 32, cudaMemcpyHostToDevice);

        size_t current_count = n;
        uint8_t *d_curr_in = d_red_in;
        uint8_t *d_curr_out = d_red_out;
        
        uint64_t t_red_0 = now_ns();
        while (current_count > 1) {
            size_t next_count = current_count / 2;
            unsigned blocks_red = (next_count + 255) / 256;
            merkle_reduce_kernel<<<blocks_red, 256>>>(d_curr_in, d_curr_out, next_count);
            cudaDeviceSynchronize();
            current_count = next_count;
            uint8_t *temp = d_curr_in;
            d_curr_in = d_curr_out;
            d_curr_out = temp;
        }
        uint64_t t_red_1 = now_ns();
        reduction_kernel_ms = (t_red_1 - t_red_0) * 1e-6;
    }

    // 6a. Steady State Leaf Hashing (excluding host/device alloc/free)
    std::vector<double> steady_trials_ms(iterations);
    for (int i = 0; i < iterations; i++) {
        uint64_t t_steady_0 = now_ns();
        cudaMemcpy(d_in, leaves, total_bytes, cudaMemcpyHostToDevice);
        sha256_batch_kernel<<<blocks, threads_per_block>>>(d_in, leaf_bytes, d_out, batch);
        cudaMemcpy(digests, d_out, 32 * batch, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        uint64_t t_steady_1 = now_ns();
        steady_trials_ms[i] = (t_steady_1 - t_steady_0) * 1e-6;
    }
    std::sort(steady_trials_ms.begin(), steady_trials_ms.end());
    double steady_p50 = steady_trials_ms[iterations / 2];
    double steady_p95 = steady_trials_ms[iterations * 95 / 100];
    double steady_p99 = steady_trials_ms[iterations * 99 / 100];
    double steady_avg = std::accumulate(steady_trials_ms.begin(), steady_trials_ms.end(), 0.0) / iterations;

    // 6b. Steady State Full-Tree Build (using pre-allocated persistent buffers)
    std::vector<double> ft_trials_ms(iterations);
    uint8_t root_digest[32];
    for (int i = 0; i < iterations; i++) {
        uint64_t t_ft_0 = now_ns();
        cudaMemcpy(d_in, leaves, total_bytes, cudaMemcpyHostToDevice);
        sha256_batch_kernel<<<blocks, threads_per_block>>>(d_in, leaf_bytes, d_out, batch);
        if (n > 1) {
            cudaMemcpy(d_red_in, d_out, batch * 32, cudaMemcpyDeviceToDevice);
            if (n > batch) {
                cudaMemset(d_red_in + batch * 32, 0, (n - batch) * 32);
            }
            size_t current_count = n;
            uint8_t *d_curr_in = d_red_in;
            uint8_t *d_curr_out = d_red_out;
            while (current_count > 1) {
                size_t next_count = current_count / 2;
                unsigned blocks_red = (next_count + 255) / 256;
                merkle_reduce_kernel<<<blocks_red, 256>>>(d_curr_in, d_curr_out, next_count);
                current_count = next_count;
                uint8_t *temp = d_curr_in;
                d_curr_in = d_curr_out;
                d_curr_out = temp;
            }
            cudaMemcpy(root_digest, d_curr_in, 32, cudaMemcpyDeviceToHost);
        } else {
            cudaMemcpy(root_digest, d_out, 32, cudaMemcpyDeviceToHost);
        }
        cudaDeviceSynchronize();
        uint64_t t_ft_1 = now_ns();
        ft_trials_ms[i] = (t_ft_1 - t_ft_0) * 1e-6;
    }
    std::sort(ft_trials_ms.begin(), ft_trials_ms.end());
    double ft_p50 = ft_trials_ms[iterations / 2];
    double ft_p95 = ft_trials_ms[iterations * 95 / 100];
    double ft_p99 = ft_trials_ms[iterations * 99 / 100];
    double ft_avg = std::accumulate(ft_trials_ms.begin(), ft_trials_ms.end(), 0.0) / iterations;

    // 6c. Cold Full-Tree Build (including GPU alloc/free in measurement)
    double cold_ft_ms = 0;
    {
        uint64_t t_cold_0 = now_ns();
        uint8_t *d_cold_in = nullptr;
        uint8_t *d_cold_out = nullptr;
        cudaMalloc(&d_cold_in, total_bytes);
        cudaMalloc(&d_cold_out, batch * 32);
        cudaMemcpy(d_cold_in, leaves, total_bytes, cudaMemcpyHostToDevice);
        sha256_batch_kernel<<<blocks, threads_per_block>>>(d_cold_in, leaf_bytes, d_cold_out, batch);
        if (n > 1) {
            uint8_t *d_cold_red_in = nullptr;
            uint8_t *d_cold_red_out = nullptr;
            cudaMalloc(&d_cold_red_in, n * 32);
            cudaMalloc(&d_cold_red_out, (n / 2) * 32);
            cudaMemcpy(d_cold_red_in, d_cold_out, batch * 32, cudaMemcpyDeviceToDevice);
            if (n > batch) {
                cudaMemset(d_cold_red_in + batch * 32, 0, (n - batch) * 32);
            }
            size_t current_count = n;
            uint8_t *d_curr_in = d_cold_red_in;
            uint8_t *d_curr_out = d_cold_red_out;
            while (current_count > 1) {
                size_t next_count = current_count / 2;
                unsigned blocks_red = (next_count + 255) / 256;
                merkle_reduce_kernel<<<blocks_red, 256>>>(d_curr_in, d_curr_out, next_count);
                current_count = next_count;
                uint8_t *temp = d_curr_in;
                d_curr_in = d_curr_out;
                d_curr_out = temp;
            }
            cudaMemcpy(root_digest, d_curr_in, 32, cudaMemcpyDeviceToHost);
            cudaFree(d_cold_red_in);
            cudaFree(d_cold_red_out);
        } else {
            cudaMemcpy(root_digest, d_cold_out, 32, cudaMemcpyDeviceToHost);
        }
        cudaFree(d_cold_in);
        cudaFree(d_cold_out);
        cudaDeviceSynchronize();
        uint64_t t_cold_1 = now_ns();
        cold_ft_ms = (t_cold_1 - t_cold_0) * 1e-6;
    }

    cudaFree(d_in);
    cudaFree(d_out);
    if (n > 1) {
        cudaFree(d_red_in);
        cudaFree(d_red_out);
    }

    // 7. CPU Single-threaded throughput (measured cleanly)
    uint64_t ct0 = now_ns();
    cpu_thread_worker_single(batch, iterations / 10 ? iterations / 10 : 1);
    uint64_t ct1 = now_ns();
    double cpu_sec = (ct1 - ct0) * 1e-9;
    double cpu_single_thr = (double)(batch * (iterations / 10 ? iterations / 10 : 1)) / cpu_sec;

    // 8. CPU Multi-threaded (all-core) throughput using pinned threads & barrier sync
    unsigned int num_cores = std::thread::hardware_concurrency();
    if (num_cores == 0) num_cores = 1;

    Barrier start_barrier(num_cores + 1);
    Barrier end_barrier(num_cores + 1);

    std::vector<std::thread> threads;
    threads.reserve(num_cores);
    for (unsigned int c = 0; c < num_cores; c++) {
        threads.emplace_back(cpu_thread_worker_partitioned,
                             c, num_cores, batch, iterations / 10 ? iterations / 10 : 1,
                             &start_barrier, &end_barrier);
        pin_thread_to_core(threads.back(), c);
    }

    // Synchronize to eliminate thread startup latency from the timer
    start_barrier.wait();
    uint64_t mt0 = now_ns();

    // Wait for all threads to complete work
    end_barrier.wait();
    uint64_t mt1 = now_ns();

    for (auto &t : threads) t.join();

    double cpu_multi_sec = (mt1 - mt0) * 1e-9;
    double cpu_multi_thr = (double)(batch * (iterations / 10 ? iterations / 10 : 1)) / cpu_multi_sec;

    // 9. CPU Full-Tree (single-threaded) — only for small batches due to CPU cost
    double cpu_ft_single_ms = 0;
    double cpu_ft_multi_ms = 0;
    if (batch <= 4096) {
        cpu_ft_single_ms = cpu_compute_full_tree_blocking(leaves, batch, leaf_bytes, root_digest);
        cpu_ft_multi_ms = cpu_compute_full_tree_parallel(leaves, batch, leaf_bytes, num_cores, root_digest);
    }

    double gpu_throughput = (double)batch / (steady_avg * 1e-3);
    double ft_throughput = (double)batch / (ft_avg * 1e-3);

    printf("batch %zu results:\n", batch);
    printf("  CPU Single-Core      : %.3f M leaves/s\n", cpu_single_thr * 1e-6);
    printf("  CPU All-Core (x%u)    : %.3f M leaves/s (stride-partitioned, pinned)\n", num_cores, cpu_multi_thr * 1e-6);
    if (batch <= 4096) {
        printf("  CPU Full-Tree (1-core): %.3f ms total (hash + reduce)\n", cpu_ft_single_ms);
        printf("  CPU Full-Tree (x%u)   : %.3f ms total (parallel hash + reduce)\n", num_cores, cpu_ft_multi_ms);
    } else {
        printf("  CPU Full-Tree        : skipped (batch too large for CPU measurement)\n");
    }
    printf("  GPU Steady-State Leaf: %.3f M leaves/s (latency p50=%.3fms, p95=%.3fms, p99=%.3fms)\n",
           gpu_throughput * 1e-6, steady_p50, steady_p95, steady_p99);
    printf("  GPU Full-Tree (Cold) : %.3f ms (alloc + copy + leaf + reduce + copy_back + free)\n", cold_ft_ms);
    printf("  GPU Full-Tree (Steady): %.3f M leaves/s (latency p50=%.3fms, p95=%.3fms, p99=%.3fms)\n",
           ft_throughput * 1e-6, ft_p50, ft_p95, ft_p99);
    printf("  GPU Breakdown (ms)   : alloc=%.3f | h2d=%.3f | leaf_kernel=%.3f | reduction=%.3f | d2h=%.3f\n",
           alloc_ms, h2d_ms, leaf_kernel_ms, reduction_kernel_ms, d2h_ms);

    cudaEventDestroy(start_ev);
    cudaEventDestroy(stop_ev);
    free(leaves);
    free(digests);
}

int main(int argc, char **argv) {
    bool only_tests = false;
    bool quick_mode = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--only-tests") == 0) {
            only_tests = true;
        }
        if (strcmp(argv[i], "--quick") == 0) {
            quick_mode = true;
        }
    }

    printf("==================================================\n");
    printf(" AEGIS-Q GPU Integrity Plane Correctness Verification\n");
    printf("==================================================\n");

    bool success = true;
    success &= run_leaf_parity_test_size(16);
    success &= run_leaf_parity_test_size(32);
    success &= run_leaf_parity_test_size(64);
    success &= run_merkle_parity_test(1);
    success &= run_merkle_parity_test(2);
    success &= run_merkle_parity_test(3); // non-power of two
    success &= run_merkle_parity_test(4);
    success &= run_merkle_parity_test(15); // non-power of two
    success &= run_merkle_parity_test(16);
    success &= run_merkle_parity_test(256);
    success &= run_empty_tree_test();

    if (!success) {
        printf("\n[ERROR] Parity or correctness tests failed!\n");
        return EXIT_FAILURE;
    }

    printf("\n[SUCCESS] All correctness tests passed.\n");

    if (only_tests) {
        return EXIT_SUCCESS;
    }

    printf("\n==================================================\n");
    if (quick_mode) {
        printf(" Quick Sanity-Check Benchmarks (batch=1, 16)\n");
    } else {
        printf(" Rigorous Throughput & Latency Benchmarks (100 trials)\n");
    }
    printf("==================================================\n");

    size_t batches[] = {1, 16, 256, 4096, 65536, 1048576};
    for (size_t b : batches) {
        if (quick_mode && b > 16) break;  // Quick mode: only batch=1 and batch=16
        run_performance_bench(b);
        printf("\n");
    }

    return EXIT_SUCCESS;
}
