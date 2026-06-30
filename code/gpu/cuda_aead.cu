#include "cuda_aead.h"

#include <cuda_runtime.h>
#include <cerrno>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

static __thread skim_cuda_um_stats_t g_um_stats;
static std::mutex g_um_alloc_mu;
static std::unordered_map<void*, size_t> g_um_alloc_sizes;

/*
 * One mount owns one synchronous submission lane today.  Keeping its stream
 * and staging buffers alive removes allocator/stream churn from the I/O
 * critical path while preserving the caller's completion semantics.  A later
 * multi-stream executor can shard this object; it must not reintroduce an
 * unbounded per-request allocation path.
 */
struct persistent_executor_t {
    cudaStream_t stream = nullptr;
    unsigned char *key = nullptr;
    unsigned char *round_keys = nullptr;
    unsigned char *nonces = nullptr;
    unsigned char *aads = nullptr;
    unsigned char *tags = nullptr;
    size_t *offsets = nullptr;
    size_t *lengths = nullptr;
    unsigned char *in = nullptr;
    unsigned char *out = nullptr;
    size_t byte_capacity = 0;
    size_t job_capacity = 0;
};
static std::mutex g_executor_mu;
static persistent_executor_t g_executor;

static void executor_release_locked()
{
    if (g_executor.stream) cudaStreamSynchronize(g_executor.stream);
    if (g_executor.out) cudaFree(g_executor.out);
    if (g_executor.in) cudaFree(g_executor.in);
    if (g_executor.lengths) cudaFree(g_executor.lengths);
    if (g_executor.offsets) cudaFree(g_executor.offsets);
    if (g_executor.nonces) cudaFree(g_executor.nonces);
    if (g_executor.tags) cudaFree(g_executor.tags);
    if (g_executor.aads) cudaFree(g_executor.aads);
    if (g_executor.round_keys) cudaFree(g_executor.round_keys);
    if (g_executor.key) cudaFree(g_executor.key);
    if (g_executor.stream) cudaStreamDestroy(g_executor.stream);
    g_executor = persistent_executor_t{};
}

static size_t round_capacity(size_t value)
{
    size_t cap = 1;
    while (cap < value && cap <= SIZE_MAX / 2) cap <<= 1;
    return cap < value ? value : cap;
}

static int executor_ensure_locked(size_t bytes, size_t jobs)
{
    if (!skim_cuda_aead_available()) return -1;
    if (g_executor.stream && g_executor.byte_capacity >= bytes &&
        g_executor.job_capacity >= jobs)
        return 0;

    const size_t byte_capacity = round_capacity(bytes);
    const size_t job_capacity = round_capacity(jobs);
    executor_release_locked();
    cudaError_t rc = cudaStreamCreateWithFlags(&g_executor.stream, cudaStreamNonBlocking);
    rc = rc == cudaSuccess ? cudaMallocManaged(&g_executor.key, 32) : rc;
    rc = rc == cudaSuccess ? cudaMallocManaged(&g_executor.round_keys, 240) : rc;
    rc = rc == cudaSuccess ? cudaMallocManaged(&g_executor.nonces, job_capacity * 12) : rc;
    rc = rc == cudaSuccess ? cudaMallocManaged(&g_executor.aads, job_capacity * 28) : rc;
    rc = rc == cudaSuccess ? cudaMallocManaged(&g_executor.tags, job_capacity * 16) : rc;
    rc = rc == cudaSuccess ? cudaMallocManaged(&g_executor.offsets, job_capacity * sizeof(size_t)) : rc;
    rc = rc == cudaSuccess ? cudaMallocManaged(&g_executor.lengths, job_capacity * sizeof(size_t)) : rc;
    rc = rc == cudaSuccess ? cudaMallocManaged(&g_executor.in, byte_capacity) : rc;
    rc = rc == cudaSuccess ? cudaMallocManaged(&g_executor.out, byte_capacity) : rc;
    if (rc != cudaSuccess) {
        executor_release_locked();
        return -1;
    }
    g_executor.byte_capacity = byte_capacity;
    g_executor.job_capacity = job_capacity;
    g_um_stats.managed_alloc_bytes += 32 + 240 + job_capacity * (12 + 28 + 16 + 2 * sizeof(size_t)) + 2 * byte_capacity;
    return 0;
}

int skim_cuda_executor_init(size_t max_batch_bytes, size_t max_jobs)
{
    std::lock_guard<std::mutex> lock(g_executor_mu);
    return executor_ensure_locked(max_batch_bytes, max_jobs);
}

void skim_cuda_executor_shutdown(void)
{
    std::lock_guard<std::mutex> lock(g_executor_mu);
    executor_release_locked();
}

__device__ static unsigned char gf_mul(unsigned char a, unsigned char b)
{
    unsigned char r = 0;
    for (int i = 0; i < 8; ++i) {
        if (b & 1) r ^= a;
        unsigned char hi = a & 0x80;
        a <<= 1;
        if (hi) a ^= 0x1b;
        b >>= 1;
    }
    return r;
}

__device__ static unsigned char aes_sbox(unsigned char x)
{
    unsigned char inv = 1;
    if (x == 0) inv = 0;
    else {
        /* x^254 in GF(2^8): compact and table-free, suitable for validation. */
        unsigned char base = x;
        unsigned int exponent = 254;
        while (exponent) {
            if (exponent & 1) inv = gf_mul(inv, base);
            base = gf_mul(base, base);
            exponent >>= 1;
        }
    }
    return (unsigned char)(inv ^ ((inv << 1) | (inv >> 7)) ^
           ((inv << 2) | (inv >> 6)) ^ ((inv << 3) | (inv >> 5)) ^
           ((inv << 4) | (inv >> 4)) ^ 0x63);
}

__device__ static void aes_expand_256(const unsigned char key[32], unsigned char round_keys[240])
{
    for (int i = 0; i < 32; ++i) round_keys[i] = key[i];
    unsigned char temp[4];
    unsigned char rcon = 1;
    int bytes = 32;
    while (bytes < 240) {
        for (int i = 0; i < 4; ++i) temp[i] = round_keys[bytes - 4 + i];
        if (bytes % 32 == 0) {
            unsigned char t = temp[0];
            temp[0] = aes_sbox(temp[1]) ^ rcon;
            temp[1] = aes_sbox(temp[2]);
            temp[2] = aes_sbox(temp[3]);
            temp[3] = aes_sbox(t);
            rcon = gf_mul(rcon, 2);
        } else if (bytes % 32 == 16) {
            for (int i = 0; i < 4; ++i) temp[i] = aes_sbox(temp[i]);
        }
        for (int i = 0; i < 4; ++i) {
            round_keys[bytes] = round_keys[bytes - 32] ^ temp[i];
            ++bytes;
        }
    }
}

__device__ static void aes_encrypt_block_round_keys(const unsigned char round_keys[240], unsigned char state[16])
{
    for (int i = 0; i < 16; ++i) state[i] ^= round_keys[i];
    for (int round = 1; round < 14; ++round) {
        for (int i = 0; i < 16; ++i) state[i] = aes_sbox(state[i]);
        unsigned char t[16];
        t[0]=state[0]; t[1]=state[5]; t[2]=state[10]; t[3]=state[15];
        t[4]=state[4]; t[5]=state[9]; t[6]=state[14]; t[7]=state[3];
        t[8]=state[8]; t[9]=state[13]; t[10]=state[2]; t[11]=state[7];
        t[12]=state[12]; t[13]=state[1]; t[14]=state[6]; t[15]=state[11];
        for (int c = 0; c < 4; ++c) {
            int p = 4 * c;
            unsigned char a0=t[p], a1=t[p+1], a2=t[p+2], a3=t[p+3];
            state[p] = gf_mul(a0,2) ^ gf_mul(a1,3) ^ a2 ^ a3;
            state[p+1] = a0 ^ gf_mul(a1,2) ^ gf_mul(a2,3) ^ a3;
            state[p+2] = a0 ^ a1 ^ gf_mul(a2,2) ^ gf_mul(a3,3);
            state[p+3] = gf_mul(a0,3) ^ a1 ^ a2 ^ gf_mul(a3,2);
        }
        for (int i = 0; i < 16; ++i) state[i] ^= round_keys[round * 16 + i];
    }
    for (int i = 0; i < 16; ++i) state[i] = aes_sbox(state[i]);
    unsigned char t[16] = {state[0],state[5],state[10],state[15],state[4],state[9],state[14],state[3],
                           state[8],state[13],state[2],state[7],state[12],state[1],state[6],state[11]};
    for (int i = 0; i < 16; ++i) state[i] = t[i] ^ round_keys[224 + i];
}

__global__ static void aes256_expand_key_kernel(const unsigned char *key, unsigned char *round_keys)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
        aes_expand_256(key, round_keys);
}

__global__ static void um_smoke_touch_kernel(unsigned char *buf, size_t size)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        buf[i] = (unsigned char)(buf[i] ^ 0x5aU);
}

__global__ static void aes256_ctr_batch_kernel(const unsigned char *round_keys,
                                               const unsigned char *nonces,
                                               const unsigned char *in,
                                               unsigned char *out,
                                               const size_t *offsets,
                                               const size_t *lengths,
                                               size_t count)
{
    size_t job = (size_t)blockIdx.y;
    size_t lane = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (job >= count) return;
    size_t base = offsets[job];
    size_t length = lengths[job];
    size_t offset = lane * 16;
    if (offset >= length) return;
    const unsigned char *nonce = nonces + job * 12;
    unsigned char counter[16];
    for (int i = 0; i < 12; ++i) counter[i] = nonce[i];
    unsigned int n = (unsigned int)(lane + 2);
    counter[12] = (unsigned char)(n >> 24); counter[13] = (unsigned char)(n >> 16);
    counter[14] = (unsigned char)(n >> 8); counter[15] = (unsigned char)n;
    aes_encrypt_block_round_keys(round_keys, counter);
    size_t take = length - offset < 16 ? length - offset : 16;
    for (size_t i = 0; i < take; ++i)
        out[base + offset + i] = in[base + offset + i] ^ counter[i];
}

__device__ static void gcm_shift_right(unsigned char value[16])
{
    unsigned char carry = 0;
    for (int i = 0; i < 16; ++i) {
        unsigned char next = (unsigned char)(value[i] & 1U);
        value[i] = (unsigned char)((value[i] >> 1) | (carry << 7));
        carry = next;
    }
}

__device__ static void gcm_multiply_device(const unsigned char x[16], const unsigned char h[16], unsigned char out[16])
{
    unsigned char z[16] = {0};
    unsigned char v[16];
    for (int i = 0; i < 16; ++i) v[i] = h[i];
    for (int bit = 0; bit < 128; ++bit) {
        if ((x[bit / 8] >> (7 - (bit % 8))) & 1U)
            for (int i = 0; i < 16; ++i) z[i] ^= v[i];
        int lsb = v[15] & 1U;
        gcm_shift_right(v);
        if (lsb) v[0] ^= 0xe1U;
    }
    for (int i = 0; i < 16; ++i) out[i] = z[i];
}

__device__ static void gcm_absorb_device(unsigned char state[16], const unsigned char h[16],
                                          const unsigned char *data, size_t length)
{
    while (length) {
        unsigned char block[16] = {0};
        size_t take = length < 16 ? length : 16;
        for (size_t i = 0; i < take; ++i) block[i] = data[i];
        for (int i = 0; i < 16; ++i) state[i] ^= block[i];
        unsigned char next[16];
        gcm_multiply_device(state, h, next);
        for (int i = 0; i < 16; ++i) state[i] = next[i];
        data += take;
        length -= take;
    }
}

__global__ static void aes256_gcm_tag_batch_kernel(const unsigned char *round_keys,
                                                     const unsigned char *nonces,
                                                     const unsigned char *aads,
                                                     const unsigned char *ciphertext,
                                                     const size_t *offsets,
                                                     const size_t *lengths,
                                                     unsigned char *tags,
                                                     size_t count)
{
    size_t job = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (job >= count) return;
    unsigned char h[16] = {0};
    unsigned char j0[16] = {0};
    for (int i = 0; i < 12; ++i) j0[i] = nonces[job * 12 + i];
    j0[15] = 1;
    aes_encrypt_block_round_keys(round_keys, h);
    aes_encrypt_block_round_keys(round_keys, j0);
    unsigned char state[16] = {0};
    gcm_absorb_device(state, h, aads + job * 28, 28);
    gcm_absorb_device(state, h, ciphertext + offsets[job], lengths[job]);
    unsigned char lengths_block[16] = {0};
    unsigned long long aad_bits = 28ULL * 8ULL;
    unsigned long long cipher_bits = (unsigned long long)lengths[job] * 8ULL;
    for (int i = 7; i >= 0; --i) {
        lengths_block[i] = (unsigned char)aad_bits;
        aad_bits >>= 8;
        lengths_block[8 + i] = (unsigned char)cipher_bits;
        cipher_bits >>= 8;
    }
    for (int i = 0; i < 16; ++i) state[i] ^= lengths_block[i];
    unsigned char final_state[16];
    gcm_multiply_device(state, h, final_state);
    for (int i = 0; i < 16; ++i) tags[job * 16 + i] = j0[i] ^ final_state[i];
}

int skim_cuda_aead_available(void)
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

int skim_cuda_aead_is_uma(void)
{
    int count = 0;
    cudaDeviceProp prop;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0 &&
           cudaGetDeviceProperties(&prop, 0) == cudaSuccess &&
           prop.managedMemory && prop.integrated;
}

int skim_cuda_is_available(void)
{
    return skim_cuda_aead_available();
}

void skim_cuda_um_stats_reset(void)
{
    memset(&g_um_stats, 0, sizeof(g_um_stats));
}

skim_cuda_um_stats_t skim_cuda_um_stats_snapshot(void)
{
    return g_um_stats;
}

int skim_cuda_current_device(void)
{
    int dev = 0;
    return cudaGetDevice(&dev) == cudaSuccess ? dev : 0;
}

int skim_cuda_um_self_test(void)
{
    if (!skim_cuda_aead_available())
        return -1;
    skim_cuda_um_stats_reset();
    const size_t size = 16U * 1024U * 1024U;
    unsigned char *buf = (unsigned char *)skim_cuda_managed_alloc(size);
    if (!buf)
        return -1;
    for (size_t i = 0; i < size; ++i)
        buf[i] = (unsigned char)(i & 0xffU);
    int dev = skim_cuda_current_device();
    int rc = skim_cuda_mem_prefetch(buf, size, dev);
    if (rc == 0) {
        const int threads = 256;
        const int blocks = (int)((size + (size_t)threads - 1U) / (size_t)threads);
        um_smoke_touch_kernel<<<blocks, threads>>>(buf, size);
        rc = cudaDeviceSynchronize() == cudaSuccess ? 0 : -1;
    }
    if (rc == 0) {
        rc = skim_cuda_mem_prefetch_host(buf, size);
        if (rc == 0)
            rc = cudaDeviceSynchronize() == cudaSuccess ? 0 : -1;
    }
    if (rc == 0) {
        for (size_t i = 0; i < size; i += 4096U) {
            unsigned char expected = (unsigned char)((i & 0xffU) ^ 0x5aU);
            if (buf[i] != expected) {
                rc = -1;
                break;
            }
        }
    }
    skim_cuda_managed_free(buf);
    return rc;
}

void *skim_cuda_managed_alloc(size_t size)
{
    void *ptr = nullptr;
    if (cudaMallocManaged(&ptr, size) != cudaSuccess)
        return nullptr;
    g_um_stats.managed_alloc_bytes += size;
    {
        std::lock_guard<std::mutex> lock(g_um_alloc_mu);
        g_um_alloc_sizes[ptr] = size;
    }
    return ptr;
}

void skim_cuda_managed_free(void *ptr)
{
    if (ptr) {
        size_t bytes = 0;
        {
            std::lock_guard<std::mutex> lock(g_um_alloc_mu);
            auto it = g_um_alloc_sizes.find(ptr);
            if (it != g_um_alloc_sizes.end()) {
                bytes = it->second;
                g_um_alloc_sizes.erase(it);
            }
        }
        g_um_stats.managed_free_bytes += bytes;
        cudaFree(ptr);
    }
}

int skim_cuda_mem_prefetch(void *ptr, size_t size, int device)
{
    if (!ptr || size == 0)
        return 0;
    cudaMemLocation loc = { cudaMemLocationTypeDevice, device };
    if (cudaMemPrefetchAsync(ptr, size, loc, 0) != cudaSuccess)
        return -1;
    g_um_stats.prefetch_to_device_bytes += size;
    g_um_stats.prefetch_device_calls += 1;
    return 0;
}

int skim_cuda_mem_prefetch_host(void *ptr, size_t size)
{
    if (!ptr || size == 0)
        return 0;
    cudaMemLocation loc = { cudaMemLocationTypeHost, 0 };
    if (cudaMemPrefetchAsync(ptr, size, loc, 0) != cudaSuccess)
        return -1;
    g_um_stats.prefetch_to_host_bytes += size;
    g_um_stats.prefetch_host_calls += 1;
    return 0;
}

int skim_cuda_aes256_gcm_ctr(const uint8_t key[32], const uint8_t nonce[12],
                             const uint8_t *in, uint8_t *out, size_t length)
{
    if (!key || !nonce || !in || !out || !skim_cuda_aead_available()) return -1;
    if (length == 0) return 0;
    size_t offset = 0, length_entry = length;
    return skim_cuda_aes256_gcm_ctr_batch(key, nonce, in, out,
                                          &offset, &length_entry, 1);
}

int skim_cuda_aes256_gcm_ctr_batch(const uint8_t key[32], const uint8_t *nonces,
                                   const uint8_t *in, uint8_t *out,
                                   const size_t *offsets, const size_t *lengths,
                                   size_t count)
{
    if (!key || !nonces || !in || !out || !offsets || !lengths || count == 0 || !skim_cuda_aead_available())
        return -1;

    size_t total_bytes = 0;
    size_t max_length = 0;
    for (size_t i = 0; i < count; ++i) {
        if (lengths[i] == 0) continue;
        if (offsets[i] > total_bytes) total_bytes = offsets[i];
        if (lengths[i] > max_length) max_length = lengths[i];
        size_t end = offsets[i] + lengths[i];
        if (end > total_bytes) total_bytes = end;
    }
    if (total_bytes == 0) return 0;

    std::lock_guard<std::mutex> lock(g_executor_mu);
    if (executor_ensure_locked(total_bytes, count) != 0)
        return -1;

    memcpy(g_executor.key, key, 32);
    memcpy(g_executor.nonces, nonces, count * 12);
    memcpy(g_executor.offsets, offsets, count * sizeof(size_t));
    memcpy(g_executor.lengths, lengths, count * sizeof(size_t));
    memcpy(g_executor.in, in, total_bytes);

    int dev = 0;
    cudaGetDevice(&dev);
    cudaMemLocation gpu_location = { cudaMemLocationTypeDevice, dev };
    cudaMemLocation cpu_location = { cudaMemLocationTypeHost, 0 };
    cudaError_t rc = cudaMemPrefetchAsync(g_executor.key, 32, gpu_location, 0, g_executor.stream);
    rc = rc == cudaSuccess ? cudaMemPrefetchAsync(g_executor.round_keys, 240, gpu_location, 0, g_executor.stream) : rc;
    rc = rc == cudaSuccess ? cudaMemPrefetchAsync(g_executor.nonces, count * 12, gpu_location, 0, g_executor.stream) : rc;
    rc = rc == cudaSuccess ? cudaMemPrefetchAsync(g_executor.offsets, count * sizeof(size_t), gpu_location, 0, g_executor.stream) : rc;
    rc = rc == cudaSuccess ? cudaMemPrefetchAsync(g_executor.lengths, count * sizeof(size_t), gpu_location, 0, g_executor.stream) : rc;
    rc = rc == cudaSuccess ? cudaMemPrefetchAsync(g_executor.in, total_bytes, gpu_location, 0, g_executor.stream) : rc;
    if (rc != cudaSuccess) return -1;

    aes256_expand_key_kernel<<<1, 1, 0, g_executor.stream>>>(g_executor.key, g_executor.round_keys);
    rc = cudaGetLastError();
    if (rc != cudaSuccess) return -1;
    dim3 grid((unsigned int)((max_length + 255) / 256), (unsigned int)count, 1);
    aes256_ctr_batch_kernel<<<grid, 256, 0, g_executor.stream>>>(g_executor.round_keys, g_executor.nonces,
        g_executor.in, g_executor.out, g_executor.offsets, g_executor.lengths, count);
    rc = cudaGetLastError();
    rc = rc == cudaSuccess ? cudaMemPrefetchAsync(g_executor.out, total_bytes, cpu_location, 0, g_executor.stream) : rc;
    rc = rc == cudaSuccess ? cudaStreamSynchronize(g_executor.stream) : rc;
    if (rc != cudaSuccess) return -1;
    memcpy(out, g_executor.out, total_bytes);
    return 0;
}

int skim_cuda_aes256_gcm_batch(const uint8_t key[32], const uint8_t *nonces,
                               const uint8_t *aads, const uint8_t *in, uint8_t *out,
                               const size_t *offsets, const size_t *lengths,
                               uint8_t *tags, size_t count)
{
    if (!key || !nonces || !aads || !in || !out || !offsets || !lengths || !tags ||
        count == 0 || !skim_cuda_aead_available())
        return -1;
    size_t total_bytes = 0, max_length = 0;
    for (size_t i = 0; i < count; ++i) {
        size_t end = offsets[i] + lengths[i];
        if (end > total_bytes) total_bytes = end;
        if (lengths[i] > max_length) max_length = lengths[i];
    }
    if (total_bytes == 0) return 0;

    std::lock_guard<std::mutex> lock(g_executor_mu);
    if (executor_ensure_locked(total_bytes, count) != 0) return -1;
    memcpy(g_executor.key, key, 32);
    memcpy(g_executor.nonces, nonces, count * 12);
    memcpy(g_executor.aads, aads, count * 28);
    memcpy(g_executor.offsets, offsets, count * sizeof(size_t));
    memcpy(g_executor.lengths, lengths, count * sizeof(size_t));
    memcpy(g_executor.in, in, total_bytes);

    int dev = 0;
    cudaGetDevice(&dev);
    cudaMemLocation gpu = { cudaMemLocationTypeDevice, dev };
    cudaMemLocation host = { cudaMemLocationTypeHost, 0 };
    cudaError_t rc = cudaMemPrefetchAsync(g_executor.key, 32, gpu, 0, g_executor.stream);
    rc = rc == cudaSuccess ? cudaMemPrefetchAsync(g_executor.round_keys, 240, gpu, 0, g_executor.stream) : rc;
    rc = rc == cudaSuccess ? cudaMemPrefetchAsync(g_executor.nonces, count * 12, gpu, 0, g_executor.stream) : rc;
    rc = rc == cudaSuccess ? cudaMemPrefetchAsync(g_executor.aads, count * 28, gpu, 0, g_executor.stream) : rc;
    rc = rc == cudaSuccess ? cudaMemPrefetchAsync(g_executor.offsets, count * sizeof(size_t), gpu, 0, g_executor.stream) : rc;
    rc = rc == cudaSuccess ? cudaMemPrefetchAsync(g_executor.lengths, count * sizeof(size_t), gpu, 0, g_executor.stream) : rc;
    rc = rc == cudaSuccess ? cudaMemPrefetchAsync(g_executor.in, total_bytes, gpu, 0, g_executor.stream) : rc;
    if (rc != cudaSuccess) return -1;
    aes256_expand_key_kernel<<<1, 1, 0, g_executor.stream>>>(g_executor.key, g_executor.round_keys);
    rc = cudaGetLastError();
    if (rc != cudaSuccess) return -1;
    dim3 grid((unsigned int)((max_length + 255) / 256), (unsigned int)count, 1);
    aes256_ctr_batch_kernel<<<grid, 256, 0, g_executor.stream>>>(g_executor.round_keys, g_executor.nonces,
        g_executor.in, g_executor.out, g_executor.offsets, g_executor.lengths, count);
    rc = cudaGetLastError();
    if (rc != cudaSuccess) return -1;
    const unsigned int tag_threads = 64;
    aes256_gcm_tag_batch_kernel<<<(unsigned int)((count + tag_threads - 1) / tag_threads), tag_threads, 0, g_executor.stream>>>(
        g_executor.round_keys, g_executor.nonces, g_executor.aads, g_executor.out,
        g_executor.offsets, g_executor.lengths, g_executor.tags, count);
    rc = cudaGetLastError();
    rc = rc == cudaSuccess ? cudaMemPrefetchAsync(g_executor.out, total_bytes, host, 0, g_executor.stream) : rc;
    rc = rc == cudaSuccess ? cudaMemPrefetchAsync(g_executor.tags, count * 16, host, 0, g_executor.stream) : rc;
    rc = rc == cudaSuccess ? cudaStreamSynchronize(g_executor.stream) : rc;
    if (rc != cudaSuccess) return -1;
    memcpy(out, g_executor.out, total_bytes);
    memcpy(tags, g_executor.tags, count * 16);
    return 0;
}

int skim_cuda_shake128_encrypt_batch(const uint8_t *in, uint8_t *out,
                                     const size_t *offsets, const size_t *lengths,
                                     size_t count)
{
    (void)in;
    (void)out;
    (void)offsets;
    (void)lengths;
    (void)count;
    return -ENOTSUP;
}
