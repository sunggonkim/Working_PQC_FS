#include "cuda_aead.h"

#include <openssl/crypto.h>
#include <openssl/evp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static constexpr size_t kBlockBytes = 4096;
static constexpr size_t kAadBytes = 28;
static constexpr size_t kNonceBytes = 12;
static constexpr size_t kTagBytes = 16;

static void fill_pattern(std::vector<uint8_t> &buf, uint32_t seed)
{
    uint32_t x = seed;
    for (auto &byte : buf) {
        x = x * 1664525u + 1013904223u;
        byte = static_cast<uint8_t>(x >> 24);
    }
}

static int cpu_gcm_batch(const uint8_t key[32],
                         const std::vector<uint8_t> &nonces,
                         const std::vector<uint8_t> &aads,
                         const std::vector<uint8_t> &input,
                         std::vector<uint8_t> &output,
                         std::vector<uint8_t> &tags,
                         const std::vector<size_t> &offsets,
                         const std::vector<size_t> &lengths)
{
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx)
        return -1;
    int rc = 0;
    for (size_t i = 0; i < lengths.size(); ++i) {
        int out_len = 0;
        int ok = EVP_CIPHER_CTX_reset(ctx) == 1 &&
                 EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), nullptr, key,
                                    nonces.data() + i * kNonceBytes) == 1 &&
                 EVP_EncryptUpdate(ctx, nullptr, &out_len,
                                   aads.data() + i * kAadBytes,
                                   static_cast<int>(kAadBytes)) == 1 &&
                 EVP_EncryptUpdate(ctx, output.data() + offsets[i], &out_len,
                                   input.data() + offsets[i],
                                   static_cast<int>(lengths[i])) == 1 &&
                 EVP_EncryptFinal_ex(ctx, output.data() + offsets[i] + out_len,
                                     &out_len) == 1 &&
                 EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG,
                                     static_cast<int>(kTagBytes),
                                     tags.data() + i * kTagBytes) == 1;
        if (!ok) {
            rc = -1;
            break;
        }
    }
    EVP_CIPHER_CTX_free(ctx);
    return rc;
}

static uint64_t elapsed_ns(std::chrono::steady_clock::time_point start,
                           std::chrono::steady_clock::time_point end)
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

static double mib_per_s(size_t bytes, uint64_t ns)
{
    if (ns == 0)
        return 0.0;
    return (static_cast<double>(bytes) / (1024.0 * 1024.0)) /
           (static_cast<double>(ns) / 1000000000.0);
}

int main(int argc, char **argv)
{
    int reps = 7;
    int warmups = 2;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc)
            reps = std::max(1, std::atoi(argv[++i]));
        else if (std::strcmp(argv[i], "--warmups") == 0 && i + 1 < argc)
            warmups = std::max(0, std::atoi(argv[++i]));
    }

    const int gpu_available = skim_cuda_aead_available();
    std::printf("mode,count,total_bytes,rep,ns,mib_s,verified,gpu_available\n");

    uint8_t key[32];
    for (size_t i = 0; i < sizeof(key); ++i)
        key[i] = static_cast<uint8_t>(0x40u + i);

    const size_t counts[] = {1, 4, 16, 64, 256};
    for (size_t count : counts) {
        const size_t total = count * kBlockBytes;
        std::vector<uint8_t> input(total), cpu_out(total), gpu_out(total);
        std::vector<uint8_t> cpu_tags(count * kTagBytes), gpu_tags(count * kTagBytes);
        std::vector<uint8_t> nonces(count * kNonceBytes), aads(count * kAadBytes);
        std::vector<size_t> offsets(count), lengths(count);

        fill_pattern(input, static_cast<uint32_t>(0x1234u + count));
        fill_pattern(nonces, static_cast<uint32_t>(0x5678u + count));
        fill_pattern(aads, static_cast<uint32_t>(0x9abcu + count));
        for (size_t i = 0; i < count; ++i) {
            offsets[i] = i * kBlockBytes;
            lengths[i] = kBlockBytes;
        }

        for (int rep = -warmups; rep < reps; ++rep) {
            std::fill(cpu_out.begin(), cpu_out.end(), 0);
            std::fill(cpu_tags.begin(), cpu_tags.end(), 0);
            auto start = std::chrono::steady_clock::now();
            int rc = cpu_gcm_batch(key, nonces, aads, input, cpu_out, cpu_tags,
                                   offsets, lengths);
            auto end = std::chrono::steady_clock::now();
            if (rc != 0)
                return 2;
            if (rep >= 0) {
                uint64_t ns = elapsed_ns(start, end);
                std::printf("cpu,%zu,%zu,%d,%llu,%.6f,1,%d\n", count, total, rep,
                            static_cast<unsigned long long>(ns),
                            mib_per_s(total, ns), gpu_available);
            }
        }

        if (!gpu_available)
            continue;

        for (int rep = -warmups; rep < reps; ++rep) {
            std::fill(gpu_out.begin(), gpu_out.end(), 0);
            std::fill(gpu_tags.begin(), gpu_tags.end(), 0);
            auto start = std::chrono::steady_clock::now();
            int rc = skim_cuda_aes256_gcm_batch(key, nonces.data(), aads.data(),
                                                input.data(), gpu_out.data(),
                                                offsets.data(), lengths.data(),
                                                gpu_tags.data(), count);
            auto end = std::chrono::steady_clock::now();
            if (rc != 0)
                return 3;
            const int verified =
                std::memcmp(cpu_out.data(), gpu_out.data(), total) == 0 &&
                std::memcmp(cpu_tags.data(), gpu_tags.data(), count * kTagBytes) == 0;
            if (!verified)
                return 4;
            if (rep >= 0) {
                uint64_t ns = elapsed_ns(start, end);
                std::printf("gpu,%zu,%zu,%d,%llu,%.6f,%d,%d\n", count, total, rep,
                            static_cast<unsigned long long>(ns),
                            mib_per_s(total, ns), verified, gpu_available);
            }
        }
    }

    OPENSSL_cleanse(key, sizeof(key));
    return 0;
}
