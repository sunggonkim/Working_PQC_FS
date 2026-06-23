#ifndef SKIM_CUDA_AEAD_H
#define SKIM_CUDA_AEAD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * AES-256-GCM data path for the authenticated-block backend.  The caller
 * owns nonce construction, AAD processing, tag generation, and persistence;
 * this backend performs only the GCM CTR transform.  Splitting the operation
 * this way keeps one byte-level storage format across CPU and CUDA paths.
 */
int skim_cuda_aes256_gcm_ctr(const uint8_t key[32], const uint8_t nonce[12],
                             const uint8_t *in, uint8_t *out, size_t length);
int skim_cuda_aes256_gcm_ctr_batch(const uint8_t key[32], const uint8_t *nonces,
                                   const uint8_t *in, uint8_t *out,
                                   const size_t *offsets, const size_t *lengths,
                                   size_t count);
/* Complete, byte-compatible AES-256-GCM batch executor.  AAD is 28 bytes per
 * job and tags is 16 bytes per job. */
int skim_cuda_aes256_gcm_batch(const uint8_t key[32], const uint8_t *nonces,
                               const uint8_t *aads, const uint8_t *in, uint8_t *out,
                               const size_t *offsets, const size_t *lengths,
                               uint8_t *tags, size_t count);
/* Placeholder for the heterogeneous SHAKE128 lane requested in the roadmap.
 * The current repository still routes authenticated storage I/O through the
 * byte-identical AES-GCM format; this symbol exists so the metadata and
 * scheduling layers can compile against the future interface. */
int skim_cuda_shake128_encrypt_batch(const uint8_t *in, uint8_t *out,
                                     const size_t *offsets, const size_t *lengths,
                                     size_t count);
int skim_cuda_aead_available(void);
int skim_cuda_is_available(void);
int skim_cuda_aead_is_uma(void);
int skim_cuda_current_device(void);
int skim_cuda_um_self_test(void);
void *skim_cuda_managed_alloc(size_t size);
void skim_cuda_managed_free(void *ptr);
int skim_cuda_mem_prefetch(void *ptr, size_t size, int device);
int skim_cuda_mem_prefetch_host(void *ptr, size_t size);

/* Mount-lifetime CUDA executor.  The FUSE path initializes this once with its
 * largest coalesced batch; calls below reuse its stream and managed buffers. */
int skim_cuda_executor_init(size_t max_batch_bytes, size_t max_jobs);
void skim_cuda_executor_shutdown(void);

typedef struct {
    uint64_t managed_alloc_bytes;
    uint64_t managed_free_bytes;
    uint64_t prefetch_to_device_bytes;
    uint64_t prefetch_to_host_bytes;
    uint64_t prefetch_device_calls;
    uint64_t prefetch_host_calls;
} skim_cuda_um_stats_t;

void skim_cuda_um_stats_reset(void);
skim_cuda_um_stats_t skim_cuda_um_stats_snapshot(void);

#ifdef __cplusplus
}
#endif

#endif
