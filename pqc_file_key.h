/**
 * pqc_file_key.h — ML-KEM-768 File Key Lifecycle Management (M4)
 * 
 * Purpose:
 *   Implements the gate functions for ML-KEM-768 file-key lifecycle:
 *   - File creation: ML-KEM-768 key establishment and encapsulation
 *   - File recovery: Decapsulation on remount
 *   - Key rotation: Epoch-based key updates with deadline tracking
 *   - Zeroization: Secure key material destruction (CPU + GPU)
 *
 * Security Model:
 *   - Each file has a unique ML-KEM-768 keypair established at creation.
 *   - Encapsulated ciphertext (1184 bytes) stored in authenticated metadata.
 *   - TPM 2.0 / RPMB seals the device recipient key (M7 implementation).
 *   - Rotation policy enforces epoch-based access control.
 *   - GPU key residency documented (M6: CUPTI instrumentation).
 *
 * Copyright 2025 AEGIS-Q Authors. Apache-2.0.
 */

#ifndef PQC_FILE_KEY_H
#define PQC_FILE_KEY_H

#include <stdint.h>
#include <time.h>
#include "pqc_block_job.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * pqc_file_key_create()
 *
 * Gate function: called during file creation (pqc_fuse open with O_CREAT).
 *
 * Workflow:
 *   1. Generate ML-KEM-768 public/secret keypair (CPU).
 *   2. Encapsulate a random shared secret using the public key.
 *   3. Store ciphertext in pqc_file_key_metadata_t.
 *   4. (M7 future) Provision TPM-sealed device key path.
 *   5. Record creation timestamp and initial epoch.
 *
 * On success:
 *   - key_md->state set to PQC_KEY_STATE_CREATED
 *   - key_md->ciphertext and ciphertext_len populated
 *   - key_md->current_epoch = 0
 *   - key_md->created_ns set to current time
 *   - The file can now be opened with the ML-KEM-provisioned key.
 *
 * Returns: 0 on success, -1 on error (allocation, RNG failure, TPM error).
 *
 * M4 Gate Criterion:
 *   "File creation, remount, rotation, and recovery decrypt only with
 *    valid ML-KEM-provisioned metadata."
 */
int pqc_file_key_create(pqc_file_key_metadata_t *key_md, uint64_t file_id);

/**
 * pqc_file_key_decrypt_shared_secret()
 *
 * Gate function: called during file read to recover the file key.
 *
 * Workflow:
 *   1. Retrieve the encapsulated ciphertext from file metadata.
 *   2. Perform ML-KEM-768 decapsulation (CPU by default, GPU if admitted).
 *   3. Return the 32-byte shared secret.
 *   4. (Optional) Validate that decapsulation result is deterministic.
 *
 * GPU Eligibility (M4 → M5 integration):
 *   - Single decapsulation is too small for GPU launch overhead.
 *   - M5 admission controller will batch decapsulations across files
 *     during file recovery or periodic key refresh.
 *   - GPU route is decided on batched decaps (not single-file recovery).
 *
 * Returns: 0 on success (output_ss filled), -1 on decapsulation error.
 *
 * M4 Gate Criterion:
 *   "File creation, remount, rotation, and recovery decrypt only with
 *    valid ML-KEM-provisioned metadata."
 */
int pqc_file_key_decrypt_shared_secret(
    const pqc_file_key_metadata_t *key_md,
    uint8_t out_shared_secret[32]);

/**
 * pqc_file_key_rotate()
 *
 * Gate function: called periodically or on admin request to enforce
 * forward secrecy.
 *
 * Workflow:
 *   1. Increment epoch counter.
 *   2. Generate new ML-KEM-768 keypair.
 *   3. Encapsulate new shared secret.
 *   4. Store new ciphertext as "current".
 *   5. Save old ciphertext as "previous" (for grace period).
 *   6. Update next_rotation_deadline_ns based on policy.
 *   7. Zeroize old key material:
 *      - CPU-side: OPENSSL_cleanse or equivalent.
 *      - GPU-side: cudaMemset if key was GPU-resident.
 *   8. Log GPU ops count to document key residency exposure.
 *
 * Rotation Policy (M4 skeleton; to be defined in config):
 *   - ROTATION_INTERVAL_NS: e.g., 86400e9 ns = 1 day.
 *   - GRACE_PERIOD_NS: time to keep old key active (e.g., 1 hour).
 *
 * Returns: 0 on success, -1 on error.
 *
 * M4 Gate Criterion:
 *   "Key material lifecycle is tested for zeroization and is documented
 *    honestly for GPU-resident memory."
 */
int pqc_file_key_rotate(pqc_file_key_metadata_t *key_md);

/**
 * pqc_file_key_zeroize()
 *
 * Gate function: called on file deletion or archive to destroy key material.
 *
 * Workflow:
 *   1. Zeroize current and previous ciphertexts.
 *   2. Zeroize any cached shared secrets (CPU memory).
 *   3. If GPU key residency > threshold (M6 telemetry):
 *      - Issue cudaMemset on GPU buffers.
 *      - Wait for all GPU operations to complete.
 *      - Document GPU zeroization time and success.
 *   4. Mark state as PQC_KEY_STATE_ZEROIZED.
 *   5. Record zeroized_ns timestamp.
 *
 * GPU Zeroization Policy (M4 research question):
 *   - GPU key material is stored in device-side allocations (not UVM).
 *   - Can be zeroized in < 100 µs on modern GPU (memory bandwidth ~500 GB/s).
 *   - Document: "GPU key zeroization took X ms; Y % of GPU peak BW."
 *
 * Returns: 0 on success, -1 on error (GPU timeout, TPM error).
 *
 * M4 Gate Criterion:
 *   "Key material lifecycle is tested for zeroization and is documented
 *    honestly for GPU-resident memory."
 */
int pqc_file_key_zeroize(pqc_file_key_metadata_t *key_md);

/**
 * pqc_file_key_verify_epoch()
 *
 * Utility: validate that an inode/block access uses a non-stale epoch.
 * Called during read/write to ensure the access is using the current epoch.
 *
 * Returns: 0 if epoch is current, 1 if epoch is previous (grace period),
 *          -1 if epoch is expired or invalid.
 */
int pqc_file_key_verify_epoch(const pqc_file_key_metadata_t *key_md,
                              uint64_t access_epoch);

/* ─────────────────────────────────────────────────────────────────────────
 *  Metadata Persistence (xattr)
 * ───────────────────────────────────────────────────────────────────────── */

/**
 * pqc_file_key_save_xattr()
 *
 * Serialize and store metadata to file xattr.
 * Xattr name: "user.pqc_metadata"
 *
 * Format (binary blob):
 *   magic(4) | version(1) | state(1) | epoch(8) | prev_epoch(8) |
 *   epoch_deadline_ns(8) | ciphertext_len(2) | ciphertext(1184) |
 *   tpm_handle(4) | created_ns(8) | last_rotated_ns(8) | checksum(4)
 *
 * Returns: 0 on success, -1 on xattr write error.
 */
int pqc_file_key_save_xattr(int fd, const pqc_file_key_metadata_t *key_md);

/**
 * pqc_file_key_load_xattr()
 *
 * Retrieve and deserialize metadata from file xattr.
 * Validates checksum and version.
 *
 * Returns: 0 on success (metadata populated), -1 on error or xattr not found.
 */
int pqc_file_key_load_xattr(int fd, pqc_file_key_metadata_t *key_md);

/**
 * pqc_file_key_delete_xattr()
 *
 * Remove metadata xattr from file.
 * Called on file deletion.
 *
 * Returns: 0 on success, 1 if xattr not found, -1 on error.
 */
int pqc_file_key_delete_xattr(int fd);

#ifdef __cplusplus
}
#endif

#endif /* PQC_FILE_KEY_H */
