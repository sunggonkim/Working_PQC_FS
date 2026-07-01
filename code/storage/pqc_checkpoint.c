#include "pqc_checkpoint.h"

#include "pqc_anchor_worker.h"
#include "pqc_publish.h"
#include "pqc_test_hooks.h"

#include <errno.h>

#include <oqs/oqs.h>

void pqc_checkpoint_make_anchor_state(uint64_t sequence,
                                      uint64_t logical_size,
                                      uint64_t max_generation,
                                      pqc_anchor_state_t *out)
{
    if (!out)
        return;

    out->epoch = max_generation;
    out->sequence = sequence;
    out->logical_size = logical_size;
}

int pqc_checkpoint_record_file_anchor(uint64_t file_id,
                                      uint64_t sequence,
                                      uint64_t logical_size,
                                      uint64_t max_generation,
                                      pqc_anchor_state_t *out_state)
{
    int rc = pqc_anchor_record_file(file_id, max_generation, sequence,
                                    logical_size);
    if (rc != 0)
        return rc;
    pqc_checkpoint_make_anchor_state(sequence, logical_size, max_generation,
                                     out_state);
    return 0;
}

int pqc_checkpoint_store_and_stage_anchor(const char *path,
                                          uint64_t file_id,
                                          uint64_t sequence,
                                          uint64_t logical_size,
                                          uint64_t max_generation)
{
    int rc = pqc_publish_checkpoint_store_xattr(path, file_id, sequence,
                                                logical_size, max_generation);
    if (rc != 0)
        return rc;

    pqc_fault_cutpoint("checkpoint_xattr_after");

    pqc_anchor_backend_t backend = pqc_anchor_backend();
    if (backend == PQC_ANCHOR_BACKEND_DISABLED)
        return 0;

    pqc_anchor_state_t state = {0};
    rc = pqc_checkpoint_record_file_anchor(file_id, sequence, logical_size,
                                           max_generation, &state);
    if (rc != 0)
        return rc;

    pqc_anchor_worker_stage(&state);
    if (backend == PQC_ANCHOR_BACKEND_HARDWARE) {
        rc = pqc_anchor_store(&state);
        if (rc != 0)
            return rc;
    }
    return 0;
}

int pqc_checkpoint_store_and_stage_anchor_final(
    const char *path,
    uint64_t file_id,
    uint64_t sequence,
    uint64_t logical_size,
    uint64_t max_generation,
    int reservation_matches_final)
{
    if (!path)
        return -EINVAL;

    if (reservation_matches_final && sequence == max_generation &&
        max_generation != 0 &&
        pqc_anchor_backend() == PQC_ANCHOR_BACKEND_DISABLED) {
        /*
         * Strict writeback reserved this exact checkpoint before encryption.
         * For in-place overwrites, final publication would rewrite identical
         * xattr bytes.  Keep the generation high-water and fault boundary while
         * avoiding a redundant setxattr on the default no-anchor path.
        */
        pqc_fault_cutpoint("checkpoint_xattr_after");
        pqc_fault_cutpoint("checkpoint_xattr_reused_after");
        return 0;
    }

    return pqc_checkpoint_store_and_stage_anchor(path, file_id, sequence,
                                                 logical_size, max_generation);
}

int pqc_checkpoint_reserve_generation(const char *path,
                                      uint64_t file_id,
                                      uint64_t logical_size,
                                      uint64_t reserved_generation)
{
    int rc = pqc_publish_checkpoint_store_xattr(path, file_id,
                                                reserved_generation,
                                                logical_size,
                                                reserved_generation);
    if (rc != 0)
        return rc;

    pqc_fault_cutpoint("generation_reservation_xattr_after");

    pqc_anchor_backend_t backend = pqc_anchor_backend();
    if (backend == PQC_ANCHOR_BACKEND_DISABLED)
        return 0;

    pqc_anchor_state_t state = {0};
    rc = pqc_checkpoint_record_file_anchor(file_id, reserved_generation,
                                           logical_size, reserved_generation,
                                           &state);
    if (rc != 0)
        return rc;

    /*
     * Reservation records keep generation reuse from moving backward, but a
     * file-backed freshness anchor should not be flushed for this intermediate
     * state.  Foreground fsync publishes the final checkpoint state below via
     * pqc_checkpoint_store_and_stage_anchor().  Hardware anchors keep the
     * fail-closed reservation behavior because losing the generation advance is
     * worse than rejecting the mount after an interrupted publication.
     */
    if (backend == PQC_ANCHOR_BACKEND_HARDWARE)
        return pqc_anchor_store(&state);
    return 0;
}

int pqc_checkpoint_load_and_verify_anchor(const char *path,
                                          uint64_t expected_file_id,
                                          pqc_checkpoint_t *out)
{
    if (!out)
        return -EINVAL;

    pqc_checkpoint_t ckpt = {0};
    int rc = pqc_publish_checkpoint_load_xattr(path, expected_file_id, &ckpt);
    if (rc != 0)
        return rc;

    if (pqc_anchor_backend() == PQC_ANCHOR_BACKEND_DISABLED) {
        *out = ckpt;
        return 0;
    }

    pqc_anchor_state_t expected = {0};
    rc = pqc_checkpoint_record_file_anchor(ckpt.file_id, ckpt.sequence,
                                           ckpt.logical_size,
                                           ckpt.max_generation, &expected);
    if (rc != 0) {
        OQS_MEM_cleanse(&ckpt, sizeof(ckpt));
        return rc;
    }
    int anchor_rc = pqc_anchor_load(&expected);
    if (anchor_rc != 0) {
        OQS_MEM_cleanse(&ckpt, sizeof(ckpt));
        return anchor_rc;
    }

    *out = ckpt;
    return 0;
}
