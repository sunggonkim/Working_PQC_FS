#include "pqc_strict_publish.h"

#include "pqc_checkpoint.h"
#include "pqc_durability.h"
#include "pqc_journal.h"
#include "pqc_publish.h"
#include "pqc_test_hooks.h"

#include <errno.h>
#include <string.h>
#include <unistd.h>

static int publish_pwrite_full(int fd, const uint8_t *buf, size_t len,
                               off_t offset)
{
    size_t done = 0;
    while (done < len) {
        ssize_t n = pwrite(fd, buf + done, len - done, offset + (off_t)done);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            return -errno;
        }
        if (n == 0)
            return -EIO;
        done += (size_t)n;
    }
    return 0;
}

int pqc_strict_publish_commit(const pqc_strict_publish_request_t *req)
{
    if (!req || !req->marker_path || !req->blocks || !req->cipher_batch ||
        !req->data_sidecar_dirty || !req->journal_sidecar_dirty ||
        !req->data_sidecar_dirty_epoch || !req->journal_sidecar_dirty_epoch ||
        !req->logical_size)
        return -EINVAL;
    if (req->block_count == 0 || req->reserved_generation == 0)
        return -EINVAL;

    int res = 0;
    int data_fsync_deferred = 0;
    if (req->data_sidecar_end_after_append_valid)
        *req->data_sidecar_end_after_append_valid = 0;
    if (req->journal_sidecar_end_after_append_valid)
        *req->journal_sidecar_end_after_append_valid = 0;

    /* Phase 1: append the encrypted batch.  None is visible yet because the
     * journal or epoch log has not published a mapping. */
    size_t cipher_bytes = 0;
    for (size_t bi = 0; bi < req->block_count; ++bi) {
        pqc_crypto_block_desc_t *block = &req->blocks[bi];
        size_t end = block->output_offset + (size_t)block->length;
        if (end < block->output_offset) {
            res = -EOVERFLOW;
            break;
        }
        if (end > cipher_bytes)
            cipher_bytes = end;
    }
    off_t batch_pos = -1;
    if (res == 0 && req->data_sidecar_append_offset_valid) {
        if (req->data_sidecar_append_offset > (uint64_t)INT64_MAX) {
            res = -EOVERFLOW;
        } else {
            batch_pos = (off_t)req->data_sidecar_append_offset;
        }
    } else if (res == 0) {
        batch_pos = lseek(req->data_fd, 0, SEEK_END);
        if (batch_pos < 0)
            res = -errno;
    }
    if (res == 0)
        res = publish_pwrite_full(req->data_fd, req->cipher_batch,
                                  cipher_bytes, batch_pos);
    if (res == 0 && req->data_sidecar_end_after_append &&
        req->data_sidecar_end_after_append_valid) {
        *req->data_sidecar_end_after_append =
            (uint64_t)batch_pos + (uint64_t)cipher_bytes;
        *req->data_sidecar_end_after_append_valid = 1;
    }
    for (size_t bi = 0; res == 0 && bi < req->block_count; ++bi) {
        pqc_crypto_block_desc_t *block = &req->blocks[bi];
        ++*req->data_sidecar_dirty_epoch;
        *req->data_sidecar_dirty = 1;
        block->ciphertext_offset =
            (uint64_t)batch_pos + (uint64_t)block->output_offset;
        pqc_fault_cutpoint("data_write_after_pwrite");
    }

    /* Phase 2: establish the data-before-metadata durability boundary once.
     * Strict mode uses a direct data-sidecar fdatasync.  Epoch mode may hand
     * this boundary to a later syncfs barrier that covers the data sidecar and
     * epoch log on the same filesystem. */
    if (res == 0) {
        if (req->skip_data_fsync) {
            if (req->after_data_fsync) {
                res = req->after_data_fsync(
                    req, req->after_data_fsync_opaque);
            } else if (req->defer_data_fsync) {
                pqc_fault_cutpoint("data_fsync_deferred_after");
            } else {
                res = -EINVAL;
            }
            if (res == 0) {
                if (req->after_metadata_publish) {
                    data_fsync_deferred = 1;
                } else if (!req->defer_data_fsync) {
                    *req->data_sidecar_dirty = 0;
                    pqc_fault_cutpoint("data_fsync_skipped_epoch_after");
                }
            }
        } else {
            res = pqc_durability_fdatasync(
                req->data_fd, PQC_DURABILITY_SITE_DATA_SIDECAR);
            if (res == 0) {
                *req->data_sidecar_dirty = 0;
                pqc_fault_cutpoint("data_fsync_after");
                if (req->after_data_fsync) {
                    res = req->after_data_fsync(
                        req, req->after_data_fsync_opaque);
                }
            }
        }
    }

    /* Phase 3: append all committed mappings, then publish them with one
     * journal barrier.  Recovery ignores any torn tail without a full record. */
    if (res == 0 && req->skip_journal_append) {
        if (req->journal_sidecar_epoch_repairable)
            *req->journal_sidecar_epoch_repairable = 1;
        if (req->journal_sidecar_epoch_repairable_epoch)
            *req->journal_sidecar_epoch_repairable_epoch =
                *req->journal_sidecar_dirty_epoch;
        pqc_fault_cutpoint("journal_append_skipped_epoch_after");
    } else {
        if (req->block_count > PQC_WRITEBACK_MAX_BLOCKS) {
            res = -E2BIG;
        } else {
            block_mapping_t mappings[PQC_WRITEBACK_MAX_BLOCKS];
            int journal_end_after_append_reported = 0;
            for (size_t bi = 0; bi < req->block_count; ++bi) {
                const pqc_crypto_block_desc_t *block = &req->blocks[bi];
                mappings[bi] = (block_mapping_t) {
                    .logical_block = block->block,
                    .generation = block->generation,
                    .ciphertext_offset = block->ciphertext_offset,
                    .plaintext_length = block->length,
                    .algorithm_id = req->algorithm_id,
                };
                memcpy(mappings[bi].tag, block->tag,
                       sizeof(mappings[bi].tag));
            }
            if (req->journal_sidecar_append_offset_valid) {
                res = pqc_journal_append_mappings_with_highwater_at_unsynced(
                    req->journal_fd, mappings, req->block_count,
                    req->reserved_generation,
                    req->journal_sidecar_append_offset,
                    req->journal_sidecar_end_after_append);
                journal_end_after_append_reported = 1;
            } else {
                res = pqc_journal_append_mappings_with_highwater_unsynced(
                    req->journal_fd, mappings, req->block_count,
                    req->reserved_generation);
            }
            if (res == 0 && journal_end_after_append_reported &&
                req->journal_sidecar_end_after_append &&
                req->journal_sidecar_end_after_append_valid)
                *req->journal_sidecar_end_after_append_valid = 1;
        }
        for (size_t bi = 0; res == 0 && bi < req->block_count; ++bi) {
            const pqc_crypto_block_desc_t *block = &req->blocks[bi];
            (void)block;
            ++*req->journal_sidecar_dirty_epoch;
            *req->journal_sidecar_dirty = 1;
            if (req->journal_sidecar_epoch_repairable)
                *req->journal_sidecar_epoch_repairable = 0;
            if (req->journal_sidecar_epoch_repairable_epoch)
                *req->journal_sidecar_epoch_repairable_epoch = 0;
            pqc_fault_cutpoint("journal_append_after");
        }
        if (res == 0) {
            ++*req->journal_sidecar_dirty_epoch;
            *req->journal_sidecar_dirty = 1;
            pqc_fault_cutpoint("journal_highwater_append_after");
        }
    }

    if (res == 0 && req->skip_journal_append) {
        pqc_fault_cutpoint("journal_fsync_skipped_epoch_after");
    } else if (res == 0 && req->skip_journal_fsync) {
        if (req->defer_journal_fsync) {
            pqc_fault_cutpoint("journal_fsync_deferred_after");
        } else {
            if (req->journal_sidecar_epoch_repairable)
                *req->journal_sidecar_epoch_repairable = 1;
            if (req->journal_sidecar_epoch_repairable_epoch)
                *req->journal_sidecar_epoch_repairable_epoch =
                    *req->journal_sidecar_dirty_epoch;
        }
        pqc_fault_cutpoint("journal_fsync_skipped_epoch_after");
    } else if (res == 0) {
        res = pqc_durability_fdatasync(
            req->journal_fd, PQC_DURABILITY_SITE_JOURNAL_SIDECAR);
        if (res == 0) {
            *req->journal_sidecar_dirty = 0;
            if (req->journal_sidecar_epoch_repairable)
                *req->journal_sidecar_epoch_repairable = 0;
            if (req->journal_sidecar_epoch_repairable_epoch)
                *req->journal_sidecar_epoch_repairable_epoch = 0;
            pqc_fault_cutpoint("journal_fsync_after");
        }
    }

    if (res == 0) {
        uint64_t previous_logical_size = *req->logical_size;
        *req->logical_size = req->final_size;
        if (req->final_size != previous_logical_size) {
            res = pqc_publish_logical_size_store(req->marker_path,
                                                 req->final_size);
            pqc_fault_cutpoint("logical_size_xattr_after");
        }
        if (res == 0) {
            res = pqc_checkpoint_store_and_stage_anchor(
                req->marker_path, req->file_id, req->reserved_generation,
                *req->logical_size, req->reserved_generation);
        }
        if (res == 0 && req->final_size != previous_logical_size &&
            ftruncate(req->storage_fd,
                                  (off_t)req->final_size) != 0)
            res = -errno;
    }

    if (res == 0 && req->after_metadata_publish)
        res = req->after_metadata_publish(
            req, req->after_metadata_publish_opaque);

    if (res == 0 && data_fsync_deferred && !req->defer_data_fsync) {
        *req->data_sidecar_dirty = 0;
        pqc_fault_cutpoint("data_fsync_skipped_epoch_after");
    }

    return res;
}
