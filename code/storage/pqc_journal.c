#include "pqc_journal.h"

#include "pqc_durability.h"

#include <errno.h>
#include <stddef.h>
#include <string.h>
#include <unistd.h>

#include <openssl/evp.h>

#define PQC_JOURNAL_HIGHWATER_BLOCK UINT64_MAX

static int journal_digest_with_ctx(journal_record_t *record, EVP_MD_CTX *md)
{
    unsigned int out_len = 0;
    if (!record || !md)
        return -EINVAL;
    int ok = EVP_DigestInit_ex(md, EVP_sha256(), NULL) == 1 &&
             EVP_DigestUpdate(md, record,
                              offsetof(journal_record_t, digest)) == 1 &&
             EVP_DigestFinal_ex(md, record->digest, &out_len) == 1 &&
             out_len == sizeof(record->digest);
    return ok ? 0 : -EIO;
}

static int journal_digest(journal_record_t *record)
{
    EVP_MD_CTX *md = EVP_MD_CTX_new();
    if (!md)
        return -ENOMEM;
    int rc = journal_digest_with_ctx(record, md);
    EVP_MD_CTX_free(md);
    return rc;
}

static int journal_record_valid(journal_record_t *record)
{
    uint8_t saved[sizeof(record->digest)];
    if (record->magic != PQC_JOURNAL_MAGIC ||
        record->version != PQC_JOURNAL_VERSION ||
        record->committed != PQC_JOURNAL_COMMITTED ||
        record->mapping.plaintext_length > PQC_LOGICAL_BLOCK_SIZE ||
        record->mapping.algorithm_id != PQC_ALGO_AES_256_GCM)
        return 0;
    memcpy(saved, record->digest, sizeof(saved));
    return journal_digest(record) == 0 &&
           memcmp(saved, record->digest, sizeof(saved)) == 0;
}

static int journal_record_is_data_mapping(const journal_record_t *record)
{
    return record && record->mapping.logical_block !=
        PQC_JOURNAL_HIGHWATER_BLOCK;
}

static int journal_record_is_highwater(const journal_record_t *record,
                                       off_t record_offset)
{
    return record &&
        record->mapping.logical_block == PQC_JOURNAL_HIGHWATER_BLOCK &&
        record->mapping.plaintext_length == 0 &&
        record->mapping.ciphertext_offset == (uint64_t)record_offset;
}

static int journal_write_full(int fd, const void *buf, size_t len)
{
    const uint8_t *p = (const uint8_t *)buf;
    size_t done = 0;
    while (done < len) {
        ssize_t n = write(fd, p + done, len - done);
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

static int journal_record_from_mapping(journal_record_t *record,
                                       const block_mapping_t *mapping)
{
    if (!record || !mapping ||
        mapping->plaintext_length > PQC_LOGICAL_BLOCK_SIZE ||
        mapping->algorithm_id != PQC_ALGO_AES_256_GCM)
        return -EINVAL;
    *record = (journal_record_t) {
        .magic = PQC_JOURNAL_MAGIC,
        .version = PQC_JOURNAL_VERSION,
        .committed = PQC_JOURNAL_COMMITTED,
        .mapping = *mapping,
    };
    return journal_digest(record);
}

static int journal_record_from_mapping_with_ctx(journal_record_t *record,
                                                const block_mapping_t *mapping,
                                                EVP_MD_CTX *md)
{
    if (!record || !mapping ||
        mapping->plaintext_length > PQC_LOGICAL_BLOCK_SIZE ||
        mapping->algorithm_id != PQC_ALGO_AES_256_GCM)
        return -EINVAL;
    *record = (journal_record_t) {
        .magic = PQC_JOURNAL_MAGIC,
        .version = PQC_JOURNAL_VERSION,
        .committed = PQC_JOURNAL_COMMITTED,
        .mapping = *mapping,
    };
    return journal_digest_with_ctx(record, md);
}

int pqc_journal_append_mapping(int journal_fd, const block_mapping_t *mapping)
{
    journal_record_t record;
    int rc = journal_record_from_mapping(&record, mapping);
    if (rc)
        return rc;
    rc = journal_write_full(journal_fd, &record, sizeof(record));
    if (rc != 0)
        return rc;
    return pqc_durability_fdatasync(
        journal_fd, PQC_DURABILITY_SITE_JOURNAL_SIDECAR);
}

int pqc_journal_append_mapping_unsynced(int journal_fd,
                                        const block_mapping_t *mapping)
{
    journal_record_t record;
    int rc = journal_record_from_mapping(&record, mapping);
    if (rc)
        return rc;
    return journal_write_full(journal_fd, &record, sizeof(record));
}

int pqc_journal_append_highwater_unsynced(int journal_fd,
                                          uint64_t max_generation)
{
    if (journal_fd < 0 || max_generation == 0)
        return -EINVAL;
    off_t pos = lseek(journal_fd, 0, SEEK_END);
    if (pos < 0)
        return -errno;
    journal_record_t record = {
        .magic = PQC_JOURNAL_MAGIC,
        .version = PQC_JOURNAL_VERSION,
        .committed = PQC_JOURNAL_COMMITTED,
        .mapping = {
            .logical_block = PQC_JOURNAL_HIGHWATER_BLOCK,
            .generation = max_generation,
            .ciphertext_offset = (uint64_t)pos,
            .plaintext_length = 0,
            .algorithm_id = PQC_ALGO_AES_256_GCM,
        },
    };
    int rc = journal_digest(&record);
    if (rc)
        return rc;
    return journal_write_full(journal_fd, &record, sizeof(record));
}

int pqc_journal_append_mappings_with_highwater_unsynced(
    int journal_fd,
    const block_mapping_t *mappings,
    size_t count,
    uint64_t max_generation)
{
    if (journal_fd < 0 || !mappings || count == 0 || max_generation == 0)
        return -EINVAL;
    if (count > PQC_WRITEBACK_MAX_BLOCKS)
        return -E2BIG;
    off_t pos = lseek(journal_fd, 0, SEEK_END);
    if (pos < 0)
        return -errno;

    return pqc_journal_append_mappings_with_highwater_at_unsynced(
        journal_fd, mappings, count, max_generation, (uint64_t)pos, NULL);
}

int pqc_journal_append_mappings_with_highwater_at_unsynced(
    int journal_fd,
    const block_mapping_t *mappings,
    size_t count,
    uint64_t max_generation,
    uint64_t append_offset,
    uint64_t *end_after_append)
{
    if (journal_fd < 0 || !mappings || count == 0 || max_generation == 0)
        return -EINVAL;
    if (count > PQC_WRITEBACK_MAX_BLOCKS)
        return -E2BIG;
    uint64_t mapping_bytes = (uint64_t)count * sizeof(journal_record_t);
    uint64_t total_bytes = mapping_bytes + sizeof(journal_record_t);
    if (mapping_bytes / sizeof(journal_record_t) != (uint64_t)count ||
        append_offset > UINT64_MAX - total_bytes)
        return -EOVERFLOW;
    journal_record_t records[PQC_WRITEBACK_MAX_BLOCKS + 1U];
    EVP_MD_CTX *md = EVP_MD_CTX_new();
    if (!md)
        return -ENOMEM;
    for (size_t i = 0; i < count; ++i) {
        int rc = journal_record_from_mapping_with_ctx(
            &records[i], &mappings[i], md);
        if (rc != 0) {
            EVP_MD_CTX_free(md);
            return rc;
        }
    }
    records[count] = (journal_record_t) {
        .magic = PQC_JOURNAL_MAGIC,
        .version = PQC_JOURNAL_VERSION,
        .committed = PQC_JOURNAL_COMMITTED,
        .mapping = {
            .logical_block = PQC_JOURNAL_HIGHWATER_BLOCK,
            .generation = max_generation,
            .ciphertext_offset =
                append_offset + mapping_bytes,
            .plaintext_length = 0,
            .algorithm_id = PQC_ALGO_AES_256_GCM,
        },
    };
    int rc = journal_digest_with_ctx(&records[count], md);
    EVP_MD_CTX_free(md);
    if (rc != 0)
        return rc;
    rc = journal_write_full(journal_fd, records, (size_t)total_bytes);
    if (rc == 0 && end_after_append)
        *end_after_append = append_offset + total_bytes;
    return rc;
}

int pqc_journal_lookup_mapping(int journal_fd, uint64_t logical_block,
                               block_mapping_t *out)
{
    return pqc_journal_lookup_mapping_committed(journal_fd, logical_block,
                                                UINT64_MAX, out);
}

int pqc_journal_lookup_mapping_committed(int journal_fd,
                                         uint64_t logical_block,
                                         uint64_t max_generation,
                                         block_mapping_t *out)
{
    if (!out)
        return -EINVAL;
    journal_record_t record;
    block_mapping_t best = {0};
    int found = 0;
    off_t pos = 0;
    ssize_t n = 0;
    while ((n = pread(journal_fd, &record, sizeof(record), pos)) ==
           (ssize_t)sizeof(record)) {
        pos += (off_t)sizeof(record);
        if (journal_record_valid(&record) &&
            journal_record_is_data_mapping(&record) &&
            record.mapping.logical_block == logical_block &&
            record.mapping.generation <= max_generation &&
            (!found || record.mapping.generation > best.generation)) {
            best = record.mapping;
            found = 1;
        }
    }
    if (n < 0)
        return -errno;
    if (!found)
        return -ENOENT;
    *out = best;
    return 0;
}

void pqc_journal_lookup_view_init(pqc_journal_lookup_view_t *view,
                                  uint64_t first_block,
                                  uint64_t last_block,
                                  uint64_t max_generation)
{
    if (!view)
        return;
    memset(view, 0, sizeof(*view));
    view->first_block = first_block;
    view->last_block = last_block;
    view->max_generation = max_generation;
    if (last_block < first_block) {
        view->initialized = 1;
        return;
    }
    uint64_t count = last_block - first_block + 1U;
    if (count > PQC_JOURNAL_LOOKUP_VIEW_MAX_BLOCKS) {
        view->overflow = 1;
        return;
    }
    view->slot_count = (size_t)count;
}

void pqc_journal_lookup_view_clear(pqc_journal_lookup_view_t *view)
{
    if (!view)
        return;
    memset(view, 0, sizeof(*view));
}

static int pqc_journal_lookup_view_prepare(
    int journal_fd,
    pqc_journal_lookup_view_t *view)
{
    if (journal_fd < 0 || !view)
        return -EINVAL;
    if (view->overflow)
        return -E2BIG;
    if (view->initialized)
        return 0;

    view->initialized = 1;
    journal_record_t record;
    off_t pos = 0;
    ssize_t n = 0;
    while ((n = pread(journal_fd, &record, sizeof(record), pos)) ==
           (ssize_t)sizeof(record)) {
        pos += (off_t)sizeof(record);
        if (!journal_record_valid(&record) ||
            !journal_record_is_data_mapping(&record) ||
            record.mapping.generation > view->max_generation ||
            record.mapping.logical_block < view->first_block ||
            record.mapping.logical_block > view->last_block)
            continue;
        size_t idx = (size_t)(record.mapping.logical_block -
                              view->first_block);
        if (idx >= view->slot_count)
            continue;
        if (!view->present[idx] ||
            record.mapping.generation > view->mappings[idx].generation) {
            view->mappings[idx] = record.mapping;
            view->present[idx] = 1;
        }
    }
    return n < 0 ? -errno : 0;
}

int pqc_journal_lookup_mapping_committed_view(
    int journal_fd,
    pqc_journal_lookup_view_t *view,
    uint64_t logical_block,
    uint64_t max_generation,
    block_mapping_t *out)
{
    if (!out)
        return -EINVAL;
    if (!view || view->overflow ||
        logical_block < view->first_block ||
        logical_block > view->last_block ||
        max_generation != view->max_generation) {
        return pqc_journal_lookup_mapping_committed(
            journal_fd, logical_block, max_generation, out);
    }

    int rc = pqc_journal_lookup_view_prepare(journal_fd, view);
    if (rc != 0)
        return rc;
    size_t idx = (size_t)(logical_block - view->first_block);
    if (idx >= view->slot_count || !view->present[idx])
        return -ENOENT;
    *out = view->mappings[idx];
    return 0;
}

int pqc_journal_tail_highwater_generation(int journal_fd,
                                          uint64_t *max_generation)
{
    if (journal_fd < 0 || !max_generation)
        return -EINVAL;
    off_t end = lseek(journal_fd, 0, SEEK_END);
    if (end < 0)
        return -errno;
    return pqc_journal_tail_highwater_generation_at(
        journal_fd, (uint64_t)end, max_generation);
}

int pqc_journal_tail_highwater_generation_at(int journal_fd,
                                             uint64_t journal_end,
                                             uint64_t *max_generation)
{
    if (journal_fd < 0 || !max_generation)
        return -EINVAL;
    if (journal_end > (uint64_t)INT64_MAX)
        return -EOVERFLOW;
    off_t end = (off_t)journal_end;
    if (end < (off_t)sizeof(journal_record_t))
        return -ENOENT;
    off_t pos = end - (off_t)sizeof(journal_record_t);
    journal_record_t record;
    ssize_t n = pread(journal_fd, &record, sizeof(record), pos);
    if (n != (ssize_t)sizeof(record))
        return n < 0 ? -errno : -ENOENT;
    if (!journal_record_valid(&record) ||
        !journal_record_is_highwater(&record, pos))
        return -ENOENT;
    *max_generation = record.mapping.generation;
    return 0;
}

uint64_t pqc_journal_max_generation(int journal_fd)
{
    journal_record_t record;
    uint64_t max = 0;
    off_t pos = 0;
    while (pread(journal_fd, &record, sizeof(record), pos) ==
           (ssize_t)sizeof(record)) {
        pos += (off_t)sizeof(record);
        if (journal_record_valid(&record) && record.mapping.generation > max)
            max = record.mapping.generation;
    }
    return max;
}
