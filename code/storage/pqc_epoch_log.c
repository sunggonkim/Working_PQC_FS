#include "pqc_epoch_log.h"

#include "pqc_checkpoint.h"
#include "pqc_config.h"
#include "pqc_durability.h"
#include "pqc_journal.h"
#include "pqc_posix.h"
#include "pqc_trace_sink.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include <openssl/crypto.h>
#include <openssl/evp.h>

static pqc_trace_sink_t g_epoch_log_trace_sink =
    PQC_TRACE_SINK_INITIALIZER;

static void put_u32_le(uint8_t *dst, uint32_t value)
{
    dst[0] = (uint8_t)(value & 0xffU);
    dst[1] = (uint8_t)((value >> 8) & 0xffU);
    dst[2] = (uint8_t)((value >> 16) & 0xffU);
    dst[3] = (uint8_t)((value >> 24) & 0xffU);
}

static void put_u64_le(uint8_t *dst, uint64_t value)
{
    for (size_t i = 0; i < 8; ++i)
        dst[i] = (uint8_t)((value >> (8U * i)) & 0xffU);
}

static uint32_t get_u32_le(const uint8_t *src)
{
    return ((uint32_t)src[0]) |
           ((uint32_t)src[1] << 8) |
           ((uint32_t)src[2] << 16) |
           ((uint32_t)src[3] << 24);
}

static uint64_t get_u64_le(const uint8_t *src)
{
    uint64_t value = 0;
    for (size_t i = 0; i < 8; ++i)
        value |= ((uint64_t)src[i]) << (8U * i);
    return value;
}

static int epoch_log_digest_with_ctx(const uint8_t *buf, size_t len,
                                     uint8_t out[32], EVP_MD_CTX *md)
{
    unsigned int out_len = 0;
    if (!buf || !out || !md)
        return -EINVAL;
    int ok = EVP_DigestInit_ex(md, EVP_sha256(), NULL) == 1 &&
             EVP_DigestUpdate(md, buf, len) == 1 &&
             EVP_DigestFinal_ex(md, out, &out_len) == 1 &&
             out_len == 32U;
    return ok ? 0 : -EIO;
}

static int epoch_log_digest(const uint8_t *buf, size_t len,
                            uint8_t out[32])
{
    EVP_MD_CTX *md = EVP_MD_CTX_new();
    if (!md)
        return -ENOMEM;
    int rc = epoch_log_digest_with_ctx(buf, len, out, md);
    EVP_MD_CTX_free(md);
    return rc;
}

static int epoch_log_validate_record(const pqc_epoch_log_record_t *record)
{
    if (!record)
        return -EINVAL;
    if (record->record_type != PQC_EPOCH_LOG_RECORD_BLOCK &&
        record->record_type != PQC_EPOCH_LOG_RECORD_COMMIT)
        return -EINVAL;
    if (record->algorithm_id != PQC_ALGO_AES_256_GCM)
        return -EINVAL;
    if (record->plaintext_length > PQC_LOGICAL_BLOCK_SIZE)
        return -EINVAL;
    if (record->record_type == PQC_EPOCH_LOG_RECORD_COMMIT &&
        record->plaintext_length != 0)
        return -EINVAL;
    return 0;
}

static int epoch_log_records_have_same_key(
    const pqc_epoch_log_record_t *a,
    const pqc_epoch_log_record_t *b)
{
    return a && b &&
           a->file_id == b->file_id &&
           a->logical_block == b->logical_block &&
           a->generation == b->generation;
}

static int epoch_log_key_seen_before_fd(int fd,
                                        off_t before_offset,
                                        const pqc_epoch_log_record_t *record,
                                        int *seen)
{
    if (fd < 0 || !record || !seen)
        return -EINVAL;
    *seen = 0;
    off_t offset = 0;
    while (offset < before_offset) {
        uint8_t encoded[PQC_EPOCH_LOG_RECORD_SIZE];
        ssize_t n = pread(fd, encoded, sizeof(encoded), offset);
        if (n < 0)
            return -errno;
        if (n != (ssize_t)sizeof(encoded))
            return -EIO;
        pqc_epoch_log_record_t previous;
        int rc = pqc_epoch_log_decode_record(encoded, sizeof(encoded),
                                             &previous);
        if (rc != 0)
            return rc;
        if (previous.record_type == PQC_EPOCH_LOG_RECORD_BLOCK &&
            epoch_log_records_have_same_key(&previous, record)) {
            *seen = 1;
            return 0;
        }
        offset += (off_t)sizeof(encoded);
    }
    return 0;
}

static void epoch_log_trace_write_line(const char *line, size_t len)
{
    (void)pqc_trace_sink_write_env(&g_epoch_log_trace_sink,
                                   "PQC_PUBLICATION_TRACE_PATH",
                                   line, len);
}

static int epoch_log_trace_enabled(void)
{
    return pqc_trace_sink_enabled_env(&g_epoch_log_trace_sink,
                                      "PQC_PUBLICATION_TRACE_PATH");
}

void pqc_epoch_log_trace_shutdown(void)
{
    pqc_trace_sink_close(&g_epoch_log_trace_sink);
}

static void epoch_log_trace_compaction(
    const char *log_path, int rc,
    const pqc_epoch_log_replay_summary_t *summary)
{
    if (!epoch_log_trace_enabled())
        return;
    char line[8192];
    int n = snprintf(
        line, sizeof(line),
        "{\"event\":\"epoch_replay_compact\",\"rc\":%d,"
        "\"log_path\":\"%s\",\"decoded_records\":%zu,"
        "\"block_records\":%zu,\"commit_records\":%zu,"
        "\"committed_records\":%zu,\"uncommitted_records\":%zu,"
        "\"duplicate_generation_records\":%zu,"
        "\"journal_repair_records\":%zu,"
        "\"torn_tail_bytes\":%zu,\"committed_epoch\":%llu,"
        "\"logical_size_after\":%llu,\"max_generation\":%llu}\n",
        rc,
        log_path ? log_path : "",
        summary ? summary->decoded_records : 0,
        summary ? summary->block_records : 0,
        summary ? summary->commit_records : 0,
        summary ? summary->committed_records : 0,
        summary ? summary->uncommitted_records : 0,
        summary ? summary->duplicate_generation_records : 0,
        summary ? summary->journal_repair_records : 0,
        summary ? summary->torn_tail_bytes : 0,
        (unsigned long long)(summary ? summary->committed_epoch : 0),
        (unsigned long long)(summary ? summary->logical_size_after : 0),
        (unsigned long long)(summary ? summary->max_generation : 0));
    if (n > 0 && (size_t)n < sizeof(line))
        epoch_log_trace_write_line(line, (size_t)n);
}

static int epoch_log_journal_has_mapping(int journal_fd,
                                         const pqc_epoch_log_record_t *record,
                                         int *has_mapping)
{
    if (!has_mapping)
        return -EINVAL;
    *has_mapping = 0;
    if (journal_fd < 0)
        return -EBADF;
    block_mapping_t existing;
    memset(&existing, 0, sizeof(existing));
    int rc = pqc_journal_lookup_mapping_committed(
        journal_fd, record->logical_block, record->generation, &existing);
    if (rc == -ENOENT)
        return 0;
    if (rc != 0)
        return rc;
    if (existing.generation != record->generation)
        return 0;
    *has_mapping =
        existing.ciphertext_offset == record->ciphertext_offset &&
        existing.plaintext_length == record->plaintext_length &&
        existing.algorithm_id == record->algorithm_id &&
        memcmp(existing.tag, record->tag, sizeof(existing.tag)) == 0;
    return 0;
}

static int epoch_log_repair_journal_prefix(
    int log_fd, int journal_fd, uint64_t expected_file_id,
    const pqc_epoch_log_replay_summary_t *summary,
    size_t *repair_count)
{
    if (!summary || journal_fd < 0)
        return -EINVAL;
    if (repair_count)
        *repair_count = 0;
    off_t offset = 0;
    while ((uint64_t)offset < summary->last_commit_offset) {
        uint8_t encoded[PQC_EPOCH_LOG_RECORD_SIZE];
        ssize_t n = pread(log_fd, encoded, sizeof(encoded), offset);
        if (n < 0)
            return -errno;
        if (n != (ssize_t)sizeof(encoded))
            return -EIO;
        pqc_epoch_log_record_t record;
        int rc = pqc_epoch_log_decode_record(encoded, sizeof(encoded),
                                             &record);
        if (rc != 0)
            return rc;
        if (expected_file_id != 0 && record.file_id != expected_file_id)
            return -ESTALE;
        if (record.record_type == PQC_EPOCH_LOG_RECORD_BLOCK) {
            int has_mapping = 0;
            rc = epoch_log_journal_has_mapping(journal_fd, &record,
                                               &has_mapping);
            if (rc != 0)
                return rc;
            if (!has_mapping) {
                block_mapping_t map = {
                    .logical_block = record.logical_block,
                    .generation = record.generation,
                    .ciphertext_offset = record.ciphertext_offset,
                    .plaintext_length = record.plaintext_length,
                    .algorithm_id = record.algorithm_id,
                };
                memcpy(map.tag, record.tag, sizeof(map.tag));
                rc = pqc_journal_append_mapping_unsynced(journal_fd, &map);
                if (rc != 0)
                    return rc;
                if (repair_count)
                    ++*repair_count;
            }
        }
        offset += (off_t)sizeof(encoded);
    }
    if (repair_count && *repair_count > 0) {
        return pqc_durability_fdatasync(
            journal_fd, PQC_DURABILITY_SITE_JOURNAL_SIDECAR);
    }
    return 0;
}

static int epoch_log_encode_record_with_ctx(
    const pqc_epoch_log_record_t *record,
    uint8_t *out, size_t out_len,
    size_t *written,
    EVP_MD_CTX *md)
{
    int rc = epoch_log_validate_record(record);
    if (rc != 0)
        return rc;
    if (!out || !md)
        return -EINVAL;
    if (out_len < PQC_EPOCH_LOG_RECORD_SIZE)
        return -EMSGSIZE;

    put_u64_le(out + 0, PQC_EPOCH_LOG_MAGIC);
    put_u32_le(out + 8, PQC_EPOCH_LOG_VERSION);
    put_u32_le(out + 12, record->record_type);
    put_u32_le(out + 16, record->flags);
    put_u32_le(out + 20, record->algorithm_id);
    put_u64_le(out + 24, record->epoch);
    put_u64_le(out + 32, record->sequence);
    put_u64_le(out + 40, record->file_id);
    put_u64_le(out + 48, record->logical_block);
    put_u64_le(out + 56, record->generation);
    put_u64_le(out + 64, record->ciphertext_offset);
    put_u64_le(out + 72, record->logical_size_after);
    put_u32_le(out + 80, record->plaintext_length);
    put_u32_le(out + 84, 0);
    memcpy(out + 88, record->tag, PQC_AEAD_TAG_SIZE);

    rc = epoch_log_digest_with_ctx(
        out, PQC_EPOCH_LOG_RECORD_DIGEST_OFFSET,
        out + PQC_EPOCH_LOG_RECORD_DIGEST_OFFSET, md);
    if (rc != 0)
        return rc;
    if (written)
        *written = PQC_EPOCH_LOG_RECORD_SIZE;
    return 0;
}

int pqc_epoch_log_encode_record(const pqc_epoch_log_record_t *record,
                                uint8_t *out, size_t out_len,
                                size_t *written)
{
    EVP_MD_CTX *md = EVP_MD_CTX_new();
    if (!md)
        return -ENOMEM;
    int rc = epoch_log_encode_record_with_ctx(record, out, out_len, written,
                                             md);
    EVP_MD_CTX_free(md);
    return rc;
}

int pqc_epoch_log_decode_record(const uint8_t *buf, size_t buf_len,
                                pqc_epoch_log_record_t *out)
{
    if (!buf || !out)
        return -EINVAL;
    if (buf_len < PQC_EPOCH_LOG_RECORD_SIZE)
        return -EMSGSIZE;
    if (get_u64_le(buf + 0) != PQC_EPOCH_LOG_MAGIC ||
        get_u32_le(buf + PQC_EPOCH_LOG_RECORD_VERSION_OFFSET) !=
            PQC_EPOCH_LOG_VERSION)
        return -EPROTO;

    uint8_t digest[32];
    int rc = epoch_log_digest(buf, PQC_EPOCH_LOG_RECORD_DIGEST_OFFSET,
                              digest);
    if (rc != 0)
        return rc;
    if (CRYPTO_memcmp(digest, buf + PQC_EPOCH_LOG_RECORD_DIGEST_OFFSET,
                      sizeof(digest)) != 0)
        return -EBADMSG;

    pqc_epoch_log_record_t decoded = {
        .record_type = get_u32_le(buf + 12),
        .flags = get_u32_le(buf + 16),
        .algorithm_id = get_u32_le(buf + 20),
        .epoch = get_u64_le(buf + 24),
        .sequence = get_u64_le(buf + 32),
        .file_id = get_u64_le(buf + 40),
        .logical_block = get_u64_le(buf + 48),
        .generation = get_u64_le(buf + 56),
        .ciphertext_offset = get_u64_le(buf + 64),
        .logical_size_after = get_u64_le(buf + 72),
        .plaintext_length = get_u32_le(buf + 80),
    };
    memcpy(decoded.tag, buf + 88, PQC_AEAD_TAG_SIZE);
    rc = epoch_log_validate_record(&decoded);
    if (rc != 0)
        return rc;
    *out = decoded;
    return 0;
}

int pqc_epoch_log_append_record_fd(int fd,
                                   const pqc_epoch_log_record_t *record)
{
    return pqc_epoch_log_append_records_fd(fd, record, 1);
}

int pqc_epoch_log_append_records_fd(int fd,
                                    const pqc_epoch_log_record_t *records,
                                    size_t count)
{
    if (fd < 0)
        return -EBADF;
    if (!records || count == 0)
        return -EINVAL;
    if (count > PQC_WRITEBACK_MAX_BLOCKS + 1U)
        return -E2BIG;

    uint8_t encoded[(PQC_WRITEBACK_MAX_BLOCKS + 1U) *
                    PQC_EPOCH_LOG_RECORD_SIZE];
    EVP_MD_CTX *md = EVP_MD_CTX_new();
    if (!md)
        return -ENOMEM;
    size_t total = 0;
    for (size_t i = 0; i < count; ++i) {
        size_t written = 0;
        int rc = epoch_log_encode_record_with_ctx(
            &records[i], encoded + total, sizeof(encoded) - total,
            &written, md);
        if (rc != 0) {
            size_t used = total;
            if (sizeof(encoded) - total >= PQC_EPOCH_LOG_RECORD_SIZE)
                used += PQC_EPOCH_LOG_RECORD_SIZE;
            OPENSSL_cleanse(encoded, used);
            EVP_MD_CTX_free(md);
            return rc;
        }
        total += written;
    }
    EVP_MD_CTX_free(md);

    size_t done = 0;
    while (done < total) {
        ssize_t n = write(fd, encoded + done, total - done);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            OPENSSL_cleanse(encoded, total);
            return -errno;
        }
        if (n == 0) {
            OPENSSL_cleanse(encoded, total);
            return -EIO;
        }
        done += (size_t)n;
    }
    OPENSSL_cleanse(encoded, total);
    return 0;
}

int pqc_epoch_log_replay_fd(int fd, uint64_t expected_file_id,
                            pqc_epoch_log_replay_summary_t *out)
{
    if (fd < 0 || !out)
        return -EINVAL;

    pqc_epoch_log_replay_summary_t summary;
    memset(&summary, 0, sizeof(summary));
    off_t offset = 0;
    int rc = 0;

    for (;;) {
        uint8_t encoded[PQC_EPOCH_LOG_RECORD_SIZE];
        ssize_t n = pread(fd, encoded, sizeof(encoded), offset);
        if (n < 0) {
            rc = -errno;
            break;
        }
        if (n == 0)
            break;
        if (n != (ssize_t)sizeof(encoded)) {
            summary.torn_tail_bytes = (size_t)n;
            break;
        }

        pqc_epoch_log_record_t record;
        rc = pqc_epoch_log_decode_record(encoded, sizeof(encoded), &record);
        if (rc != 0)
            break;
        if (expected_file_id != 0 && record.file_id != expected_file_id) {
            rc = -ESTALE;
            break;
        }
        if (summary.file_id == 0)
            summary.file_id = record.file_id;

        ++summary.decoded_records;
        if (record.record_type == PQC_EPOCH_LOG_RECORD_BLOCK) {
            int seen = 0;
            rc = epoch_log_key_seen_before_fd(fd, offset, &record, &seen);
            if (rc != 0)
                break;
            if (seen) {
                ++summary.duplicate_generation_records;
                rc = -EEXIST;
                break;
            }
            ++summary.block_records;
            if (summary.max_generation < record.generation)
                summary.max_generation = record.generation;
        } else if (record.record_type == PQC_EPOCH_LOG_RECORD_COMMIT) {
            ++summary.commit_records;
            summary.committed_epoch = record.epoch;
            summary.committed_sequence = record.sequence;
            summary.logical_size_after = record.logical_size_after;
            if (summary.max_generation < record.generation)
                summary.max_generation = record.generation;
            summary.committed_records = summary.decoded_records;
            summary.last_commit_offset =
                (uint64_t)offset + PQC_EPOCH_LOG_RECORD_SIZE;
        } else {
            rc = -EPROTO;
            break;
        }
        offset += (off_t)sizeof(encoded);
    }

    summary.uncommitted_records =
        summary.decoded_records >= summary.committed_records
            ? summary.decoded_records - summary.committed_records
            : 0;
    *out = summary;
    return rc;
}

int pqc_epoch_log_open_replay_path(const char *marker_path,
                                   uint64_t expected_file_id,
                                   int *fd_out,
                                   pqc_epoch_log_replay_summary_t *out)
{
    if (fd_out)
        *fd_out = -1;
    if (!marker_path || !fd_out || !out)
        return -EINVAL;

    char log_path[4096 + 16];
    int rc = pqc_sidecar_path(log_path, sizeof(log_path), marker_path,
                              ".pqcepoch");
    if (rc != 0)
        return rc;
    int fd = open(log_path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return errno == ENOENT ? -ENOENT : -errno;

    rc = pqc_epoch_log_replay_fd(fd, expected_file_id, out);
    if (rc != 0) {
        int saved_rc = rc;
        int close_errno = 0;
        if (close(fd) != 0)
            close_errno = errno;
        return saved_rc ? saved_rc : -close_errno;
    }
    *fd_out = fd;
    return 0;
}

int pqc_epoch_log_replay_path(const char *marker_path,
                              uint64_t expected_file_id,
                              pqc_epoch_log_replay_summary_t *out)
{
    int fd = -1;
    int rc = pqc_epoch_log_open_replay_path(marker_path, expected_file_id,
                                            &fd, out);
    int close_errno = 0;
    if (fd >= 0 && close(fd) != 0)
        close_errno = errno;
    if (rc == 0 && close_errno != 0)
        rc = -close_errno;
    return rc;
}

int pqc_epoch_log_compact_checkpoint(const char *marker_path,
                                     uint64_t expected_file_id,
                                     int journal_fd,
                                     uint64_t journal_max_generation,
                                     pqc_epoch_log_replay_summary_t *out)
{
    if (!marker_path || expected_file_id == 0)
        return -EINVAL;

    char log_path[4096 + 16];
    int rc = pqc_sidecar_path(log_path, sizeof(log_path), marker_path,
                              ".pqcepoch");
    if (rc != 0)
        return rc;
    int fd = open(log_path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return errno == ENOENT ? -ENOENT : -errno;

    pqc_epoch_log_replay_summary_t summary;
    memset(&summary, 0, sizeof(summary));
    rc = pqc_epoch_log_replay_fd(fd, expected_file_id, &summary);
    if (rc != 0) {
        int saved_rc = rc;
        int close_errno = 0;
        if (close(fd) != 0)
            close_errno = errno;
        rc = saved_rc ? saved_rc : -close_errno;
        if (out)
            *out = summary;
        epoch_log_trace_compaction(log_path, rc, &summary);
        return rc;
    }
    if (summary.committed_records == 0) {
        rc = -ENOENT;
    } else if (journal_max_generation < summary.max_generation) {
        if (journal_fd < 0) {
            rc = -EAGAIN;
        } else {
            size_t repaired = 0;
            rc = epoch_log_repair_journal_prefix(fd, journal_fd,
                                                 expected_file_id, &summary,
                                                 &repaired);
            summary.journal_repair_records = repaired;
        }
        if (rc == 0)
            journal_max_generation = summary.max_generation;
    }
    if (rc == 0 && journal_max_generation >= summary.max_generation) {
        rc = pqc_checkpoint_store_and_stage_anchor(
            marker_path, expected_file_id, summary.committed_epoch,
            summary.logical_size_after, summary.max_generation);
    }
    int close_errno = 0;
    if (close(fd) != 0)
        close_errno = errno;
    if (rc == 0 && close_errno != 0)
        rc = -close_errno;
    if (out)
        *out = summary;
    epoch_log_trace_compaction(log_path, rc, &summary);
    return rc;
}

int pqc_epoch_log_lookup_mapping_committed_fd(
    int fd,
    const pqc_epoch_log_replay_summary_t *summary,
    uint64_t expected_file_id,
    uint64_t logical_block,
    uint64_t max_generation,
    block_mapping_t *out)
{
    if (fd < 0 || !summary || !out)
        return -EINVAL;
    if (summary->last_commit_offset == 0)
        return -ENOENT;

    block_mapping_t best;
    memset(&best, 0, sizeof(best));
    int found = 0;
    off_t offset = 0;
    int rc = 0;
    while ((uint64_t)offset < summary->last_commit_offset) {
        uint8_t encoded[PQC_EPOCH_LOG_RECORD_SIZE];
        ssize_t n = pread(fd, encoded, sizeof(encoded), offset);
        if (n < 0) {
            rc = -errno;
            break;
        }
        if (n != (ssize_t)sizeof(encoded)) {
            rc = -EIO;
            break;
        }

        pqc_epoch_log_record_t record;
        rc = pqc_epoch_log_decode_record(encoded, sizeof(encoded), &record);
        if (rc != 0)
            break;
        if (expected_file_id != 0 && record.file_id != expected_file_id) {
            rc = -ESTALE;
            break;
        }
        if (record.record_type == PQC_EPOCH_LOG_RECORD_BLOCK &&
            record.logical_block == logical_block &&
            record.generation <= max_generation &&
            (!found || record.generation > best.generation)) {
            best.logical_block = record.logical_block;
            best.generation = record.generation;
            best.ciphertext_offset = record.ciphertext_offset;
            best.plaintext_length = record.plaintext_length;
            best.algorithm_id = record.algorithm_id;
            memcpy(best.tag, record.tag, sizeof(best.tag));
            found = 1;
        }
        offset += (off_t)sizeof(encoded);
    }

    if (rc != 0)
        return rc;
    if (!found)
        return -ENOENT;
    *out = best;
    return 0;
}

void pqc_epoch_log_lookup_view_init(pqc_epoch_log_lookup_view_t *view,
                                    uint64_t first_block,
                                    uint64_t last_block,
                                    uint64_t max_generation,
                                    uint64_t file_id)
{
    if (!view)
        return;
    memset(view, 0, sizeof(*view));
    view->first_block = first_block;
    view->last_block = last_block;
    view->max_generation = max_generation;
    view->file_id = file_id;
    if (last_block < first_block) {
        view->initialized = 1;
        return;
    }
    uint64_t count = last_block - first_block + 1U;
    if (count > PQC_EPOCH_LOG_LOOKUP_VIEW_MAX_BLOCKS) {
        view->overflow = 1;
        return;
    }
    view->slot_count = (size_t)count;
}

void pqc_epoch_log_lookup_view_clear(pqc_epoch_log_lookup_view_t *view)
{
    if (!view)
        return;
    memset(view, 0, sizeof(*view));
}

static int pqc_epoch_log_lookup_view_prepare_fd(
    int fd,
    const pqc_epoch_log_replay_summary_t *summary,
    pqc_epoch_log_lookup_view_t *view,
    uint64_t expected_file_id)
{
    if (fd < 0 || !summary || !view)
        return -EINVAL;
    if (view->overflow)
        return -E2BIG;
    if (view->initialized)
        return 0;

    view->initialized = 1;
    view->summary = *summary;
    if (summary->last_commit_offset == 0) {
        view->available = 0;
        return 0;
    }
    view->available = 1;

    off_t offset = 0;
    int rc = 0;
    while ((uint64_t)offset < summary->last_commit_offset) {
        uint8_t encoded[PQC_EPOCH_LOG_RECORD_SIZE];
        ssize_t n = pread(fd, encoded, sizeof(encoded), offset);
        if (n < 0) {
            rc = -errno;
            break;
        }
        if (n != (ssize_t)sizeof(encoded)) {
            rc = -EIO;
            break;
        }

        pqc_epoch_log_record_t record;
        rc = pqc_epoch_log_decode_record(encoded, sizeof(encoded), &record);
        if (rc != 0)
            break;
        if (expected_file_id != 0 && record.file_id != expected_file_id) {
            rc = -ESTALE;
            break;
        }
        if (record.record_type != PQC_EPOCH_LOG_RECORD_BLOCK ||
            record.generation > view->max_generation ||
            record.logical_block < view->first_block ||
            record.logical_block > view->last_block) {
            offset += (off_t)sizeof(encoded);
            continue;
        }
        size_t idx = (size_t)(record.logical_block - view->first_block);
        if (idx >= view->slot_count) {
            offset += (off_t)sizeof(encoded);
            continue;
        }
        if (!view->present[idx] ||
            record.generation > view->mappings[idx].generation) {
            block_mapping_t map = {
                .logical_block = record.logical_block,
                .generation = record.generation,
                .ciphertext_offset = record.ciphertext_offset,
                .plaintext_length = record.plaintext_length,
                .algorithm_id = record.algorithm_id,
            };
            memcpy(map.tag, record.tag, sizeof(map.tag));
            view->mappings[idx] = map;
            view->present[idx] = 1;
        }
        offset += (off_t)sizeof(encoded);
    }
    return rc;
}

int pqc_epoch_log_lookup_mapping_committed_view(
    const pqc_epoch_log_lookup_view_t *view,
    uint64_t expected_file_id,
    uint64_t logical_block,
    uint64_t max_generation,
    block_mapping_t *out)
{
    if (!out)
        return -EINVAL;
    if (!view || !view->initialized || view->overflow ||
        !view->available ||
        expected_file_id != view->file_id ||
        max_generation != view->max_generation ||
        logical_block < view->first_block ||
        logical_block > view->last_block)
        return -ENOENT;

    size_t idx = (size_t)(logical_block - view->first_block);
    if (idx >= view->slot_count || !view->present[idx])
        return -ENOENT;
    *out = view->mappings[idx];
    return 0;
}

int pqc_epoch_log_lookup_mapping_committed_fd_view(
    int fd,
    const pqc_epoch_log_replay_summary_t *summary,
    pqc_epoch_log_lookup_view_t *view,
    uint64_t expected_file_id,
    uint64_t logical_block,
    uint64_t max_generation,
    block_mapping_t *out)
{
    if (!view || view->overflow ||
        logical_block < view->first_block ||
        logical_block > view->last_block ||
        max_generation != view->max_generation ||
        expected_file_id != view->file_id) {
        return pqc_epoch_log_lookup_mapping_committed_fd(
            fd, summary, expected_file_id, logical_block, max_generation,
            out);
    }
    int rc = pqc_epoch_log_lookup_view_prepare_fd(
        fd, summary, view, expected_file_id);
    if (rc != 0)
        return rc;
    return pqc_epoch_log_lookup_mapping_committed_view(
        view, expected_file_id, logical_block, max_generation, out);
}

int pqc_epoch_log_lookup_mapping_committed(const char *marker_path,
                                           uint64_t expected_file_id,
                                           uint64_t logical_block,
                                           uint64_t max_generation,
                                           block_mapping_t *out)
{
    int fd = -1;
    pqc_epoch_log_replay_summary_t summary;
    memset(&summary, 0, sizeof(summary));
    int rc = pqc_epoch_log_open_replay_path(marker_path, expected_file_id,
                                            &fd, &summary);
    if (rc != 0)
        return rc;

    rc = pqc_epoch_log_lookup_mapping_committed_fd(
        fd, &summary, expected_file_id, logical_block, max_generation, out);
    int close_errno = 0;
    if (close(fd) != 0)
        close_errno = errno;
    if (rc == 0 && close_errno != 0)
        rc = -close_errno;
    return rc;
}
