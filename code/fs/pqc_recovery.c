#include "pqc_recovery.h"

#include "pqc_crypto.h"
#include "pqc_epoch_log.h"
#include "pqc_journal.h"

#include <errno.h>
#include <string.h>
#include <unistd.h>

static int recovery_pread_full(int fd, uint8_t *buf, size_t len,
                               off_t offset)
{
    size_t done = 0;
    while (done < len) {
        ssize_t n = pread(fd, buf + done, len - done,
                          offset + (off_t)done);
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

void pqc_recovery_epoch_fallback_view_init(
    pqc_recovery_epoch_fallback_view_t *view)
{
    if (!view)
        return;
    memset(view, 0, sizeof(*view));
    view->fd = -1;
}

void pqc_recovery_epoch_fallback_view_set_lookup(
    pqc_recovery_epoch_fallback_view_t *view,
    pqc_epoch_log_lookup_view_t *lookup_view)
{
    if (!view)
        return;
    view->lookup_view = lookup_view;
}

void pqc_recovery_epoch_fallback_view_close(
    pqc_recovery_epoch_fallback_view_t *view)
{
    if (!view)
        return;
    if (view->fd >= 0) {
        (void)close(view->fd);
        view->fd = -1;
    }
    view->initialized = 0;
    view->available = 0;
    view->lookup_view = NULL;
    memset(&view->summary, 0, sizeof(view->summary));
}

static int recovery_epoch_fallback_view_prepare(
    const char *marker_path,
    uint64_t file_id,
    pqc_recovery_epoch_fallback_view_t *view)
{
    if (!marker_path)
        return -ENOENT;
    if (!view)
        return -EINVAL;
    if (view->initialized)
        return view->available ? 0 : -ENOENT;

    view->initialized = 1;
    int fd = -1;
    int rc = pqc_epoch_log_open_replay_path(marker_path, file_id, &fd,
                                            &view->summary);
    if (rc == -ENOENT) {
        view->available = 0;
        return -ENOENT;
    }
    if (rc != 0) {
        view->available = 0;
        return rc;
    }
    view->fd = fd;
    view->available = 1;
    return 0;
}

int pqc_recovery_load_authenticated_block(int journal_fd, int data_fd,
                                          const uint8_t *key, size_t key_len,
                                          uint64_t file_id, uint64_t block,
                                          uint8_t plain[PQC_LOGICAL_BLOCK_SIZE])
{
    return pqc_recovery_load_authenticated_block_committed(
        journal_fd, data_fd, key, key_len, file_id, block, UINT64_MAX, plain);
}

int pqc_recovery_load_authenticated_block_committed(
    int journal_fd,
    int data_fd,
    const uint8_t *key,
    size_t key_len,
    uint64_t file_id,
    uint64_t block,
    uint64_t max_generation,
    uint8_t plain[PQC_LOGICAL_BLOCK_SIZE])
{
    if (!key || !plain)
        return -EINVAL;

    memset(plain, 0, PQC_LOGICAL_BLOCK_SIZE);
    block_mapping_t map;
    int rc = pqc_journal_lookup_mapping_committed(journal_fd, block,
                                                  max_generation, &map);
    if (rc == -ENOENT)
        return 0;
    if (rc)
        return rc;

    uint8_t cipher[PQC_LOGICAL_BLOCK_SIZE] = {0};
    rc = recovery_pread_full(data_fd, cipher, map.plaintext_length,
                             (off_t)map.ciphertext_offset);
    if (rc != 0)
        return rc;
    if (map.algorithm_id != PQC_ALGO_AES_256_GCM)
        return -EINVAL;
    return pqc_crypto_crypt_block_gcm(key, key_len, file_id, block,
                                      map.generation, map.plaintext_length,
                                      cipher, plain, map.tag, 0, 0);
}

int pqc_recovery_load_authenticated_block_committed_epoch_fallback(
    int journal_fd,
    int data_fd,
    const char *marker_path,
    const uint8_t *key,
    size_t key_len,
    uint64_t file_id,
    uint64_t block,
    uint64_t max_generation,
    uint8_t plain[PQC_LOGICAL_BLOCK_SIZE])
{
    pqc_recovery_epoch_fallback_view_t view;
    pqc_recovery_epoch_fallback_view_init(&view);
    int rc = pqc_recovery_load_authenticated_block_committed_epoch_view(
        journal_fd, data_fd, marker_path, NULL, &view, key, key_len, file_id,
        block, max_generation, plain);
    pqc_recovery_epoch_fallback_view_close(&view);
    return rc;
}

int pqc_recovery_lookup_mapping_committed_epoch_view(
    int journal_fd,
    const char *marker_path,
    pqc_journal_lookup_view_t *journal_view,
    pqc_recovery_epoch_fallback_view_t *view,
    uint64_t file_id,
    uint64_t block,
    uint64_t max_generation,
    block_mapping_t *out)
{
    if (!out)
        return -EINVAL;

    block_mapping_t map;
    int rc = journal_view
        ? pqc_journal_lookup_mapping_committed_view(
            journal_fd, journal_view, block, max_generation, &map)
        : pqc_journal_lookup_mapping_committed(journal_fd, block,
                                               max_generation, &map);
    if (rc == -ENOENT && marker_path) {
        if (view) {
            if (view->lookup_view && view->lookup_view->initialized) {
                rc = pqc_epoch_log_lookup_mapping_committed_view(
                    view->lookup_view, file_id, block, max_generation, &map);
            } else {
                rc = recovery_epoch_fallback_view_prepare(marker_path,
                                                          file_id, view);
                if (rc == 0) {
                    rc = view->lookup_view
                        ? pqc_epoch_log_lookup_mapping_committed_fd_view(
                            view->fd, &view->summary, view->lookup_view,
                            file_id, block, max_generation, &map)
                        : pqc_epoch_log_lookup_mapping_committed_fd(
                            view->fd, &view->summary, file_id, block,
                            max_generation, &map);
                }
            }
        } else {
            rc = pqc_epoch_log_lookup_mapping_committed(
                marker_path, file_id, block, max_generation, &map);
        }
    }
    if (rc == -ENOENT)
        return -ENOENT;
    if (rc)
        return rc;
    if (map.algorithm_id != PQC_ALGO_AES_256_GCM ||
        map.plaintext_length > PQC_LOGICAL_BLOCK_SIZE)
        return -EINVAL;
    *out = map;
    return 0;
}

int pqc_recovery_load_authenticated_block_committed_epoch_view(
    int journal_fd,
    int data_fd,
    const char *marker_path,
    pqc_journal_lookup_view_t *journal_view,
    pqc_recovery_epoch_fallback_view_t *view,
    const uint8_t *key,
    size_t key_len,
    uint64_t file_id,
    uint64_t block,
    uint64_t max_generation,
    uint8_t plain[PQC_LOGICAL_BLOCK_SIZE])
{
    if (!key || !plain)
        return -EINVAL;

    memset(plain, 0, PQC_LOGICAL_BLOCK_SIZE);
    block_mapping_t map;
    int rc = pqc_recovery_lookup_mapping_committed_epoch_view(
        journal_fd, marker_path, journal_view, view, file_id, block,
        max_generation, &map);
    if (rc == -ENOENT)
        return 0;
    if (rc)
        return rc;
    uint8_t cipher[PQC_LOGICAL_BLOCK_SIZE] = {0};
    rc = recovery_pread_full(data_fd, cipher, map.plaintext_length,
                             (off_t)map.ciphertext_offset);
    if (rc != 0)
        return rc;
    return pqc_crypto_crypt_block_gcm(key, key_len, file_id, block,
                                      map.generation, map.plaintext_length,
                                      cipher, plain, map.tag, 0, 0);
}
