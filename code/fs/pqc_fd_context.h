#ifndef PQC_FD_CONTEXT_H
#define PQC_FD_CONTEXT_H

#include "pqc_lock_profile.h"
#include "pqc_epoch_log.h"
#include "pqc_format.h"
#include "pqc_journal.h"
#include "pqc_state.h"

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>
#include <time.h>

#define PQC_MAX_FD    4096
#define COALESCE_SIZE PQC_WRITEBACK_COALESCE_SIZE

typedef struct {
    int             valid;
    uint8_t         ss[64];
    size_t          ss_len;
    uint64_t        file_id;
    file_state_t   *state;
    int             data_fd;
    int             journal_fd;
    int             epoch_log_fd;
    int             epoch_log_syncfs_domain_known;
    int             epoch_log_same_syncfs_domain;
    int             data_sidecar_dirty;
    int             journal_sidecar_dirty;
    int             epoch_log_dirty;
    uint64_t        data_sidecar_dirty_epoch;
    uint64_t        journal_sidecar_dirty_epoch;
    uint64_t        epoch_log_dirty_epoch;
    int             journal_sidecar_epoch_repairable;
    uint64_t        journal_sidecar_epoch_repairable_epoch;
    uint64_t        fsync_metadata_epoch;
    uint64_t        fsync_metadata_synced_epoch;
    int             fsync_metadata_epoch_repairable;
    uint64_t        fsync_metadata_epoch_repairable_epoch;
    uint64_t        logical_size;
    char            logical_path[4096];
    char            marker_path[4096];
    char            epoch_log_path[4096 + 16];
    pthread_mutex_t fd_lock;
    uint8_t        *wbuf;
    uint8_t        *snapshot_buf;
    size_t          snapshot_buf_capacity;
    int             snapshot_buf_in_use;
    uint8_t        *flush_plain_batch;
    uint8_t        *flush_cipher_batch;
    size_t          flush_batch_capacity;
    int             flush_scratch_in_use;
    uint8_t        *read_plain_batch;
    uint8_t        *read_cipher_batch;
    size_t          read_batch_capacity;
    int             read_scratch_in_use;
    pqc_journal_lookup_view_t *read_journal_cache;
    int             read_journal_cache_valid;
    uint64_t        read_journal_cache_epoch;
    pqc_epoch_log_lookup_view_t *read_epoch_cache;
    int             read_epoch_cache_valid;
    uint64_t        read_epoch_cache_epoch;
    size_t          wbuf_used;
    off_t           wbuf_base_off;
    uint64_t        pending_jobs;
    uint64_t        pending_writeback_jobs;
    pthread_cond_t  pending_cv;
    int             tier;
    int             qos_class;
    uint64_t        key_epoch;
    time_t          last_rekey;
} pqc_fd_ctx_t;

typedef struct {
    int data_fd;
    int journal_fd;
    int epoch_log_fd;
    int sync_data;
    int sync_journal;
    int sync_epoch_log;
    int sync_data_epoch_domain_known;
    int sync_data_epoch_same_domain;
    uint64_t data_epoch;
    uint64_t journal_epoch;
    uint64_t epoch_log_epoch;
} pqc_fd_sidecar_sync_t;

void pqc_fd_context_table_init(void);
int pqc_fd_context_index_for_fd(int fd);
size_t pqc_fd_context_capacity(void);
pqc_fd_ctx_t *pqc_fd_context_at_index(size_t idx);
pqc_fd_ctx_t *pqc_fd_context_for_fd(int fd);
int pqc_fd_context_path_is_open(const char *marker_path);
int pqc_fd_context_logical_path_is_open(const char *logical_path);
int pqc_fd_context_any_open(void);
int pqc_fd_context_all_open_markers_hidden(void);
int pqc_fd_context_rename_path(const char *from_marker_path,
                               const char *to_marker_path);
int pqc_fd_context_rename_logical_path(const char *from_logical_path,
                                       const char *to_logical_path);

int pqc_fd_context_set(int fd,
                       const char *logical_path,
                       const char *marker_path,
                       const uint8_t *ss,
                       size_t ss_len,
                       uint64_t fid,
                       int open_flags);
void pqc_fd_context_clear(int fd);

void pqc_fd_context_wait_pending_locked(pqc_fd_ctx_t *ctx,
                                        pqc_lock_profile_scope_t *scope,
                                        const char *site);
void pqc_fd_context_wait_writeback_locked(pqc_fd_ctx_t *ctx,
                                          pqc_lock_profile_scope_t *scope,
                                          const char *site);
void pqc_fd_context_pending_job_begin(void *arg);
void pqc_fd_context_pending_job_end(void *arg);
void pqc_fd_context_writeback_job_begin(void *arg);
void pqc_fd_context_writeback_job_end(void *arg);
int pqc_fd_context_prepare_dirty_sidecar_sync_locked(
    pqc_fd_ctx_t *ctx,
    pqc_fd_sidecar_sync_t *sync);
int pqc_fd_context_run_dirty_sidecar_sync(pqc_fd_sidecar_sync_t *sync);
void pqc_fd_context_finish_dirty_sidecar_sync_locked(
    pqc_fd_ctx_t *ctx,
    const pqc_fd_sidecar_sync_t *sync,
    int sync_rc);
void pqc_fd_context_mark_fsync_dirty(pqc_fd_ctx_t *ctx);
void pqc_fd_context_invalidate_read_journal_cache(pqc_fd_ctx_t *ctx);
int pqc_fd_context_read_journal_cache_snapshot_locked(
    pqc_fd_ctx_t *ctx,
    uint64_t first_block,
    uint64_t last_block,
    uint64_t max_generation,
    pqc_journal_lookup_view_t *out,
    uint64_t *epoch_out);
void pqc_fd_context_read_journal_cache_store_locked(
    pqc_fd_ctx_t *ctx,
    const pqc_journal_lookup_view_t *view,
    uint64_t epoch_snapshot);
void pqc_fd_context_invalidate_read_epoch_cache(pqc_fd_ctx_t *ctx);
int pqc_fd_context_read_epoch_cache_snapshot_locked(
    pqc_fd_ctx_t *ctx,
    uint64_t first_block,
    uint64_t last_block,
    uint64_t max_generation,
    uint64_t file_id,
    pqc_epoch_log_lookup_view_t *out,
    uint64_t *epoch_out);
void pqc_fd_context_read_epoch_cache_store_locked(
    pqc_fd_ctx_t *ctx,
    const pqc_epoch_log_lookup_view_t *view,
    uint64_t epoch_snapshot);
void pqc_fd_context_mark_fsync_epoch_repairable(pqc_fd_ctx_t *ctx);
int pqc_fd_context_fsync_dirty_epoch(const pqc_fd_ctx_t *ctx,
                                     uint64_t *epoch_out);
int pqc_fd_context_fsync_epoch_repairable(const pqc_fd_ctx_t *ctx,
                                          uint64_t epoch);
void pqc_fd_context_mark_fsync_synced(pqc_fd_ctx_t *ctx, uint64_t epoch);

#endif /* PQC_FD_CONTEXT_H */
