#ifndef PQC_DURABILITY_H
#define PQC_DURABILITY_H

#include <stdint.h>

typedef enum {
    PQC_DURABILITY_SITE_USER_FILE = 0,
    PQC_DURABILITY_SITE_DATA_SIDECAR,
    PQC_DURABILITY_SITE_JOURNAL_SIDECAR,
    PQC_DURABILITY_SITE_EPOCH_LOG,
    PQC_DURABILITY_SITE_ANCHOR_FILE,
    PQC_DURABILITY_SITE_MARKER_METADATA,
    PQC_DURABILITY_SITE_KEYRING_METADATA,
    PQC_DURABILITY_SITE_PARENT_DIR,
    PQC_DURABILITY_SITE_OTHER,
    PQC_DURABILITY_SITE_COUNT
} pqc_durability_site_t;

typedef struct {
    uint64_t fdatasync_calls;
    uint64_t fdatasync_failures;
    uint64_t fdatasync_total_ns;
    uint64_t fsync_calls;
    uint64_t fsync_failures;
    uint64_t fsync_total_ns;
    uint64_t syncfs_calls;
    uint64_t syncfs_failures;
    uint64_t syncfs_total_ns;
    uint64_t site_calls[PQC_DURABILITY_SITE_COUNT];
    uint64_t site_failures[PQC_DURABILITY_SITE_COUNT];
} pqc_durability_stats_t;

const char *pqc_durability_site_name(pqc_durability_site_t site);
int pqc_durability_fdatasync(int fd, pqc_durability_site_t site);
int pqc_durability_fsync(int fd, pqc_durability_site_t site);
int pqc_durability_syncfs(int fd, pqc_durability_site_t site);
void pqc_durability_begin_mounted_operations(void);
void pqc_durability_end_mounted_operations(void);
void pqc_durability_stats_snapshot(pqc_durability_stats_t *out);
void pqc_durability_mounted_stats_snapshot(pqc_durability_stats_t *out);
void pqc_durability_stats_reset(void);
void pqc_durability_log_summary(void);
int pqc_durability_self_test(void);

#endif /* PQC_DURABILITY_H */
