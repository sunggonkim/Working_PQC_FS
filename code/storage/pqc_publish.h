#ifndef PQC_PUBLISH_H
#define PQC_PUBLISH_H

#include <stdint.h>

#include "pqc_format.h"

/*
 * Strict-mode publication record helpers.  This module owns durable xattr
 * encoding/decoding for logical size and checkpoint records; higher-level
 * anchor ordering and fault cutpoints remain with the caller during mechanical
 * decomposition.
 */

int pqc_publish_logical_size_load(const char *path, uint64_t *size);
int pqc_publish_logical_size_store(const char *path, uint64_t size);

int pqc_publish_checkpoint_store_xattr(const char *path, uint64_t file_id,
                                       uint64_t sequence,
                                       uint64_t logical_size,
                                       uint64_t max_generation);
int pqc_publish_checkpoint_load_xattr(const char *path,
                                      uint64_t expected_file_id,
                                      pqc_checkpoint_t *out);

#endif /* PQC_PUBLISH_H */
