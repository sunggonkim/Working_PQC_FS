#ifndef PQC_EPOCH_PUBLISH_H
#define PQC_EPOCH_PUBLISH_H

#include "pqc_strict_publish.h"

typedef enum {
    PQC_PUBLICATION_MODE_STRICT = 0,
    PQC_PUBLICATION_MODE_EPOCH_UNAVAILABLE = 1,
    PQC_PUBLICATION_MODE_EPOCH_REDO_LOG = 2,
} pqc_publication_mode_t;

int pqc_publication_mode_parse(const char *raw,
                               pqc_publication_mode_t *out);
const char *pqc_publication_mode_name(pqc_publication_mode_t mode);
int pqc_publication_mode_from_config(pqc_publication_mode_t *out);
int pqc_publication_dispatch_commit(const pqc_strict_publish_request_t *req);
void pqc_publication_trace_shutdown(void);

#endif /* PQC_EPOCH_PUBLISH_H */
