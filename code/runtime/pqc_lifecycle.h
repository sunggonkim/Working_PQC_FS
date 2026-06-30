#ifndef PQC_LIFECYCLE_H
#define PQC_LIFECYCLE_H

#ifndef FUSE_USE_VERSION
#define FUSE_USE_VERSION 31
#endif

#include <fuse.h>

void *pqc_fuse_init(struct fuse_conn_info *conn, struct fuse_config *cfg);
void pqc_destroy(void *private_data);

#endif /* PQC_LIFECYCLE_H */
