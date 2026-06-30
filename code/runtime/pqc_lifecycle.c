#include "pqc_lifecycle.h"

#include "pqc_anchor.h"
#include "pqc_anchor_worker.h"
#include "pqc_metrics.h"
#include "pqc_qos.h"
#include "pqc_runtime.h"

void pqc_destroy(void *private_data)
{
    (void)private_data;
    pqc_log("Shutting down PQC-FUSE filesystem...");
    pqc_anchor_worker_stop();
    (void)pqc_qos_stop_gpu_monitor();
    (void)pqc_anchor_worker_flush_now();
    (void)pqc_anchor_worker_flush_pending_on_shutdown();
    (void)pqc_anchor_finalize();
    pqc_runtime_cleanup();
    pqc_metrics_close();
}

void *pqc_fuse_init(struct fuse_conn_info *conn, struct fuse_config *cfg)
{
    if (cfg) {
        cfg->kernel_cache = 0;
        cfg->auto_cache = 0;
    }
    if (!conn)
        return NULL;
#ifdef FUSE_CAP_WRITEBACK_CACHE
    conn->want &= ~FUSE_CAP_WRITEBACK_CACHE;
#endif
#ifdef FUSE_CAP_ASYNC_DIO
    if (conn->capable & FUSE_CAP_ASYNC_DIO)
        conn->want |= FUSE_CAP_ASYNC_DIO;
#endif
#ifdef FUSE_DIRECT_IO_ALLOW_MMAP
    conn->want &= ~FUSE_DIRECT_IO_ALLOW_MMAP;
#endif
    return NULL;
}
