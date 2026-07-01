#include "pqc_runtime.h"

#include "cuda_aead.h"
#include "pqc_admission.h"
#include "pqc_anchor.h"
#include "pqc_anchor_worker.h"
#include "pqc_config.h"
#include "pqc_durability.h"
#include "pqc_epoch_log.h"
#include "pqc_epoch_publish.h"
#include "pqc_fd_context.h"
#include "pqc_format.h"
#include "pqc_fuse_trace.h"
#include "pqc_keyring.h"
#include "pqc_lock_profile.h"
#include "pqc_metrics.h"
#include "pqc_parallel_commit.h"
#include "pqc_plane_trace.h"
#include "pqc_qos.h"
#include "pqc_rekey.h"
#include "pqc_scheduler.h"
#include "pqc_storage_path.h"
#include "pqc_test_hooks.h"

#include <errno.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static OQS_KEM *g_kem = NULL;
static uint8_t *g_public_key = NULL;
static uint8_t *g_secret_key = NULL;
static atomic_int g_runtime_initialized = ATOMIC_VAR_INIT(0);
static atomic_int g_runtime_admission_started = ATOMIC_VAR_INIT(0);

static void runtime_free_key_material(void)
{
    if (g_kem) {
        if (g_secret_key)
            OQS_MEM_cleanse(g_secret_key, g_kem->length_secret_key);
        free(g_public_key);
        free(g_secret_key);
        OQS_KEM_free(g_kem);
    } else {
        free(g_public_key);
        free(g_secret_key);
    }
    g_public_key = NULL;
    g_secret_key = NULL;
    g_kem = NULL;
}

static void runtime_log_shutdown_stats(void)
{
    pqc_rekey_queue_stats_t rekey_stats;
    pqc_rekey_queue_stats_snapshot(&rekey_stats);
    pqc_log("Rekey queue stats: accepted=%llu duplicate=%llu invalid_fd=%llu queue_full=%llu",
            (unsigned long long)rekey_stats.accepted,
            (unsigned long long)rekey_stats.duplicate,
            (unsigned long long)rekey_stats.invalid_fd,
            (unsigned long long)rekey_stats.queue_full);

    pqc_scheduler_stats_t sched_stats;
    pqc_scheduler_stats_snapshot(&sched_stats);
    pqc_log("Scheduler stats: submitted=%llu cpu=%llu gpu=%llu bytes_cpu=%llu bytes_gpu=%llu migration_ns=%llu",
            (unsigned long long)sched_stats.submitted,
            (unsigned long long)sched_stats.cpu_executed,
            (unsigned long long)sched_stats.gpu_executed,
            (unsigned long long)sched_stats.bytes_cpu,
            (unsigned long long)sched_stats.bytes_gpu,
            (unsigned long long)sched_stats.gpu_migration_ns);
    pqc_durability_log_summary();
}

static void runtime_cleanup_resources(int final_outputs)
{
    if (pqc_qos_stop_admission_telemetry() && final_outputs)
        pqc_log("Admission telemetry file monitor stopped");
    if (pqc_qos_stop_gpu_monitor() && final_outputs)
        pqc_log("GPU load monitor stopped");
    if (pqc_rekey_worker_stop() > 0 && final_outputs)
        pqc_log("PQC background rekey worker stopped");

    skim_cuda_executor_shutdown();

    pqc_anchor_worker_stop();
    (void)pqc_anchor_worker_flush_now();
    (void)pqc_anchor_worker_flush_pending_on_shutdown();
    (void)pqc_anchor_finalize();

    if (final_outputs)
        runtime_log_shutdown_stats();

    pqc_keyring_clear_master_key();
    pqc_parallel_commit_runtime_shutdown();
    pqc_publication_trace_shutdown();
    pqc_epoch_log_trace_shutdown();
    pqc_anchor_trace_shutdown();
    runtime_free_key_material();
    if (atomic_exchange_explicit(&g_runtime_admission_started, 0,
                                 memory_order_acq_rel))
        pqc_admission_shutdown();
    pqc_lock_profile_shutdown();
}

OQS_KEM *pqc_runtime_kem(void)
{
    return g_kem;
}

const uint8_t *pqc_runtime_public_key(void)
{
    return g_public_key;
}

const uint8_t *pqc_runtime_secret_key(void)
{
    return g_secret_key;
}

int pqc_runtime_init(void)
{
    int init_rc = -1;
    pqc_fuse_trace_reset();
    pqc_plane_trace_reset();
    int lock_profile_rc = pqc_lock_profile_init_from_config();
    if (lock_profile_rc != 0)
        pqc_log("Lock profile trace disabled: %s", strerror(-lock_profile_rc));
    pqc_durability_init_from_config();
    (void)pqc_fault_cutpoint_enabled_slow();
    pqc_rekey_init_from_config();
    int keyplane_rekey_enabled = pqc_rekey_write_trigger_enabled();
    pqc_anchor_init_from_config();
    pqc_anchor_worker_init_from_config();

    const char *admission_trace =
        pqc_config_get_nonempty("PQC_ADMISSION_TRACE_PATH");
    const char *telemetry_path =
        pqc_config_get_nonempty("PQC_TELEMETRY_FILE");
    int qos_throttle_requested =
        pqc_config_enabled("PQC_ENABLE_QOS_THROTTLE_ON_WRITE");
    int explicit_gpu_monitor =
        pqc_config_enabled("PQC_ENABLE_GPU_LOAD_MONITOR") ||
        pqc_config_present("PQC_GPU_LOAD_PATH");
    int admission_needed =
        keyplane_rekey_enabled || admission_trace != NULL ||
        telemetry_path != NULL ||
        qos_throttle_requested;
    int qos_monitor_needed =
        qos_throttle_requested || telemetry_path != NULL ||
        explicit_gpu_monitor;
    if (admission_needed) {
        if (pqc_admission_init(admission_trace) != 0)
            goto fail;
        atomic_store_explicit(&g_runtime_admission_started, 1,
                              memory_order_release);
    } else {
        pqc_log("Elastic admission controller disabled for this mount");
    }
    int publication_rc = pqc_publication_init_from_config();
    if (publication_rc != 0) {
        fprintf(stderr, "[PQC-FUSE] FATAL: Publication mode failed to initialize: %s\n",
                strerror(-publication_rc));
        goto fail;
    }
    if (admission_needed) {
        pqc_scheduler_reload_runtime_policy_from_env();
        pqc_scheduler_set_data_accounting_enabled(1);
    } else {
        pqc_scheduler_set_data_accounting_enabled(0);
        pqc_log("Elastic scheduler policy reload skipped for this mount");
    }
    int parallel_commit_rc = pqc_parallel_commit_runtime_init_from_config();
    if (parallel_commit_rc != 0) {
        fprintf(stderr, "[PQC-FUSE] FATAL: Parallel commit mode failed to initialize: %s\n",
                strerror(-parallel_commit_rc));
        goto fail;
    }
    if (pqc_parallel_commit_runtime_enabled())
        pqc_log("Parallel commit mode enabled");

    pqc_fd_context_table_init();

    if (keyplane_rekey_enabled) {
        /* Try ML-KEM-768 first (NIST standardized name), fallback to Kyber768 */
        g_kem = OQS_KEM_new(OQS_KEM_alg_ml_kem_768);
        if (!g_kem)
            g_kem = OQS_KEM_new(OQS_KEM_alg_kyber_768);
        if (!g_kem) {
            fprintf(stderr, "[PQC-FUSE] FATAL: Neither ML-KEM-768 nor Kyber-768 available in liboqs!\n");
            fprintf(stderr, "[PQC-FUSE] Rebuild liboqs with KEM_ml_kem_768 or KEM_kyber_768 enabled.\n");
            goto fail;
        }

        pqc_log("KEM algorithm : %s", g_kem->method_name);
        pqc_log("  public key  : %zu bytes", g_kem->length_public_key);
        pqc_log("  secret key  : %zu bytes", g_kem->length_secret_key);
        pqc_log("  ciphertext  : %zu bytes", g_kem->length_ciphertext);
        pqc_log("  shared secret: %zu bytes", g_kem->length_shared_secret);

        g_public_key = malloc(g_kem->length_public_key);
        g_secret_key = malloc(g_kem->length_secret_key);
        if (!g_public_key || !g_secret_key) {
            fprintf(stderr, "[PQC-FUSE] FATAL: Failed to allocate key memory\n");
            goto fail;
        }

        double t0 = pqc_metrics_time_us();
        OQS_STATUS rc = OQS_KEM_keypair(g_kem, g_public_key, g_secret_key);
        double t1 = pqc_metrics_time_us();

        if (rc != OQS_SUCCESS) {
            fprintf(stderr, "[PQC-FUSE] FATAL: Key generation failed\n");
            goto fail;
        }

        pqc_log("Keypair generated in %.2f µs", t1 - t0);
    } else {
        pqc_log("PQC key-plane rekey disabled for this mount");
    }

    /* Derive master key */
    const char *pass = pqc_config_get_nonempty("PQC_MASTER_PASSWORD");
    if (!pass) {
        fprintf(stderr, "[PQC-FUSE] FATAL: PQC_MASTER_PASSWORD is required.\n");
        goto fail;
    }
    if (pqc_keyring_derive_master_key(pass) != 0) {
        fprintf(stderr, "[PQC-FUSE] FATAL: Failed to derive Master Key\n");
        goto fail;
    }
    pqc_log("Master Key derived successfully (%s)",
            pqc_keyring_kdf_name());
    int anchor_reconstruct_rc =
        pqc_anchor_reconstruct_committed_map_from_storage(
            pqc_storage_path_root());
    if (anchor_reconstruct_rc != 0) {
        fprintf(stderr,
                "[PQC-FUSE] FATAL: Failed to reconstruct freshness map: %s\n",
                strerror(-anchor_reconstruct_rc));
        goto fail;
    }
    if (admission_needed) {
        pqc_scheduler_policy_t sched_policy;
        pqc_scheduler_runtime_policy_snapshot(&sched_policy);
        pqc_log("Scheduler policy: gpu_min_bytes=%zu coherence_penalty_ns=%llu",
                sched_policy.gpu_min_bytes,
                (unsigned long long)sched_policy.coherence_penalty_ns);
    }

    if (qos_monitor_needed) {
        pqc_qos_monitor_status_t qos_status;
        pqc_qos_start_monitors(&qos_status);
        if (qos_status.gpu_monitor_started) {
            pqc_log("GPU load monitor started: %s", qos_status.gpu_load_path);
        } else if (qos_status.gpu_monitor_configured) {
            pqc_log("GPU load monitor disabled: %s",
                    strerror(qos_status.gpu_monitor_errno ?
                             qos_status.gpu_monitor_errno : errno));
        }
        if (qos_status.telemetry_monitor_configured) {
            if (qos_status.telemetry_monitor_started) {
                pqc_log("Admission telemetry file monitor started: %s",
                        qos_status.telemetry_path);
            } else {
                pqc_log("Admission telemetry file monitor disabled: %s",
                        strerror(qos_status.telemetry_monitor_errno ?
                                 qos_status.telemetry_monitor_errno : errno));
            }
        }
    } else {
        pqc_qos_disable_monitors_for_mount();
        pqc_log("QoS monitor startup skipped for this mount");
    }

    if (pqc_anchor_backend() != PQC_ANCHOR_BACKEND_DISABLED) {
        int anchor_rc = pqc_anchor_probe();
        if (anchor_rc != 0) {
            fprintf(stderr,
                    "[PQC-FUSE] FATAL: Freshness anchor probe failed: %s\n",
                    strerror(-anchor_rc));
            goto fail;
        }

        int anchor_worker_rc = pqc_anchor_worker_start_if_configured();
        if (anchor_worker_rc > 0) {
            pqc_log("Freshness anchor worker started");
        } else if (anchor_worker_rc < 0) {
            pqc_log("Freshness anchor worker failed to start: %s",
                    strerror(-anchor_worker_rc));
        }
    } else {
        pqc_log("Freshness anchor probe/worker skipped for this mount");
    }

    if (keyplane_rekey_enabled) {
        int rekey_rc = pqc_rekey_worker_start(g_kem, g_public_key);
        if (rekey_rc == 0) {
            pqc_log("PQC background rekey worker started");
        } else {
            pqc_log("PQC background rekey worker failed to start: %s",
                    strerror(-rekey_rc));
        }
    }

    atomic_store_explicit(&g_runtime_initialized, 1, memory_order_release);
    return 0;

fail:
    runtime_cleanup_resources(0);
    return init_rc;
}

void pqc_runtime_cleanup(void)
{
    if (!atomic_exchange_explicit(&g_runtime_initialized, 0,
                                  memory_order_acq_rel))
        return;

    runtime_cleanup_resources(1);
    if (pqc_config_dump_if_requested() != 0)
        pqc_log("Runtime config dump failed");
    if (pqc_plane_trace_dump_if_requested() != 0)
        pqc_log("Plane trace dump failed");
    if (pqc_fuse_trace_dump_if_requested() != 0)
        pqc_log("FUSE trace dump failed");
}
