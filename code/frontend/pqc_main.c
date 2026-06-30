#include "pqc_fuse.h"

#include "cuda_aead.h"
#include "pqc_anchor.h"
#include "pqc_config.h"
#include "pqc_durability.h"
#include "pqc_metrics.h"
#include "pqc_runtime.h"
#include "pqc_scheduler.h"
#include "pqc_selftest.h"
#include "pqc_storage_path.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(const char *progname)
{
    fprintf(stderr,
        "\n"
        "╔══════════════════════════════════════════════════════════════════╗\n"
        "║  PQC-FUSE: Post-Quantum Cryptography FUSE Filesystem          ║\n"
        "║  Authenticated AES-GCM storage; optional ML-KEM-768 helpers  ║\n"
        "╚══════════════════════════════════════════════════════════════════╝\n"
        "\n"
        "Usage: %s <storage_dir> <mountpoint> [FUSE options]\n"
        "\n"
        "  <storage_dir>  : Physical directory where encrypted files are stored\n"
        "  <mountpoint>   : Virtual mount point (where apps write data)\n"
        "\n"
        "Example:\n"
        "  mkdir -p ~/pqc_edge_workspace/{mnt_secure,storage_physical}\n"
        "  %s ~/pqc_edge_workspace/storage_physical ~/pqc_edge_workspace/mnt_secure -f\n"
        "\n"
        "  -f : Run in foreground (recommended for profiling)\n"
        "\n",
        progname, progname);
}

int main(int argc, char *argv[])
{
    if (argc == 2 && strcmp(argv[1], "--self-test") == 0) {
        int sched_rc = pqc_selftest_scheduler();
        int ckpt_rc = pqc_selftest_checkpoint();
        int anchor_rc = pqc_anchor_self_test();
        int parallel_commit_rc = pqc_selftest_parallel_commit();
        int rekey_lifecycle_rc = pqc_selftest_rekey_lifecycle();
        int rekey_queue_rc = pqc_selftest_rekey_queue_accounting();
        int anchor_worker_rc = pqc_selftest_anchor_worker_lifecycle();
        int anchor_overflow_rc = pqc_selftest_anchor_committed_map_overflow();
        int anchor_pending_clear_rc = pqc_selftest_anchor_pending_clear();
        int lock_profile_rc = pqc_selftest_lock_profile();
        int durability_rc = pqc_selftest_durability();
        int key_lifecycle_rc = pqc_selftest_key_lifecycle();
        int trust_boundary_rc = pqc_selftest_trust_boundary();
        int keyring_kdf_rc = pqc_selftest_keyring_kdf();
        int epoch_log_rc = pqc_selftest_epoch_log();
        int epoch_publish_rc = pqc_selftest_epoch_publish();
        int admission_init_rc = pqc_selftest_admission_init_modes();
        int gen_rc = pqc_selftest_generation_replay();
        int rc = pqc_selftest_crypto() || pqc_selftest_journal() ||
                 gen_rc || ckpt_rc || sched_rc || anchor_rc ||
                 parallel_commit_rc || rekey_lifecycle_rc ||
                 anchor_worker_rc ||
                 anchor_overflow_rc ||
                 anchor_pending_clear_rc ||
                 lock_profile_rc ||
                 durability_rc ||
                 key_lifecycle_rc ||
                 trust_boundary_rc ||
                 keyring_kdf_rc ||
                 rekey_queue_rc || epoch_log_rc || epoch_publish_rc ||
                 admission_init_rc;
        fprintf(stderr, "PQC-FUSE scheduler self-test: %s\n",
                sched_rc == 0 ? "PASS" : "FAIL");
        fprintf(stderr, "PQC-FUSE parallel commit self-test: %s\n",
                parallel_commit_rc == 0 ? "PASS" : "FAIL");
        fprintf(stderr, "PQC-FUSE rekey lifecycle self-test: %s\n",
                rekey_lifecycle_rc == 0 ? "PASS" : "FAIL");
        fprintf(stderr, "PQC-FUSE rekey queue accounting self-test: %s\n",
                rekey_queue_rc == 0 ? "PASS" : "FAIL");
        fprintf(stderr, "PQC-FUSE anchor worker lifecycle self-test: %s\n",
                anchor_worker_rc == 0 ? "PASS" : "FAIL");
        fprintf(stderr, "PQC-FUSE anchor map overflow self-test: %s\n",
                anchor_overflow_rc == 0 ? "PASS" : "FAIL");
        fprintf(stderr, "PQC-FUSE anchor pending clear self-test: %s\n",
                anchor_pending_clear_rc == 0 ? "PASS" : "FAIL");
        fprintf(stderr, "PQC-FUSE lock profile self-test: %s\n",
                lock_profile_rc == 0 ? "PASS" : "FAIL");
        fprintf(stderr, "PQC-FUSE durability sync self-test: %s\n",
                durability_rc == 0 ? "PASS" : "FAIL");
        fprintf(stderr, "PQC-FUSE key lifecycle self-test: %s\n",
                key_lifecycle_rc == 0 ? "PASS" : "FAIL");
        fprintf(stderr, "PQC-FUSE trust boundary self-test: %s\n",
                trust_boundary_rc == 0 ? "PASS" : "FAIL");
        fprintf(stderr, "PQC-FUSE keyring KDF self-test: %s\n",
                keyring_kdf_rc == 0 ? "PASS" : "FAIL");
        fprintf(stderr, "PQC-FUSE epoch log self-test: %s\n",
                epoch_log_rc == 0 ? "PASS" : "FAIL");
        fprintf(stderr, "PQC-FUSE epoch publish self-test: %s\n",
                epoch_publish_rc == 0 ? "PASS" : "FAIL");
        fprintf(stderr, "PQC-FUSE admission init self-test: %s\n",
                admission_init_rc == 0 ? "PASS" : "FAIL");
        fprintf(stderr, "PQC-FUSE generation replay self-test: %s\n",
                gen_rc == 0 ? "PASS" : "FAIL");
        fprintf(stderr, "PQC-FUSE checkpoint self-test: %s\n",
                ckpt_rc == 0 ? "PASS" : "FAIL");
        fprintf(stderr, "PQC-FUSE anchor self-test: %s\n",
                anchor_rc == 0 ? "PASS" : "FAIL");
        fprintf(stderr, "PQC-FUSE storage self-test: %s\n", rc == 0 ? "PASS" : "FAIL");
        (void)pqc_config_dump_if_requested();
        return rc == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (argc == 2 && strcmp(argv[1], "--scheduler-smoke") == 0) {
        pqc_scheduler_smoke_report(stderr);
        (void)pqc_config_dump_if_requested();
        return EXIT_SUCCESS;
    }
    if (argc == 2 && strcmp(argv[1], "--admission-telemetry-smoke") == 0) {
        int rc = pqc_selftest_admission_telemetry_smoke();
        (void)pqc_config_dump_if_requested();
        return rc == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (argc == 2 && strcmp(argv[1], "--anchor-self-test") == 0) {
        int anchor_rc = pqc_anchor_self_test();
        fprintf(stderr, "PQC-FUSE anchor self-test: %s\n",
                anchor_rc == 0 ? "PASS" : "FAIL");
        (void)pqc_config_dump_if_requested();
        return anchor_rc == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (argc == 2 && strcmp(argv[1], "--um-smoke") == 0) {
        int um_rc = skim_cuda_um_self_test();
        skim_cuda_um_stats_t s = skim_cuda_um_stats_snapshot();
        fprintf(stdout,
                "{\"self_test_rc\":%d,\"managed_alloc_bytes\":%llu,\"managed_free_bytes\":%llu,"
                "\"prefetch_to_device_bytes\":%llu,\"prefetch_to_host_bytes\":%llu,"
                "\"prefetch_device_calls\":%llu,\"prefetch_host_calls\":%llu}\n",
                um_rc,
                (unsigned long long)s.managed_alloc_bytes,
                (unsigned long long)s.managed_free_bytes,
                (unsigned long long)s.prefetch_to_device_bytes,
                (unsigned long long)s.prefetch_to_host_bytes,
                (unsigned long long)s.prefetch_device_calls,
                (unsigned long long)s.prefetch_host_calls);
        (void)pqc_config_dump_if_requested();
        return EXIT_SUCCESS;
    }

    fprintf(stderr,
        "\n"
        "  ┌─────────────────────────────────────────────────────┐\n"
        "  │  PQC-FUSE v0.1 — Authenticated Encrypted Storage   │\n"
        "  │  CPU data plane; optional batch GPU helpers         │\n"
        "  └─────────────────────────────────────────────────────┘\n\n");

    if (argc < 3) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    char *storage_dir = realpath(argv[1], NULL);
    if (!storage_dir) {
        fprintf(stderr, "[PQC-FUSE] ERROR: Storage directory '%s' does not exist.\n", argv[1]);
        fprintf(stderr, "[PQC-FUSE] Create it with: mkdir -p %s\n", argv[1]);
        return EXIT_FAILURE;
    }
    pqc_storage_path_set_root(storage_dir);
    free(storage_dir);

    (void)pqc_metrics_open_for_storage(pqc_storage_path_root());

    pqc_log("════════════════════════════════════════════════════════════");
    pqc_log("  PQC-FUSE Session Started");
    pqc_log("  Storage dir : %s", pqc_storage_path_root());
    pqc_log("  Mount point : %s", argv[2]);
    pqc_log("════════════════════════════════════════════════════════════");

    if (pqc_runtime_init() != 0) {
        fprintf(stderr, "[PQC-FUSE] FATAL: PQC initialization failed.\n");
        (void)pqc_config_dump_if_requested();
        return EXIT_FAILURE;
    }

    int fuse_argc = argc - 1;
    char **fuse_argv = malloc(sizeof(char *) * (size_t)fuse_argc);
    if (!fuse_argv) {
        pqc_runtime_cleanup();
        pqc_metrics_close();
        return EXIT_FAILURE;
    }
    fuse_argv[0] = argv[0];
    for (int i = 2; i < argc; i++)
        fuse_argv[i - 1] = argv[i];

    pqc_log("Starting FUSE main loop...");
    pqc_durability_begin_mounted_operations();
    int ret = fuse_main(fuse_argc, fuse_argv, pqc_fuse_operations(), NULL);
    pqc_durability_end_mounted_operations();

    free(fuse_argv);
    pqc_runtime_cleanup();
    pqc_metrics_close();

    return ret;
}
