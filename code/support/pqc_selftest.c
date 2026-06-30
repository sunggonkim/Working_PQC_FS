#include "pqc_selftest.h"

#include "cuda_aead.h"
#include "pqc_admission.h"
#include "pqc_anchor.h"
#include "pqc_anchor_worker.h"
#include "pqc_block_job.h"
#include "pqc_checkpoint.h"
#include "pqc_config.h"
#include "pqc_crypto.h"
#include "pqc_durability.h"
#include "pqc_epoch_log.h"
#include "pqc_epoch_publish.h"
#include "pqc_format.h"
#include "pqc_journal.h"
#include "pqc_key_lifecycle.h"
#include "pqc_keyring.h"
#include "pqc_lock_profile.h"
#include "pqc_parallel_commit.h"
#include "pqc_rekey.h"
#include "pqc_trust_boundary.h"

#include <errno.h>
#include <oqs/oqs.h>
#include <openssl/crypto.h>
#include <pthread.h>
#include <openssl/rand.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int pqc_selftest_crypto(void)
{
    uint8_t key[32], plain[PQC_LOGICAL_BLOCK_SIZE], cipher[PQC_LOGICAL_BLOCK_SIZE];
    uint8_t recovered[PQC_LOGICAL_BLOCK_SIZE], tag[PQC_AEAD_TAG_SIZE];
    if (RAND_bytes(key, sizeof(key)) != 1 || RAND_bytes(plain, sizeof(plain)) != 1)
        return -1;
    if (pqc_crypto_crypt_block_gcm(key, sizeof(key), 7, 11, 1, sizeof(plain), plain, cipher, tag, 1, 0) != 0 ||
        pqc_crypto_crypt_block_gcm(key, sizeof(key), 7, 11, 1, sizeof(plain), cipher, recovered, tag, 0, 0) != 0 ||
        memcmp(plain, recovered, sizeof(plain)) != 0)
        return -1;
    tag[0] ^= 1;
    int tamper_rejected = pqc_crypto_crypt_block_gcm(key, sizeof(key), 7, 11, 1, sizeof(plain),
                                           cipher, recovered, tag, 0, 0) == -EBADMSG;
    if (tamper_rejected && skim_cuda_aead_available()) {
        uint8_t cpu_reference[PQC_LOGICAL_BLOCK_SIZE], reference_tag[PQC_AEAD_TAG_SIZE];
        uint8_t gpu_cipher[PQC_LOGICAL_BLOCK_SIZE], gpu_plain[PQC_LOGICAL_BLOCK_SIZE], gpu_tag[PQC_AEAD_TAG_SIZE];
        size_t offsets[1] = {0}, lengths[1] = {sizeof(plain)};
        uint8_t nonces[PQC_AEAD_NONCE_SIZE];
        uint8_t aad[28];
        pqc_crypto_build_block_aad(aad, 7, 11, 2, sizeof(plain));
        int gpu_ok = pqc_crypto_derive_block_nonce(7, 11, 2, nonces) == 0 &&
                     pqc_crypto_crypt_block_gcm(key, sizeof(key), 7, 11, 2, sizeof(plain), plain, cpu_reference, reference_tag, 1, 0) == 0 &&
                     skim_cuda_aes256_gcm_batch(key, nonces, aad, plain, gpu_cipher,
                                                 offsets, lengths, gpu_tag, 1) == 0 &&
                     memcmp(cpu_reference, gpu_cipher, sizeof(cpu_reference)) == 0 &&
                     CRYPTO_memcmp(reference_tag, gpu_tag, sizeof(reference_tag)) == 0 &&
                     pqc_crypto_crypt_block_gcm(key, sizeof(key), 7, 11, 2, sizeof(plain), gpu_cipher, gpu_plain, gpu_tag, 0, 1) == 0 &&
                     memcmp(plain, gpu_plain, sizeof(plain)) == 0;
        tamper_rejected = gpu_ok;
        OPENSSL_cleanse(cpu_reference, sizeof(cpu_reference)); OPENSSL_cleanse(reference_tag, sizeof(reference_tag));
        OPENSSL_cleanse(gpu_cipher, sizeof(gpu_cipher)); OPENSSL_cleanse(gpu_plain, sizeof(gpu_plain));
        OPENSSL_cleanse(gpu_tag, sizeof(gpu_tag));
        OPENSSL_cleanse(nonces, sizeof(nonces));
    }
    if (tamper_rejected) {
        enum { batch_count = 3 };
        static const uint32_t batch_lengths[batch_count] = {64U, 127U, 19U};
        pqc_crypto_block_desc_t batch[batch_count];
        uint8_t batch_plain[256] = {0};
        uint8_t batch_cipher[sizeof(batch_plain)] = {0};
        uint8_t batch_recovered[sizeof(batch_plain)] = {0};
        size_t packed = 0;
        memset(batch, 0, sizeof(batch));
        for (size_t i = 0; i < batch_count; ++i) {
            batch[i].block = 21U + i;
            batch[i].generation = 5U + i;
            batch[i].length = batch_lengths[i];
            batch[i].input_offset = packed;
            batch[i].output_offset = packed;
            for (uint32_t j = 0; j < batch[i].length; ++j)
                batch_plain[packed + j] = (uint8_t)(0x31U + i + j);
            if (pqc_crypto_derive_block_nonce(7, batch[i].block,
                                              batch[i].generation,
                                              batch[i].nonce) != 0) {
                tamper_rejected = 0;
                break;
            }
            pqc_crypto_build_block_aad(batch[i].aad, 7, batch[i].block,
                                       batch[i].generation, batch[i].length);
            packed += batch[i].length;
        }
        if (tamper_rejected &&
            pqc_crypto_crypt_block_batch_gcm(key, sizeof(key), 7, batch,
                                             batch_count, batch_plain,
                                             batch_cipher) != 0) {
            tamper_rejected = 0;
        }
        for (size_t i = 0; tamper_rejected && i < batch_count; ++i) {
            if (pqc_crypto_crypt_block_gcm(
                    key, sizeof(key), 7, batch[i].block, batch[i].generation,
                    batch[i].length, batch_cipher + batch[i].output_offset,
                    batch_recovered + batch[i].input_offset, batch[i].tag,
                    0, 0) != 0 ||
                memcmp(batch_plain + batch[i].input_offset,
                       batch_recovered + batch[i].input_offset,
                       batch[i].length) != 0)
                tamper_rejected = 0;
        }
        OPENSSL_cleanse(batch, sizeof(batch));
        OPENSSL_cleanse(batch_plain, sizeof(batch_plain));
        OPENSSL_cleanse(batch_cipher, sizeof(batch_cipher));
        OPENSSL_cleanse(batch_recovered, sizeof(batch_recovered));
    }
    OQS_MEM_cleanse(key, sizeof(key)); OQS_MEM_cleanse(plain, sizeof(plain));
    OQS_MEM_cleanse(recovered, sizeof(recovered));
    return tamper_rejected ? 0 : -1;
}

int pqc_selftest_durability(void)
{
    return pqc_durability_self_test();
}

int pqc_selftest_journal(void)
{
    FILE *tmp = tmpfile();
    if (!tmp) return -1;
    int fd = fileno(tmp);
    block_mapping_t first = { .logical_block = 4, .generation = 1,
                              .ciphertext_offset = 4096, .plaintext_length = 17 };
    block_mapping_t latest = { .logical_block = 4, .generation = 2,
                               .ciphertext_offset = 8192, .plaintext_length = 31 };
    int ok = pqc_journal_append_mapping(fd, &first) == 0 &&
             pqc_journal_append_mapping(fd, &latest) == 0;
    block_mapping_t recovered = {0};
    ok = ok && pqc_journal_lookup_mapping(fd, 4, &recovered) == 0 &&
         recovered.generation == latest.generation &&
         recovered.ciphertext_offset == latest.ciphertext_offset;
    ok = ok && write(fd, "tail", 4) == 4 &&
         pqc_journal_lookup_mapping(fd, 4, &recovered) == 0 &&
         recovered.generation == latest.generation;
    fclose(tmp);
    return ok ? 0 : -1;
}

int pqc_selftest_generation_replay(void)
{
    FILE *tmp = tmpfile();
    if (!tmp) return -1;
    int fd = fileno(tmp);

    block_mapping_t gen1 = { .logical_block = 9, .generation = 1,
                             .ciphertext_offset = 4096, .plaintext_length = 32,
                             .algorithm_id = PQC_ALGO_AES_256_GCM };
    block_mapping_t gen2 = { .logical_block = 9, .generation = 2,
                             .ciphertext_offset = 8192, .plaintext_length = 32,
                             .algorithm_id = PQC_ALGO_AES_256_GCM };
    block_mapping_t replayed_gen1 = { .logical_block = 9, .generation = 1,
                                      .ciphertext_offset = 12288, .plaintext_length = 32,
                                      .algorithm_id = PQC_ALGO_AES_256_GCM };
    block_mapping_t recovered = {0};
    int ok = pqc_journal_append_mapping(fd, &gen1) == 0 &&
             pqc_journal_append_mapping(fd, &gen2) == 0 &&
             pqc_journal_append_mapping(fd, &replayed_gen1) == 0 &&
             pqc_journal_lookup_mapping(fd, 9, &recovered) == 0 &&
             recovered.generation == gen2.generation &&
             recovered.ciphertext_offset == gen2.ciphertext_offset &&
             pqc_journal_max_generation(fd) == gen2.generation;

    uint8_t key[32];
    uint8_t plain[32];
    uint8_t cipher[32];
    uint8_t recovered_plain[32];
    uint8_t tag[PQC_AEAD_TAG_SIZE];
    for (size_t i = 0; i < sizeof(key); ++i) key[i] = (uint8_t)(0xa0U + i);
    for (size_t i = 0; i < sizeof(plain); ++i) plain[i] = (uint8_t)(0x30U + i);
    memset(cipher, 0, sizeof(cipher));
    memset(recovered_plain, 0, sizeof(recovered_plain));
    memset(tag, 0, sizeof(tag));

    int enc_rc = pqc_crypto_crypt_block_gcm(key, sizeof(key), UINT64_C(0xabcdef0123456789),
                                 9, 2, sizeof(plain), plain, cipher, tag, 1, 0);
    int wrong_gen_rc = pqc_crypto_crypt_block_gcm(key, sizeof(key), UINT64_C(0xabcdef0123456789),
                                       9, 1, sizeof(cipher), cipher, recovered_plain,
                                       tag, 0, 0);
    int right_gen_rc = pqc_crypto_crypt_block_gcm(key, sizeof(key), UINT64_C(0xabcdef0123456789),
                                       9, 2, sizeof(cipher), cipher, recovered_plain,
                                       tag, 0, 0);
    ok = ok && enc_rc == 0 && wrong_gen_rc != 0 && right_gen_rc == 0 &&
         CRYPTO_memcmp(plain, recovered_plain, sizeof(plain)) == 0;

    OPENSSL_cleanse(key, sizeof(key));
    OPENSSL_cleanse(plain, sizeof(plain));
    OPENSSL_cleanse(cipher, sizeof(cipher));
    OPENSSL_cleanse(recovered_plain, sizeof(recovered_plain));
    OPENSSL_cleanse(tag, sizeof(tag));
    fclose(tmp);
    return ok ? 0 : -1;
}

int pqc_selftest_checkpoint(void)
{
    char path[] = "/tmp/skim_ckpt_selftestXXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return -1;
    close(fd);
    int ok = pqc_keyring_derive_master_key("checkpoint-self-test") == 0;
    if (ok) {
        ok = pqc_checkpoint_store_and_stage_anchor(path, 42, 7, 8192, 11) == 0;
        if (ok && pqc_anchor_backend() == PQC_ANCHOR_BACKEND_HARDWARE)
            ok = pqc_anchor_flush() == 0;
        pqc_checkpoint_t ckpt = {0};
        ok = ok && pqc_checkpoint_load_and_verify_anchor(path, 42, &ckpt) == 0 &&
             ckpt.sequence == 7 && ckpt.logical_size == 8192 && ckpt.max_generation == 11;
    }
    unlink(path);
    pqc_keyring_clear_master_key();
    return ok ? 0 : -1;
}

int pqc_selftest_key_lifecycle(void)
{
    return pqc_key_lifecycle_self_test();
}

int pqc_selftest_trust_boundary(void)
{
    return pqc_trust_boundary_self_test();
}

int pqc_selftest_keyring_kdf(void)
{
    return pqc_keyring_kdf_self_test();
}

int pqc_selftest_anchor_worker_lifecycle(void)
{
    return pqc_anchor_worker_lifecycle_self_test();
}

int pqc_selftest_anchor_committed_map_overflow(void)
{
    return pqc_anchor_committed_map_overflow_self_test();
}

int pqc_selftest_anchor_pending_clear(void)
{
    return pqc_anchor_pending_clear_self_test();
}

int pqc_selftest_lock_profile(void)
{
    return pqc_lock_profile_self_test(stderr);
}

int pqc_selftest_scheduler(void)
{
    pqc_scheduler_policy_t policy = {
        .gpu_min_bytes = 8192,
        .gpu_queue_penalty_ns = 25000,
        .coherence_penalty_ns = 65536,
        .gpu_max_inflight_jobs = 2,
        .gpu_max_wait_ns = 25000,
        .cpu_load_bias = 1.0,
        .gpu_queue_bias = 1.0,
    };
    pqc_block_job_t small = {0}, large = {0};
    pqc_block_job_init(&small, 1, 0, 1, 0, 4096, PQC_JOB_FLAG_ENCRYPT | PQC_JOB_FLAG_READMOD, PQC_PLANE_KEY);
    pqc_block_job_init(&large, 1, 1, 2, 4096, 16384, PQC_JOB_FLAG_ENCRYPT | PQC_JOB_FLAG_READMOD | PQC_JOB_FLAG_GPU_ELIGIBLE, PQC_PLANE_KEY);
    small.cpu_queue_depth = 2;
    large.gpu_queue_depth = 0;
    small.coherence_cost_ns = 0;
    large.coherence_cost_ns = 1024;
    large.ai_qos_budget_ns = 1000000;
    if (pqc_block_job_choose_target(&small, &policy, 0.5, 0.0) != PQC_JOB_CPU)
        return -1;
    if (pqc_block_job_choose_target(&large, &policy, 0.5, 0.0) != PQC_JOB_GPU)
        return -1;
    return 0;
}

typedef struct {
    pthread_mutex_t lock;
    pthread_cond_t cv;
    int ready_count;
    int begun_count;
    int start;
    int worker_count;
} pqc_parallel_commit_test_gate_t;

typedef struct {
    pqc_parallel_commit_coordinator_t *coordinator;
    pqc_parallel_commit_test_gate_t *gate;
    uint64_t file_id;
    uint64_t bytes;
    int rc;
    pqc_parallel_commit_ticket_t ticket;
} pqc_parallel_commit_test_arg_t;

static void *parallel_commit_test_worker(void *opaque)
{
    pqc_parallel_commit_test_arg_t *arg =
        (pqc_parallel_commit_test_arg_t *)opaque;
    pthread_mutex_lock(&arg->gate->lock);
    arg->gate->ready_count++;
    if (arg->gate->ready_count == arg->gate->worker_count)
        pthread_cond_broadcast(&arg->gate->cv);
    while (!arg->gate->start)
        pthread_cond_wait(&arg->gate->cv, &arg->gate->lock);
    pthread_mutex_unlock(&arg->gate->lock);

    arg->rc = pqc_parallel_commit_begin(arg->coordinator, arg->file_id,
                                        arg->bytes, &arg->ticket);
    if (arg->ticket.role == PQC_PARALLEL_COMMIT_ROLE_LEADER) {
        usleep(1000);
        int finish_rc = pqc_parallel_commit_finish(arg->coordinator,
                                                   &arg->ticket, 0);
        if (finish_rc != 0)
            arg->rc = finish_rc;
    }
    return NULL;
}

static void *parallel_commit_runtime_shutdown_worker(void *opaque)
{
    pqc_parallel_commit_test_arg_t *arg =
        (pqc_parallel_commit_test_arg_t *)opaque;
    pthread_mutex_lock(&arg->gate->lock);
    arg->gate->ready_count++;
    if (arg->gate->ready_count == arg->gate->worker_count)
        pthread_cond_broadcast(&arg->gate->cv);
    while (!arg->gate->start)
        pthread_cond_wait(&arg->gate->cv, &arg->gate->lock);
    pthread_mutex_unlock(&arg->gate->lock);

    arg->rc = pqc_parallel_commit_runtime_begin(arg->file_id, arg->bytes,
                                                &arg->ticket);
    pthread_mutex_lock(&arg->gate->lock);
    arg->gate->begun_count++;
    pthread_cond_broadcast(&arg->gate->cv);
    pthread_mutex_unlock(&arg->gate->lock);

    if (arg->ticket.role == PQC_PARALLEL_COMMIT_ROLE_LEADER) {
        usleep(50000);
        int finish_rc =
            pqc_parallel_commit_runtime_finish(&arg->ticket, 0);
        if (finish_rc != 0)
            arg->rc = finish_rc;
    }
    return NULL;
}

int pqc_selftest_parallel_commit(void)
{
    enum { worker_count = 4 };
    pqc_parallel_commit_config_t config = {
        .shard_count = 4,
        .max_group_size = worker_count,
        .max_wait_ns = 500000000ULL,
    };
    pqc_parallel_commit_coordinator_t *coordinator = NULL;
    int rc = pqc_parallel_commit_init(&coordinator, &config);
    if (rc != 0)
        return rc;

    pqc_parallel_commit_test_gate_t gate = {
        .lock = PTHREAD_MUTEX_INITIALIZER,
        .cv = PTHREAD_COND_INITIALIZER,
        .ready_count = 0,
        .start = 0,
        .worker_count = worker_count,
    };
    pqc_parallel_commit_test_arg_t args[worker_count];
    pthread_t threads[worker_count];
    int created = 0;
    memset(args, 0, sizeof(args));

    for (int i = 0; i < worker_count; ++i) {
        args[i].coordinator = coordinator;
        args[i].gate = &gate;
        args[i].file_id = UINT64_C(0xfeed000000000001);
        args[i].bytes = 4096;
        if (pthread_create(&threads[i], NULL, parallel_commit_test_worker,
                           &args[i]) != 0) {
            rc = -1;
            goto out_join_partial;
        }
        created++;
    }

    pthread_mutex_lock(&gate.lock);
    while (gate.ready_count != worker_count)
        pthread_cond_wait(&gate.cv, &gate.lock);
    gate.start = 1;
    pthread_cond_broadcast(&gate.cv);
    pthread_mutex_unlock(&gate.lock);

out_join_partial:
    if (rc != 0) {
        pthread_mutex_lock(&gate.lock);
        gate.start = 1;
        pthread_cond_broadcast(&gate.cv);
        pthread_mutex_unlock(&gate.lock);
    }
    for (int i = 0; i < created; ++i)
        pthread_join(threads[i], NULL);

    int leaders = 0;
    int followers = 0;
    int ok = (rc == 0);
    for (int i = 0; i < worker_count; ++i) {
        ok = ok && args[i].rc == 0;
        ok = ok && args[i].ticket.shard_count == config.shard_count;
        ok = ok && args[i].ticket.observed_queue_depth >= 1;
        ok = ok && args[i].ticket.observed_queue_depth <= worker_count;
        ok = ok && args[i].ticket.group_size == worker_count;
        ok = ok && args[i].ticket.group_bytes == worker_count * 4096ULL;
        if (args[i].ticket.role == PQC_PARALLEL_COMMIT_ROLE_LEADER)
            leaders++;
        else if (args[i].ticket.role == PQC_PARALLEL_COMMIT_ROLE_FOLLOWER)
            followers++;
    }

    pqc_parallel_commit_stats_t stats;
    pqc_parallel_commit_stats_snapshot(coordinator, &stats);
    ok = ok && leaders == 1 && followers == worker_count - 1;
    ok = ok && stats.shard_count == config.shard_count;
    ok = ok && stats.total_epochs == 1;
    ok = ok && stats.total_requests == worker_count;
    ok = ok && stats.total_leaders == 1;
    ok = ok && stats.total_followers == worker_count - 1;
    ok = ok && stats.max_observed_group_size == worker_count;
    ok = ok && stats.max_observed_queue_depth == worker_count;
    ok = ok && stats.full_group_epochs == 1;

    pthread_cond_destroy(&gate.cv);
    pthread_mutex_destroy(&gate.lock);
    pqc_parallel_commit_destroy(coordinator);
    if (!ok)
        return -1;

    setenv("PQC_PARALLEL_COMMIT_MODE", "epoch-gated-strict", 1);
    setenv("PQC_PARALLEL_COMMIT_SHARDS", "1", 1);
    setenv("PQC_PARALLEL_COMMIT_GROUP_MAX", "1", 1);
    setenv("PQC_PARALLEL_COMMIT_WAIT_NS", "0", 1);
    unsetenv("PQC_PARALLEL_COMMIT_TRACE_PATH");

    rc = pqc_parallel_commit_runtime_init_from_config();
    if (rc != 0) {
        unsetenv("PQC_PARALLEL_COMMIT_MODE");
        unsetenv("PQC_PARALLEL_COMMIT_SHARDS");
        unsetenv("PQC_PARALLEL_COMMIT_GROUP_MAX");
        unsetenv("PQC_PARALLEL_COMMIT_WAIT_NS");
        return -1;
    }

    pqc_parallel_commit_test_gate_t runtime_gate = {
        .lock = PTHREAD_MUTEX_INITIALIZER,
        .cv = PTHREAD_COND_INITIALIZER,
        .ready_count = 0,
        .begun_count = 0,
        .start = 0,
        .worker_count = 1,
    };
    pqc_parallel_commit_test_arg_t runtime_arg = {
        .gate = &runtime_gate,
        .file_id = UINT64_C(0xfeed000000000002),
        .bytes = 4096,
    };
    pthread_t runtime_thread;
    if (pthread_create(&runtime_thread, NULL,
                       parallel_commit_runtime_shutdown_worker,
                       &runtime_arg) != 0) {
        pqc_parallel_commit_runtime_shutdown();
        pthread_cond_destroy(&runtime_gate.cv);
        pthread_mutex_destroy(&runtime_gate.lock);
        unsetenv("PQC_PARALLEL_COMMIT_MODE");
        unsetenv("PQC_PARALLEL_COMMIT_SHARDS");
        unsetenv("PQC_PARALLEL_COMMIT_GROUP_MAX");
        unsetenv("PQC_PARALLEL_COMMIT_WAIT_NS");
        return -1;
    }

    pthread_mutex_lock(&runtime_gate.lock);
    while (runtime_gate.ready_count != runtime_gate.worker_count)
        pthread_cond_wait(&runtime_gate.cv, &runtime_gate.lock);
    runtime_gate.start = 1;
    pthread_cond_broadcast(&runtime_gate.cv);
    while (runtime_gate.begun_count != runtime_gate.worker_count)
        pthread_cond_wait(&runtime_gate.cv, &runtime_gate.lock);
    pthread_mutex_unlock(&runtime_gate.lock);

    pqc_parallel_commit_runtime_shutdown();
    pthread_join(runtime_thread, NULL);
    ok = runtime_arg.rc == 0 &&
         runtime_arg.ticket.role == PQC_PARALLEL_COMMIT_ROLE_LEADER &&
         runtime_arg.ticket.runtime_ref_held == 0 &&
         runtime_arg.ticket.runtime_coordinator == NULL &&
         !pqc_parallel_commit_runtime_enabled();

    pthread_cond_destroy(&runtime_gate.cv);
    pthread_mutex_destroy(&runtime_gate.lock);
    unsetenv("PQC_PARALLEL_COMMIT_MODE");
    unsetenv("PQC_PARALLEL_COMMIT_SHARDS");
    unsetenv("PQC_PARALLEL_COMMIT_GROUP_MAX");
    unsetenv("PQC_PARALLEL_COMMIT_WAIT_NS");
    return ok ? 0 : -1;
}

int pqc_selftest_rekey_lifecycle(void)
{
    OQS_KEM *kem = OQS_KEM_new(OQS_KEM_alg_ml_kem_768);
    if (!kem)
        kem = OQS_KEM_new(OQS_KEM_alg_kyber_768);
    if (!kem)
        return -1;

    uint8_t *public_key = malloc(kem->length_public_key);
    uint8_t *secret_key = malloc(kem->length_secret_key);
    int ok = public_key && secret_key &&
             OQS_KEM_keypair(kem, public_key, secret_key) == OQS_SUCCESS;

    int stop_before_start = -1;
    int start_first = -1;
    int start_duplicate = -1;
    int stop_first = -1;
    int stop_duplicate = -1;
    int start_after_stop = -1;
    int stop_after_restart = -1;

    if (ok) {
        stop_before_start = pqc_rekey_worker_stop();
        start_first = pqc_rekey_worker_start(kem, public_key);
        start_duplicate = pqc_rekey_worker_start(kem, public_key);
        stop_first = pqc_rekey_worker_stop();
        stop_duplicate = pqc_rekey_worker_stop();
        start_after_stop = pqc_rekey_worker_start(kem, public_key);
        stop_after_restart = pqc_rekey_worker_stop();
        ok = stop_before_start == 0 &&
             start_first == 0 &&
             start_duplicate == 0 &&
             stop_first == 1 &&
             stop_duplicate == 0 &&
             start_after_stop == 0 &&
             stop_after_restart == 1;
    }

    (void)pqc_rekey_worker_stop();
    if (secret_key)
        OQS_MEM_cleanse(secret_key, kem->length_secret_key);
    free(public_key);
    free(secret_key);
    OQS_KEM_free(kem);
    return ok ? 0 : -1;
}

int pqc_selftest_rekey_queue_accounting(void)
{
    return pqc_rekey_queue_accounting_selftest();
}

int pqc_selftest_epoch_log(void)
{
    pqc_epoch_log_record_t record = {
        .record_type = PQC_EPOCH_LOG_RECORD_BLOCK,
        .flags = 3,
        .algorithm_id = PQC_ALGO_AES_256_GCM,
        .epoch = 7,
        .sequence = 9,
        .file_id = UINT64_C(0xabcdef0123456789),
        .logical_block = 11,
        .generation = 13,
        .ciphertext_offset = 4096,
        .logical_size_after = 8192,
        .plaintext_length = 123,
    };
    for (size_t i = 0; i < sizeof(record.tag); ++i)
        record.tag[i] = (uint8_t)(0xa0U + i);

    uint8_t encoded[PQC_EPOCH_LOG_RECORD_SIZE];
    size_t written = 0;
    pqc_epoch_log_record_t decoded = {0};
    int ok =
        pqc_epoch_log_encode_record(&record, encoded, sizeof(encoded),
                                    &written) == 0 &&
        written == PQC_EPOCH_LOG_RECORD_SIZE &&
        pqc_epoch_log_decode_record(encoded, sizeof(encoded), &decoded) == 0 &&
        decoded.record_type == record.record_type &&
        decoded.flags == record.flags &&
        decoded.algorithm_id == record.algorithm_id &&
        decoded.epoch == record.epoch &&
        decoded.sequence == record.sequence &&
        decoded.file_id == record.file_id &&
        decoded.logical_block == record.logical_block &&
        decoded.generation == record.generation &&
        decoded.ciphertext_offset == record.ciphertext_offset &&
        decoded.logical_size_after == record.logical_size_after &&
        decoded.plaintext_length == record.plaintext_length &&
        CRYPTO_memcmp(decoded.tag, record.tag, sizeof(record.tag)) == 0;

    uint8_t tampered[PQC_EPOCH_LOG_RECORD_SIZE];
    memcpy(tampered, encoded, sizeof(tampered));
    tampered[PQC_EPOCH_LOG_RECORD_SIZE - 1U] ^= 0x5aU;
    ok = ok &&
         pqc_epoch_log_decode_record(tampered, sizeof(tampered), &decoded) ==
             -EBADMSG;

    memcpy(tampered, encoded, sizeof(tampered));
    tampered[PQC_EPOCH_LOG_RECORD_VERSION_OFFSET] ^= 0x7fU;
    ok = ok &&
         pqc_epoch_log_decode_record(tampered, sizeof(tampered), &decoded) ==
             -EPROTO;

    record.plaintext_length = PQC_LOGICAL_BLOCK_SIZE + 1U;
    ok = ok &&
         pqc_epoch_log_encode_record(&record, encoded, sizeof(encoded),
                                     &written) == -EINVAL &&
         pqc_epoch_log_decode_record(encoded,
                                     PQC_EPOCH_LOG_RECORD_SIZE - 1U,
                                     &decoded) == -EMSGSIZE;

    FILE *tmp = tmpfile();
    if (!tmp)
        return -1;
    int fd = fileno(tmp);
    pqc_epoch_log_record_t block1 = record;
    block1.record_type = PQC_EPOCH_LOG_RECORD_BLOCK;
    block1.epoch = 1;
    block1.sequence = 1;
    block1.logical_block = 0;
    block1.generation = 1;
    block1.logical_size_after = 4096;
    block1.plaintext_length = 64;
    pqc_epoch_log_record_t commit1 = block1;
    commit1.record_type = PQC_EPOCH_LOG_RECORD_COMMIT;
    commit1.plaintext_length = 0;
    pqc_epoch_log_record_t block2 = block1;
    block2.epoch = 2;
    block2.sequence = 2;
    block2.logical_block = 1;
    block2.generation = 2;
    block2.logical_size_after = 8192;
    pqc_epoch_log_record_t commit2 = block2;
    commit2.record_type = PQC_EPOCH_LOG_RECORD_COMMIT;
    commit2.plaintext_length = 0;

    ok = ok &&
         pqc_epoch_log_append_record_fd(fd, &block1) == 0 &&
         pqc_epoch_log_append_record_fd(fd, &commit1) == 0 &&
         pqc_epoch_log_append_record_fd(fd, &block2) == 0 &&
         pqc_epoch_log_append_record_fd(fd, &commit2) == 0 &&
         pqc_epoch_log_encode_record(&block2, encoded, sizeof(encoded),
                                     &written) == 0 &&
         write(fd, encoded, 17) == 17;
    pqc_epoch_log_replay_summary_t summary;
    memset(&summary, 0, sizeof(summary));
    ok = ok &&
         pqc_epoch_log_replay_fd(fd, record.file_id, &summary) == 0 &&
         summary.block_records == 2 &&
         summary.commit_records == 2 &&
         summary.committed_records == 4 &&
         summary.uncommitted_records == 0 &&
         summary.torn_tail_bytes == 17 &&
         summary.committed_epoch == 2 &&
         summary.logical_size_after == 8192 &&
         summary.max_generation == 2;
    fclose(tmp);

    FILE *dup = tmpfile();
    if (!dup)
        return -1;
    int dup_fd = fileno(dup);
    ok = ok &&
         pqc_epoch_log_append_record_fd(dup_fd, &block1) == 0 &&
         pqc_epoch_log_append_record_fd(dup_fd, &block1) == 0 &&
         pqc_epoch_log_append_record_fd(dup_fd, &commit1) == 0;
    memset(&summary, 0, sizeof(summary));
    ok = ok &&
         pqc_epoch_log_replay_fd(dup_fd, record.file_id, &summary) ==
             -EEXIST &&
         summary.duplicate_generation_records == 1;
    fclose(dup);
    return ok ? 0 : -1;
}

int pqc_selftest_epoch_publish(void)
{
    pqc_publication_mode_t mode = PQC_PUBLICATION_MODE_STRICT;
    int ok =
        pqc_publication_mode_parse(NULL, &mode) == 0 &&
        mode == PQC_PUBLICATION_MODE_STRICT &&
        pqc_publication_mode_parse("", &mode) == 0 &&
        mode == PQC_PUBLICATION_MODE_STRICT &&
        pqc_publication_mode_parse("strict", &mode) == 0 &&
        mode == PQC_PUBLICATION_MODE_STRICT &&
        pqc_publication_mode_parse("epoch", &mode) == 0 &&
        mode == PQC_PUBLICATION_MODE_EPOCH_REDO_LOG &&
        pqc_publication_mode_parse("epoch-redo-log", &mode) == 0 &&
        mode == PQC_PUBLICATION_MODE_EPOCH_REDO_LOG &&
        pqc_publication_mode_parse("epoch-skeleton", &mode) == 0 &&
        mode == PQC_PUBLICATION_MODE_EPOCH_UNAVAILABLE &&
        pqc_publication_mode_parse("invalid", &mode) == -EINVAL &&
        pqc_publication_mode_parse("strict", NULL) == -EINVAL;
    return ok ? 0 : -1;
}

int pqc_selftest_admission_init_modes(void)
{
    if (pqc_admission_init(NULL) != 0)
        return -1;
    pqc_admission_shutdown();

    if (pqc_admission_init("") != 0)
        return -1;
    pqc_admission_shutdown();

    char trace_path[] = "/tmp/aegisq_admission_init_traceXXXXXX";
    int fd = mkstemp(trace_path);
    if (fd < 0)
        return -1;
    close(fd);
    if (unlink(trace_path) != 0)
        return -1;

    if (pqc_admission_init(trace_path) != 0)
        return -1;
    pqc_admission_shutdown();

    struct stat st;
    int ok = stat(trace_path, &st) == 0 && st.st_size > 0;
    (void)unlink(trace_path);
    return ok ? 0 : -1;
}

int pqc_selftest_admission_telemetry_smoke(void)
{
    const char *trace_path =
        pqc_config_nonempty_or_default("PQC_ADMISSION_TRACE_PATH",
                                       "/tmp/aegisq_admission_telemetry_smoke_trace.jsonl");

    if (pqc_admission_init(trace_path) != 0)
        return -1;

    const double mem_bw = pqc_config_double_or_default("PQC_TELEMETRY_MEM_BANDWIDTH", 0.0);
    const double tensor_core = pqc_config_double_or_default("PQC_TELEMETRY_TENSOR_CORE", 0.0);
    const uint64_t ai_budget_ns =
        pqc_config_u64_or_default("PQC_ADMISSION_SMOKE_AI_BUDGET_NS", 2000000ULL);
    const uint64_t cpu_queue_depth =
        pqc_config_u64_or_default("PQC_ADMISSION_SMOKE_CPU_QUEUE_DEPTH", 1ULL);
    const uint64_t gpu_queue_depth =
        pqc_config_u64_or_default("PQC_ADMISSION_SMOKE_GPU_QUEUE_DEPTH", 1ULL);
    const uint64_t uma_cost_ns =
        pqc_config_u64_or_default("PQC_ADMISSION_SMOKE_UMA_COST_NS", 0ULL);
    const size_t bytes_total =
        (size_t)pqc_config_u64_or_default("PQC_ADMISSION_SMOKE_BYTES", 131072ULL);
    const uint64_t batch_age_ns =
        pqc_config_u64_or_default("PQC_ADMISSION_SMOKE_BATCH_AGE_NS", 0ULL);
    const uint64_t stale_sleep_us =
        pqc_config_u64_or_default("PQC_ADMISSION_SMOKE_STALE_SLEEP_US", 0ULL);

    pqc_admission_update_telemetry(mem_bw, tensor_core);
    pqc_admission_update_ai_budget(ai_budget_ns, 0);
    if (stale_sleep_us > 0)
        usleep((useconds_t)stale_sleep_us);

    pqc_admission_context_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.batch_count = 1;
    ctx.bytes_total = bytes_total;
    ctx.batch_age_ns = batch_age_ns;
    ctx.gpu_kernel_est_ns = pqc_config_u64_or_default("PQC_ADMISSION_SMOKE_GPU_KERNEL_NS", 100000ULL);
    ctx.gpu_h2d_staging_ns = pqc_config_u64_or_default("PQC_ADMISSION_SMOKE_H2D_NS", 100000ULL);
    ctx.gpu_d2h_staging_ns = pqc_config_u64_or_default("PQC_ADMISSION_SMOKE_D2H_NS", 100000ULL);
    ctx.cpu_queue_depth = cpu_queue_depth;
    ctx.gpu_queue_depth = gpu_queue_depth;
    ctx.cpu_load_avg = pqc_config_double_or_default("PQC_ADMISSION_SMOKE_CPU_LOAD", 0.0);
    ctx.gpu_load_avg = pqc_config_double_or_default("PQC_ADMISSION_SMOKE_GPU_LOAD", 0.0);
    ctx.ai_inference_deadline_ns =
        pqc_config_u64_or_default("PQC_ADMISSION_SMOKE_DEADLINE_NS", 10000000ULL);
    ctx.uma_migration_cost_ns = uma_cost_ns;
    ctx.uma_migration_bytes_est =
        pqc_config_u64_or_default("PQC_ADMISSION_SMOKE_UMA_BYTES", 0ULL);

    int rc = pqc_admit(&ctx);
    pqc_admission_stats_t stats;
    memset(&stats, 0, sizeof(stats));
    pqc_scheduler_trace_stats(&stats);

    fprintf(stdout,
            "{\"event\":\"admission_telemetry_smoke\","
            "\"rc\":%d,"
            "\"trace_path\":\"%s\","
            "\"telemetry_mem_bandwidth_util\":%.4f,"
            "\"telemetry_tensor_core_util\":%.4f,"
            "\"ai_budget_ns\":%llu,"
            "\"batch_age_ns\":%llu,"
            "\"deadline_ns\":%llu,"
            "\"producer_slack_age_ns\":%llu,"
            "\"producer_slack_stale_after_ns\":%llu,"
            "\"producer_slack_stale\":%s,"
            "\"cpu_queue_depth\":%llu,"
            "\"gpu_queue_depth\":%llu,"
            "\"bytes_total\":%zu,"
            "\"chosen_target\":\"%s\","
            "\"decision_reason\":%u,"
            "\"deferral_reason\":%u,"
            "\"stats_total\":%llu,"
            "\"stats_gpu\":%llu,"
            "\"stats_cpu\":%llu}\n",
            rc,
            trace_path,
            mem_bw,
            tensor_core,
            (unsigned long long)ai_budget_ns,
            (unsigned long long)ctx.batch_age_ns,
            (unsigned long long)ctx.ai_inference_deadline_ns,
            (unsigned long long)ctx.producer_slack_age_ns,
            (unsigned long long)ctx.producer_slack_stale_after_ns,
            ctx.producer_slack_stale ? "true" : "false",
            (unsigned long long)cpu_queue_depth,
            (unsigned long long)gpu_queue_depth,
            bytes_total,
            ctx.chosen_target == PQC_JOB_GPU ? "GPU" : "CPU",
            (unsigned int)ctx.decision_reason,
            (unsigned int)ctx.deferral_reason,
            (unsigned long long)stats.total_requests,
            (unsigned long long)stats.gpu_admitted_count,
            (unsigned long long)stats.cpu_routed_count);

    pqc_admission_shutdown();
    return rc;
}
