#include "pqc_trust_boundary.h"

#include <errno.h>
#include <stdint.h>
#include <string.h>

static const pqc_trust_boundary_entry_t g_trust_boundary[] = {
    {
        .subject = PQC_TRUST_SUBJECT_DAEMON_PROCESS,
        .status = PQC_TRUST_STATUS_IMPLEMENTED_BOUNDARY,
        .name = "daemon-process",
        .trusted_component = "pqc_fuse userspace daemon and OpenSSL/OQS libraries",
        .implemented_boundary =
            "authenticated encryption, generation checks, explicit key cleanup, "
            "and fail-closed metadata parsing inside the mounted daemon path",
        .excluded_attacker =
            "attacker that can read or modify daemon memory, ptrace the process, "
            "or replace linked crypto libraries",
        .failure_boundary =
            "daemon memory compromise can disclose mount and per-file secrets",
        .non_claim_guard =
            "do not claim process compromise resistance or deployment hardening",
        .claims_defense = 1,
        .deployment_ready = 0,
    },
    {
        .subject = PQC_TRUST_SUBJECT_KERNEL_DRIVER_FUSE,
        .status = PQC_TRUST_STATUS_NON_CLAIM,
        .name = "kernel-driver-fuse-stack",
        .trusted_component = "Linux kernel, VFS, FUSE driver, filesystem driver, and block stack",
        .implemented_boundary =
            "no production code defends against a compromised kernel, driver, "
            "or FUSE stack",
        .excluded_attacker =
            "compromised kernel, malicious FUSE driver, malicious filesystem "
            "driver, or malicious block driver",
        .failure_boundary =
            "kernel or driver compromise can observe, reorder, suppress, or forge I/O",
        .non_claim_guard =
            "must not claim compromised-kernel, driver, or FUSE-stack defense",
        .claims_defense = 0,
        .deployment_ready = 0,
    },
    {
        .subject = PQC_TRUST_SUBJECT_BACKING_STORAGE,
        .status = PQC_TRUST_STATUS_EXTERNAL_TCB,
        .name = "backing-storage",
        .trusted_component = "local lower filesystem and configured anchor backend",
        .implemented_boundary =
            "ciphertext/tag verification and committed-prefix anchor checks for "
            "the stated daemon/crash model",
        .excluded_attacker =
            "administrator that can roll back the complete lower storage root and "
            "all file-backed anchors without a hardware freshness source",
        .failure_boundary =
            "file-backed anchors are replayable; TPM-backed freshness depends on provisioning",
        .non_claim_guard =
            "must not claim full rollback resistance from file-backed anchors",
        .claims_defense = 1,
        .deployment_ready = 0,
    },
    {
        .subject = PQC_TRUST_SUBJECT_PRIVILEGED_LOCAL_ATTACKER,
        .status = PQC_TRUST_STATUS_NON_CLAIM,
        .name = "privileged-local-attacker",
        .trusted_component = "host OS privilege separation",
        .implemented_boundary =
            "no production path defends against root, CAP_SYS_ADMIN, ptrace, "
            "or direct lower-storage access by a privileged local attacker",
        .excluded_attacker =
            "root user, host administrator, or process with equivalent local privilege",
        .failure_boundary =
            "privileged local access can read daemon memory, alter mounts, and modify storage",
        .non_claim_guard =
            "must not claim privileged local attacker defense",
        .claims_defense = 0,
        .deployment_ready = 0,
    },
    {
        .subject = PQC_TRUST_SUBJECT_MULTI_TENANT_GPU,
        .status = PQC_TRUST_STATUS_NON_CLAIM,
        .name = "multi-tenant-gpu",
        .trusted_component = "exclusive or administratively isolated GPU execution environment",
        .implemented_boundary =
            "no production code implements GPU tenant isolation, MIG partitioning, "
            "or scheduler-enforced cross-tenant secrecy",
        .excluded_attacker =
            "co-resident GPU tenant, hostile CUDA process, or shared-device observer",
        .failure_boundary =
            "multi-tenant GPU deployment requires platform isolation outside AEGIS-Q",
        .non_claim_guard =
            "must not claim multi-tenant GPU isolation or cross-tenant defense",
        .claims_defense = 0,
        .deployment_ready = 0,
    },
    {
        .subject = PQC_TRUST_SUBJECT_GPU_SIDE_CHANNEL,
        .status = PQC_TRUST_STATUS_NON_CLAIM,
        .name = "gpu-side-channel",
        .trusted_component = "single-tenant GPU timing and resource domain assumption",
        .implemented_boundary =
            "no production code implements timing, cache, memory-bus, power, "
            "thermal, or occupancy side-channel defenses",
        .excluded_attacker =
            "local observer measuring GPU timing, cache effects, memory traffic, "
            "power, temperature, or scheduler occupancy",
        .failure_boundary =
            "GPU side-channel resistance is out of scope for this implementation",
        .non_claim_guard =
            "must not claim side-channel protection",
        .claims_defense = 0,
        .deployment_ready = 0,
    },
    {
        .subject = PQC_TRUST_SUBJECT_DEPLOYMENT_READINESS,
        .status = PQC_TRUST_STATUS_NON_CLAIM,
        .name = "deployment-readiness",
        .trusted_component = "research prototype configuration and operator discipline",
        .implemented_boundary =
            "production code includes explicit checks and fail-closed paths, but "
            "does not provide deployment certification or a hardening profile",
        .excluded_attacker =
            "production adversary model requiring complete POSIX coverage, "
            "privileged-attacker defense, side-channel defense, or certified crash safety",
        .failure_boundary =
            "the code is not a general-purpose deployment-ready secure filesystem",
        .non_claim_guard =
            "must not claim ready for deployment or general-purpose filesystem status",
        .claims_defense = 0,
        .deployment_ready = 0,
    },
};

static int text_present(const char *text)
{
    return text && text[0] != '\0';
}

static int text_contains(const char *text, const char *needle)
{
    return text && needle && strstr(text, needle) != NULL;
}

const pqc_trust_boundary_entry_t *pqc_trust_boundary_entries(
    size_t *count_out)
{
    if (count_out)
        *count_out = sizeof(g_trust_boundary) / sizeof(g_trust_boundary[0]);
    return g_trust_boundary;
}

const pqc_trust_boundary_entry_t *pqc_trust_boundary_find(
    pqc_trust_subject_t subject)
{
    size_t count = 0;
    const pqc_trust_boundary_entry_t *entries =
        pqc_trust_boundary_entries(&count);
    for (size_t i = 0; i < count; ++i) {
        if (entries[i].subject == subject)
            return &entries[i];
    }
    return NULL;
}

const char *pqc_trust_subject_name(pqc_trust_subject_t subject)
{
    switch (subject) {
    case PQC_TRUST_SUBJECT_DAEMON_PROCESS:
        return "daemon-process";
    case PQC_TRUST_SUBJECT_KERNEL_DRIVER_FUSE:
        return "kernel-driver-fuse-stack";
    case PQC_TRUST_SUBJECT_BACKING_STORAGE:
        return "backing-storage";
    case PQC_TRUST_SUBJECT_PRIVILEGED_LOCAL_ATTACKER:
        return "privileged-local-attacker";
    case PQC_TRUST_SUBJECT_MULTI_TENANT_GPU:
        return "multi-tenant-gpu";
    case PQC_TRUST_SUBJECT_GPU_SIDE_CHANNEL:
        return "gpu-side-channel";
    case PQC_TRUST_SUBJECT_DEPLOYMENT_READINESS:
        return "deployment-readiness";
    default:
        return "unknown";
    }
}

const char *pqc_trust_status_name(pqc_trust_status_t status)
{
    switch (status) {
    case PQC_TRUST_STATUS_IMPLEMENTED_BOUNDARY:
        return "implemented-boundary";
    case PQC_TRUST_STATUS_EXTERNAL_TCB:
        return "external-tcb";
    case PQC_TRUST_STATUS_NON_CLAIM:
        return "non-claim";
    default:
        return "unknown";
    }
}

static int require_non_claim(pqc_trust_subject_t subject,
                             const char *required_guard)
{
    const pqc_trust_boundary_entry_t *entry =
        pqc_trust_boundary_find(subject);
    if (!entry)
        return -ENOENT;
    if (entry->status != PQC_TRUST_STATUS_NON_CLAIM ||
        entry->claims_defense != 0 || entry->deployment_ready != 0)
        return -EINVAL;
    if (!text_contains(entry->non_claim_guard, required_guard))
        return -EINVAL;
    return 0;
}

int pqc_trust_boundary_self_test(void)
{
    uint32_t seen = 0;
    size_t count = 0;
    const pqc_trust_boundary_entry_t *entries =
        pqc_trust_boundary_entries(&count);
    if (!entries || count != PQC_TRUST_SUBJECT_COUNT)
        return -EINVAL;

    for (size_t i = 0; i < count; ++i) {
        const pqc_trust_boundary_entry_t *entry = &entries[i];
        if (entry->subject < 0 || entry->subject >= PQC_TRUST_SUBJECT_COUNT)
            return -ERANGE;
        uint32_t bit = UINT32_C(1) << (uint32_t)entry->subject;
        if ((seen & bit) != 0)
            return -EEXIST;
        seen |= bit;
        if (!text_present(entry->name) ||
            !text_present(entry->trusted_component) ||
            !text_present(entry->implemented_boundary) ||
            !text_present(entry->excluded_attacker) ||
            !text_present(entry->failure_boundary) ||
            !text_present(entry->non_claim_guard))
            return -EINVAL;
        if (strcmp(entry->name, pqc_trust_subject_name(entry->subject)) != 0)
            return -EINVAL;
        if (strcmp(pqc_trust_status_name(entry->status), "unknown") == 0)
            return -EINVAL;
        if (entry->status == PQC_TRUST_STATUS_NON_CLAIM &&
            entry->claims_defense)
            return -EINVAL;
        if (entry->deployment_ready)
            return -EINVAL;
    }

    uint32_t expected = (UINT32_C(1) << PQC_TRUST_SUBJECT_COUNT) - 1U;
    if (seen != expected)
        return -ENOENT;
    if (require_non_claim(PQC_TRUST_SUBJECT_KERNEL_DRIVER_FUSE,
                          "kernel") != 0)
        return -EINVAL;
    if (require_non_claim(PQC_TRUST_SUBJECT_PRIVILEGED_LOCAL_ATTACKER,
                          "privileged") != 0)
        return -EINVAL;
    if (require_non_claim(PQC_TRUST_SUBJECT_MULTI_TENANT_GPU,
                          "multi-tenant") != 0)
        return -EINVAL;
    if (require_non_claim(PQC_TRUST_SUBJECT_GPU_SIDE_CHANNEL,
                          "side-channel") != 0)
        return -EINVAL;
    if (require_non_claim(PQC_TRUST_SUBJECT_DEPLOYMENT_READINESS,
                          "ready for deployment") != 0)
        return -EINVAL;

    return 0;
}
