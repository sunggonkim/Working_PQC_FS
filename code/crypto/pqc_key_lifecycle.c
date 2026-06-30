#include "pqc_key_lifecycle.h"

#include <errno.h>
#include <stdint.h>
#include <string.h>

static const pqc_key_lifecycle_entry_t g_key_lifecycle[] = {
    {
        .material = PQC_KEY_MATERIAL_MOUNT_KEY,
        .plane = PQC_KEY_LIFECYCLE_PLANE_KEY,
        .status = PQC_KEY_LIFECYCLE_STATUS_IMPLEMENTED,
        .name = "mount-key",
        .owner_module = "pqc_keyring.c",
        .producer = "pqc_keyring_derive_master_key",
        .storage = "process memory g_master_key; .pqc_kdf stores KDF metadata",
        .protector = "OpenSSL scrypt for new roots; PBKDF2 legacy compatibility only",
        .rotation = "cleared on runtime cleanup and rederived on mount",
        .recovery = "password plus canonical .pqc_kdf metadata",
        .failure_boundary = "invalid KDF metadata rejects; no TPM/PCR key release",
        .hardware_released = 0,
        .data_plane_critical = 0,
    },
    {
        .material = PQC_KEY_MATERIAL_FILE_ENVELOPE_SECRET,
        .plane = PQC_KEY_LIFECYCLE_PLANE_KEY,
        .status = PQC_KEY_LIFECYCLE_STATUS_IMPLEMENTED,
        .name = "per-file-envelope-secret",
        .owner_module = "pqc_keyring.c,pqc_fd_context.c,pqc_rekey.c",
        .producer = "pqc_keyring_metadata_store/load and rekey worker",
        .storage = "user.pqc_metadata xattr plus fd-lifecycle memory",
        .protector = "HMAC-SHA256 metadata check and mount-key/file-id wrapping",
        .rotation = "explicit rekey worker refresh when configured",
        .recovery = "xattr unwrap under the current mount key and file id",
        .failure_boundary = "missing or tampered xattr rejects authenticated open",
        .hardware_released = 0,
        .data_plane_critical = 0,
    },
    {
        .material = PQC_KEY_MATERIAL_DATA_BLOCK_KEY,
        .plane = PQC_KEY_LIFECYCLE_PLANE_DATA,
        .status = PQC_KEY_LIFECYCLE_STATUS_IMPLEMENTED,
        .name = "aes-gcm-data-block-key",
        .owner_module = "pqc_crypto.c,pqc_writeback.c,pqc_file_io.c",
        .producer = "fd-context per-file secret loaded from keyring metadata",
        .storage = "fd-lifecycle memory only",
        .protector = "AES-256-GCM nonce/AAD bound to file id, block, generation, length",
        .rotation = "inherits per-file envelope-secret refresh",
        .recovery = "reload per-file envelope secret before decrypting blocks",
        .failure_boundary = "tag mismatch returns authentication failure",
        .hardware_released = 0,
        .data_plane_critical = 1,
    },
    {
        .material = PQC_KEY_MATERIAL_MOUNT_KEM_KEYPAIR,
        .plane = PQC_KEY_LIFECYCLE_PLANE_KEY,
        .status = PQC_KEY_LIFECYCLE_STATUS_IMPLEMENTED,
        .name = "mount-lifetime-kem-keypair",
        .owner_module = "pqc_runtime.c,pqc_rekey.c",
        .producer = "OQS_KEM_keypair at runtime init",
        .storage = "mount-lifetime process memory",
        .protector = "process isolation and explicit cleanup zeroization",
        .rotation = "new keypair on each mount",
        .recovery = "not persisted; remount generates a new keypair",
        .failure_boundary = "keypair failure aborts runtime init",
        .hardware_released = 0,
        .data_plane_critical = 0,
    },
    {
        .material = PQC_KEY_MATERIAL_FRESHNESS_ANCHOR,
        .plane = PQC_KEY_LIFECYCLE_PLANE_FRESHNESS,
        .status = PQC_KEY_LIFECYCLE_STATUS_IMPLEMENTED,
        .name = "committed-prefix-freshness-anchor",
        .owner_module = "pqc_anchor.c,pqc_checkpoint.c,pqc_anchor_worker.c",
        .producer = "committed-prefix root over per-file generation state",
        .storage = "file backend or administrator-provisioned TPM NV backend",
        .protector = "prefix-anchor digest and fail-closed load checks",
        .rotation = "advanced by checkpoint/freshness-anchor publication",
        .recovery = "anchor load compares stored prefix against reconstructed state",
        .failure_boundary = "file backend is replayable; TPM path depends on provisioning",
        .hardware_released = 0,
        .data_plane_critical = 0,
    },
    {
        .material = PQC_KEY_MATERIAL_TPM_PCR_POLICY,
        .plane = PQC_KEY_LIFECYCLE_PLANE_FRESHNESS,
        .status = PQC_KEY_LIFECYCLE_STATUS_NON_CLAIM,
        .name = "persistent-tpm-pcr-key-release-policy",
        .owner_module = "pqc_anchor.c",
        .producer = "not implemented in the mounted production path",
        .storage = "none in this revision",
        .protector = "none; persistent PCR-bound key release is a non-claim",
        .rotation = "not implemented",
        .recovery = "not implemented",
        .failure_boundary = "must not be used to claim sealed-key or PCR-bound recovery",
        .hardware_released = 0,
        .data_plane_critical = 0,
    },
};

static int text_present(const char *text)
{
    return text && text[0] != '\0';
}

const pqc_key_lifecycle_entry_t *pqc_key_lifecycle_entries(size_t *count_out)
{
    if (count_out)
        *count_out = sizeof(g_key_lifecycle) / sizeof(g_key_lifecycle[0]);
    return g_key_lifecycle;
}

const pqc_key_lifecycle_entry_t *pqc_key_lifecycle_find(
    pqc_key_material_t material)
{
    size_t count = 0;
    const pqc_key_lifecycle_entry_t *entries =
        pqc_key_lifecycle_entries(&count);
    for (size_t i = 0; i < count; ++i) {
        if (entries[i].material == material)
            return &entries[i];
    }
    return NULL;
}

const char *pqc_key_lifecycle_plane_name(pqc_key_lifecycle_plane_t plane)
{
    switch (plane) {
    case PQC_KEY_LIFECYCLE_PLANE_DATA:
        return "data";
    case PQC_KEY_LIFECYCLE_PLANE_KEY:
        return "key";
    case PQC_KEY_LIFECYCLE_PLANE_FRESHNESS:
        return "freshness";
    default:
        return "unknown";
    }
}

const char *pqc_key_lifecycle_status_name(pqc_key_lifecycle_status_t status)
{
    switch (status) {
    case PQC_KEY_LIFECYCLE_STATUS_IMPLEMENTED:
        return "implemented";
    case PQC_KEY_LIFECYCLE_STATUS_EXTERNAL:
        return "external";
    case PQC_KEY_LIFECYCLE_STATUS_NON_CLAIM:
        return "non-claim";
    default:
        return "unknown";
    }
}

const char *pqc_key_material_name(pqc_key_material_t material)
{
    switch (material) {
    case PQC_KEY_MATERIAL_MOUNT_KEY:
        return "mount-key";
    case PQC_KEY_MATERIAL_FILE_ENVELOPE_SECRET:
        return "per-file-envelope-secret";
    case PQC_KEY_MATERIAL_DATA_BLOCK_KEY:
        return "aes-gcm-data-block-key";
    case PQC_KEY_MATERIAL_MOUNT_KEM_KEYPAIR:
        return "mount-lifetime-kem-keypair";
    case PQC_KEY_MATERIAL_FRESHNESS_ANCHOR:
        return "committed-prefix-freshness-anchor";
    case PQC_KEY_MATERIAL_TPM_PCR_POLICY:
        return "persistent-tpm-pcr-key-release-policy";
    default:
        return "unknown";
    }
}

int pqc_key_lifecycle_self_test(void)
{
    uint32_t seen = 0;
    size_t count = 0;
    const pqc_key_lifecycle_entry_t *entries =
        pqc_key_lifecycle_entries(&count);
    if (!entries || count != PQC_KEY_MATERIAL_COUNT)
        return -EINVAL;

    for (size_t i = 0; i < count; ++i) {
        const pqc_key_lifecycle_entry_t *entry = &entries[i];
        if (entry->material < 0 || entry->material >= PQC_KEY_MATERIAL_COUNT)
            return -ERANGE;
        uint32_t bit = UINT32_C(1) << (uint32_t)entry->material;
        if ((seen & bit) != 0)
            return -EEXIST;
        seen |= bit;
        if (!text_present(entry->name) ||
            !text_present(entry->owner_module) ||
            !text_present(entry->producer) ||
            !text_present(entry->storage) ||
            !text_present(entry->protector) ||
            !text_present(entry->rotation) ||
            !text_present(entry->recovery) ||
            !text_present(entry->failure_boundary))
            return -EINVAL;
        if (strcmp(entry->name, pqc_key_material_name(entry->material)) != 0)
            return -EINVAL;
        if (strcmp(pqc_key_lifecycle_plane_name(entry->plane), "unknown") == 0)
            return -EINVAL;
        if (strcmp(pqc_key_lifecycle_status_name(entry->status), "unknown") == 0)
            return -EINVAL;
        if (entry->material == PQC_KEY_MATERIAL_DATA_BLOCK_KEY &&
            (entry->plane != PQC_KEY_LIFECYCLE_PLANE_DATA ||
             !entry->data_plane_critical))
            return -EINVAL;
        if (entry->material != PQC_KEY_MATERIAL_DATA_BLOCK_KEY &&
            entry->data_plane_critical)
            return -EINVAL;
        if (entry->material == PQC_KEY_MATERIAL_TPM_PCR_POLICY &&
            entry->status != PQC_KEY_LIFECYCLE_STATUS_NON_CLAIM)
            return -EINVAL;
        if (entry->hardware_released)
            return -EINVAL;
    }

    uint32_t expected = (UINT32_C(1) << PQC_KEY_MATERIAL_COUNT) - 1U;
    return seen == expected ? 0 : -ENOENT;
}
