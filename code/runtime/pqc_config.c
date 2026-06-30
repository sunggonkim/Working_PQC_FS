#include "pqc_config.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PQC_CONFIG_MAX_ENTRIES 256
#define PQC_CONFIG_NAME_MAX 96
#define PQC_CONFIG_ACCESSOR_MAX 32
#define PQC_CONFIG_STATUS_MAX 32
#define PQC_CONFIG_VALUE_MAX 192

typedef struct {
    char name[PQC_CONFIG_NAME_MAX];
    char accessor[PQC_CONFIG_ACCESSOR_MAX];
    char parse_status[PQC_CONFIG_STATUS_MAX];
    char raw_value[PQC_CONFIG_VALUE_MAX];
    char effective_value[PQC_CONFIG_VALUE_MAX];
    char fallback_value[PQC_CONFIG_VALUE_MAX];
    unsigned long long access_count;
    int present;
    int redacted;
} pqc_config_entry_t;

static pthread_mutex_t g_config_lock = PTHREAD_MUTEX_INITIALIZER;
static pqc_config_entry_t g_config_entries[PQC_CONFIG_MAX_ENTRIES];
static size_t g_config_entry_count = 0;

static int config_lock(void)
{
    return pthread_mutex_lock(&g_config_lock);
}

static int config_unlock(void)
{
    return pthread_mutex_unlock(&g_config_lock);
}

static const char *pqc_config_getenv_unrecorded(const char *name)
{
    return name ? getenv(name) : NULL;
}

static int pqc_config_is_sensitive(const char *name)
{
    if (!name) return 0;
    return strstr(name, "PASSWORD") ||
           strstr(name, "SECRET") ||
           strstr(name, "TOKEN") ||
           strstr(name, "PRIVATE");
}

static void pqc_config_copy(char *dst, size_t dst_len, const char *src)
{
    if (!dst || dst_len == 0) return;
    if (!src) src = "";
    snprintf(dst, dst_len, "%s", src);
}

static void pqc_config_format_u64(char *dst, size_t dst_len, uint64_t value)
{
    snprintf(dst, dst_len, "%llu", (unsigned long long)value);
}

static void pqc_config_format_long(char *dst, size_t dst_len, long value)
{
    snprintf(dst, dst_len, "%ld", value);
}

static void pqc_config_format_double(char *dst, size_t dst_len, double value)
{
    snprintf(dst, dst_len, "%.17g", value);
}

static pqc_config_entry_t *pqc_config_find_or_create_locked(const char *name)
{
    if (!name || !*name) return NULL;
    for (size_t i = 0; i < g_config_entry_count; i++) {
        if (strcmp(g_config_entries[i].name, name) == 0)
            return &g_config_entries[i];
    }
    if (g_config_entry_count >= PQC_CONFIG_MAX_ENTRIES)
        return NULL;
    pqc_config_entry_t *entry = &g_config_entries[g_config_entry_count++];
    memset(entry, 0, sizeof(*entry));
    pqc_config_copy(entry->name, sizeof(entry->name), name);
    return entry;
}

static void pqc_config_record(const char *name, const char *accessor,
                              const char *raw_value, const char *parse_status,
                              const char *effective_value,
                              const char *fallback_value)
{
    if (!name || !*name) return;
    if (config_lock() != 0)
        return;
    pqc_config_entry_t *entry = pqc_config_find_or_create_locked(name);
    if (entry) {
        entry->access_count++;
        entry->present = raw_value != NULL;
        entry->redacted = pqc_config_is_sensitive(name);
        pqc_config_copy(entry->accessor, sizeof(entry->accessor), accessor);
        pqc_config_copy(entry->parse_status, sizeof(entry->parse_status),
                        parse_status);
        if (entry->redacted && raw_value) {
            pqc_config_copy(entry->raw_value, sizeof(entry->raw_value),
                            "<redacted>");
        } else {
            pqc_config_copy(entry->raw_value, sizeof(entry->raw_value),
                            raw_value);
        }
        pqc_config_copy(entry->effective_value, sizeof(entry->effective_value),
                        effective_value);
        pqc_config_copy(entry->fallback_value, sizeof(entry->fallback_value),
                        fallback_value);
    }
    (void)config_unlock();
}

const char *pqc_config_getenv(const char *name)
{
    const char *value = pqc_config_getenv_unrecorded(name);
    pqc_config_record(name, "string", value,
                      value ? "present" : "unset", value, "");
    return value;
}

const char *pqc_config_get_nonempty(const char *name)
{
    const char *value = pqc_config_getenv_unrecorded(name);
    const char *effective = (value && *value) ? value : "";
    pqc_config_record(name, "string-nonempty", value,
                      !value ? "unset" : (*value ? "present" : "empty"),
                      effective, "");
    return (value && *value) ? value : NULL;
}

const char *pqc_config_nonempty_or_default(const char *name,
                                           const char *fallback)
{
    const char *value = pqc_config_getenv_unrecorded(name);
    const char *effective = (value && *value) ? value : fallback;
    pqc_config_record(name, "string-default", value,
                      !value ? "unset" : (*value ? "present" : "empty"),
                      effective, fallback);
    return effective;
}

int pqc_config_present(const char *name)
{
    const char *value = pqc_config_getenv_unrecorded(name);
    int present = value != NULL;
    pqc_config_record(name, "present", value,
                      present ? "set" : "unset",
                      present ? "1" : "0", "0");
    return present;
}

int pqc_config_enabled(const char *name)
{
    const char *value = pqc_config_getenv_unrecorded(name);
    int enabled = value && *value && strcmp(value, "0") != 0;
    pqc_config_record(name, "bool-enabled", value,
                      !value ? "unset" : (!*value ? "empty" :
                      (strcmp(value, "0") == 0 ? "zero-disabled" : "enabled")),
                      enabled ? "1" : "0", "0");
    return enabled;
}

uint64_t pqc_config_u64_or_default(const char *name, uint64_t fallback)
{
    const char *value = pqc_config_getenv_unrecorded(name);
    char effective[PQC_CONFIG_VALUE_MAX];
    char fallback_text[PQC_CONFIG_VALUE_MAX];
    pqc_config_format_u64(fallback_text, sizeof(fallback_text), fallback);
    if (!value || !*value) {
        pqc_config_record(name, "u64", value, value ? "empty" : "unset",
                          fallback_text, fallback_text);
        return fallback;
    }
    char *end = NULL;
    unsigned long long parsed = strtoull(value, &end, 10);
    if (end == value || !end || *end != '\0') {
        pqc_config_record(name, "u64", value, "invalid", fallback_text,
                          fallback_text);
        return fallback;
    }
    pqc_config_format_u64(effective, sizeof(effective), (uint64_t)parsed);
    pqc_config_record(name, "u64", value, "valid", effective, fallback_text);
    return (uint64_t)parsed;
}

uint64_t pqc_config_u64_legacy_or_default(const char *name, uint64_t fallback)
{
    const char *value = pqc_config_getenv_unrecorded(name);
    char effective[PQC_CONFIG_VALUE_MAX];
    char fallback_text[PQC_CONFIG_VALUE_MAX];
    pqc_config_format_u64(fallback_text, sizeof(fallback_text), fallback);
    if (!value) {
        pqc_config_record(name, "u64-legacy-prefix", value, "unset",
                          fallback_text, fallback_text);
        return fallback;
    }
    char *end = NULL;
    unsigned long long parsed = strtoull(value, &end, 10);
    const char *status = "valid-prefix";
    if (!*value)
        status = "empty-zero";
    else if (end == value)
        status = "invalid-zero";
    else if (end && *end == '\0')
        status = "valid";
    pqc_config_format_u64(effective, sizeof(effective), (uint64_t)parsed);
    pqc_config_record(name, "u64-legacy-prefix", value, status, effective,
                      fallback_text);
    return (uint64_t)parsed;
}

uint64_t pqc_config_u64_base_or_default(const char *name, uint64_t fallback,
                                        int base)
{
    const char *value = pqc_config_getenv_unrecorded(name);
    char effective[PQC_CONFIG_VALUE_MAX];
    char fallback_text[PQC_CONFIG_VALUE_MAX];
    pqc_config_format_u64(fallback_text, sizeof(fallback_text), fallback);
    if (!value || !*value) {
        pqc_config_record(name, "u64-base", value, value ? "empty" : "unset",
                          fallback_text, fallback_text);
        return fallback;
    }
    char *end = NULL;
    unsigned long parsed = strtoul(value, &end, base);
    if (end == value || !end || *end != '\0') {
        pqc_config_record(name, "u64-base", value, "invalid", fallback_text,
                          fallback_text);
        return fallback;
    }
    pqc_config_format_u64(effective, sizeof(effective), (uint64_t)parsed);
    pqc_config_record(name, "u64-base", value, "valid", effective,
                      fallback_text);
    return (uint64_t)parsed;
}

long pqc_config_long_or_default(const char *name, long fallback)
{
    const char *value = pqc_config_getenv_unrecorded(name);
    char effective[PQC_CONFIG_VALUE_MAX];
    char fallback_text[PQC_CONFIG_VALUE_MAX];
    pqc_config_format_long(fallback_text, sizeof(fallback_text), fallback);
    if (!value || !*value) {
        pqc_config_record(name, "long", value, value ? "empty" : "unset",
                          fallback_text, fallback_text);
        return fallback;
    }
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    if (end == value || !end || *end != '\0') {
        pqc_config_record(name, "long", value, "invalid", fallback_text,
                          fallback_text);
        return fallback;
    }
    pqc_config_format_long(effective, sizeof(effective), parsed);
    pqc_config_record(name, "long", value, "valid", effective, fallback_text);
    return parsed;
}

long pqc_config_long_legacy_or_default(const char *name, long fallback)
{
    const char *value = pqc_config_getenv_unrecorded(name);
    char effective[PQC_CONFIG_VALUE_MAX];
    char fallback_text[PQC_CONFIG_VALUE_MAX];
    pqc_config_format_long(fallback_text, sizeof(fallback_text), fallback);
    if (!value) {
        pqc_config_record(name, "long-legacy-prefix", value, "unset",
                          fallback_text, fallback_text);
        return fallback;
    }
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    const char *status = "valid-prefix";
    if (!*value)
        status = "empty-zero";
    else if (end == value)
        status = "invalid-zero";
    else if (end && *end == '\0')
        status = "valid";
    pqc_config_format_long(effective, sizeof(effective), parsed);
    pqc_config_record(name, "long-legacy-prefix", value, status, effective,
                      fallback_text);
    return parsed;
}

long pqc_config_positive_long_or_default(const char *name, long fallback)
{
    const char *value = pqc_config_getenv_unrecorded(name);
    char effective[PQC_CONFIG_VALUE_MAX];
    char fallback_text[PQC_CONFIG_VALUE_MAX];
    pqc_config_format_long(fallback_text, sizeof(fallback_text), fallback);
    if (!value || !*value) {
        pqc_config_record(name, "positive-long-prefix", value,
                          value ? "empty" : "unset", fallback_text,
                          fallback_text);
        return fallback;
    }
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    if (end == value || parsed <= 0) {
        pqc_config_record(name, "positive-long-prefix", value, "invalid",
                          fallback_text, fallback_text);
        return fallback;
    }
    pqc_config_format_long(effective, sizeof(effective), parsed);
    pqc_config_record(name, "positive-long-prefix", value, "valid-prefix",
                      effective, fallback_text);
    return parsed;
}

double pqc_config_double_or_default(const char *name, double fallback)
{
    const char *value = pqc_config_getenv_unrecorded(name);
    char effective[PQC_CONFIG_VALUE_MAX];
    char fallback_text[PQC_CONFIG_VALUE_MAX];
    pqc_config_format_double(fallback_text, sizeof(fallback_text), fallback);
    if (!value || !*value) {
        pqc_config_record(name, "double", value, value ? "empty" : "unset",
                          fallback_text, fallback_text);
        return fallback;
    }
    char *end = NULL;
    double parsed = strtod(value, &end);
    if (end == value || !end || *end != '\0') {
        pqc_config_record(name, "double", value, "invalid", fallback_text,
                          fallback_text);
        return fallback;
    }
    pqc_config_format_double(effective, sizeof(effective), parsed);
    pqc_config_record(name, "double", value, "valid", effective,
                      fallback_text);
    return parsed;
}

double pqc_config_double_legacy_or_default(const char *name, double fallback)
{
    const char *value = pqc_config_getenv_unrecorded(name);
    char effective[PQC_CONFIG_VALUE_MAX];
    char fallback_text[PQC_CONFIG_VALUE_MAX];
    pqc_config_format_double(fallback_text, sizeof(fallback_text), fallback);
    if (!value) {
        pqc_config_record(name, "double-legacy-prefix", value, "unset",
                          fallback_text, fallback_text);
        return fallback;
    }
    char *end = NULL;
    double parsed = strtod(value, &end);
    const char *status = "valid-prefix";
    if (!*value)
        status = "empty-zero";
    else if (end == value)
        status = "invalid-zero";
    else if (end && *end == '\0')
        status = "valid";
    pqc_config_format_double(effective, sizeof(effective), parsed);
    pqc_config_record(name, "double-legacy-prefix", value, status, effective,
                      fallback_text);
    return parsed;
}

double pqc_config_double_prefix_or_default(const char *name, double fallback)
{
    const char *value = pqc_config_getenv_unrecorded(name);
    char effective[PQC_CONFIG_VALUE_MAX];
    char fallback_text[PQC_CONFIG_VALUE_MAX];
    pqc_config_format_double(fallback_text, sizeof(fallback_text), fallback);
    if (!value || !*value) {
        pqc_config_record(name, "double-prefix", value,
                          value ? "empty" : "unset", fallback_text,
                          fallback_text);
        return fallback;
    }
    char *end = NULL;
    double parsed = strtod(value, &end);
    if (end == value) {
        pqc_config_record(name, "double-prefix", value, "invalid",
                          fallback_text, fallback_text);
        return fallback;
    }
    pqc_config_format_double(effective, sizeof(effective), parsed);
    pqc_config_record(name, "double-prefix", value, "valid-prefix",
                      effective, fallback_text);
    return parsed;
}

static void pqc_config_json_string(FILE *fp, const char *value)
{
    fputc('"', fp);
    if (value) {
        for (const unsigned char *p = (const unsigned char *)value; *p; p++) {
            switch (*p) {
            case '\\':
                fputs("\\\\", fp);
                break;
            case '"':
                fputs("\\\"", fp);
                break;
            case '\n':
                fputs("\\n", fp);
                break;
            case '\r':
                fputs("\\r", fp);
                break;
            case '\t':
                fputs("\\t", fp);
                break;
            default:
                if (*p < 0x20)
                    fprintf(fp, "\\u%04x", (unsigned int)*p);
                else
                    fputc(*p, fp);
                break;
            }
        }
    }
    fputc('"', fp);
}

int pqc_config_dump_file(const char *path)
{
    if (!path || !*path) return -1;

    pqc_config_entry_t snapshot[PQC_CONFIG_MAX_ENTRIES];
    size_t snapshot_count;
    if (config_lock() != 0)
        return -1;
    snapshot_count = g_config_entry_count;
    if (snapshot_count > PQC_CONFIG_MAX_ENTRIES)
        snapshot_count = PQC_CONFIG_MAX_ENTRIES;
    memcpy(snapshot, g_config_entries,
           snapshot_count * sizeof(snapshot[0]));
    if (config_unlock() != 0)
        return -1;

    FILE *fp = fopen(path, "w");
    if (!fp) return -1;

    time_t now = time(NULL);
    fprintf(fp, "{\n");
    fprintf(fp, "  \"schema_version\": 1,\n");
    fprintf(fp, "  \"generated_by\": \"pqc_config_dump_file\",\n");
    fprintf(fp, "  \"unix_time\": %lld,\n", (long long)now);
    fprintf(fp, "  \"entry_count\": %zu,\n", snapshot_count);
    fprintf(fp, "  \"negative_claim_guard\": ");
    pqc_config_json_string(fp, "This runtime config dump records observed knobs only; it does not prove performance, correctness, security, or deployment claims.");
    fprintf(fp, ",\n");
    fprintf(fp, "  \"variables\": [\n");
    for (size_t i = 0; i < snapshot_count; i++) {
        const pqc_config_entry_t *entry = &snapshot[i];
        fprintf(fp, "    {\n");
        fprintf(fp, "      \"name\": ");
        pqc_config_json_string(fp, entry->name);
        fprintf(fp, ",\n      \"accessor\": ");
        pqc_config_json_string(fp, entry->accessor);
        fprintf(fp, ",\n      \"present\": %s,\n",
                entry->present ? "true" : "false");
        fprintf(fp, "      \"redacted\": %s,\n",
                entry->redacted ? "true" : "false");
        fprintf(fp, "      \"parse_status\": ");
        pqc_config_json_string(fp, entry->parse_status);
        fprintf(fp, ",\n      \"raw_value\": ");
        pqc_config_json_string(fp, entry->raw_value);
        fprintf(fp, ",\n      \"effective_value\": ");
        pqc_config_json_string(fp, entry->effective_value);
        fprintf(fp, ",\n      \"fallback_value\": ");
        pqc_config_json_string(fp, entry->fallback_value);
        fprintf(fp, ",\n      \"access_count\": %llu\n",
                entry->access_count);
        fprintf(fp, "    }%s\n",
                (i + 1 == snapshot_count) ? "" : ",");
    }
    fprintf(fp, "  ]\n");
    fprintf(fp, "}\n");

    int rc = ferror(fp) ? -1 : 0;
    if (fclose(fp) != 0) rc = -1;
    return rc;
}

int pqc_config_dump_if_requested(void)
{
    const char *path = pqc_config_get_nonempty("PQC_CONFIG_DUMP_PATH");
    if (!path) return 0;
    return pqc_config_dump_file(path);
}
