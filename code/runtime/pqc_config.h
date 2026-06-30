#ifndef PQC_CONFIG_H
#define PQC_CONFIG_H

#include <stdint.h>

const char *pqc_config_getenv(const char *name);
const char *pqc_config_get_nonempty(const char *name);
const char *pqc_config_nonempty_or_default(const char *name,
                                           const char *fallback);
int pqc_config_present(const char *name);
int pqc_config_enabled(const char *name);

uint64_t pqc_config_u64_or_default(const char *name, uint64_t fallback);
uint64_t pqc_config_u64_legacy_or_default(const char *name, uint64_t fallback);
uint64_t pqc_config_u64_base_or_default(const char *name, uint64_t fallback,
                                        int base);
long pqc_config_long_or_default(const char *name, long fallback);
long pqc_config_long_legacy_or_default(const char *name, long fallback);
long pqc_config_positive_long_or_default(const char *name, long fallback);
double pqc_config_double_or_default(const char *name, double fallback);
double pqc_config_double_legacy_or_default(const char *name, double fallback);
double pqc_config_double_prefix_or_default(const char *name, double fallback);

int pqc_config_dump_file(const char *path);
int pqc_config_dump_if_requested(void);

#endif /* PQC_CONFIG_H */
