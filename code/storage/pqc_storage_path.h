#ifndef PQC_STORAGE_PATH_H
#define PQC_STORAGE_PATH_H

#include <stddef.h>

void pqc_storage_path_set_root(const char *root);
const char *pqc_storage_path_root(void);
void pqc_storage_path_resolve(char *dest, size_t dest_size, const char *path);

#endif /* PQC_STORAGE_PATH_H */
