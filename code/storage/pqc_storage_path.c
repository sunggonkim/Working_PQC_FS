#include "pqc_storage_path.h"

#include <stdio.h>

static char g_storage_dir[4096] = {0};

void pqc_storage_path_set_root(const char *root)
{
    snprintf(g_storage_dir, sizeof(g_storage_dir), "%s", root ? root : "");
}

const char *pqc_storage_path_root(void)
{
    return g_storage_dir;
}

void pqc_storage_path_resolve(char *dest, size_t dest_size, const char *path)
{
    const char *suffix = path ? path : "";
    const char *sep = (suffix[0] == '\0' || suffix[0] == '/') ? "" : "/";
    snprintf(dest, dest_size, "%s%s%s", g_storage_dir, sep, suffix);
}
