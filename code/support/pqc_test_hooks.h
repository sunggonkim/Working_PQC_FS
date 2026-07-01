#ifndef PQC_TEST_HOOKS_H
#define PQC_TEST_HOOKS_H

#include <stdatomic.h>

extern atomic_int g_pqc_fault_cutpoint_enabled;

int pqc_fault_cutpoint_enabled_slow(void);
void pqc_fault_cutpoint_slow(const char *name);

static inline int pqc_fault_cutpoint_enabled_fast(void)
{
    int enabled = atomic_load_explicit(&g_pqc_fault_cutpoint_enabled,
                                       memory_order_acquire);
    if (enabled >= 0)
        return enabled;
    return pqc_fault_cutpoint_enabled_slow();
}

#define pqc_fault_cutpoint(name)                                      \
    do {                                                              \
        if (pqc_fault_cutpoint_enabled_fast())                        \
            pqc_fault_cutpoint_slow((name));                          \
    } while (0)

#endif /* PQC_TEST_HOOKS_H */
