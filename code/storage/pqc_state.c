#include "pqc_state.h"

#include "pqc_lock_profile.h"

#include <errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* Shared state is keyed by the backing inode, never by a reusable FD number. */
static pthread_mutex_t g_file_state_table_lock = PTHREAD_MUTEX_INITIALIZER;
static file_state_t *g_file_states = NULL;

static int file_state_table_lock(pqc_lock_profile_scope_t *scope,
                                 const char *site)
{
    return pqc_profiled_mutex_lock(&g_file_state_table_lock,
                                   "file_state_table_lock", site, scope);
}

static int file_state_table_unlock(pqc_lock_profile_scope_t *scope,
                                   const char *site)
{
    return pqc_profiled_mutex_unlock(&g_file_state_table_lock,
                                     "file_state_table_lock", site, scope);
}

static size_t mapping_cache_index(uint64_t logical_block, size_t probe)
{
    uint64_t h = logical_block * 11400714819323198485ull;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdull;
    h ^= h >> 33;
    return (size_t)((h + probe) %
                    PQC_FILE_STATE_MAPPING_CACHE_CAPACITY);
}

static int mapping_cache_alloc_locked(file_state_t *state)
{
    if (!state)
        return -EINVAL;
    if (state->mapping_cache)
        return 0;
    state->mapping_cache = calloc(PQC_FILE_STATE_MAPPING_CACHE_CAPACITY,
                                  sizeof(*state->mapping_cache));
    if (!state->mapping_cache) {
        state->mapping_cache_complete = 0;
        return -ENOMEM;
    }
    state->mapping_cache_count = 0;
    return 0;
}

void pqc_file_state_mapping_cache_mark_complete_empty_locked(file_state_t *state)
{
    if (!state)
        return;
    /*
     * An empty complete mapping does not need a cache allocation.  This keeps
     * first-open initialization from allocating and clearing a full cache while
     * the per-file commit lock is held.
     */
    free(state->mapping_cache);
    state->mapping_cache = NULL;
    state->mapping_cache_count = 0;
    state->mapping_cache_complete = 1;
}

void pqc_file_state_mapping_cache_mark_unknown_locked(file_state_t *state)
{
    if (!state)
        return;
    state->mapping_cache_complete = 0;
}

int pqc_file_state_mapping_cache_store_locked(file_state_t *state,
                                              const block_mapping_t *mappings,
                                              size_t count)
{
    if (!state || (!mappings && count > 0))
        return -EINVAL;
    if (count == 0)
        return 0;
    if (mapping_cache_alloc_locked(state) != 0)
        return -ENOMEM;

    for (size_t i = 0; i < count; ++i) {
        const block_mapping_t *mapping = &mappings[i];
        int stored = 0;
        for (size_t probe = 0;
             probe < PQC_FILE_STATE_MAPPING_CACHE_CAPACITY;
             ++probe) {
            size_t idx = mapping_cache_index(mapping->logical_block, probe);
            pqc_file_state_mapping_slot_t *slot =
                &state->mapping_cache[idx];
            if (!slot->present) {
                slot->present = 1;
                slot->mapping = *mapping;
                ++state->mapping_cache_count;
                stored = 1;
                break;
            }
            if (slot->mapping.logical_block == mapping->logical_block) {
                if (mapping->generation >= slot->mapping.generation)
                    slot->mapping = *mapping;
                stored = 1;
                break;
            }
        }
        if (!stored) {
            state->mapping_cache_complete = 0;
            return -ENOSPC;
        }
    }
    return 0;
}

int pqc_file_state_mapping_cache_lookup(file_state_t *state,
                                        uint64_t logical_block,
                                        uint64_t max_generation,
                                        block_mapping_t *out)
{
    if (!state || !out)
        return -EINVAL;

    pqc_lock_profile_scope_t scope;
    if (pqc_profiled_mutex_lock(&state->commit_lock, "commit_lock",
                                __func__, &scope) != 0)
        return -EAGAIN;
    pqc_file_state_mapping_slot_t *cache = state->mapping_cache;
    int complete = state->mapping_cache_complete;
    if (!cache) {
        (void)pqc_profiled_mutex_unlock(&state->commit_lock, "commit_lock",
                                        __func__, &scope);
        return complete ? -ENOENT : -EAGAIN;
    }
    for (size_t probe = 0; probe < PQC_FILE_STATE_MAPPING_CACHE_CAPACITY;
         ++probe) {
        size_t idx = mapping_cache_index(logical_block, probe);
        pqc_file_state_mapping_slot_t *slot = &cache[idx];
        if (!slot->present) {
            (void)pqc_profiled_mutex_unlock(&state->commit_lock,
                                            "commit_lock", __func__, &scope);
            return complete ? -ENOENT : -EAGAIN;
        }
        if (slot->mapping.logical_block != logical_block)
            continue;
        if (slot->mapping.generation <= max_generation) {
            *out = slot->mapping;
            (void)pqc_profiled_mutex_unlock(&state->commit_lock,
                                            "commit_lock", __func__, &scope);
            return 0;
        }
        (void)pqc_profiled_mutex_unlock(&state->commit_lock, "commit_lock",
                                        __func__, &scope);
        return -EAGAIN;
    }
    (void)pqc_profiled_mutex_unlock(&state->commit_lock, "commit_lock",
                                    __func__, &scope);
    return -EAGAIN;
}

file_state_t *pqc_file_state_acquire(int fd)
{
    struct stat st;
    if (fstat(fd, &st) != 0)
        return NULL;

    pqc_lock_profile_scope_t table_scope;
    if (file_state_table_lock(&table_scope, __func__) != 0)
        return NULL;
    for (file_state_t *p = g_file_states; p; p = p->next) {
        if (p->dev == st.st_dev && p->ino == st.st_ino) {
            ++p->refs;
            (void)file_state_table_unlock(&table_scope, __func__);
            return p;
        }
    }

    file_state_t *p = calloc(1, sizeof(*p));
    if (p) {
        p->dev = st.st_dev;
        p->ino = st.st_ino;
        p->refs = 1;
        pthread_mutex_init(&p->commit_lock, NULL);
        pthread_cond_init(&p->publish_cv, NULL);
        p->next = g_file_states;
        g_file_states = p;
    }
    (void)file_state_table_unlock(&table_scope, __func__);
    return p;
}

void pqc_file_state_release(file_state_t *state)
{
    if (!state)
        return;

    file_state_t *retired = NULL;
    pqc_lock_profile_scope_t table_scope;
    if (file_state_table_lock(&table_scope, __func__) != 0)
        return;
    if (--state->refs == 0) {
        file_state_t **p = &g_file_states;
        while (*p && *p != state)
            p = &(*p)->next;
        if (*p) {
            *p = state->next;
            retired = state;
        }
    }
    (void)file_state_table_unlock(&table_scope, __func__);

    if (retired) {
        free(retired->mapping_cache);
        pthread_cond_destroy(&retired->publish_cv);
        pthread_mutex_destroy(&retired->commit_lock);
        free(retired);
    }
}
