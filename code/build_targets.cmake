# CPU/FUSE production target ownership for AEGIS-Q.
include(${CODE_DIR}/include_paths.cmake)

# ── Target 1: CPU Baseline (pqc_fuse) ──────────────────────────────────────
if(HAS_OQS)
    include(${CODE_DIR}/source_manifest.cmake)
    add_library(pqc_fuse_core STATIC ${PQC_FUSE_SOURCES})

    add_executable(pqc_fuse ${PQC_FUSE_ENTRY_SOURCES})
    set_target_properties(pqc_fuse PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
endif()
