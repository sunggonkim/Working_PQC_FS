include(${CODE_DIR}/common/includes.cmake)
include(${CODE_DIR}/frontend/includes.cmake)
include(${CODE_DIR}/crypto/includes.cmake)
include(${CODE_DIR}/storage/includes.cmake)
include(${CODE_DIR}/runtime/includes.cmake)
include(${CODE_DIR}/fs/includes.cmake)
include(${CODE_DIR}/support/includes.cmake)
include(${CODE_DIR}/gpu/includes.cmake)

set(PQC_PRODUCTION_INCLUDE_DIRS
    ${PQC_COMMON_INCLUDE_DIRS}
    ${PQC_FRONTEND_INCLUDE_DIRS}
    ${PQC_CRYPTO_INCLUDE_DIRS}
    ${PQC_STORAGE_INCLUDE_DIRS}
    ${PQC_RUNTIME_INCLUDE_DIRS}
    ${PQC_FILESYSTEM_INCLUDE_DIRS}
    ${PQC_SUPPORT_INCLUDE_DIRS}
    ${PQC_GPU_INCLUDE_DIRS}
)

set(PQC_GPU_PUBLIC_INCLUDE_DIRS
    ${PQC_COMMON_INCLUDE_DIRS}
    ${PQC_GPU_INCLUDE_DIRS}
)

set(PQC_EXPERIMENT_INCLUDE_DIRS
    ${PQC_GPU_PUBLIC_INCLUDE_DIRS}
)
