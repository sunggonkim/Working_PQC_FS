cmake_minimum_required(VERSION 3.18)
project(pqc_fuse_fs
    VERSION 0.2.0
    DESCRIPTION "PQC-based FUSE Filesystem — CPU Baseline & GPU Accelerated"
    LANGUAGES C CXX
)

# ── Compiler flags ──────────────────────────────────────────────────────────
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
add_compile_options(-Wall -Wextra -O2)

# ── Find libfuse3 ───────────────────────────────────────────────────────────
find_package(PkgConfig REQUIRED)
pkg_check_modules(FUSE3 REQUIRED IMPORTED_TARGET fuse3)

# ── Find OpenSSL (for SHAKE128 XOF stream cipher) ───────────────────────────
find_package(OpenSSL REQUIRED)

# ── Find liboqs ─────────────────────────────────────────────────────────────
find_library(OQS_LIBRARY NAMES oqs PATHS /usr/local/lib /usr/lib)
find_path(OQS_INCLUDE_DIR NAMES oqs/oqs.h PATHS /usr/local/include /usr/include)

if(OQS_LIBRARY AND OQS_INCLUDE_DIR)
    add_library(OQS::oqs SHARED IMPORTED)
    set_target_properties(OQS::oqs PROPERTIES
        IMPORTED_LOCATION "${OQS_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${OQS_INCLUDE_DIR}"
    )
    set(HAS_OQS TRUE)
else()
    message(WARNING "liboqs not found — CPU baseline (pqc_fuse) will NOT be built")
    set(HAS_OQS FALSE)
endif()

# ── Target 1: CPU Baseline (pqc_fuse) ──────────────────────────────────────
if(HAS_OQS)
    add_executable(pqc_fuse pqc_fuse.c)
    target_link_libraries(pqc_fuse PRIVATE PkgConfig::FUSE3 OQS::oqs OpenSSL::Crypto pthread m)
endif()

# ── Target 2: GPU Accelerated (pqc_fuse_gpu) ───────────────────────────────
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    # Auto-detect GPU architecture: include Thor (sm_110, Blackwell) + Jetson legacy
    set(CMAKE_CUDA_ARCHITECTURES "72;87;110" CACHE STRING "CUDA architectures")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall")

    add_executable(pqc_fuse_gpu pqc_fuse.cu)

    # FUSE headers and libs
    target_include_directories(pqc_fuse_gpu PRIVATE ${FUSE3_INCLUDE_DIRS} ${OQS_INCLUDE_DIR})
    target_link_libraries(pqc_fuse_gpu PRIVATE ${FUSE3_LIBRARIES} ${OQS_LIBRARY} OpenSSL::Crypto pthread m)
    target_compile_definitions(pqc_fuse_gpu PRIVATE ${FUSE3_CFLAGS_OTHER})

    # Set CUDA properties
    set_target_properties(pqc_fuse_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    message(STATUS "  ✅ GPU target (pqc_fuse_gpu) will be built")
    message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(WARNING "CUDA not found — GPU target (pqc_fuse_gpu) will NOT be built")
endif()

# ── Summary ─────────────────────────────────────────────────────────────────
message(STATUS "──────────────────────────────────────────")
message(STATUS " PQC FUSE Build Configuration v${PROJECT_VERSION}")
message(STATUS "──────────────────────────────────────────")
message(STATUS " FUSE3     : ${FUSE3_VERSION}")
if(HAS_OQS)
    message(STATUS " liboqs    : ${OQS_LIBRARY}")
    message(STATUS " CPU target: pqc_fuse ✅")
else()
    message(STATUS " CPU target: SKIPPED (no liboqs)")
endif()
if(CMAKE_CUDA_COMPILER)
    message(STATUS " CUDA      : ${CMAKE_CUDA_COMPILER_VERSION}")
    message(STATUS " GPU target: pqc_fuse_gpu ✅")
endif()
message(STATUS "──────────────────────────────────────────")
