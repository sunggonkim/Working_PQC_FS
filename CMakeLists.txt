cmake_minimum_required(VERSION 3.18)
project(pqc_fuse_fs
    VERSION 0.3.0
    DESCRIPTION "AEGIS-Q: PQC Secure Storage — CPU/GPU Elastic Lane"
    LANGUAGES C CXX
)

# ── Compiler flags ──────────────────────────────────────────────────────────
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
add_compile_options(
    $<$<COMPILE_LANGUAGE:C>:-Wall>
    $<$<COMPILE_LANGUAGE:C>:-Wextra>
    $<$<COMPILE_LANGUAGE:C>:-Werror>
    $<$<COMPILE_LANGUAGE:C>:-O2>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Werror>
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
)

enable_testing()

# ── Find libfuse3 ───────────────────────────────────────────────────────────
find_package(PkgConfig REQUIRED)
pkg_check_modules(FUSE3 REQUIRED IMPORTED_TARGET fuse3)

# ── Find OpenSSL ────────────────────────────────────────────────────────────
find_package(OpenSSL REQUIRED)

# ── Find liburing ───────────────────────────────────────────────────────────
find_library(URING_LIBRARY NAMES uring PATHS /usr/local/lib /usr/lib)
find_path(URING_INCLUDE_DIR NAMES liburing.h PATHS /usr/local/include /usr/include)

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
    message(WARNING "liboqs not found — CPU PQC operations will not be available")
    set(HAS_OQS FALSE)
endif()

# ── Find cuPQC SDK ──────────────────────────────────────────────────────────
find_path(CUPQC_INCLUDE_DIR NAMES cupqc/cupqc.hpp
    PATHS /usr/local/include /opt/cupqc/include)
find_library(CUPQC_PK_LIB NAMES cupqc-pk
    PATHS /usr/local/lib /opt/cupqc/lib)
find_library(CUPQC_HASH_LIB NAMES cupqc-hash
    PATHS /usr/local/lib /opt/cupqc/lib)

if(CUPQC_INCLUDE_DIR AND CUPQC_PK_LIB)
    set(HAS_CUPQC TRUE)
else()
    message(WARNING "cuPQC SDK not found — GPU PQC elastic lane will not be built")
    set(HAS_CUPQC FALSE)
endif()

add_subdirectory(code)

# ── Summary ─────────────────────────────────────────────────────────────────
message(STATUS "──────────────────────────────────────────────────────────")
message(STATUS " AEGIS-Q Build Configuration v${PROJECT_VERSION}")
message(STATUS "──────────────────────────────────────────────────────────")
message(STATUS " FUSE3      : ${FUSE3_VERSION}")
if(HAS_OQS)
    message(STATUS " liboqs     : ${OQS_LIBRARY}  ✅")
else()
    message(STATUS " liboqs     : NOT FOUND  ❌")
endif()
message(STATUS "──────────────────────────────────────────────────────────")
