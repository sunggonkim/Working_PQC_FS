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

# ── Target 1: CPU Baseline (pqc_fuse) ──────────────────────────────────────
if(HAS_OQS)
    set(PQC_FUSE_SOURCES pqc_fuse.c pqc_anchor.c pqc_admission.c)
    add_executable(pqc_fuse ${PQC_FUSE_SOURCES})
    target_link_libraries(pqc_fuse PRIVATE PkgConfig::FUSE3 OQS::oqs OpenSSL::Crypto pthread m dl)
endif()

# ── Target 2: CUDA GPU backends ────────────────────────────────────────────
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    # Thor = Blackwell sm_120; Jetson Orin = sm_87; Ampere = sm_80
    set(CMAKE_CUDA_ARCHITECTURES "110" CACHE STRING "CUDA architectures")
    set(CMAKE_CUDA_STANDARD 17)

    # ── 2a: AES-GCM backend (Data Plane — CPU fast lane confirmed superior) ──
    add_library(skim_cuda_aead STATIC cuda_aead.cu)
    set_target_properties(skim_cuda_aead PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        INTERPROCEDURAL_OPTIMIZATION ON
    )

    if(HAS_OQS)
        target_link_libraries(pqc_fuse PRIVATE skim_cuda_aead)
        target_compile_definitions(pqc_fuse PRIVATE SKIM_HAVE_CUDA_AEAD=1)
        set_target_properties(pqc_fuse PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    endif()

    # ── 2b: ML-KEM-768 GPU Elastic Lane (cuPQC) ────────────────────────────
    if(HAS_CUPQC)
        add_library(skim_cuda_pqc SHARED cuda_pqc.cu)
        set_target_properties(skim_cuda_pqc PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
            INTERPROCEDURAL_OPTIMIZATION ON
        )
        target_include_directories(skim_cuda_pqc PRIVATE ${CUPQC_INCLUDE_DIR} ${CUPQC_INCLUDE_DIR}/cupqc)
        target_link_libraries(skim_cuda_pqc PRIVATE ${CUPQC_PK_LIB})

        # bench_gpu_pqc — standalone benchmark binary (fills workload_map.csv GPU rows)
        add_executable(bench_gpu_pqc experiments/bench_gpu_pqc.cu)
        set_target_properties(bench_gpu_pqc PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
            INTERPROCEDURAL_OPTIMIZATION ON
        )
        target_include_directories(bench_gpu_pqc PRIVATE ${CUPQC_INCLUDE_DIR} ${CUPQC_INCLUDE_DIR}/cupqc)
        target_link_libraries(bench_gpu_pqc PRIVATE ${CUPQC_PK_LIB})

        # Link GPU PQC into pqc_fuse
        if(HAS_OQS)
            target_link_libraries(pqc_fuse PRIVATE skim_cuda_pqc)
            target_compile_definitions(pqc_fuse PRIVATE SKIM_HAVE_CUDA_PQC=1)
        endif()

        message(STATUS "  ✅ GPU ML-KEM-768 elastic lane (cuPQC) will be built")
    else()
        message(STATUS "  ⚠️  cuPQC not found — GPU PQC elastic lane skipped")
    endif()

    # ── 2c: Integrity Plane (BLAKE3/SHA-256 batch) ─────────────────────────
    if(CUPQC_HASH_LIB)
        # Compile cuda_integrity.cu directly into pqc_fuse if HAS_OQS
        if(HAS_OQS)
            target_sources(pqc_fuse PRIVATE cuda_integrity.cu)
            target_compile_definitions(pqc_fuse PRIVATE SKIM_HAVE_CUDA_INTEGRITY=1)
            set_target_properties(pqc_fuse PROPERTIES
                CUDA_SEPARABLE_COMPILATION OFF
                CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
                INTERPROCEDURAL_OPTIMIZATION OFF
            )
        endif()

        # bench_gpu_integrity — standalone benchmark binary
        add_executable(bench_gpu_integrity experiments/bench_gpu_integrity.cu cuda_integrity.cu)
        set_target_properties(bench_gpu_integrity PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
            INTERPROCEDURAL_OPTIMIZATION OFF
        )
        target_include_directories(bench_gpu_integrity PRIVATE ${CUPQC_INCLUDE_DIR} ${CUPQC_INCLUDE_DIR}/cupqc ${CMAKE_SOURCE_DIR})
        target_link_libraries(bench_gpu_integrity PRIVATE ${CUPQC_HASH_LIB} ${CUPQC_PK_LIB} OpenSSL::Crypto pthread)

        message(STATUS "  ✅ GPU SHA-256 integrity-plane (cuPQC-HASH) will be built")
        
        enable_testing()
        
        # M1 Gate: CTest parity and crash-state tests (no skipped GPU correctness test)
        add_test(NAME integrity_correctness COMMAND bench_gpu_integrity --only-tests)
        set_tests_properties(integrity_correctness PROPERTIES
            PASS_REGULAR_EXPRESSION "SUCCESS"
            FAIL_REGULAR_EXPRESSION "ERROR|FAIL"
            TIMEOUT 30
        )
        
        add_test(NAME integrity_quick_bench COMMAND bench_gpu_integrity --quick)
        set_tests_properties(integrity_quick_bench PROPERTIES
            PASS_REGULAR_EXPRESSION "All-Core"
            FAIL_REGULAR_EXPRESSION "error|failed"
            TIMEOUT 60
        )
    else()
        message(STATUS "  ⚠️  cuPQC-HASH not found — integrity GPU target skipped")
    endif()

else()
    message(WARNING "CUDA compiler not found — GPU backends skipped")
endif()

# ── Microbenchmark / utility targets (existing) ─────────────────────────────
if(CMAKE_CUDA_COMPILER AND HAS_OQS)
    if(URING_LIBRARY AND URING_INCLUDE_DIR)
        # repro_malloc_register — conservative cudaHostRegister smoke
        add_executable(repro_malloc_register experiments/repro_malloc_register.cu)
        set_target_properties(repro_malloc_register PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
            INTERPROCEDURAL_OPTIMIZATION ON
        )
        target_link_libraries(repro_malloc_register PRIVATE OpenSSL::Crypto pthread)

        # io_uring_uvm — conservative O_DIRECT + pinned-buffer smoke
        add_executable(io_uring_uvm experiments/io_uring_uvm.cu)
        set_target_properties(io_uring_uvm PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
            INTERPROCEDURAL_OPTIMIZATION ON
        )
        target_include_directories(io_uring_uvm PRIVATE ${URING_INCLUDE_DIR})
        target_link_libraries(io_uring_uvm PRIVATE ${URING_LIBRARY} OpenSSL::Crypto pthread)
    else()
        message(WARNING "liburing not found — UMA smoke executables will not be built")
    endif()

    # workload_map_bench: fills CPU rows in artifacts/workload_map.csv
    if(EXISTS "${CMAKE_SOURCE_DIR}/experiments/workload_map_bench.cpp")
        add_executable(workload_map_bench experiments/workload_map_bench.cpp)
        target_link_libraries(workload_map_bench PRIVATE OQS::oqs OpenSSL::Crypto skim_cuda_aead pthread)
    endif()
endif()

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
if(CMAKE_CUDA_COMPILER)
    message(STATUS " CUDA       : ${CMAKE_CUDA_COMPILER_VERSION}  arch=${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS " Data Plane : AES-GCM CUDA (skim_cuda_aead) — CPU superior, kept for UMA staging")
    if(HAS_CUPQC)
        message(STATUS " Key Plane  : ML-KEM-768 cuPQC (skim_cuda_pqc + bench_gpu_pqc)  ✅")
    else()
        message(STATUS " Key Plane  : cuPQC NOT FOUND — run experiments/bench_gpu_pqc.cu manually  ❌")
    endif()
    if(CUPQC_HASH_LIB)
        message(STATUS " Integrity  : SHA-256 cuPQC-HASH (skim_cuda_integrity + bench_gpu_integrity)  ✅")
    else()
        message(STATUS " Integrity  : cuPQC-HASH NOT FOUND  ❌")
    endif()
else()
    message(STATUS " CUDA       : NOT FOUND  ❌")
endif()
message(STATUS "──────────────────────────────────────────────────────────")
