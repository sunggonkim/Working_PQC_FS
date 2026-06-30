# Runtime CUDA backend library declarations.
if(CMAKE_CUDA_COMPILER)
    add_library(skim_cuda_aead STATIC ${CODE_DIR}/gpu/cuda_aead.cu)
    target_include_directories(skim_cuda_aead PUBLIC ${PQC_GPU_PUBLIC_INCLUDE_DIRS})
    set_target_properties(skim_cuda_aead PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        INTERPROCEDURAL_OPTIMIZATION ON
    )

    if(HAS_CUPQC)
        add_library(skim_cuda_pqc SHARED ${CODE_DIR}/gpu/cuda_pqc.cu)
        set_target_properties(skim_cuda_pqc PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
            INTERPROCEDURAL_OPTIMIZATION ON
        )
        target_include_directories(skim_cuda_pqc PRIVATE ${PQC_GPU_PUBLIC_INCLUDE_DIRS} ${CUPQC_INCLUDE_DIR} ${CUPQC_INCLUDE_DIR}/cupqc)
        target_link_libraries(skim_cuda_pqc PRIVATE ${CUPQC_PK_LIB})

        message(STATUS "  ✅ GPU ML-KEM-768 elastic lane (cuPQC) will be built")
    else()
        message(STATUS "  ⚠️  cuPQC not found — GPU PQC elastic lane skipped")
    endif()

    if(CUPQC_HASH_LIB)
        message(STATUS "  ✅ GPU SHA-256 integrity-plane (cuPQC-HASH) will be built")
    else()
        message(STATUS "  ⚠️  cuPQC-HASH not found — integrity GPU target skipped")
    endif()
else()
    message(WARNING "CUDA compiler not found — GPU backends skipped")
endif()
