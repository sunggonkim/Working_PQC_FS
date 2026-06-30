# Production-core CUDA feature attachment policy.
if(CMAKE_CUDA_COMPILER AND HAS_OQS)
    target_link_libraries(pqc_fuse_core PUBLIC skim_cuda_aead)
    target_compile_definitions(pqc_fuse_core PRIVATE SKIM_HAVE_CUDA_AEAD=1)
    set_target_properties(pqc_fuse_core PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

    if(HAS_CUPQC)
        target_link_libraries(pqc_fuse_core PUBLIC skim_cuda_pqc)
        target_compile_definitions(pqc_fuse_core PRIVATE SKIM_HAVE_CUDA_PQC=1)
    endif()

    if(CUPQC_HASH_LIB)
        target_sources(pqc_fuse_core PRIVATE ${CODE_DIR}/gpu/cuda_integrity.cu)
        target_compile_definitions(pqc_fuse_core PRIVATE SKIM_HAVE_CUDA_INTEGRITY=1)
        set_target_properties(pqc_fuse_core PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
            INTERPROCEDURAL_OPTIMIZATION OFF
        )
    endif()
endif()
