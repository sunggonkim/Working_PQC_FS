if(CMAKE_CUDA_COMPILER)
    if(HAS_CUPQC)
        add_executable(bench_gpu_pqc ${EXPERIMENT_CODE_DIR}/bench_gpu_pqc.cu)
        set_target_properties(bench_gpu_pqc PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
            INTERPROCEDURAL_OPTIMIZATION ON
        )
        target_include_directories(bench_gpu_pqc PRIVATE ${PQC_EXPERIMENT_INCLUDE_DIRS} ${CUPQC_INCLUDE_DIR} ${CUPQC_INCLUDE_DIR}/cupqc)
        target_link_libraries(bench_gpu_pqc PRIVATE ${CUPQC_PK_LIB})
    endif()

    add_executable(gpu_burner ${EXPERIMENT_CODE_DIR}/gpu_burner.cu)
    set_target_properties(gpu_burner PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        INTERPROCEDURAL_OPTIMIZATION ON
    )

    add_executable(bench_gpu_dataplane_negative_control
        ${EXPERIMENT_CODE_DIR}/bench_gpu_dataplane_negative_control.cu)
    set_target_properties(bench_gpu_dataplane_negative_control PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        INTERPROCEDURAL_OPTIMIZATION ON
    )
    target_include_directories(bench_gpu_dataplane_negative_control PRIVATE ${PQC_EXPERIMENT_INCLUDE_DIRS})
    target_link_libraries(bench_gpu_dataplane_negative_control PRIVATE skim_cuda_aead OpenSSL::Crypto)

    if(CUPQC_HASH_LIB)
        add_executable(bench_gpu_integrity
            ${EXPERIMENT_CODE_DIR}/bench_gpu_integrity.cu
            ${CODE_DIR}/gpu/cuda_integrity.cu)
        set_target_properties(bench_gpu_integrity PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
            INTERPROCEDURAL_OPTIMIZATION OFF
        )
        target_include_directories(bench_gpu_integrity PRIVATE ${PQC_EXPERIMENT_INCLUDE_DIRS} ${CUPQC_INCLUDE_DIR} ${CUPQC_INCLUDE_DIR}/cupqc)
        target_link_libraries(bench_gpu_integrity PRIVATE ${CUPQC_HASH_LIB} ${CUPQC_PK_LIB} OpenSSL::Crypto pthread)
    endif()
endif()
