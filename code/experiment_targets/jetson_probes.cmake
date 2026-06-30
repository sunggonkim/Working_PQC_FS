if(CMAKE_CUDA_COMPILER)
    add_executable(jetson_memory_contract_probe
        ${EXPERIMENT_CODE_DIR}/jetson_memory_contract_probe.cu)
    set_target_properties(jetson_memory_contract_probe PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        INTERPROCEDURAL_OPTIMIZATION ON
    )

    add_executable(cuda_qos_contract_probe
        ${EXPERIMENT_CODE_DIR}/cuda_qos_contract_probe.cu)
    set_target_properties(cuda_qos_contract_probe PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        INTERPROCEDURAL_OPTIMIZATION ON
    )
endif()
