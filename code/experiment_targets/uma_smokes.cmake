if(CMAKE_CUDA_COMPILER AND HAS_OQS)
    if(URING_LIBRARY AND URING_INCLUDE_DIR)
        add_executable(repro_malloc_register ${EXPERIMENT_CODE_DIR}/repro_malloc_register.cu)
        target_include_directories(repro_malloc_register PRIVATE ${PQC_EXPERIMENT_INCLUDE_DIRS})
        set_target_properties(repro_malloc_register PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
            INTERPROCEDURAL_OPTIMIZATION ON
        )
        target_link_libraries(repro_malloc_register PRIVATE OpenSSL::Crypto pthread)

        add_executable(io_uring_uvm ${EXPERIMENT_CODE_DIR}/io_uring_uvm.cu)
        target_include_directories(io_uring_uvm PRIVATE ${PQC_EXPERIMENT_INCLUDE_DIRS})
        set_target_properties(io_uring_uvm PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
            INTERPROCEDURAL_OPTIMIZATION ON
        )
        target_include_directories(io_uring_uvm PRIVATE ${PQC_EXPERIMENT_INCLUDE_DIRS} ${URING_INCLUDE_DIR})
        target_link_libraries(io_uring_uvm PRIVATE ${URING_LIBRARY} OpenSSL::Crypto pthread)
    else()
        message(WARNING "liburing not found — UMA smoke executables will not be built")
    endif()
endif()
