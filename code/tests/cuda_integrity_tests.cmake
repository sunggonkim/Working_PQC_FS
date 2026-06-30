if(CMAKE_CUDA_COMPILER AND CUPQC_HASH_LIB)
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
endif()
