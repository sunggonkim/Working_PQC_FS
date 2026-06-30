if(HAS_OQS AND Python3_Interpreter_FOUND)
    add_test(
        NAME secure_inference_log_macro_strict
        COMMAND ${CMAKE_COMMAND} -E env
            "PQC_FUSE_BIN=$<TARGET_FILE:pqc_fuse>"
            "${Python3_EXECUTABLE}"
            "${EXPERIMENT_CODE_DIR}/run_secure_inference_log_macro_smoke.py"
            --records 2
            --payload-bytes 4096
            --publication-mode strict
    )
    set_tests_properties(secure_inference_log_macro_strict PROPERTIES
        PASS_REGULAR_EXPRESSION "\"overall_pass\": true"
        FAIL_REGULAR_EXPRESSION "\"overall_pass\": false|Traceback|ERROR|FAIL"
        TIMEOUT 30
    )

    add_test(
        NAME secure_inference_log_macro_epoch
        COMMAND ${CMAKE_COMMAND} -E env
            "PQC_FUSE_BIN=$<TARGET_FILE:pqc_fuse>"
            "${Python3_EXECUTABLE}"
            "${EXPERIMENT_CODE_DIR}/run_secure_inference_log_macro_smoke.py"
            --records 2
            --payload-bytes 4096
            --publication-mode epoch-redo-log
    )
    set_tests_properties(secure_inference_log_macro_epoch PROPERTIES
        PASS_REGULAR_EXPRESSION "\"overall_pass\": true"
        FAIL_REGULAR_EXPRESSION "\"overall_pass\": false|Traceback|ERROR|FAIL"
        TIMEOUT 30
    )
endif()
