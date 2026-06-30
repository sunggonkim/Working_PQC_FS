if(Python3_Interpreter_FOUND)
    add_test(
        NAME lock_contract_regression
        COMMAND ${Python3_EXECUTABLE}
            "${EXPERIMENT_CODE_DIR}/check_lock_contract_regression.py"
            --repo-root "${CMAKE_SOURCE_DIR}"
    )
    set_tests_properties(lock_contract_regression PROPERTIES
        PASS_REGULAR_EXPRESSION "\"overall_pass\": true"
        FAIL_REGULAR_EXPRESSION "\"overall_pass\": false|Traceback|ERROR|FAIL"
        TIMEOUT 30
    )
endif()
