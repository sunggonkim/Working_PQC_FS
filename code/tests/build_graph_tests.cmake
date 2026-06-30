if(Python3_Interpreter_FOUND)
    add_test(
        NAME build_graph_ownership
        COMMAND ${Python3_EXECUTABLE}
            "${EXPERIMENT_CODE_DIR}/check_build_graph_ownership.py"
            --repo-root "${CMAKE_SOURCE_DIR}"
    )
    set_tests_properties(build_graph_ownership PROPERTIES
        PASS_REGULAR_EXPRESSION "\"overall_pass\": true"
        FAIL_REGULAR_EXPRESSION "\"overall_pass\": false|Traceback|ERROR|FAIL"
        TIMEOUT 30
    )
endif()
