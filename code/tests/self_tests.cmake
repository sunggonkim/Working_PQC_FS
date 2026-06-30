if(HAS_OQS)
    add_test(NAME pqc_fuse_self_test COMMAND pqc_fuse --self-test)
    set_tests_properties(pqc_fuse_self_test PROPERTIES
        PASS_REGULAR_EXPRESSION "generation replay self-test: PASS"
        FAIL_REGULAR_EXPRESSION "FAIL"
        TIMEOUT 30
    )
endif()
