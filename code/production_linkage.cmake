# Production include and external dependency policy for AEGIS-Q runtime targets.
if(HAS_OQS)
    target_include_directories(pqc_fuse_core PRIVATE ${PQC_PRODUCTION_INCLUDE_DIRS})
    target_link_libraries(pqc_fuse_core
        PUBLIC
            PkgConfig::FUSE3
            OQS::oqs
            OpenSSL::Crypto
            pthread
            m
            dl
    )

    target_include_directories(pqc_fuse PRIVATE ${PQC_PRODUCTION_INCLUDE_DIRS})
    target_link_libraries(pqc_fuse PRIVATE pqc_fuse_core)
endif()
