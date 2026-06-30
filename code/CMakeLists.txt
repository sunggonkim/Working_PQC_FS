# Code-owned build entry point for the production tree and code-local runners.
set(CODE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(EXPERIMENT_CODE_DIR "${CODE_DIR}/experiments")
find_package(Python3 COMPONENTS Interpreter QUIET)

include(${CODE_DIR}/toolchain.cmake)
include(${CODE_DIR}/build_targets.cmake)
include(${CODE_DIR}/production_linkage.cmake)
include(${CODE_DIR}/cuda_backends.cmake)
include(${CODE_DIR}/cuda_feature_wiring.cmake)
include(${CODE_DIR}/experiment_targets.cmake)
include(${CODE_DIR}/tests.cmake)
include(${CODE_DIR}/summary.cmake)
