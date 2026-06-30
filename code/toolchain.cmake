include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    # Thor = Blackwell sm_120; Jetson Orin = sm_87; Ampere = sm_80
    set(CMAKE_CUDA_ARCHITECTURES "110" CACHE STRING "CUDA architectures")
    set(CMAKE_CUDA_STANDARD 17)
endif()
