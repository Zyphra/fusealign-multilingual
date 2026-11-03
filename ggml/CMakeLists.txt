cmake_minimum_required(VERSION 3.18)
project(fusealign-cpp C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if (GGML_CUDA AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 90 CACHE STRING "" FORCE)  # H100
endif()

# ggml: upstream submodule ggml/ggml, pinned to v0.27.0 (a05c9922).

set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(ggml)

# Network on a ggml backend.
add_library(fusealign_model STATIC src/model.cpp)
target_include_directories(fusealign_model PUBLIC src)
target_link_libraries(fusealign_model PUBLIC ggml)

# Host-side pieces: VAD, long-form routing, text, resampling, WAV.
add_library(fusealign_host STATIC src/vad.cpp src/longform.cpp src/text.cpp src/resample.cpp src/wav.cpp)
target_include_directories(fusealign_host PUBLIC src)
target_link_libraries(fusealign_host PUBLIC ggml)

# Public API.
add_library(fusealign STATIC src/fusealign.cpp)
target_include_directories(fusealign PUBLIC src)
target_link_libraries(fusealign PUBLIC fusealign_model fusealign_host)

add_executable(fusealign-cli src/cli.cpp)
target_include_directories(fusealign-cli PRIVATE vendor)
target_link_libraries(fusealign-cli PRIVATE fusealign)

enable_testing()
add_subdirectory(tests/model)
add_subdirectory(tests/host)
