cmake_minimum_required(VERSION 3.16)

project(leanllm
    VERSION 0.1.0
    LANGUAGES C CXX
)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# llama.cpp dependency

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG 07822bddf80d73f1168e592c52e69caaff820f9c
)

# llama.cpp dependency

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(llama_cpp)

# LeanLLM library

add_library(leanllm SHARED
    src/leanllm.c
)

if(WIN32)
    set_target_properties(leanllm PROPERTIES
        PREFIX ""
    )
endif()

target_include_directories(leanllm
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(leanllm
    PRIVATE
        LEANLLM_BUILD
)

target_link_libraries(leanllm
    PRIVATE
        llama
)

if(MSVC)
    target_compile_options(leanllm PRIVATE /W4)
else()
    target_compile_options(leanllm PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Examples

add_executable(simple
    examples/simple.c
)

set_target_properties(simple PROPERTIES
    LINKER_LANGUAGE CXX
)

target_link_libraries(simple
    PRIVATE
        leanllm
)