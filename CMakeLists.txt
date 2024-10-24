cmake_minimum_required(VERSION 3.10)
project(PVFinder CUDA CXX)

# Set CUDA architecture and standards
#set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

# Optionally set compilers (can be overridden by user)
if (NOT CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER "/usr/bin/gcc")
endif()

if (NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "/usr/bin/g++")
endif()

# Unset any Anaconda-related paths
unset(ENV{CONDA_PREFIX})
unset(ENV{CONDA_DEFAULT_ENV})

# CUDA and cuDNN paths (can be set via environment variables or CMake options)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.1" CACHE PATH "Path to CUDA toolkit")
set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/include")
set(CUDNN_INCLUDE_DIR "/usr/include" CACHE PATH "Path to cuDNN include")
set(CUDNN_LIBRARY_DIR "/usr/lib/x86_64-linux-gnu" CACHE PATH "Path to cuDNN libraries")

# Find cnpy with fallback to local installation directory
find_path(CNPY_INCLUDE_DIR cnpy.h
    PATHS
    $ENV{CNPY_INCLUDE_DIR}
    ${CMAKE_INSTALL_PREFIX}/include
    ~/local/include
    /usr/include
    /usr/local/include
)

find_library(CNPY_LIBRARY
    NAMES cnpy
    PATHS
    $ENV{CNPY_LIBRARY_DIR}
    ${CMAKE_INSTALL_PREFIX}/lib
    ~/local/lib
    /usr/lib
    /usr/local/lib
)

if(NOT CNPY_LIBRARY)
    message(FATAL_ERROR "cnpy library not found. Please install it first in ~/local or specify its path using CNPY_INCLUDE_DIR and CNPY_LIBRARY_DIR.")
endif()

# Find ZLIB (can be set via environment variables or installed locally)
find_path(ZLIB_INCLUDE_DIR
    NAMES zlib.h
    PATHS
    $ENV{ZLIB_INCLUDE_DIR}
    ${CMAKE_INSTALL_PREFIX}/include
    ~/local/include
    /usr/include
)

find_library(ZLIB_LIBRARY
    NAMES z
    PATHS
    $ENV{ZLIB_LIBRARY_DIR}
    ${CMAKE_INSTALL_PREFIX}/lib
    ~/local/lib
    /usr/lib
)

if(NOT ZLIB_LIBRARY)
    message(FATAL_ERROR "ZLIB library not found. Please install it or provide its path.")
endif()

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${CUDNN_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/src        # Base src directory for headers
    ${CMAKE_SOURCE_DIR}/src/layers # For layers' headers
    ${CNPY_INCLUDE_DIR}
    ${ZLIB_INCLUDE_DIR}
)

# Library directories
link_directories(
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    ${CUDNN_LIBRARY_DIR}
)

# Source files
file(GLOB_RECURSE SOURCES 
    "src/*.cu"
    "src/layers/*.cu"
)



# Common CUDA properties
set(CUDA_COMPILE_OPTIONS
    --use_fast_math
    -lineinfo
    --expt-relaxed-constexpr
    -arch=sm_80

)

# Set properties for all CUDA files
foreach(CUDA_SOURCE ${SOURCES})
    set_source_files_properties(
        ${CUDA_SOURCE}
        PROPERTIES
        LANGUAGE CUDA
    )
endforeach()

# Add executable
add_executable(pvfinder ${SOURCES})

target_include_directories(pvfinder
    PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${CUDNN_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/include  
    ${CMAKE_SOURCE_DIR}/src      
    ${CMAKE_SOURCE_DIR}/src/layers 
    ${CNPY_INCLUDE_DIR}
    ${ZLIB_INCLUDE_DIR}
)

target_link_libraries(pvfinder
    PRIVATE
    cudart
    cublas
    cudnn
    ${CNPY_LIBRARY}
    ${ZLIB_LIBRARY}
)

target_compile_options(pvfinder
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMPILE_OPTIONS}>
)

set(CMAKE_BUILD_TYPE Debug)

# Print configuration
message(STATUS "Using system GCC: ${CMAKE_C_COMPILER}")
message(STATUS "Using system G++: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA Include: ${CUDA_INCLUDE_DIRS}")
message(STATUS "cuDNN Include: ${CUDNN_INCLUDE_DIR}")
message(STATUS "cnpy Include: ${CNPY_INCLUDE_DIR}")
message(STATUS "cnpy Library: ${CNPY_LIBRARY}")
message(STATUS "ZLIB Include: ${ZLIB_INCLUDE_DIR}")
message(STATUS "ZLIB Library: ${ZLIB_LIBRARY}")
message(STATUS "CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
