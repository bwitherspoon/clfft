# At least cmake 3.1 required for CMAKE_CXX_STANDARD and friends
cmake_minimum_required(VERSION 3.1)

if (NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build: Debug Release RelWithDebInfo MinSizeRel")
endif()

project(OCL-FFT CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  list(APPEND CMAKE_CXX_FLAGS "-march=native -Wall")
endif()

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(OpenCL REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/src)

add_subdirectory(src)
