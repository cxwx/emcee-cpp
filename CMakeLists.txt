cmake_minimum_required(VERSION 3.14)
project(emcee-cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Header-only library
add_library(emcee INTERFACE)
target_include_directories(emcee INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Examples
option(EMCEE_BUILD_EXAMPLES "Build example programs" ON)

if(EMCEE_BUILD_EXAMPLES)
    add_executable(gaussian examples/gaussian.cpp)
    target_link_libraries(gaussian PRIVATE emcee Threads::Threads)
    find_package(Threads REQUIRED)
endif()
