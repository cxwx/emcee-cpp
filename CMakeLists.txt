cmake_minimum_required(VERSION 3.14)
project(emcee-cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)

find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)

# Header-only library
add_library(emcee INTERFACE)
add_library(emcee-cpp::emcee ALIAS emcee)
target_include_directories(emcee INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(emcee INTERFACE Eigen3::Eigen Threads::Threads)

# Install headers
install(DIRECTORY include/emcee
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    PATTERN ".*" EXCLUDE
)

# Install CMake targets and config
install(TARGETS emcee
    EXPORT emcee-cpp-targets
)

install(EXPORT emcee-cpp-targets
    FILE emcee-cpp-targets.cmake
    NAMESPACE emcee-cpp::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/emcee-cpp
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/emcee-cpp-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/emcee-cpp-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/emcee-cpp
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/emcee-cpp-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/emcee-cpp-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/emcee-cpp-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/emcee-cpp
)

# Examples
option(EMCEE_BUILD_EXAMPLES "Build example programs" ON)

if(EMCEE_BUILD_EXAMPLES)
    add_executable(gaussian examples/gaussian.cpp)
    target_link_libraries(gaussian PRIVATE emcee)
endif()
