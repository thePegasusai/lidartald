# CMake version requirement
# Version 3.26 required for modern CUDA support and advanced configuration options
cmake_minimum_required(VERSION 3.26)

# Project declaration with version and language support
project(lidar_core VERSION 1.0 LANGUAGES CXX CUDA)

# Set C++ standard to C++20 for advanced language features
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard to C++17 (maximum supported by CUDA 12.0)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Build type configuration
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find required packages
find_package(PCL 1.12.0 REQUIRED COMPONENTS 
    common 
    io 
    filters 
    features 
    segmentation 
    gpu
)

find_package(CUDA 12.0 REQUIRED)
find_package(Eigen3 3.4.0 REQUIRED)

# Verify CUDA compute capability
include(CheckLanguage)
check_language(CUDA)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75) # Targeting Turing architecture (RTX 20xx, Quadro RTX)
endif()

# Set optimization flags for Release build
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -march=native -fopenmp")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -O3")

# Define source files
set(LIDAR_CORE_SOURCES
    src/point_cloud.cpp
    src/feature_detector.cpp
    src/surface_classifier.cpp
    src/lidar_processor.cpp
)

# Create shared library target
add_library(lidar_core SHARED ${LIDAR_CORE_SOURCES})

# Configure include directories
target_include_directories(lidar_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${PCL_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Set compile definitions
target_compile_definitions(lidar_core
    PUBLIC
        ENABLE_CUDA
        PCL_NO_PRECOMPILE
    PRIVATE
        EIGEN_NO_DEBUG
)

# Link dependencies
target_link_libraries(lidar_core
    PUBLIC
        ${PCL_LIBRARIES}
        CUDA::cudart
        CUDA::cuda_driver
        Eigen3::Eigen
)

# Set library properties
set_target_properties(lidar_core PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Installation rules
include(GNUInstallDirs)

install(TARGETS lidar_core
    EXPORT lidar_core-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lidar_core
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Export targets
install(EXPORT lidar_core-targets
    FILE lidar_core-targets.cmake
    NAMESPACE lidar_core::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lidar_core
)

# Create and install package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "lidar_core-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/lidar_core-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/lidar_core-config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lidar_core
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/lidar_core-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/lidar_core-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lidar_core
)

# Export build tree targets
export(EXPORT lidar_core-targets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/lidar_core-targets.cmake"
    NAMESPACE lidar_core::
)

# Set up testing if enabled
option(BUILD_TESTING "Build tests" ON)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()