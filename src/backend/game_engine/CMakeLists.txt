# CMake version requirement for modern CUDA and Vulkan support
cmake_minimum_required(VERSION 3.26)

# Project declaration
project(game_engine VERSION 1.0 LANGUAGES CXX CUDA)

# Configure C++ standard and build type
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard (max supported by CUDA 12.0)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86) # Targeting Ampere architecture

# Force Release build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find required packages with specific versions
find_package(Vulkan 1.3.0 REQUIRED)
find_package(glm 0.9.9.8 REQUIRED)
find_package(Bullet 3.24 REQUIRED COMPONENTS BulletDynamics BulletCollision LinearMath)
find_package(CUDA 12.0 REQUIRED)
find_package(spdlog 1.11.0 REQUIRED)
find_package(OpenMP REQUIRED)

# Add lidar_core dependency
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../lidar_core ${CMAKE_CURRENT_BINARY_DIR}/lidar_core)

# Define source files
set(GAME_ENGINE_SOURCES
    src/physics.cpp
    src/renderer.cpp
    src/state_manager.cpp
)

# Create shared library target
add_library(game_engine SHARED ${GAME_ENGINE_SOURCES})

# Configure include directories
target_include_directories(game_engine
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${Vulkan_INCLUDE_DIRS}
        ${GLM_INCLUDE_DIRS}
        ${BULLET_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Set advanced compiler flags for maximum performance
set(OPTIMIZATION_FLAGS
    -O3                  # Maximum optimization
    -DNDEBUG            # Disable assertions
    -march=native       # CPU-specific optimizations
    -ffast-math         # Fast math operations
    -flto               # Link-time optimization
    -mavx2              # AVX2 SIMD instructions
    -mfma              # FMA instructions
    -fopenmp           # OpenMP support
)

# Configure compile definitions
target_compile_definitions(game_engine
    PUBLIC
        ENABLE_VULKAN
        GLM_FORCE_RADIANS
        GLM_FORCE_DEPTH_ZERO_TO_ONE
        ENABLE_CUDA
        USE_SIMD
        ENABLE_MULTITHREADING
    PRIVATE
        VK_USE_PLATFORM_XCB_KHR
)

# Set compile options
target_compile_options(game_engine
    PRIVATE
        ${OPTIMIZATION_FLAGS}
        -Wall
        -Wextra
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# Enable Link Time Optimization
set_target_properties(game_engine PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Link dependencies
target_link_libraries(game_engine
    PUBLIC
        Vulkan::Vulkan
        ${GLM_LIBRARIES}
        ${BULLET_LIBRARIES}
        CUDA::cudart
        CUDA::cublas
        spdlog::spdlog
        lidar_core
        OpenMP::OpenMP_CXX
)

# Enable precompiled headers
target_precompile_headers(game_engine
    PRIVATE
        <vulkan/vulkan.hpp>
        <glm/glm.hpp>
        <btBulletDynamicsCommon.h>
        <cuda_runtime.h>
        <spdlog/spdlog.h>
)

# Installation rules
include(GNUInstallDirs)

install(TARGETS game_engine
    EXPORT game_engine-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/game_engine
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Export targets
install(EXPORT game_engine-targets
    FILE game_engine-targets.cmake
    NAMESPACE game_engine::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/game_engine
)

# Create and install package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "game_engine-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/game_engine-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/game_engine-config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/game_engine
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/game_engine-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/game_engine-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/game_engine
)

# Export build tree targets
export(EXPORT game_engine-targets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/game_engine-targets.cmake"
    NAMESPACE game_engine::
)