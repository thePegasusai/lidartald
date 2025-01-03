# CMake version requirement for modern CUDA and Vulkan support
cmake_minimum_required(VERSION 3.26)

# Project declaration with version and language support
project(tald_unia_backend 
    VERSION 1.0
    LANGUAGES CXX
    DESCRIPTION "TALD UNIA Backend Components"
)

# Set C++ standard to C++20 for advanced language features
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Force Release build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set optimization flags for maximum performance
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -march=native -flto -mavx2 -mfma")

# Find Corrosion for Rust integration
find_package(Corrosion REQUIRED)

# Add core C++ components
add_subdirectory(lidar_core)
add_subdirectory(game_engine)

# Import Rust fleet manager using Corrosion
corrosion_import_crate(
    MANIFEST_PATH fleet_manager/Cargo.toml
    PROFILE release
    FEATURES "production"
    LOCKED
)

# Define project-wide compile definitions
add_compile_definitions(
    TALD_UNIA_VERSION="${PROJECT_VERSION}"
    ENABLE_CUDA
    ENABLE_VULKAN
    ENABLE_MULTITHREADING
)

# Set include directories
set(TALD_UNIA_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${LIDAR_CORE_INCLUDE_DIRS}
)

# Export project configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/tald_unia_backend-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tald_unia_backend-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/tald_unia_backend-config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tald_unia_backend
)

# Installation rules
include(GNUInstallDirs)

install(TARGETS lidar_core game_engine fleet_manager
    EXPORT tald_unia_backend-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT tald_unia_backend-targets
    FILE tald_unia_backend-targets.cmake
    NAMESPACE tald_unia::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tald_unia_backend
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/tald_unia_backend-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/tald_unia_backend-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tald_unia_backend
)

# Set up testing if enabled
option(BUILD_TESTING "Build tests" ON)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# Export variables for parent scope
set(TALD_UNIA_VERSION ${PROJECT_VERSION} CACHE INTERNAL "")
set(TALD_UNIA_INCLUDE_DIRS ${TALD_UNIA_INCLUDE_DIRS} CACHE INTERNAL "")