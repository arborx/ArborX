cmake_minimum_required(VERSION 3.22)
project(ArborX CXX)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.30.0)
  message(STATUS "Setting policy CMP0167 to use FindBoost module")
  cmake_policy(SET CMP0167 NEW)
endif()

# use gnu standard install directories
include(GNUInstallDirs)

find_package(Kokkos 4.5 REQUIRED CONFIG)
message(STATUS "Found Kokkos: ${Kokkos_DIR} (version \"${Kokkos_VERSION}\")")

# We use minimum compiler versions from Kokkos, with the exception for NVCC (>= 11.5)
if(Kokkos_CXX_COMPILER_ID STREQUAL "NVIDIA" AND Kokkos_CXX_COMPILER_VERSION VERSION_LESS 11.5.0)
  message(FATAL_ERROR "NVCC versions prior to 11.5 are not supported by ArborX")
endif()

add_library(ArborX INTERFACE)
target_link_libraries(ArborX INTERFACE Kokkos::kokkos)
set_target_properties(ArborX PROPERTIES INTERFACE_COMPILE_FEATURES cxx_std_20)
# As all executables using ArborX depend on it, depending on record_hash allows
# updating hash each time executable is rebuilt, including when called from
# within a subdirectory.
add_dependencies(ArborX record_hash)

if(Kokkos_ENABLE_HIP AND NOT Kokkos_ENABLE_ROCTHRUST)
  message(WARNING "Kokkos found does not have rocThrust enabled.\nThis will negatively impact performance on AMD GPUs.")
endif()

# Refer to the alias target in examples and benchmarks so they can be built
# against an installed ArborX
add_library(ArborX::ArborX ALIAS ArborX)

option(ARBORX_ENABLE_MPI "Enable MPI support" OFF)
if(ARBORX_ENABLE_MPI)
  find_package(MPI REQUIRED)
  target_link_libraries(ArborX INTERFACE MPI::MPI_CXX)
endif()

include(CMakeDependentOption)

cmake_dependent_option(ARBORX_ENABLE_CUDA_AWARE_MPI
                       "Allow using device data in MPI communication"
                       OFF "ARBORX_ENABLE_MPI" OFF)
if(ARBORX_USE_CUDA_AWARE_MPI)
  set(ARBORX_ENABLE_GPU_AWARE_MPI ON)
    message(DEPRECATION "ARBORX_USE_CUDA_AWARE_MPI is deprecated, use ARBORX_ENABLE_GPU_AWARE_MPI instead")
endif()

cmake_dependent_option(ARBORX_ENABLE_GPU_AWARE_MPI
                       "Allow using device data in MPI communication"
                       OFF "ARBORX_ENABLE_MPI" OFF)

target_include_directories(ArborX INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/geometry>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/cluster>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/distributed>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/interpolation>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/spatial>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ArborX>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ArborX/geometry>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ArborX/cluster>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ArborX/distributed>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ArborX/interpolation>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ArborX/spatial>
)

install(TARGETS ArborX
  EXPORT ArborXTargets
  ARCHIVE LIBRARY PUBLIC_HEADER
)

install(EXPORT ArborXTargets
  NAMESPACE ArborX::
  DESTINATION ${CMAKE_INSTALL_DATADIR}/cmake/ArborX
)

set(ARBORX_VERSION_MAJOR 2)
set(ARBORX_VERSION_MINOR 0)
set(ARBORX_VERSION_PATCH 99)
math(EXPR ARBORX_VERSION "${ARBORX_VERSION_MAJOR} * 10000 + ${ARBORX_VERSION_MINOR} * 100 + ${ARBORX_VERSION_PATCH}")
set(ARBORX_VERSION_STRING "${ARBORX_VERSION_MAJOR}.${ARBORX_VERSION_MINOR}.${ARBORX_VERSION_PATCH}")
message(STATUS "ArborX version: ${ARBORX_VERSION_STRING}")

# Make sure that the git hash in ArborX_Version.hpp is considered to be always
# out of date, and thus is updated every recompile.
add_custom_target(
  record_hash ALL VERBATIM
  COMMAND ${CMAKE_COMMAND}
    -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
    -DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
    -DARBORX_VERSION_STRING=${ARBORX_VERSION_STRING}
    -P cmake/SetupVersion.cmake
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
# Also run the record_hash command during configuration stage to have a visible
# ArborX_Version.hpp at all times.
execute_process(
  COMMAND ${CMAKE_COMMAND}
    -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
    -DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
    -DARBORX_VERSION_STRING=${ARBORX_VERSION_STRING}
    -P cmake/SetupVersion.cmake
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/ArborX_Config.hpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/include/ArborX_Config.hpp)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ArborXSettings.cmake.in
               ${CMAKE_CURRENT_BINARY_DIR}/ArborXSettings.cmake
               @ONLY)

include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/ArborXConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/ArborXConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_DATADIR}/cmake/ArborX
)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/ArborXConfigVersion.cmake
  VERSION ${ARBORX_VERSION_STRING}
  COMPATIBILITY SameMajorVersion
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/ArborXConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/ArborXConfigVersion.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/ArborXSettings.cmake
  DESTINATION ${CMAKE_INSTALL_DATADIR}/cmake/ArborX )

install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ArborX
        FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY ${PROJECT_BINARY_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ArborX
        FILES_MATCHING PATTERN "*.hpp")

if(NOT CMAKE_BUILD_TYPE)
  set(default_build_type "RelWithDebInfo")
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING
    "Choose the type of build, options are: Debug, Release, RelWithDebInfo and MinSizeRel."
    FORCE)
endif()

option(ARBORX_ENABLE_TESTS "Enable tests" OFF)
option(ARBORX_ENABLE_EXAMPLES "Enable examples" OFF)
option(ARBORX_ENABLE_BENCHMARKS "Enable benchmarks" OFF)

if(ARBORX_ENABLE_TESTS OR ARBORX_ENABLE_EXAMPLES OR ARBORX_ENABLE_BENCHMARKS)
  enable_testing()
endif()

cmake_dependent_option(ARBORX_ENABLE_HEADER_SELF_CONTAINMENT_TESTS "Enable header self-containment unit tests" ON "ARBORX_ENABLE_TESTS" OFF)
if(ARBORX_ENABLE_HEADER_SELF_CONTAINMENT_TESTS)
  # Globbing all the header filenames to test for self-containment and presence of header guards
  file(GLOB_RECURSE ArborX_HEADERS RELATIVE ${CMAKE_SOURCE_DIR}/src src/*.hpp)
  # Findout what headers are using macros defined in ArborX_Config.hpp
  file(STRINGS src/ArborX_Config.hpp.in ArborX_DEFINITIONS REGEX "define ARBORX_")
  foreach(_definition ${ArborX_DEFINITIONS})
    string(REGEX REPLACE "(#define |#cmakedefine )" "" _macro ${_definition})
    list(APPEND ArborX_MACROS ${_macro})
  endforeach()
  foreach(_file ${ArborX_HEADERS})
    foreach(_macro ${ArborX_MACROS})
      file(STRINGS src/${_file} _includes_mpi REGEX "mpi.h")
      if(_includes_mpi)
        list(APPEND ArborX_HEADERS_MUST_ENABLE_MPI ${_file})
      endif()
      file(STRINGS src/${_file} _has_macro REGEX "${_macro}")
      if(_has_macro)
        list(APPEND ArborX_HEADERS_MUST_INCLUDE_CONFIG_HPP ${_file})
        continue()
      endif()
    endforeach()
  endforeach()
endif()
if(ARBORX_ENABLE_TESTS)
  add_subdirectory(test)
endif()
if(ARBORX_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()
if(ARBORX_ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
