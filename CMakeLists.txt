cmake_minimum_required(VERSION 3.12)
project(ArborX CXX)

find_package(Kokkos 3.1 REQUIRED)
if(Kokkos_ENABLE_CUDA)
  kokkos_check(OPTIONS CUDA_LAMBDA)
endif()

add_library(ArborX INTERFACE)
target_link_libraries(ArborX INTERFACE Kokkos::kokkos)
set_target_properties(ArborX PROPERTIES INTERFACE_COMPILE_FEATURES cxx_std_14)
# As all executables using ArborX depend on it, depending on record_hash allows
# updating hash each time executable is rebuilt, including when called from
# within a subdirectory.
add_dependencies(ArborX record_hash)

include(CMakeDependentOption)
cmake_dependent_option(ARBORX_ENABLE_ROCTHRUST "Enable rocThrust support" ON "Kokkos_ENABLE_HIP" OFF)
if(Kokkos_ENABLE_HIP AND ARBORX_ENABLE_ROCTHRUST)
  # Require at least rocThrust-2.10.5 (that comes with ROCm 3.9) because
  # rocPRIM dependency is not set properly in exported configuration for
  # earlier versions
  find_package(rocthrust 2.10.5 REQUIRED CONFIG)
  target_link_libraries(ArborX INTERFACE roc::rocthrust)
endif()

if(Kokkos_ENABLE_HIP AND NOT ARBORX_ENABLE_ROCTHRUST)
  message(WARNING "rocThrust is NOT enabled.\nThis will negatively impact performance on AMD GPUs.")
endif()

# Refer to the alias target in examples and benchmarks so they can be built
# against an installed ArborX
add_library(ArborX::ArborX ALIAS ArborX)

option(ARBORX_ENABLE_MPI "Enable MPI support" OFF)
if(ARBORX_ENABLE_MPI)
  find_package(MPI REQUIRED)
  target_link_libraries(ArborX INTERFACE MPI::MPI_CXX)
endif()
cmake_dependent_option(ARBORX_USE_CUDA_AWARE_MPI
                       "Allow using device data in MPI communication"
                       OFF "ARBORX_ENABLE_MPI" OFF)

target_include_directories(ArborX INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/details>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
  $<INSTALL_INTERFACE:include/details>
)

install(TARGETS ArborX
  EXPORT ArborXTargets
  ARCHIVE LIBRARY PUBLIC_HEADER
)

install(EXPORT ArborXTargets
  NAMESPACE ArborX::
  DESTINATION lib/cmake/ArborX
)

set(ARBORX_VERSION_STRING "0.9 (dev)")

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
  INSTALL_DESTINATION lib/cmake/ArborX
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/ArborXConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/ArborXSettings.cmake
  DESTINATION lib/cmake/ArborX )

if(ARBORX_ENABLE_MPI)
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/ DESTINATION include
          FILES_MATCHING PATTERN "*.hpp")
else()
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/ DESTINATION include
          FILES_MATCHING PATTERN "*.hpp"
          PATTERN "*Distribut*" EXCLUDE)
endif()
install(DIRECTORY ${PROJECT_BINARY_DIR}/include/ DESTINATION include
        FILES_MATCHING PATTERN "*.hpp")

if (NOT CMAKE_BUILD_TYPE)
  set(default_build_type "RelWithDebInfo")
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING
    "Choose the type of build, options are: Debug, Release, RelWithDebInfo and MinSizeRel."
    FORCE)
endif()

option(ARBORX_ENABLE_TESTS "Enable tests" OFF)
option(ARBORX_ENABLE_EXAMPLES "Enable examples" OFF)
option(ARBORX_ENABLE_BENCHMARKS "Enable benchmarks" OFF)

if(${ARBORX_ENABLE_TESTS} OR ${ARBORX_ENABLE_EXAMPLES})
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
