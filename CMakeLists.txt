cmake_minimum_required(VERSION 3.12)
project(ArborX CXX)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
find_package(Kokkos REQUIRED)

add_library(ArborX INTERFACE)
target_link_libraries(ArborX INTERFACE Kokkos::Kokkos)

option(ARBORX_ENABLE_MPI "Enable MPI support" OFF)
if(ARBORX_ENABLE_MPI)
  find_package(MPI REQUIRED)
  target_link_libraries(ArborX INTERFACE MPI::MPI_CXX)
endif()
#target_compile_features(ArborX INTERFACE cxx_std_14)

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

# Makes sure to store the current git hash ArborX_Version.hpp each time we
# recompile.
add_custom_target(
  record_hash ALL VERBATIM
  COMMAND ${CMAKE_COMMAND}
    -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
    -DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
    -DARBORX_VERSION_STRING=${ARBORX_VERSION_STRING}
    -P cmake/SetupVersion.cmake
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
# Also run the command when configuring
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
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindKokkos.cmake
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

option(ARBORX_ENABLE_TESTS "Enable tests" ON)
option(ARBORX_ENABLE_EXAMPLES "Enable examples" ON)

if(${ARBORX_ENABLE_TESTS} OR ${ARBORX_ENABLE_EXAMPLES})
  enable_testing()
endif()

if(${ARBORX_ENABLE_TESTS})
  add_subdirectory(test)
  # Globbing all the header filenames to test for self-containment and presence of header guards
  file(GLOB_RECURSE ArborX_HEADERS RELATIVE ${CMAKE_SOURCE_DIR}/src src/*.hpp)
endif()
if(${ARBORX_ENABLE_EXAMPLES})
  add_subdirectory(examples)
endif()
