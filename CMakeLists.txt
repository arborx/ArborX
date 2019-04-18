cmake_minimum_required(VERSION 3.13)
project(ArborX CXX)

find_package(kokkos REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options unit_test_framework)
find_package(MPI REQUIRED)

add_library(ArborX INTERFACE)
target_link_libraries(ArborX INTERFACE Kokkos::kokkos MPI::MPI_CXX)
#target_compile_features(ArborX INTERFACE cxx_std_14)

# FIXME
if(KOKKOS_DEVICES MATCHES "Cuda")
  find_package(CUDA REQUIRED)
  get_filename_component(CUDA_LIBRARY_DIR ${CUDA_cudadevrt_LIBRARY} DIRECTORY)
  target_link_directories(ArborX INTERFACE ${CUDA_LIBRARY_DIR})
endif()

enable_testing()

target_include_directories(ArborX INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/details>
  $<INSTALL_INTERFACE:include>
  $<INSTALL_INTERFACE:include/details>
)

add_subdirectory(test)
add_subdirectory(examples)

install(TARGETS ArborX
  EXPORT ArborXTargets
  ARCHIVE LIBRARY PUBLIC_HEADER
)

install(EXPORT ArborXTargets
  NAMESPACE ArborX::
  DESTINATION lib/cmake/ArborX
)

include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/ArborXConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/ArborXConfig.cmake
  INSTALL_DESTINATION lib/cmake/ArborX
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ArborXConfig.cmake
        DESTINATION lib/cmake/ArborX )

install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/ DESTINATION include)
