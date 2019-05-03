cmake_minimum_required(VERSION 3.13)
project(ArborX CXX)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
find_package(Kokkos REQUIRED)
find_package(MPI REQUIRED)

add_library(ArborX INTERFACE)
target_link_libraries(ArborX INTERFACE Kokkos::Kokkos MPI::MPI_CXX)
#target_compile_features(ArborX INTERFACE cxx_std_14)

# FIXME
if(Kokkos_DEVICES MATCHES "Cuda")
  find_package(CUDA REQUIRED)
  get_filename_component(CUDA_LIBRARY_DIR ${CUDA_cudadevrt_LIBRARY} DIRECTORY)
  target_link_directories(ArborX INTERFACE ${CUDA_LIBRARY_DIR})
endif()

option(ArborX_ENABLE_TESTS "Enable tests" ON)
option(ArborX_ENABLE_EXAMPLES "Enable examples" ON)

if(${ArborX_ENABLE_TESTS} OR ${ArborX_ENABLE_EXAMPLES})
  enable_testing()
endif()

add_subdirectory(src)
if(${ArborX_ENABLE_TESTS})
  add_subdirectory(test)
endif()
if(${ArborX_ENABLE_EXAMPLES})
  add_subdirectory(examples)
endif()
