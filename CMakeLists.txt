cmake_minimum_required(VERSION 3.13)
project(ArborX CXX)

find_package(kokkos REQUIRED)
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
# Globbing all the header filenames to test for self-containment and presence of header guards
file(GLOB_RECURSE ArborX_HEADERS RELATIVE ${CMAKE_SOURCE_DIR}/src src/*.hpp)

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(examples)
