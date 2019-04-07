cmake_minimum_required(VERSION 3.13)
project(ArborX CXX)

find_package(kokkos REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options unit_test_framework)
find_package(MPI REQUIRED)

add_library(ArborX INTERFACE)
target_link_libraries(ArborX INTERFACE Kokkos::kokkos MPI::MPI_CXX)
#target_compile_features(ArborX INTERFACE cxx_std_14)
target_link_directories(ArborX INTERFACE /usr/local/cuda/lib64)
#set_target_properties(ArborX PROPERTIES CXX_STANDARD 14 CXX_EXTENSIONS OFF CXX_STANDARD_REQUIRED ON)

enable_testing()

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(examples)
