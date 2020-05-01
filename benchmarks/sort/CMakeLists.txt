if(NOT PROJECT_NAME)
  cmake_minimum_required(VERSION 3.12)
  project(SortBenchmark CXX)

  find_package(Kokkos 3.0 REQUIRED QUIET)
  if(Kokkos_ENABLE_CUDA)
    kokkos_check(OPTIONS CUDA_LAMBDA)
  endif()

  find_package(Boost REQUIRED COMPONENTS program_options)

  set(BENCHMARK_NAME SortBenchmark)
else()
  set(BENCHMARK_NAME ArborX_SortBenchmark)
endif()

# We require version 1.4.0 or higher but the format used by Google benchmark is
# wrong and thus, we cannot check the version during the configuration step.
find_package(benchmark REQUIRED)

add_executable(${BENCHMARK_NAME}.exe sort_benchmark.cpp pss_common.hpp pss_parallel_stable_sort.hpp sort_benchmark_helpers.hpp)
target_compile_features(${BENCHMARK_NAME}.exe PUBLIC cxx_std_14)
target_link_libraries(${BENCHMARK_NAME}.exe Kokkos::kokkos benchmark::benchmark Boost::program_options)
add_test(NAME ${BENCHMARK_NAME} COMMAND ./${BENCHMARK_NAME}.exe --num-values 10000 --value-type float --benchmark_color=true)
