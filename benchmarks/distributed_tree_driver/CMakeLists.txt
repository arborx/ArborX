set(ARBORX_BENCHMARK_UTILS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/benchmarks/utils)

add_executable(ArborX_Benchmark_DistributedTree.exe distributed_tree_driver.cpp)
target_link_libraries(ArborX_Benchmark_DistributedTree.exe ArborX::ArborX Boost::program_options)
if(ARBORX_ENABLE_CALIPER)
  target_link_libraries(ArborX_Benchmark_DistributedTree.exe caliper)
endif()
target_include_directories(ArborX_Benchmark_DistributedTree.exe PRIVATE ${ARBORX_BENCHMARK_UTILS_INCLUDE_DIR})
add_test(NAME ArborX_Benchmark_DistributedTree COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:ArborX_Benchmark_DistributedTree.exe> ${MPIEXEC_POSTFLAGS})
