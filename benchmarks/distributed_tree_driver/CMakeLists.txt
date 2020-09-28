add_executable(ArborX_DistributedTree_Benchmark.exe distributed_tree_driver.cpp)
target_link_libraries(ArborX_DistributedTree_Benchmark.exe ArborX::ArborX Boost::program_options)
target_include_directories(ArborX_DistributedTree_Benchmark.exe PRIVATE ${CMAKE_SOURCE_DIR}/benchmarks/point_clouds)
add_test(NAME ArborX_DistributedTree_Benchmark COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ./ArborX_DistributedTree_Benchmark.exe ${MPIEXEC_POSTFLAGS})
