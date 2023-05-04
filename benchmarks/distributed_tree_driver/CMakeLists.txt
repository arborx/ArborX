add_executable(ArborX_Benchmark_DistributedTree.exe distributed_tree_driver.cpp)
target_link_libraries(ArborX_Benchmark_DistributedTree.exe ArborX::ArborX Boost::program_options)
target_include_directories(ArborX_Benchmark_DistributedTree.exe PRIVATE ${CMAKE_SOURCE_DIR}/benchmarks/point_clouds)
add_test(NAME ArborX_Benchmark_DistributedTree COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:ArborX_Benchmark_DistributedTree.exe> ${MPIEXEC_POSTFLAGS})
