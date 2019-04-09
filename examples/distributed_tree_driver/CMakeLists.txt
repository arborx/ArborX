add_executable(ArborX_DistributedTree.exe distributed_tree_driver.cpp)
target_link_libraries(ArborX_DistributedTree.exe ArborX Boost::program_options)
target_include_directories(ArborX_DistributedTree.exe PRIVATE ${CMAKE_SOURCE_DIR}/examples/point_clouds)
add_test(NAME ArborX_DistributedTree_Example COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ./ArborX_DistributedTree.exe ${MPIEXEC_POSTFLAGS})
