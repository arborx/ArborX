add_executable(ArborX_DistributedTree_KNN.exe distributed_knn.cpp)
target_link_libraries(ArborX_DistributedTree_KNN.exe ArborX::ArborX)
add_test(NAME ArborX_DistributedTree_KNN_Example COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ./ArborX_DistributedTree_KNN.exe ${MPIEXEC_POSTFLAGS})
