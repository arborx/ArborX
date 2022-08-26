add_executable(ArborX_Example_DistributedTree_KNN.exe distributed_knn.cpp)
target_link_libraries(ArborX_Example_DistributedTree_KNN.exe ArborX::ArborX)
add_test(NAME ArborX_Example_DistributedTree_KNN COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ./ArborX_Example_DistributedTree_KNN.exe ${MPIEXEC_POSTFLAGS})
