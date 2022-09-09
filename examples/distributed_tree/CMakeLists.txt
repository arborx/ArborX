add_executable(ArborX_DistributedTree_KNNCallback.exe distributed_knn_callback.cpp)
target_link_libraries(ArborX_DistributedTree_KNNCallback.exe ArborX::ArborX)
add_test(NAME ArborX_DistributedTree_KNNCallback_Example COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ./ArborX_DistributedTree_KNNCallback.exe ${MPIEXEC_POSTFLAGS})
