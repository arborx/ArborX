add_executable(ArborX_DistributedTree_KNN.exe distributed_knn.cpp)
target_link_libraries(ArborX_DistributedTree_KNN.exe ArborX::ArborX)
add_test(NAME ArborX_DistributedTree_KNN_Example COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ./ArborX_DistributedTree_KNN.exe ${MPIEXEC_POSTFLAGS})

add_executable(ArborX_DistributedTree_Intersects.exe distributed_intersects.cpp)
target_link_libraries(ArborX_DistributedTree_Intersects.exe ArborX::ArborX)
add_test(NAME ArborX_DistributedTree_Intersects_Example COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ./ArborX_DistributedTree_Intersects.exe ${MPIEXEC_POSTFLAGS})

add_executable(ArborX_DistributedTree_IntersectsCallback.exe distributed_intersects_callback.cpp)
target_link_libraries(ArborX_DistributedTree_IntersectsCallback.exe ArborX::ArborX)
add_test(NAME ArborX_DistributedTree_IntersectsCallback_Example COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ./ArborX_DistributedTree_IntersectsCallback.exe ${MPIEXEC_POSTFLAGS})
