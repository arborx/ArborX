##---------------------------------------------------------------------------##
## EXAMPLES
##---------------------------------------------------------------------------##

TRIBITS_ADD_EXECUTABLE(
  distributed_tree
  SOURCES distributed_tree_driver.cpp
  )
IF(Kokkos_ENABLE_Serial)
  TRIBITS_ADD_TEST(
    distributed_tree
    POSTFIX_AND_ARGS_0 serial --node=serial
    COMM serial mpi
    NUM_MPI_PROCS 2
    FAIL_REGULAR_EXPRESSION "data race;leak;runtime error"
    )
ENDIF()
IF(Kokkos_ENABLE_Cuda)
  TRIBITS_ADD_TEST(
    distributed_tree
    POSTFIX_AND_ARGS_0 cuda --node=cuda
    COMM serial mpi
    NUM_MPI_PROCS 2
    FAIL_REGULAR_EXPRESSION "data race;leak;runtime error"
    )
ENDIF()
IF(Kokkos_ENABLE_OpenMP)
  TRIBITS_ADD_TEST(
    distributed_tree
    POSTFIX_AND_ARGS_0 openmp --node=openmp
    COMM serial mpi
    NUM_MPI_PROCS 2
    NUM_TOTAL_CORES_USED 4
    ENVIRONMENT OMP_NUM_THREADS=2
    FAIL_REGULAR_EXPRESSION "data race;leak;runtime error"
    )
ENDIF()
