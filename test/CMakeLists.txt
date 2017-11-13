# ##---------------------------------------------------------------------------##
# ## TESTS
# ##---------------------------------------------------------------------------##

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  LinearBVH
  SOURCES tstLinearBVH.cpp Search_UnitTestHelpers.hpp unit_test_main.cpp
  COMM serial mpi
  NUM_MPI_PROCS 1
  STANDARD_PASS_OUTPUT
  FAIL_REGULAR_EXPRESSION "data race;leak;runtime error"
  )
TRIBITS_ADD_EXECUTABLE_AND_TEST(
  DetailsTreeConstruction
  SOURCES tstDetailsTreeConstruction.cpp unit_test_main.cpp
  COMM serial mpi
  NUM_MPI_PROCS 1
  STANDARD_PASS_OUTPUT
  FAIL_REGULAR_EXPRESSION "data race;leak;runtime error"
  )
TRIBITS_ADD_EXECUTABLE_AND_TEST(
  DetailsTreeTraversal
  SOURCES tstDetailsTreeTraversal.cpp unit_test_main.cpp
  COMM serial mpi
  NUM_MPI_PROCS 1
  STANDARD_PASS_OUTPUT
  FAIL_REGULAR_EXPRESSION "data race;leak;runtime error"
  )
TRIBITS_ADD_EXECUTABLE_AND_TEST(
  DetailsAlgorithms
  SOURCES tstDetailsAlgorithms.cpp unit_test_main.cpp
  COMM serial mpi
  NUM_MPI_PROCS 1
  STANDARD_PASS_OUTPUT
  FAIL_REGULAR_EXPRESSION "data race;leak;runtime error"
  )
TRIBITS_ADD_EXECUTABLE_AND_TEST(
  DetailsUtils
  SOURCES tstDetailsUtils.cpp unit_test_main.cpp
  COMM serial mpi
  NUM_MPI_PROCS 1
  STANDARD_PASS_OUTPUT
  FAIL_REGULAR_EXPRESSION "data race;leak;runtime error"
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  DistributedSearchTree
  SOURCES tstDistributedSearchTree.cpp Search_UnitTestHelpers.hpp unit_test_main.cpp
  COMM serial mpi
  NUM_MPI_PROCS 4
  STANDARD_PASS_OUTPUT
  FAIL_REGULAR_EXPRESSION "data race;leak;runtime error"
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  DetailsDistributedSearchTreeImpl
  SOURCES tstDetailsDistributedSearchTreeImpl.cpp unit_test_main.cpp
  COMM serial mpi
  NUM_MPI_PROCS 4
  STANDARD_PASS_OUTPUT
  FAIL_REGULAR_EXPRESSION "data race;leak;runtime error"
  )
