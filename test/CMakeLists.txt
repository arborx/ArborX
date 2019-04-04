##---------------------------------------------------------------------------##
## TESTS
##---------------------------------------------------------------------------##
TRIBITS_ADD_EXECUTABLE_AND_TEST(
  LinearBVH
  SOURCES tstLinearBVH.cpp Search_UnitTestHelpers.hpp unit_test_main.cpp
  COMM serial mpi
  NUM_MPI_PROCS 1
  STANDARD_PASS_OUTPUT
  FAIL_REGULAR_EXPRESSION "data race;leak;runtime error"
  )
TRIBITS_ADD_EXECUTABLE_AND_TEST(
  Exception
  SOURCES tstException.cpp
  COMM serial mpi
  NUM_MPI_PROCS 1
  ADDED_EXE_TARGET_NAME_OUT dbc_exe
  )
# TODO link Boost::dynamic_linking interface target to enable dynamic linking
# (adds BOOST_ALL_DYN_LINK)
target_compile_definitions(${dbc_exe} PRIVATE BOOST_TEST_MAIN BOOST_TEST_DYN_LINK)

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  HelloWorld
  SOURCES tstHelloWorld.cpp utf_main.cpp
  COMM serial mpi
  ADDED_EXE_TARGET_NAME_OUT hello_world_exe
  )
target_compile_definitions(${hello_world_exe} PRIVATE BOOST_TEST_DYN_LINK DTK_SEARCH_MPI_UNIT_TEST)

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
  DetailsBatchedQueries
  SOURCES tstDetailsBatchedQueries.cpp unit_test_main.cpp
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

IF (HAVE_DTK_BOOST)
  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    BoostRangeAdapters
    SOURCES tstBoostRangeAdapters.cpp DTK_BoostRangeAdapters.hpp unit_test_main.cpp
    COMM serial mpi
    NUM_MPI_PROCS 1
    STANDARD_PASS_OUTPUT
    FAIL_REGULAR_EXPRESSION "data race;leak;runtime error"
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    BoostGeometryAdapters
    SOURCES tstBoostGeometryAdapters.cpp DTK_BoostGeometryAdapters.hpp unit_test_main.cpp
    COMM serial mpi
    NUM_MPI_PROCS 1
    STANDARD_PASS_OUTPUT
    FAIL_REGULAR_EXPRESSION "data race;leak;runtime error"
    )
ENDIF()
