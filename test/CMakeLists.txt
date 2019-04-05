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

set(DTK_SEARCH_DEVICE_TYPES)
if(Kokkos_ENABLE_Serial)
  list(APPEND DTK_SEARCH_DEVICE_TYPES Kokkos::Serial::device_type)
endif()
if(Kokkos_ENABLE_OpenMP)
  list(APPEND DTK_SEARCH_DEVICE_TYPES Kokkos::OpenMP::device_type)
endif()
if(Kokkos_ENABLE_Cuda)
  list(APPEND DTK_SEARCH_DEVICE_TYPES Kokkos::Cuda::device_type)
endif()
string(REPLACE ";" "," DTK_SEARCH_DEVICE_TYPES "${DTK_SEARCH_DEVICE_TYPES}")
set(DTK_SEARCH_DEVICE_TYPES "std::tuple<${DTK_SEARCH_DEVICE_TYPES}>")
configure_file(DTK_EnableDeviceTypes.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/DTK_EnableDeviceTypes.hpp @ONLY)

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
    BoostAdapters
    SOURCES tstBoostGeometryAdapters.cpp tstBoostRangeAdapters.cpp utf_main.cpp
    COMM serial mpi
    NUM_MPI_PROCS 1
    ADDED_EXE_TARGET_NAME_OUT boost_adapters_exe
    )
    target_compile_definitions(${boost_adapters_exe} PRIVATE BOOST_TEST_DYN_LINK)
ENDIF()
