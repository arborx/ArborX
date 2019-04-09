##---------------------------------------------------------------------------##
## TESTS
##---------------------------------------------------------------------------##
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
  HelloWorld
  SOURCES tstHelloWorld.cpp utf_main.cpp
  COMM serial mpi
  ADDED_EXE_TARGET_NAME_OUT hello_world_exe
  )
target_compile_definitions(${hello_world_exe} PRIVATE BOOST_TEST_DYN_LINK DTK_SEARCH_MPI_UNIT_TEST)

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  Exception
  SOURCES tstException.cpp
  COMM serial mpi
  NUM_MPI_PROCS 1
  ADDED_EXE_TARGET_NAME_OUT exception_exe
  )
# TODO link Boost::dynamic_linking interface target to enable dynamic linking
# (adds BOOST_ALL_DYN_LINK)
target_compile_definitions(${exception_exe} PRIVATE BOOST_TEST_MAIN BOOST_TEST_DYN_LINK)

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  DetailsUtils
  SOURCES tstDetailsUtils.cpp utf_main.cpp
  COMM serial mpi
  NUM_MPI_PROCS 1
  ADDED_EXE_TARGET_NAME_OUT details_utils_exe
  )
target_compile_definitions(${details_utils_exe} PRIVATE BOOST_TEST_DYN_LINK)
target_include_directories(${details_utils_exe} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  DetailsAlgorithms
  SOURCES tstDetailsAlgorithms.cpp
  COMM serial mpi
  NUM_MPI_PROCS 1
  ADDED_EXE_TARGET_NAME_OUT details_algorithms_exe
  )
target_compile_definitions(${details_algorithms_exe} PRIVATE BOOST_TEST_MAIN BOOST_TEST_DYN_LINK)

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  LinearBVH
  SOURCES tstLinearBVH.cpp Search_UnitTestHelpers.hpp utf_main.cpp
  COMM serial mpi
  NUM_MPI_PROCS 1
  ADDED_EXE_TARGET_NAME_OUT linear_bvh_exe
  )
target_compile_definitions(${linear_bvh_exe} PRIVATE BOOST_TEST_DYN_LINK)
target_include_directories(${linear_bvh_exe} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  DetailsTreeConstruction
  SOURCES tstDetailsTreeConstruction.cpp utf_main.cpp
  COMM serial mpi
  NUM_MPI_PROCS 1
  ADDED_EXE_TARGET_NAME_OUT details_tree_construction_exe
  )
target_compile_definitions(${details_tree_construction_exe} PRIVATE BOOST_TEST_DYN_LINK)
target_include_directories(${details_tree_construction_exe} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  DetailsTreeTraversal
  SOURCES tstDetailsTreeTraversal.cpp
  COMM serial mpi
  NUM_MPI_PROCS 1
  ADDED_EXE_TARGET_NAME_OUT details_tree_traversal_exe
  )
target_compile_definitions(${details_tree_traversal_exe} PRIVATE BOOST_TEST_MAIN BOOST_TEST_DYN_LINK)

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  DetailsBatchedQueries
  SOURCES tstDetailsBatchedQueries.cpp utf_main.cpp
  COMM serial mpi
  NUM_MPI_PROCS 1
  ADDED_EXE_TARGET_NAME_OUT details_batched_queries_exe
  )
target_compile_definitions(${details_batched_queries_exe} PRIVATE BOOST_TEST_DYN_LINK)
target_include_directories(${details_batched_queries_exe} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  DistributedSearchTree
  SOURCES tstDistributedSearchTree.cpp Search_UnitTestHelpers.hpp utf_main.cpp
  COMM serial mpi
  NUM_MPI_PROCS 4
  ADDED_EXE_TARGET_NAME_OUT distributed_search_tree_exe
  )
target_compile_definitions(${distributed_search_tree_exe} PRIVATE BOOST_TEST_DYN_LINK DTK_SEARCH_MPI_UNIT_TEST)
target_include_directories(${distributed_search_tree_exe} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  DetailsDistributedSearchTreeImpl
  SOURCES tstDetailsDistributedSearchTreeImpl.cpp utf_main.cpp
  COMM serial mpi
  NUM_MPI_PROCS 4
  ADDED_EXE_TARGET_NAME_OUT details_distributed_search_tree_exe
  )
target_compile_definitions(${details_distributed_search_tree_exe} PRIVATE BOOST_TEST_DYN_LINK)
target_include_directories(${details_distributed_search_tree_exe} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

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
