
set(ARBORX_DEVICE_TYPES)
if(Kokkos_DEVICES MATCHES "Serial")
  list(APPEND ARBORX_DEVICE_TYPES Kokkos::Serial::device_type)
endif()
if(Kokkos_DEVICES MATCHES "OpenMP")
  list(APPEND ARBORX_DEVICE_TYPES Kokkos::OpenMP::device_type)
endif()
if(Kokkos_DEVICES MATCHES "Cuda")
  list(APPEND ARBORX_DEVICE_TYPES Kokkos::CudaSpace::device_type)
endif()
if(Kokkos_DEVICES MATCHES "Pthread")
  list(APPEND ARBORX_DEVICE_TYPES Kokkos::Threads::device_type)
endif()

if(NOT ARBORX_DEVICE_TYPES)
  message(SEND_ERROR "Kokkos_DEVICES must include at least one of 'Serial', 'OpenMP', 'Cuda' and 'Pthread'!")
endif()

string(REPLACE ";" "," ARBORX_DEVICE_TYPES "${ARBORX_DEVICE_TYPES}")
configure_file(ArborX_EnableDeviceTypes.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/ArborX_EnableDeviceTypes.hpp @ONLY)

find_package(Boost 1.67.0 REQUIRED COMPONENTS unit_test_framework)

# Compile only, nothing to run
add_executable(ArborX_DetailsConcepts.exe tstDetailsConcepts.cpp)
target_link_libraries(ArborX_DetailsConcepts.exe PRIVATE ArborX)

add_executable(ArborX_Miscellaneous.exe tstMiscellaneous.cpp tstException.cpp)
# TODO link Boost::dynamic_linking interface target to enable dynamic linking
# (adds BOOST_ALL_DYN_LINK)
target_link_libraries(ArborX_Miscellaneous.exe PRIVATE ArborX Boost::unit_test_framework)
target_compile_definitions(ArborX_Miscellaneous.exe PRIVATE BOOST_TEST_DYN_LINK)
add_test(NAME ArborX_Miscellaneous_Test COMMAND ./ArborX_Miscellaneous.exe)

add_executable(ArborX_DetailsUtils.exe tstDetailsUtils.cpp utf_main.cpp)
target_link_libraries(ArborX_DetailsUtils.exe PRIVATE ArborX Boost::unit_test_framework)
target_compile_definitions(ArborX_DetailsUtils.exe PRIVATE BOOST_TEST_DYN_LINK)
target_include_directories(ArborX_DetailsUtils.exe PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
add_test(NAME ArborX_DetailsUtils_Test COMMAND ./ArborX_DetailsUtils.exe)

add_executable(ArborX_DetailsAlgorithms.exe tstDetailsAlgorithms.cpp)
target_link_libraries(ArborX_DetailsAlgorithms.exe PRIVATE ArborX Boost::unit_test_framework)
target_compile_definitions(ArborX_DetailsAlgorithms.exe PRIVATE BOOST_TEST_MAIN BOOST_TEST_DYN_LINK)
target_include_directories(ArborX_DetailsAlgorithms.exe PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
add_test(NAME ArborX_DetailsAlgorithms_Test COMMAND ./ArborX_DetailsAlgorithms.exe)

add_executable(ArborX_LinearBVH.exe tstLinearBVH.cpp utf_main.cpp)
target_link_libraries(ArborX_LinearBVH.exe PRIVATE ArborX Boost::unit_test_framework)
target_compile_definitions(ArborX_LinearBVH.exe PRIVATE BOOST_TEST_DYN_LINK)
target_include_directories(ArborX_LinearBVH.exe PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
add_test(NAME ArborX_LinearBVH_Test COMMAND ./ArborX_LinearBVH.exe)

add_executable(ArborX_DetailsTreeConstruction.exe tstDetailsTreeConstruction.cpp utf_main.cpp)
target_link_libraries(ArborX_DetailsTreeConstruction.exe PRIVATE ArborX Boost::unit_test_framework)
target_compile_definitions(ArborX_DetailsTreeConstruction.exe PRIVATE BOOST_TEST_DYN_LINK)
target_include_directories(ArborX_DetailsTreeConstruction.exe PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
add_test(NAME ArborX_DetailsTreeConstruction_Test COMMAND ./ArborX_DetailsTreeConstruction.exe)

add_executable(ArborX_DetailsContainers.exe tstSequenceContainers.cpp tstContainerAdaptors.cpp tstHeapOperations.cpp)
target_link_libraries(ArborX_DetailsContainers.exe PRIVATE ArborX Boost::unit_test_framework)
target_compile_definitions(ArborX_DetailsContainers.exe PRIVATE BOOST_TEST_DYN_LINK)
add_test(NAME ArborX_DetailsContainers_Test COMMAND ./ArborX_DetailsContainers.exe)

add_executable(ArborX_DetailsTreeTraversal.exe tstDetailsTreeTraversal.cpp utf_main.cpp)
target_link_libraries(ArborX_DetailsTreeTraversal.exe PRIVATE ArborX Boost::unit_test_framework)
target_compile_definitions(ArborX_DetailsTreeTraversal.exe PRIVATE BOOST_TEST_DYN_LINK)
add_test(NAME ArborX_DetailsTreeTraversal_Test COMMAND ./ArborX_DetailsTreeTraversal.exe)

add_executable(ArborX_DetailsBatchedQueries.exe tstDetailsBatchedQueries.cpp utf_main.cpp)
target_link_libraries(ArborX_DetailsBatchedQueries.exe PRIVATE ArborX Boost::unit_test_framework)
target_compile_definitions(ArborX_DetailsBatchedQueries.exe PRIVATE BOOST_TEST_DYN_LINK)
target_include_directories(ArborX_DetailsBatchedQueries.exe PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
add_test(NAME ArborX_DetailsBatchedQueries_Test COMMAND ./ArborX_DetailsBatchedQueries.exe)

if(ARBORX_ENABLE_MPI)
  add_executable(ArborX_DistributedSearchTree.exe tstDistributedSearchTree.cpp utf_main.cpp)
  target_link_libraries(ArborX_DistributedSearchTree.exe PRIVATE ArborX Boost::unit_test_framework)
  target_compile_definitions(ArborX_DistributedSearchTree.exe PRIVATE BOOST_TEST_DYN_LINK ARBORX_MPI_UNIT_TEST)
  target_include_directories(ArborX_DistributedSearchTree.exe PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  add_test(NAME ArborX_DistributedSearchTree_Test COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ./ArborX_DistributedSearchTree.exe ${MPIEXEC_POSTFLAGS})

  add_executable(ArborX_DetailsDistributedSearchTreeImpl.exe tstDetailsDistributedSearchTreeImpl.cpp utf_main.cpp)
  target_link_libraries(ArborX_DetailsDistributedSearchTreeImpl.exe PRIVATE ArborX Boost::unit_test_framework)
  target_compile_definitions(ArborX_DetailsDistributedSearchTreeImpl.exe PRIVATE BOOST_TEST_DYN_LINK)
  target_include_directories(ArborX_DetailsDistributedSearchTreeImpl.exe PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  add_test(NAME ArborX_DetailsDistributedSearchTreeImpl_Test COMMAND ./ArborX_DetailsDistributedSearchTreeImpl.exe)
endif()

add_executable(ArborX_BoostAdapters.exe tstBoostGeometryAdapters.cpp tstBoostRangeAdapters.cpp utf_main.cpp)
target_link_libraries(ArborX_BoostAdapters.exe PRIVATE ArborX Boost::unit_test_framework)
target_compile_definitions(ArborX_BoostAdapters.exe PRIVATE BOOST_TEST_DYN_LINK)
add_test(NAME ArborX_BoostAdapters_Test COMMAND ./ArborX_BoostAdapters.exe)

add_subdirectory(headers_self_contained)
