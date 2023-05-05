
set(ARBORX_DEVICE_TYPES)
if(Kokkos_ENABLE_SERIAL)
  list(APPEND ARBORX_DEVICE_TYPES Kokkos::Serial::device_type)
endif()
if(Kokkos_ENABLE_OPENMP)
  list(APPEND ARBORX_DEVICE_TYPES Kokkos::OpenMP::device_type)
endif()
if(Kokkos_ENABLE_CUDA)
  list(APPEND ARBORX_DEVICE_TYPES Kokkos::CudaSpace::device_type)
endif()
if(Kokkos_ENABLE_THREADS)
  list(APPEND ARBORX_DEVICE_TYPES Kokkos::Threads::device_type)
endif()
if(Kokkos_ENABLE_HIP)
  list(APPEND ARBORX_DEVICE_TYPES Kokkos::Experimental::HIPSpace::device_type)
endif()
if(Kokkos_ENABLE_OPENMPTARGET)
  list(APPEND ARBORX_DEVICE_TYPES Kokkos::Experimental::OpenMPTarget::device_type)
endif()
if(Kokkos_ENABLE_SYCL)
  list(APPEND ARBORX_DEVICE_TYPES Kokkos::Experimental::SYCL::device_type)
endif()

string(REPLACE ";" "," ARBORX_DEVICE_TYPES "${ARBORX_DEVICE_TYPES}")

if(NOT ARBORX_DEVICE_TYPES)
  message(SEND_ERROR "Kokkos_DEVICES must include at least one of 'SERIAL', 'OPENMP', 'CUDA', 'HIP', 'OPENMPTARGET', 'SYCL' or 'THREADS'!")
endif()

configure_file(ArborX_EnableDeviceTypes.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/ArborX_EnableDeviceTypes.hpp @ONLY)

find_package(Boost 1.67.0 REQUIRED COMPONENTS unit_test_framework)
# CMake Boost version check is tricky due to multiple changes to the way Boost
# package stores version information
if(NOT DEFINED Boost_VERSION_MINOR)
  set(Boost_VERSION_MINOR ${Boost_MINOR_VERSION})
endif()
if(Kokkos_ENABLE_CUDA AND Boost_VERSION VERSION_GREATER 1.68 AND Boost_VERSION VERSION_LESS 1.75)
  message(WARNING "Boost versions 1.69 to 1.74 are known to yield build issues with NVCC")
endif()

# Compile only, nothing to run
add_executable(ArborX_Test_CompileOnly.exe
  tstCompileOnlyAccessTraits.cpp
  tstCompileOnlyCallbacks.cpp
  tstCompileOnlyTypeRequirements.cpp
  tstCompileOnlyWeightedEdges.cpp
  tstCompileOnlyMain.cpp
)
target_link_libraries(ArborX_Test_CompileOnly.exe PRIVATE ArborX)

add_executable(ArborX_Test_DetailsUtils.exe
  tstDetailsUtils.cpp
  tstDetailsKokkosExtViewHelpers.cpp
  utf_main.cpp
)
# TODO link Boost::dynamic_linking interface target to enable dynamic linking
# (adds BOOST_ALL_DYN_LINK)
target_link_libraries(ArborX_Test_DetailsUtils.exe PRIVATE ArborX Boost::unit_test_framework)
target_compile_definitions(ArborX_Test_DetailsUtils.exe PRIVATE BOOST_TEST_DYN_LINK)
target_include_directories(ArborX_Test_DetailsUtils.exe PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
add_test(NAME ArborX_Test_DetailsUtils COMMAND ArborX_Test_DetailsUtils.exe)

add_executable(ArborX_Test_Geometry.exe
  tstDetailsAlgorithms.cpp
  tstCompileOnlyGeometry.cpp
  tstRay.cpp
  tstKDOP.cpp
)
target_link_libraries(ArborX_Test_Geometry.exe PRIVATE ArborX Boost::unit_test_framework)
target_compile_definitions(ArborX_Test_Geometry.exe PRIVATE BOOST_TEST_DYN_LINK)
target_include_directories(ArborX_Test_Geometry.exe PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
add_test(NAME ArborX_Test_Geometry COMMAND ArborX_Test_Geometry.exe)

set(ARBORX_TEST_QUERY_TREE_SOURCES)
foreach(_test Callbacks Degenerate ManufacturedSolution ComparisonWithBoost)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/tstQueryTree${_test}_BVH.cpp.tmp"
    "#include <ArborX_LinearBVH.hpp>\n"
    "#define ARBORX_TEST_TREE_TYPES Tuple<ArborX::BVH>\n"
    "#define ARBORX_TEST_DEVICE_TYPES std::tuple<${ARBORX_DEVICE_TYPES}>\n"
    "#include <tstQueryTree${_test}.cpp>\n"
  )
  configure_file(
    "${CMAKE_CURRENT_BINARY_DIR}/tstQueryTree${_test}_BVH.cpp.tmp"
    "${CMAKE_CURRENT_BINARY_DIR}/tstQueryTree${_test}_BVH.cpp" COPYONLY
  )
  list(APPEND ARBORX_TEST_QUERY_TREE_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/tstQueryTree${_test}_BVH.cpp")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/tstQueryTree${_test}_BF.cpp.tmp"
    "#include <ArborX_BruteForce.hpp>\n"
    "#include <ArborX_Box.hpp>\n"
    "template <class MemorySpace> using ArborX__BruteForce = ArborX::BruteForce<MemorySpace, ArborX::Box>;\n"
    "#define ARBORX_TEST_TREE_TYPES Tuple<ArborX__BruteForce>\n"
    "#define ARBORX_TEST_DEVICE_TYPES std::tuple<${ARBORX_DEVICE_TYPES}>\n"
    "#define ARBORX_TEST_DISABLE_NEAREST_QUERY\n"
    "#define ARBORX_TEST_DISABLE_CALLBACK_EARLY_EXIT\n"
    "#include <tstQueryTree${_test}.cpp>\n"
  )
  configure_file(
    "${CMAKE_CURRENT_BINARY_DIR}/tstQueryTree${_test}_BF.cpp.tmp"
    "${CMAKE_CURRENT_BINARY_DIR}/tstQueryTree${_test}_BF.cpp" COPYONLY
  )
  list(APPEND ARBORX_TEST_QUERY_TREE_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/tstQueryTree${_test}_BF.cpp")
  foreach(_bounding_volume KDOP14 KDOP18) # purposefully ommitting KDOP6 and KDOP26 to reduce the number of instantiations
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/tstQueryTree${_test}_BVH_${_bounding_volume}.cpp.tmp"
      "#include <ArborX_LinearBVH.hpp>\n"
      "#include <ArborX_KDOP.hpp>\n"
      "using KDOP6 = ArborX::Experimental::KDOP<6>;\n"
      "using KDOP14 = ArborX::Experimental::KDOP<14>;\n"
      "using KDOP18 = ArborX::Experimental::KDOP<18>;\n"
      "using KDOP26 = ArborX::Experimental::KDOP<26>;\n"
      "template <class MemorySpace> using ArborX__BoundingVolumeHierarchy_${_bounding_volume} = ArborX::BasicBoundingVolumeHierarchy<MemorySpace, ${_bounding_volume}>;\n"
      "#define ARBORX_TEST_TREE_TYPES Tuple<ArborX__BoundingVolumeHierarchy_${_bounding_volume}>\n"
      "#define ARBORX_TEST_DEVICE_TYPES std::tuple<${ARBORX_DEVICE_TYPES}>\n"
      "#define ARBORX_TEST_DISABLE_NEAREST_QUERY\n"
      "#define ARBORX_TEST_DISABLE_SPATIAL_QUERY_INTERSECTS_SPHERE\n"
      "#include <tstQueryTree${_test}.cpp>\n"
    )
    configure_file(
      "${CMAKE_CURRENT_BINARY_DIR}/tstQueryTree${_test}_BVH_${_bounding_volume}.cpp.tmp"
      "${CMAKE_CURRENT_BINARY_DIR}/tstQueryTree${_test}_BVH_${_bounding_volume}.cpp" COPYONLY
    )
    list(APPEND ARBORX_TEST_QUERY_TREE_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/tstQueryTree${_test}_BVH_${_bounding_volume}.cpp")
  endforeach()
endforeach()
list(APPEND ARBORX_TEST_QUERY_TREE_SOURCES
  tstQueryTreeRay.cpp
  tstQueryTreeTraversalPolicy.cpp
  tstQueryTreeIntersectsKDOP.cpp
  tstKokkosToolsAnnotations.cpp
  utf_main.cpp
)
if (Kokkos_VERSION VERSION_LESS 4.0.99)
  list(APPEND ARBORX_TEST_QUERY_TREE_SOURCES tstScopedProfileRegion.cpp)
endif()
add_executable(ArborX_Test_QueryTree.exe ${ARBORX_TEST_QUERY_TREE_SOURCES})
target_link_libraries(ArborX_Test_QueryTree.exe PRIVATE ArborX Boost::unit_test_framework)
target_compile_definitions(ArborX_Test_QueryTree.exe PRIVATE BOOST_TEST_DYN_LINK)
# FIXME_SYCL oneDPL messes with namespace std, see https://github.com/oneapi-src/oneDPL/issues/576
# only needed for the tools annotation test
if(Kokkos_ENABLE_SYCL)
  target_compile_definitions(ArborX_Test_QueryTree.exe PRIVATE NANORANGE_NO_STD_FORWARD_DECLARATIONS)
endif()
target_include_directories(ArborX_Test_QueryTree.exe PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME ArborX_Test_QueryTree COMMAND ArborX_Test_QueryTree.exe)

add_executable(ArborX_Test_DetailsTreeConstruction.exe tstDetailsMortonCodes.cpp tstDetailsTreeConstruction.cpp utf_main.cpp)
target_link_libraries(ArborX_Test_DetailsTreeConstruction.exe PRIVATE ArborX Boost::unit_test_framework)
target_compile_definitions(ArborX_Test_DetailsTreeConstruction.exe PRIVATE BOOST_TEST_DYN_LINK)
target_include_directories(ArborX_Test_DetailsTreeConstruction.exe PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
add_test(NAME ArborX_Test_DetailsTreeConstruction COMMAND ArborX_Test_DetailsTreeConstruction.exe)

add_executable(ArborX_Test_DetailsContainers.exe
  tstSequenceContainers.cpp
  tstContainerAdaptors.cpp
  tstHeapOperations.cpp
  tstPriorityQueueMiscellaneous.cpp
)
target_link_libraries(ArborX_Test_DetailsContainers.exe PRIVATE ArborX Boost::unit_test_framework)
target_compile_definitions(ArborX_Test_DetailsContainers.exe PRIVATE BOOST_TEST_DYN_LINK)
add_test(NAME ArborX_Test_DetailsContainers COMMAND ArborX_Test_DetailsContainers.exe)

add_executable(ArborX_Test_DetailsBatchedQueries.exe tstDetailsBatchedQueries.cpp utf_main.cpp)
target_link_libraries(ArborX_Test_DetailsBatchedQueries.exe PRIVATE ArborX Boost::unit_test_framework)
target_compile_definitions(ArborX_Test_DetailsBatchedQueries.exe PRIVATE BOOST_TEST_DYN_LINK)
target_include_directories(ArborX_Test_DetailsBatchedQueries.exe PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
add_test(NAME ArborX_Test_DetailsBatchedQueries COMMAND ArborX_Test_DetailsBatchedQueries.exe)

add_executable(ArborX_Test_DetailsCrsGraphWrapperImpl.exe tstDetailsCrsGraphWrapperImpl.cpp utf_main.cpp)
target_link_libraries(ArborX_Test_DetailsCrsGraphWrapperImpl.exe PRIVATE ArborX Boost::unit_test_framework)
target_compile_definitions(ArborX_Test_DetailsCrsGraphWrapperImpl.exe PRIVATE BOOST_TEST_DYN_LINK)
target_include_directories(ArborX_Test_DetailsCrsGraphWrapperImpl.exe PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
add_test(NAME ArborX_Test_DetailsCrsGraphWrapperImpl COMMAND ArborX_Test_DetailsCrsGraphWrapperImpl.exe)

add_executable(ArborX_Test_Clustering.exe
  tstDBSCAN.cpp
  tstDendrogram.cpp
  utf_main.cpp
)
target_link_libraries(ArborX_Test_Clustering.exe PRIVATE ArborX Boost::unit_test_framework)
target_compile_definitions(ArborX_Test_Clustering.exe PRIVATE BOOST_TEST_DYN_LINK)
target_include_directories(ArborX_Test_Clustering.exe PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_SOURCE_DIR}/benchmarks/dbscan)
add_test(NAME ArborX_Test_Clustering COMMAND ArborX_Test_Clustering.exe)

# compare results with a dataset of 1000 points from mlpack
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/mst_golden_test_points.csv
  ${CMAKE_CURRENT_BINARY_DIR}/mst_golden_test_points.csv
  COPYONLY
)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/mst_golden_test_edges.csv
  ${CMAKE_CURRENT_BINARY_DIR}/mst_golden_test_edges.csv
  COPYONLY
)
add_executable(ArborX_Test_DetailsClusteringHelpers.exe
  tstDetailsTreeNodeLabeling.cpp
  tstDetailsMutualReachabilityDistance.cpp
  tstMinimumSpanningTree.cpp
  tstMinimumSpanningTreeGoldenTest.cpp
  tstUnionFind.cpp
  utf_main.cpp
)
target_link_libraries(ArborX_Test_DetailsClusteringHelpers.exe PRIVATE ArborX Boost::unit_test_framework)
target_compile_definitions(ArborX_Test_DetailsClusteringHelpers.exe PRIVATE BOOST_TEST_DYN_LINK)
target_include_directories(ArborX_Test_DetailsClusteringHelpers.exe PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
add_test(NAME ArborX_Test_DetailsClusteringHelpers COMMAND ArborX_Test_DetailsClusteringHelpers.exe)

add_executable(ArborX_Test_SpecializedTraversals.exe
  tstDetailsHalfTraversal.cpp
  tstDetailsExpandHalfToFull.cpp
  tstNeighborList.cpp
  utf_main.cpp
)
target_link_libraries(ArborX_Test_SpecializedTraversals.exe PRIVATE ArborX Boost::unit_test_framework)
target_compile_definitions(ArborX_Test_SpecializedTraversals.exe PRIVATE BOOST_TEST_DYN_LINK)
target_include_directories(ArborX_Test_SpecializedTraversals.exe PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
add_test(NAME ArborX_Test_SpecializedTraversals COMMAND ArborX_Test_SpecializedTraversals.exe)

if(ARBORX_ENABLE_MPI)
  add_executable(ArborX_Test_DistributedTree.exe tstDistributedTree.cpp tstKokkosToolsDistributedAnnotations.cpp utf_main.cpp)
  target_link_libraries(ArborX_Test_DistributedTree.exe PRIVATE ArborX Boost::unit_test_framework)
  target_compile_definitions(ArborX_Test_DistributedTree.exe PRIVATE BOOST_TEST_DYN_LINK ARBORX_MPI_UNIT_TEST)
  # FIXME_SYCL oneDPL messes with namespace std, see https://github.com/oneapi-src/oneDPL/issues/576
  # only needed for the tools annotation test
  if(Kokkos_ENABLE_SYCL)
    target_compile_definitions(ArborX_Test_DistributedTree.exe PRIVATE NANORANGE_NO_STD_FORWARD_DECLARATIONS)
  endif()
  target_include_directories(ArborX_Test_DistributedTree.exe PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  add_test(NAME ArborX_Test_DistributedTree COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:ArborX_Test_DistributedTree.exe> ${MPIEXEC_POSTFLAGS})

  add_executable(ArborX_Test_DetailsDistributedTreeImpl.exe tstDetailsDistributedTreeImpl.cpp utf_main.cpp)
  target_link_libraries(ArborX_Test_DetailsDistributedTreeImpl.exe PRIVATE ArborX Boost::unit_test_framework)
  target_compile_definitions(ArborX_Test_DetailsDistributedTreeImpl.exe PRIVATE BOOST_TEST_DYN_LINK ARBORX_MPI_UNIT_TEST)
  target_include_directories(ArborX_Test_DetailsDistributedTreeImpl.exe PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  add_test(NAME ArborX_Test_DetailsDistributedTreeImpl COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:ArborX_Test_DetailsDistributedTreeImpl.exe> ${MPIEXEC_POSTFLAGS})
endif()

add_executable(ArborX_Test_BoostAdapters.exe tstBoostGeometryAdapters.cpp tstBoostRangeAdapters.cpp utf_main.cpp)
target_link_libraries(ArborX_Test_BoostAdapters.exe PRIVATE ArborX Boost::unit_test_framework)
target_compile_definitions(ArborX_Test_BoostAdapters.exe PRIVATE BOOST_TEST_DYN_LINK)
add_test(NAME ArborX_Test_BoostAdapters COMMAND ArborX_Test_BoostAdapters.exe)

if(ARBORX_ENABLE_HEADER_SELF_CONTAINMENT_TESTS)
  add_subdirectory(headers_self_contained)
endif()
