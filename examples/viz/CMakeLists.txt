if(Kokkos_ENABLE_Serial AND HAVE_DTK_BOOST)
  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    example_viz
    SOURCES tree_visualization.cpp
    NUM_MPI_PROCS 1
    FAIL_REGULAR_EXPRESSION "data race;leak;runtime error"
    )
  TRIBITS_COPY_FILES_TO_BINARY_DIR(
    oak_leaf
    SOURCE_FILES leaf_cloud.txt
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../point_clouds
    DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
    EXEDEPS example_viz
    )
  TRIBITS_COPY_FILES_TO_BINARY_DIR(
    viz_scripts
    SOURCE_FILES arborx_query_sort.py
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
    DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
    EXEDEPS example_viz
    )
endif()
