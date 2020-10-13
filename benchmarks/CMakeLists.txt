find_package(Boost 1.56.0 REQUIRED COMPONENTS program_options)

add_subdirectory(bvh_driver)
if (ARBORX_ENABLE_MPI)
  add_subdirectory(distributed_tree_driver)
endif()
