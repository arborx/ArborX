find_package(Boost REQUIRED COMPONENTS program_options)

add_subdirectory(bvh_driver)
if (ARBORX_ENABLE_MPI)
  add_subdirectory(distributed_tree_driver)
endif()
add_subdirectory(viz)
