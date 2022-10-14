find_package(Boost 1.56.0 REQUIRED COMPONENTS program_options)

find_package(benchmark 1.5.4 REQUIRED)
message(STATUS "Found benchmark: ${benchmark_DIR} (version \"${benchmark_VERSION}\")")

add_subdirectory(brute_force_vs_bvh)
add_subdirectory(bvh_driver)
add_subdirectory(dbscan)
add_subdirectory(develop)
add_subdirectory(execution_space_instances)
add_subdirectory(union_find)

if (ARBORX_ENABLE_MPI)
  add_subdirectory(distributed_tree_driver)
endif()
