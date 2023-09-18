find_package(Boost 1.56.0 REQUIRED COMPONENTS program_options)

find_package(benchmark 1.5.4 REQUIRED)
message(STATUS "Found benchmark: ${benchmark_DIR} (version \"${benchmark_VERSION}\")")

add_subdirectory(brute_force_vs_bvh)
add_subdirectory(dbscan)
add_subdirectory(execution_space_instances)
add_subdirectory(tokamak)
if(NOT WIN32)
  # FIXME: for now, skip the benchmarks using Google benchmark
  # when building for Windows, as we have trouble linking it
  # with the installed version of the Google benchmark
  add_subdirectory(bvh_driver)
  add_subdirectory(develop)
  add_subdirectory(union_find)
endif()

if (ARBORX_ENABLE_MPI)
  add_subdirectory(distributed_tree_driver)
endif()
