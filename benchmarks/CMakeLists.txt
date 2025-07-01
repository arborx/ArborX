find_package(Boost 1.56.0 REQUIRED COMPONENTS program_options)

# FIXME: when we start requiring benchmark >= 1.8.0, reenable
# "readability-redundant-string-cstr" check in .clang-tidy
find_package(benchmark QUIET 1.5.4)
if(benchmark_FOUND)
  message(STATUS "Found benchmark: ${benchmark_DIR} (version \"${benchmark_VERSION}\")")
else()
  message(STATUS "benchmark not found, fetching from GitHub")
  include(FetchContent)
  set(BENCHMARK_ENABLE_TESTING OFF)

  list(APPEND CMAKE_MESSAGE_INDENT "[benchmark] ")
  FetchContent_Declare(
    googlebenchmark
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    URL https://github.com/google/benchmark/archive/refs/tags/v1.8.0.tar.gz
    URL_HASH MD5=8ddf8571d3f6198d37852bcbd964f817
  )
  FetchContent_MakeAvailable(googlebenchmark)
  list(POP_BACK CMAKE_MESSAGE_INDENT)

  # Suppress clang-tidy diagnostics on code that we do not have control over
  IF(CMAKE_CXX_CLANG_TIDY)
    set_target_properties(benchmark PROPERTIES CXX_CLANG_TIDY "")
  ENDIF()

  target_compile_options(benchmark PRIVATE -w)
  target_compile_options(benchmark_main PRIVATE -w)
ENDIF()

add_subdirectory(brute_force_vs_bvh)
add_subdirectory(cluster)
add_subdirectory(execution_space_instances)
if(NOT WIN32)
  # FIXME: for now, skip the benchmarks using Google benchmark
  # when building for Windows, as we have trouble linking it
  # with the installed version of the Google benchmark
  add_subdirectory(bvh_driver)
  add_subdirectory(develop)
  add_subdirectory(union_find)
endif()
add_subdirectory(triangulated_surface_distance)

if (ARBORX_ENABLE_MPI)
  add_subdirectory(distributed_contact)
  add_subdirectory(distributed_tree_driver)
endif()
