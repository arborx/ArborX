set(ArborX_TARGET ArborX)
set(POINT_CLOUDS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/benchmarks/point_clouds)
set(UNIT_TESTS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/test)

# We require version 1.4.0 or higher but the format used by Google benchmark is
# wrong and thus, we cannot check the version during the configuration step.
find_package(benchmark REQUIRED)

find_package(Threads REQUIRED)

add_executable(ArborX_BoundingVolumeHierarchy.exe bvh_driver.cpp)
target_link_libraries(ArborX_BoundingVolumeHierarchy.exe ${ArborX_TARGET} benchmark::benchmark Boost::program_options Threads::Threads)
target_include_directories(ArborX_BoundingVolumeHierarchy.exe PRIVATE ${PROJECT_SOURCE_DIR}/ext/nanoflann/include ${POINT_CLOUDS_INCLUDE_DIR} ${UNIT_TESTS_INCLUDE_DIR})
add_test(NAME ArborX_BoundingVolumeHierarchy_Benchmark COMMAND ./ArborX_BoundingVolumeHierarchy.exe --buffer=0 --benchmark_color=true)
if(ARBORX_PERFORMANCE_TESTING)
  target_compile_definitions(ArborX_BoundingVolumeHierarchy.exe PRIVATE ARBORX_PERFORMANCE_TESTING)
endif()
