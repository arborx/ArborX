set(ARBORX_BENCHMARK_UTILS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/benchmarks/utils)
set(UNIT_TESTS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/test)

add_executable(ArborX_Benchmark_BoundingVolumeHierarchy.exe bvh_driver.cpp)
target_link_libraries(ArborX_Benchmark_BoundingVolumeHierarchy.exe ArborX::ArborX benchmark::benchmark Boost::program_options)
target_include_directories(ArborX_Benchmark_BoundingVolumeHierarchy.exe PRIVATE ${ARBORX_BENCHMARK_UTILS_INCLUDE_DIR} ${UNIT_TESTS_INCLUDE_DIR})
add_test(NAME ArborX_Benchmark_BoundingVolumeHierarchy COMMAND ArborX_Benchmark_BoundingVolumeHierarchy.exe --buffer=0 --benchmark_color=true)
