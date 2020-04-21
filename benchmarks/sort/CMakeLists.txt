set(UNIT_TESTS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/test)

# We require version 1.4.0 or higher but the format used by Google benchmark is
# wrong and thus, we cannot check the version during the configuration step.
find_package(benchmark REQUIRED)

# find_package(Threads REQUIRED)

add_executable(ArborX_SortBenchmark.exe sort_benchmark.cpp)
target_link_libraries(ArborX_SortBenchmark.exe ArborX benchmark::benchmark Boost::program_options )#Threads::Threads)
target_include_directories(ArborX_SortBenchmark.exe PRIVATE ${UNIT_TESTS_INCLUDE_DIR})
add_test(NAME ArborX_SortBenchmark COMMAND ./ArborX_SortBenchmark.exe --num-values 10000 --value-type float --benchmark_color=true)
