add_executable(ArborX_Benchmark_Develop.exe develop.cpp)
target_link_libraries(ArborX_Benchmark_Develop.exe ArborX::ArborX benchmark::benchmark)
add_test(NAME ArborX_Benchmark_Develop COMMAND ArborX_Benchmark_Develop.exe)
