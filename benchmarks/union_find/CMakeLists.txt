add_executable(ArborX_Benchmark_UnionFind.exe union_find.cpp)
target_link_libraries(ArborX_Benchmark_UnionFind.exe ArborX::ArborX benchmark::benchmark Boost::program_options)
add_test(NAME ArborX_Benchmark_UnionFind COMMAND ArborX_Benchmark_UnionFind.exe --benchmark_color=true)
