add_executable(ArborX_Benchmark_Tokamak.exe
  tokamak.cpp
)
target_link_libraries(ArborX_Benchmark_Tokamak.exe ArborX::ArborX)
add_test(NAME ArborX_Benchmark_Tokamak.exe COMMAND ./ArborX_Benchmark_Tokamak.exe)
