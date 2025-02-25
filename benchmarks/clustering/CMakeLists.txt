add_library(cluster_benchmark_helpers
  print_timers.cpp
)
target_compile_features(cluster_benchmark_helpers PRIVATE cxx_std_17)
target_link_libraries(cluster_benchmark_helpers PRIVATE Kokkos::kokkos)

set(input_file "input.txt")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${input_file} ${CMAKE_CURRENT_BINARY_DIR}/${input_file} COPYONLY)

add_executable(ArborX_Benchmark_DBSCAN.exe dbscan.cpp)
target_include_directories(ArborX_Benchmark_DBSCAN.exe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ArborX_Benchmark_DBSCAN.exe ArborX::ArborX Boost::program_options cluster_benchmark_helpers)
add_test(NAME ArborX_Benchmark_DBSCAN COMMAND ArborX_Benchmark_DBSCAN.exe --filename=${input_file} --eps=1.4 --verify)

add_executable(ArborX_Benchmark_MST.exe mst.cpp)
target_include_directories(ArborX_Benchmark_MST.exe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ArborX_Benchmark_MST.exe ArborX::ArborX Boost::program_options cluster_benchmark_helpers)
add_test(NAME ArborX_Benchmark_HDBSCAN COMMAND ArborX_Benchmark_HDBSCAN.exe --filename=${input_file})

add_executable(ArborX_Benchmark_HDBSCAN.exe hdbscan.cpp)
target_include_directories(ArborX_Benchmark_HDBSCAN.exe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ArborX_Benchmark_HDBSCAN.exe ArborX::ArborX Boost::program_options cluster_benchmark_helpers)
add_test(NAME ArborX_Benchmark_MST COMMAND ArborX_Benchmark_MST.exe --filename=${input_file})
