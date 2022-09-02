add_executable(ArborX_BruteForceVsBVH.exe brute_force_vs_bvh.cpp)
target_link_libraries(ArborX_BruteForceVsBVH.exe ArborX::ArborX Boost::program_options)
add_test(NAME ArborX_BruteForceVsBVH_Benchmark COMMAND ./ArborX_BruteForceVsBVH.exe)
