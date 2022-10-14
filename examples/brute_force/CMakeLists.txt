add_executable(ArborX_BruteForce.exe brute_force.cpp)
target_link_libraries(ArborX_BruteForce.exe ArborX::ArborX Boost::program_options)
add_test(NAME ArborX_Example_BruteForce COMMAND ./ArborX_BruteForce.exe)
