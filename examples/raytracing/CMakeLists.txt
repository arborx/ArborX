add_executable(ArborX_RayTracing.exe example_raytracing.cpp)
target_link_libraries(ArborX_RayTracing.exe ArborX::ArborX Boost::program_options)
add_test(NAME ArborX_RayTracing_Example COMMAND ./ArborX_RayTracing.exe --spheres=1000 --rays=50000 --L=200.0)
