add_executable(ArborX_Example_RayTracing.exe example_raytracing.cpp)
target_link_libraries(ArborX_Example_RayTracing.exe ArborX::ArborX Boost::program_options)
add_test(NAME ArborX_Example_RayTracing COMMAND ArborX_Example_RayTracing.exe --nx=10 --ny=10 --nz=10 --lx=1 --ly=100000 --lz=100000 --rays=1000)
