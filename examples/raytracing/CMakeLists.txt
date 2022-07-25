add_executable(ArborX_RayTracing.exe example_raytracing.cpp)
target_link_libraries(ArborX_RayTracing.exe ArborX::ArborX Boost::program_options)
add_test(NAME ArborX_RayTracing_Example COMMAND ./ArborX_RayTracing.exe --Nx=10 --Ny=10 --Nz=10 --Lx=1 --Ly=100000 --Lz=100000 --rays=1000)
