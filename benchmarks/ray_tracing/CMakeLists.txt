add_executable(ArborX_Benchmark_RayTracing.exe ray_tracing.cpp)
target_link_libraries(ArborX_Benchmark_RayTracing.exe ArborX::ArborX)
add_test(NAME ArborX_Benchmark_RayTracing COMMAND ${MPIEXEC_EXECUTABLE}
  ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ./ArborX_Benchmark_RayTracing.exe ${MPIEXEC_POSTFLAGS})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/rays_cam-0-0_test_full.csv ${CMAKE_CURRENT_BINARY_DIR}/rays_cam-0-0_test_full.csv COPYONLY)
