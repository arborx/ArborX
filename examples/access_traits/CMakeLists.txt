if(Kokkos_ENABLE_CUDA)
  add_executable(ArborX_Example_CudaAccessTraits.exe example_cuda_access_traits.cpp)
  target_link_libraries(ArborX_Example_CudaAccessTraits.exe ArborX::ArborX)
  add_test(NAME ArborX_Example_CudaAccessTraits COMMAND ArborX_Example_CudaAccessTraits.exe)
else()
  add_executable(ArborX_Example_HostAccessTraits.exe example_host_access_traits.cpp)
  target_link_libraries(ArborX_Example_HostAccessTraits.exe ArborX::ArborX)
  add_test(NAME ArborX_Example_HostAccessTraits COMMAND ArborX_Example_HostAccessTraits.exe)
endif()
