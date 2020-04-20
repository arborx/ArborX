if(Kokkos_DEVICES MATCHES "CUDA")
  set(ArborX_TARGET ArborX)

  # Pick up .cu files
  enable_language(CUDA)

  # Separate ECL into a separate target. Otherwise, due to MPI linkage,
  # it gets MPI_CXX_COMPILE_OPTIONS that has flags like `-fexceptions` which
  # break nvcc.
  add_library(ECL ECL-CC_10.cu ECL.hpp)
  add_executable(ArborX_HaloFinder.exe halo_finder.cpp)
  # Get architecture flags from Kokkos. Cannot use Kokkos::kokkos, as
  # COMPILE_OPTIONS are not transitive. Instead, use Kokkos:kokkoscore.
  get_target_property(kokkos_compile_options Kokkos::kokkoscore INTERFACE_COMPILE_OPTIONS)
  string(REGEX MATCH "-arch=sm_.." arch_flag "${kokkos_compile_options}")
  target_compile_options(ECL PRIVATE "${arch_flag}")

  target_include_directories(ArborX_HaloFinder.exe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(ArborX_HaloFinder.exe ${ArborX_TARGET} Boost::program_options ECL)

  set(input_file "input.txt")
  add_test(NAME ArborX_HaloFinder COMMAND ./ArborX_HaloFinder.exe --filename=${input_file} --linking-length=1.4)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${input_file} ${CMAKE_CURRENT_BINARY_DIR}/${input_file} COPYONLY)
endif()
