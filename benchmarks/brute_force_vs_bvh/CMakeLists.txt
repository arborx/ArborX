set(EXPLICIT_INSTANTIATION_SOURCE_FILES)
set(TEMPLATE_PARAMETERS 3)
if(Kokkos_VERSION VERSION_GREATER_EQUAL 3.7)
  list(APPEND TEMPLATE_PARAMETERS 2 4 5 6)
endif()
foreach(DIM ${TEMPLATE_PARAMETERS})
  set(filename ${CMAKE_CURRENT_BINARY_DIR}/brute_force_vs_bvh_${DIM}.cpp)
  file(WRITE ${filename}
    "#include \"${CMAKE_CURRENT_SOURCE_DIR}/brute_force_vs_bvh_timpl.hpp\"\n"
    "template void ArborXBenchmark::run<${DIM}>(int, int, int);\n"
  )
  list(APPEND EXPLICIT_INSTANTIATION_SOURCE_FILES ${filename})
endforeach()


add_executable(ArborX_BruteForceVsBVH.exe
  ${EXPLICIT_INSTANTIATION_SOURCE_FILES}
  brute_force_vs_bvh.cpp
)
target_link_libraries(ArborX_BruteForceVsBVH.exe ArborX::ArborX Boost::program_options)
add_test(NAME ArborX_BruteForceVsBVH_Benchmark COMMAND ./ArborX_BruteForceVsBVH.exe)
