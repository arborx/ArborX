set(EXPLICIT_INSTANTIATION_SOURCE_FILES)
set(TEMPLATE_PARAMETERS 2 3 4 5 6)
foreach(DIM ${TEMPLATE_PARAMETERS})
  set(filename ${CMAKE_CURRENT_BINARY_DIR}/brute_force_vs_bvh_${DIM}.cpp)
  file(WRITE ${filename}
    "#include \"${CMAKE_CURRENT_SOURCE_DIR}/brute_force_vs_bvh_timpl.hpp\"\n"
    "template void ArborXBenchmark::run<${DIM}>(int, int, int);\n"
  )
  list(APPEND EXPLICIT_INSTANTIATION_SOURCE_FILES ${filename})
endforeach()


add_executable(ArborX_Benchmark_BruteForceVsBVH.exe
  ${EXPLICIT_INSTANTIATION_SOURCE_FILES}
  brute_force_vs_bvh.cpp
)
target_link_libraries(ArborX_Benchmark_BruteForceVsBVH.exe ArborX::ArborX Boost::program_options)
add_test(NAME ArborX_Benchmark_BruteForceVsBVH COMMAND ArborX_Benchmark_BruteForceVsBVH.exe)
