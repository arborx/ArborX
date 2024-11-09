set(EXPLICIT_INSTANTIATION_SOURCE_FILES)
set(TEMPLATE_PARAMETERS 2 3 4 5 6)
foreach(DIM ${TEMPLATE_PARAMETERS})
  set(filename ${CMAKE_CURRENT_BINARY_DIR}/dbscan_${DIM}.cpp)
  file(WRITE ${filename}
    "#include \"${CMAKE_CURRENT_SOURCE_DIR}/dbscan_timpl.hpp\"\n"
    "template bool ArborXBenchmark::run<${DIM}>(ArborXBenchmark::Parameters const&);\n"
  )
  list(APPEND EXPLICIT_INSTANTIATION_SOURCE_FILES ${filename})
endforeach()

add_executable(ArborX_Benchmark_DBSCAN.exe
  ${EXPLICIT_INSTANTIATION_SOURCE_FILES}
  print_timers.cpp
  dbscan.cpp
)
target_include_directories(ArborX_Benchmark_DBSCAN.exe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ArborX_Benchmark_DBSCAN.exe ArborX::ArborX Boost::program_options)

add_executable(ArborX_DataConverter.exe converter.cpp)
target_compile_features(ArborX_DataConverter.exe PRIVATE cxx_std_17)
target_link_libraries(ArborX_DataConverter.exe Boost::program_options)

set(input_file "input.txt")
add_test(NAME ArborX_Benchmark_DBSCAN COMMAND ArborX_Benchmark_DBSCAN.exe --filename=${input_file} --eps=1.4 --verify)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${input_file} ${CMAKE_CURRENT_BINARY_DIR}/${input_file} COPYONLY)
