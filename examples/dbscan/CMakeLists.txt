add_executable(ArborX_DBSCAN.exe dbscan.cpp)
target_include_directories(ArborX_DBSCAN.exe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ArborX_DBSCAN.exe ArborX::ArborX Boost::program_options)

add_executable(ArborX_DataConverter.exe converter.cpp)
target_compile_features(ArborX_DataConverter.exe PRIVATE cxx_std_14)
target_link_libraries(ArborX_DataConverter.exe Boost::program_options)

set(input_file "input.txt")
add_test(NAME ArborX_DBSCAN_Example COMMAND ./ArborX_DBSCAN.exe --filename=${input_file} --eps=1.4 --verify)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${input_file} ${CMAKE_CURRENT_BINARY_DIR}/${input_file} COPYONLY)
