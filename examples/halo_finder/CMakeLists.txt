add_executable(ArborX_HaloFinder.exe halo_finder.cpp)
target_include_directories(ArborX_HaloFinder.exe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ArborX_HaloFinder.exe ArborX::ArborX Boost::program_options)

set(input_file "input.txt")
add_test(NAME ArborX_HaloFinder COMMAND ./ArborX_HaloFinder.exe --filename=${input_file} --linking-length=1.4 --verify)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${input_file} ${CMAKE_CURRENT_BINARY_DIR}/${input_file} COPYONLY)
