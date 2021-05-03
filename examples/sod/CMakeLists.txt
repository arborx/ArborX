add_executable(ArborX_SOD.exe sod.cpp)
target_include_directories(ArborX_SOD.exe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ArborX_SOD.exe ArborX::ArborX Boost::program_options)
