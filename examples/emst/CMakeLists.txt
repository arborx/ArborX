add_executable(ArborX_Example_EMST.exe example_emst.cpp)
target_link_libraries(ArborX_Example_EMST.exe ArborX::ArborX)
add_test(NAME ArborX_Example_EMST COMMAND ArborX_Example_EMST.exe)
