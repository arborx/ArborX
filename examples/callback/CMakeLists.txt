add_executable(ArborX_Callback.exe example_callback.cpp)
target_link_libraries(ArborX_Callback.exe ArborX::ArborX)
add_test(NAME ArborX_Example_Callback COMMAND ./ArborX_Callback.exe)
