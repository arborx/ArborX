add_executable(ArborX_Callback.exe example_callback.cpp)
target_link_libraries(ArborX_Callback.exe ${ArborX_TARGET})
add_test(NAME ArborX_Callback_Example COMMAND ./ArborX_Callback.exe)
