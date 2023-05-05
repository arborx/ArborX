add_executable(ArborX_Example_DBSCAN.exe example_dbscan.cpp)
target_link_libraries(ArborX_Example_DBSCAN.exe ArborX::ArborX)
add_test(NAME ArborX_Example_DBSCAN COMMAND ArborX_Example_DBSCAN.exe)
