add_executable(ArborX_Intersection.exe example_intersection.cpp)
target_link_libraries(ArborX_Intersection.exe ArborX::ArborX)
add_test(NAME ArborX_Intersection_Example COMMAND ./ArborX_Intersection.exe)
