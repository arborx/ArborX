add_executable(ArborX_Example_Tokamak.exe
  tokamak.cpp
)
target_link_libraries(ArborX_Example_Tokamak.exe ArborX::ArborX)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/RK4Steps.txt
          ${CMAKE_CURRENT_SOURCE_DIR}/RZGrid.stl
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_test(NAME ArborX_Example_Tokamak.exe COMMAND ./ArborX_Example_Tokamak.exe)
