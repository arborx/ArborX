# Create tests that contain each header seperately. We do not not run these test
# but we just try to compile them.
foreach (_header ${ArborX_HEADERS})
  string(REGEX REPLACE "[\./]" "_" header_test_name ${_header})
  set(header_test_name ArborX_HeaderSelfContained_${header_test_name})
  add_executable(${header_test_name} tstHeader.cpp)
  target_link_libraries(${header_test_name} PRIVATE ArborX)
  target_compile_definitions(${header_test_name} PRIVATE ARBORX_HEADER_TEST_NAME=${_header})
endforeach()
