# Create tests that contain each header separately. We do not  run these tests
# but we just try to compile them.
foreach (_header ${ArborX_HEADERS})
  string(REGEX REPLACE "[\./]" "_" header_test_name ${_header})
  set(header_test_name ArborX_HeaderSelfContained_${header_test_name})
  add_executable(${header_test_name} tstHeader.cpp)
  if(_header IN_LIST ArborX_HEADERS_MUST_INCLUDE_CONFIG_HPP)
    target_compile_definitions(${header_test_name} PRIVATE ARBORX_HEADER_MUST_INCLUDE_CONFIG_HPP)
  endif()
  target_link_libraries(${header_test_name} PRIVATE ArborX)
  target_compile_definitions(${header_test_name} PRIVATE ARBORX_HEADER_TEST_NAME=${_header})
endforeach()
