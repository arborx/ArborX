if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  find_package(ArborX REQUIRED)
  set(ArborX_TARGET ArborX::ArborX)
  find_package(Boost REQUIRED COMPONENTS program_options)
  get_filename_component(POINT_CLOUDS_INCLUDE_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)
  get_filename_component(UNIT_TESTS_INCLUDE_DIR ${POINT_CLOUDS_INCLUDE_DIR} DIRECTORY)
  set(POINT_CLOUDS_INCLUDE_DIR ${POINT_CLOUDS_INCLUDE_DIR}/point_clouds)
  set(UNIT_TESTS_INCLUDE_DIR ${UNIT_TESTS_INCLUDE_DIR}/test)
else()
  set(ArborX_TARGET ArborX)
  set(POINT_CLOUDS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/examples/point_clouds)
  set(UNIT_TESTS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/test)
endif()

# We require version 1.4.0 or higher but the format used by Google benchmark is
# wrong and thus, we cannot check the version during the configuration step.
find_package(benchmark REQUIRED)

find_package (Threads REQUIRED)

add_executable(ArborX_BoundingVolumeHierarchy.exe bvh_driver.cpp)
target_link_libraries(ArborX_BoundingVolumeHierarchy.exe ${ArborX_TARGET} benchmark::benchmark Boost::program_options Threads::Threads)
target_include_directories(ArborX_BoundingVolumeHierarchy.exe PRIVATE ${POINT_CLOUDS_INCLUDE_DIR} ${UNIT_TESTS_INCLUDE_DIR})
add_test(NAME ArborX_BoundingVolumeHierarchy_Example COMMAND ./ArborX_BoundingVolumeHierarchy.exe)
