# We require version 1.4.0 or higher but the format used by Google benchmark is
# wrong and thus, we cannot check the version during the configuration step.
find_package(benchmark REQUIRED)

find_package (Threads REQUIRED)

add_executable(ArborX_BoundingVolumeHierarchy.exe bvh_driver.cpp)
if(CMAKE_PROJECT_NAME)
  set(ArborX_TARGET ArborX)
elseif()
  find_package(ArborX REQUIRED)
  set(ArborX_TARGET ArborX::ArborX)
  find_package(Boost REQUIRED COMPONENTS program_options)
endif()
target_link_libraries(ArborX_BoundingVolumeHierarchy.exe ${ArborX_TARGET} benchmark::benchmark Boost::program_options Threads::Threads)
target_include_directories(ArborX_BoundingVolumeHierarchy.exe PRIVATE ${CMAKE_SOURCE_DIR}/examples/point_clouds)
#set_target_properties(bvh_exe PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON CXX_EXTENSION OFF)
add_test(NAME ArborX_BoundingVolumeHierarchy_Example COMMAND ./ArborX_BoundingVolumeHierarchy.exe)
