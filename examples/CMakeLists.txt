if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.12)
  project(ArborXExamples CXX)
  find_package(ArborX REQUIRED)
  enable_testing()
endif()

add_subdirectory(access_traits)

add_subdirectory(callback)

find_package(Boost COMPONENTS program_options)
if(Boost_FOUND)
  add_subdirectory(viz)
  add_subdirectory(dbscan)
endif()
