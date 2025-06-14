set(ARBORX_BENCHMARK_UTILS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/benchmarks/utils)

set(mesh_file "unit_sphere.txt")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${mesh_file} ${CMAKE_CURRENT_BINARY_DIR}/${mesh_file} COPYONLY)

add_executable(ArborX_Benchmark_DistributedContact.exe distributed_contact.cpp)
target_link_libraries(ArborX_Benchmark_DistributedContact.exe ArborX::ArborX)
target_include_directories(ArborX_Benchmark_DistributedContact.exe PRIVATE ${ARBORX_BENCHMARK_UTILS_INCLUDE_DIR})
add_test(NAME ArborX_Benchmark_DistributedContact COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:ArborX_Benchmark_DistributedContact.exe> ${MPIEXEC_POSTFLAGS})
