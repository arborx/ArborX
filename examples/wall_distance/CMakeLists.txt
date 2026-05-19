set(mesh_file "mesh.exo")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${mesh_file} ${CMAKE_CURRENT_BINARY_DIR}/${mesh_file} COPYONLY)

add_executable(ArborX_Example_Wall_Distance.exe example_wall_distance.cpp helpers.cpp)
target_link_libraries(ArborX_Example_Wall_Distance.exe ArborX::ArborX Boost::program_options)
add_test(NAME ArborX_Example_Wall_Distance COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:ArborX_Example_Wall_Distance.exe> ${MPIEXEC_POSTFLAGS} --filename mesh.exo --wall-names bottom left top right)
