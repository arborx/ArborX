IF(EXISTS ${SOURCE_DIR}/.git)
    FIND_PACKAGE(Git)
    IF(GIT_FOUND)
        EXECUTE_PROCESS(
            COMMAND           ${GIT_EXECUTABLE} log --pretty=format:%h -n 1
            OUTPUT_VARIABLE   ARBORX_GIT_COMMIT_HASH
            )
    ENDIF()
ENDIF()
MESSAGE("ArborX hash = '${ARBORX_GIT_COMMIT_HASH}'")
MESSAGE(STATUS "ARBORX_ENABLE_MPI: ${ARBORX_ENABLE_MPI}")

configure_file(${SOURCE_DIR}/src/ArborX_Config.hpp.in
               ${BINARY_DIR}/include/ArborX_Config.hpp)

