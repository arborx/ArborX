##/****************************************************************************
## * Copyright (c) 2012-2019 by the ArborX authors                            *
## * All rights reserved.                                                     *
## *                                                                          *
## * This file is part of the ArborX library. ArborX is                       *
## * distributed under a BSD 3-clause license. For the licensing terms see    *
## * the LICENSE file in the top-level directory.                             *
## *                                                                          *
## * SPDX-License-Identifier: BSD-3-Clause                                    *
## ****************************************************************************/

# This CMake script makes sure to store the current git hash in
# ArborX_Version.hpp each time we recompile. It is important that this script
# is called by through a target created by add_custom_target so that it is
# always considered to be out-of-date.

SET(ARBORX_GIT_COMMIT_HASH "No hash available")

IF(EXISTS ${SOURCE_DIR}/.git)
  FIND_PACKAGE(Git QUIET)
  IF(GIT_FOUND)
    EXECUTE_PROCESS(
      COMMAND          ${GIT_EXECUTABLE} log --pretty=format:%h -n 1
      OUTPUT_VARIABLE  ARBORX_GIT_COMMIT_HASH)
    ENDIF()
ENDIF()
MESSAGE(STATUS "ArborX hash = '${ARBORX_GIT_COMMIT_HASH}'")

configure_file(${SOURCE_DIR}/src/ArborX_Version.hpp.in
               ${BINARY_DIR}/include/ArborX_Version.hpp)
