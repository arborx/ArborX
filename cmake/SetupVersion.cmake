##/****************************************************************************
## * Copyright (c) 2017-2022 by the ArborX authors                            *
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

set(ARBORX_GIT_COMMIT_HASH "No hash available")

if(EXISTS ${SOURCE_DIR}/.git)
  find_package(Git QUIET)
  if(GIT_FOUND)
    execute_process(
      COMMAND          ${GIT_EXECUTABLE} rev-parse --verify --short HEAD
      OUTPUT_VARIABLE  ARBORX_GIT_COMMIT_HASH
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(
      COMMAND          ${GIT_EXECUTABLE} status --porcelain --untracked-files=no
      OUTPUT_VARIABLE  ARBORX_GIT_WORKTREE_STATUS)
    if(ARBORX_GIT_WORKTREE_STATUS)
      set(ARBORX_GIT_COMMIT_HASH "${ARBORX_GIT_COMMIT_HASH}-dirty")
    endif()
  endif()
endif()
message(STATUS "ArborX hash = '${ARBORX_GIT_COMMIT_HASH}'")

configure_file(${SOURCE_DIR}/src/ArborX_Version.hpp.in
               ${BINARY_DIR}/include/ArborX_Version.hpp)
