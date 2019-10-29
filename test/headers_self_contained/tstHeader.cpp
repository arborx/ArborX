/****************************************************************************
 * Copyright (c) 2012-2019 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#define ARBORX_HEADER_TEST_STRINGIZE_IMPL(x) #x
#define ARBORX_HEADER_TEST_STRINGIZE(x) ARBORX_HEADER_TEST_STRINGIZE_IMPL(x)

#define ARBORX_HEADER_TO_TEST                                                  \
  ARBORX_HEADER_TEST_STRINGIZE(ARBORX_HEADER_TEST_NAME)

// include header twice to see if the include guards are set correctly
#include ARBORX_HEADER_TO_TEST
#include ARBORX_HEADER_TO_TEST

#if defined(ARBORX_HEADER_MUST_INCLUDE_CONFIG_HPP) &&                          \
    !defined(ARBORX_CONFIG_HPP)
#error "This header does not include ArborX_Config.hpp"
#endif

int main() { return 0; }
