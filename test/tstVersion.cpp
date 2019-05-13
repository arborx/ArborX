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
#include <ArborX_Version.hpp>

#include <boost/test/unit_test.hpp>

#include <iostream>

#define BOOST_TEST_MODULE Version

BOOST_AUTO_TEST_CASE(version)
{
  auto const arborx_version = ArborX::version();
  BOOST_TEST(!arborx_version.empty());
  std::cout << "ArborX version " << arborx_version << std::endl;

  auto const arborx_commit_hash = ArborX::gitCommitHash();
  BOOST_TEST(!arborx_commit_hash.empty());
  std::cout << "ArborX commit hash " << arborx_commit_hash << std::endl;
}
