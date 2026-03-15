/****************************************************************************
 * Copyright (c) 2025, ArborX authors                                       *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <detail/ArborX_DistributedUtils.hpp>

#include <boost/test/unit_test.hpp>

#include <array>

#define BOOST_TEST_MODULE DetailsDistributedTree

namespace tt = boost::test_tools;

BOOST_AUTO_TEST_CASE(closest_factors)
{
  using ArborX::Details::closestFactors;

  // clang-format off
  BOOST_TEST((closestFactors<1>(1)    == std::array{1}),          tt::per_element());
  BOOST_TEST((closestFactors<1>(5)    == std::array{5}),          tt::per_element());

  BOOST_TEST((closestFactors<2>(1)    == std::array{1, 1}),       tt::per_element());
  BOOST_TEST((closestFactors<2>(5)    == std::array{5, 1}),       tt::per_element());

  BOOST_TEST((closestFactors<3>(1)    == std::array{1, 1, 1}),    tt::per_element());
  BOOST_TEST((closestFactors<3>(8)    == std::array{2, 2, 2}),    tt::per_element());
  BOOST_TEST((closestFactors<3>(16)   == std::array{4, 2, 2}),    tt::per_element());
  BOOST_TEST((closestFactors<3>(108)  == std::array{6, 6, 3}),    tt::per_element());
  BOOST_TEST((closestFactors<3>(800)  == std::array{10, 10, 8}),  tt::per_element());
  BOOST_TEST((closestFactors<3>(7200) == std::array{20, 20, 18}), tt::per_element());
  BOOST_TEST((closestFactors<3>(9996) == std::array{28, 21, 17}), tt::per_element());
  // clang-format on

  BOOST_TEST(
      (closestFactors<10>(1024) == std::array{2, 2, 2, 2, 2, 2, 2, 2, 2, 2}),
      tt::per_element());
}
