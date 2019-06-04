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

#define BOOST_TEST_MODULE Miscellaneous
#include "boost_ext/TupleComparison.hpp"
#include <boost/test/unit_test.hpp>

#include <string>

#include "VectorOfTuples.hpp"

BOOST_AUTO_TEST_SUITE(VectorOfTuples)

BOOST_AUTO_TEST_CASE(heterogeneous)
{
  // NOTE assertion macro did not seem to like comas hence the variables
  auto const ret = toVectorOfTuples(
      std::vector<std::string>{"dordogne", "gironde", "landes"},
      std::vector<int>{24, 33, 40});
  std::vector<std::tuple<std::string, int>> const ref = {
      std::make_tuple("dordogne", 24), std::make_tuple("gironde", 33),
      std::make_tuple("landes", 40)};
  BOOST_TEST(ret == ref, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(single_argument)
{
  auto const ret = toVectorOfTuples(std::vector<float>{3.14f});
  std::vector<std::tuple<float>> const ref = {std::make_tuple(3.14f)};
  BOOST_TEST(ret == ref, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(not_properly_sized)
{
  BOOST_CHECK_EXCEPTION(
      toVectorOfTuples(std::vector<int>{1, 2, 3}, std::vector<double>{3.14},
                       std::vector<std::string>{"foo", "bar"}),
      std::invalid_argument, [&](std::exception const &e) {
        std::string const message = e.what();
        bool const message_contains_argument_position =
            message.find("argument 2") != std::string::npos;
        bool const message_shows_size_mismatch =
            message.find("has size 1 != 3") != std::string::npos;
        return message_contains_argument_position &&
               message_shows_size_mismatch;
      });
}

BOOST_AUTO_TEST_SUITE_END()
