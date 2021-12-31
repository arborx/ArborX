/****************************************************************************
 * Copyright (c) 2017-2022 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX_Exception.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(DesignByContract)

BOOST_AUTO_TEST_CASE(dumb)
{
  using namespace ArborX;
  BOOST_CHECK_NO_THROW(ARBORX_ASSERT(true));
  std::string const prefix = "ArborX exception: ";
  BOOST_CHECK_EXCEPTION(
      ARBORX_ASSERT(false), SearchException, [&](std::exception const &e) {
        std::string const message = e.what();
        bool const message_starts_with_prefix = message.find(prefix) == 0;
        bool const message_contains_filename =
            message.find(__FILE__) != std::string::npos;
        return message_starts_with_prefix && message_contains_filename;
      });
  std::string const message = "Keep calm and chive on!";
  BOOST_CHECK_EXCEPTION(
      throw SearchException(message), std::exception,
      [&](std::exception const &e) { return prefix + message == e.what(); });
}

BOOST_AUTO_TEST_SUITE_END()
