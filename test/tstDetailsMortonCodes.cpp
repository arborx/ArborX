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

#include <ArborX_DetailsMortonCode.hpp>

#include <boost/test/unit_test.hpp>

using namespace ArborX::Details;

BOOST_AUTO_TEST_SUITE(MortonCodes)

BOOST_AUTO_TEST_CASE(expand_bits)
{
  unsigned x = 0b110010011101;
  BOOST_TEST(expandBitsBy1(x) == 0b010100000100000101010001);
  BOOST_TEST(expandBitsBy2(x) == 0b000000001000000001001001000001);

  x = 0b11111111111111000001;
  BOOST_TEST(expandBitsBy1(x) == 0b01010101010101010101000000000001);
  BOOST_TEST(expandBitsBy2(x) == 0b001001001001000000000000000001);
}

BOOST_AUTO_TEST_CASE(morton_codes)
{
  BOOST_TEST(morton2D(0.f, 0.f) == 0x0);
  BOOST_TEST(morton2D(1.f, 1.f) == 0xffffffff);
  BOOST_TEST(morton2D(0.f, 1.f) == 0x55555555);
  BOOST_TEST(morton2D(1.f, 0.f) == 0xaaaaaaaa);

  BOOST_TEST(morton3D(0.f, 0.f, 0.f) == 0x0);
  BOOST_TEST(morton3D(1.f, 1.f, 1.f) == 0x3fffffff);
  BOOST_TEST(morton3D(0.f, 0.f, 1.f) == 0x9249249);
  BOOST_TEST(morton3D(1.f, 1.f, 0.f) == 0x36db6db6);
}

BOOST_AUTO_TEST_SUITE_END()
