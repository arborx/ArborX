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

#include <ArborX_HyperPoint.hpp>

BOOST_AUTO_TEST_SUITE(MortonCodes)

BOOST_AUTO_TEST_CASE(expand_bits)
{
  unsigned x = 0b110010011101u;
  BOOST_TEST(expandBitsBy<1>(x) == 0b010100000100000101010001u);
  BOOST_TEST(expandBitsBy<2>(x) == 0b000000001000000001001001000001u);
  BOOST_TEST(expandBitsBy<3>(x) == 0b00010000000000010001000100000001u);
  BOOST_TEST(expandBitsBy<4>(x) == 0b000000000100001000010000000001u);
  BOOST_TEST(expandBitsBy<5>(x) == 0b000001000001000001000000000001u);
  BOOST_TEST(expandBitsBy<6>(x) == 0b0000001000000100000000000001u);

  unsigned long long y = 0b11111111111111000001llu;
  BOOST_TEST(expandBitsBy<1>(y) ==
             0b0101010101010101010101010101000000000001llu);
  BOOST_TEST(expandBitsBy<2>(y) ==
             0b001001001001001001001001001001001001001001000000000000000001llu);
  BOOST_TEST(expandBitsBy<3>(y) ==
             0b000100010001000100010001000100010001000000000000000000000001llu);
  BOOST_TEST(expandBitsBy<4>(y) ==
             0b000010000100001000010000100001000000000000000000000000000001llu);
  BOOST_TEST(expandBitsBy<5>(y) ==
             0b000001000001000001000001000000000000000000000000000000000001llu);
  BOOST_TEST(
      expandBitsBy<6>(y) ==
      0b000000100000010000001000000000000000000000000000000000000000001llu);
}

BOOST_AUTO_TEST_CASE(morton_codes)
{
  using ArborX::ExperimentalHyperGeometry::Point;

  BOOST_TEST(morton32(Point{0.f, 0.f}) == 0x0u);
  BOOST_TEST(morton32(Point{1.f, 1.f}) == 0xffffffffu);
  BOOST_TEST(morton32(Point{0.f, 1.f}) == 0x55555555u);
  BOOST_TEST(morton32(Point{1.f, 0.f}) == 0xaaaaaaaau);

  BOOST_TEST(morton64(Point{0.f, 0.f}) == 0x0llu);
  BOOST_TEST(morton64(Point{1.f, 1.f}) == 0x3fffffffffffffffllu);
  BOOST_TEST(morton64(Point{0.f, 1.f}) == 0x1555555555555555llu);
  BOOST_TEST(morton64(Point{1.f, 0.f}) == 0x2aaaaaaaaaaaaaaallu);

  BOOST_TEST(morton32(Point{0.f, 0.f, 0.f}) == 0x0u);
  BOOST_TEST(morton32(Point{1.f, 1.f, 1.f}) == 0x3fffffffu);
  BOOST_TEST(morton32(Point{0.f, 0.f, 1.f}) == 0x9249249u);
  BOOST_TEST(morton32(Point{1.f, 1.f, 0.f}) == 0x36db6db6u);

  BOOST_TEST(morton64(Point{0.f, 0.f, 0.f}) == 0x0llu);
  BOOST_TEST(morton64(Point{1.f, 1.f, 1.f}) == 0x7fffffffffffffffllu);
  BOOST_TEST(morton64(Point{0.f, 0.f, 1.f}) == 0x1249249249249249llu);
  BOOST_TEST(morton64(Point{1.f, 1.f, 0.f}) == 0x6db6db6db6db6db6llu);
}

BOOST_AUTO_TEST_SUITE_END()
