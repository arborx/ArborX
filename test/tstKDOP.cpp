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

#include <ArborX_DetailsAlgorithms.hpp>
#include <ArborX_KDOP.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>

#include <tuple>
#include <type_traits>

using ArborX::Box;
using ArborX::Point;
using ArborX::Experimental::KDOP;

using KDOP_types = std::tuple<KDOP<6>, KDOP<14>, KDOP<18>, KDOP<26>>;

BOOST_AUTO_TEST_SUITE(DiscreteOrientedPolytopes)

BOOST_AUTO_TEST_CASE_TEMPLATE(conversion_to_box, KDOP_t, KDOP_types)
{
  using ArborX::Details::equals;
  KDOP_t x;
  BOOST_TEST(equals((Box)x, Box{}));
  x += Point{0, 1, 0};
  x += Point{1, 0, 1};
  BOOST_TEST(equals((Box)x, Box{{0, 0, 0}, {1, 1, 1}}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(intersects_KDOP, KDOP_t, KDOP_types)
{
  KDOP_t x;
  BOOST_TEST(!intersects(x, x));
  x += Point{1, 0, 0};
  x += Point{0, 1, 0};
  x += Point{0, 0, 1};
  BOOST_TEST(intersects(x, x));
  BOOST_TEST(!intersects(x, KDOP_t{}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(intersects_box, KDOP_t, KDOP_types)
{
  KDOP_t x;
  BOOST_TEST(!intersects(x, Box{}));
  BOOST_TEST(!intersects(x, Box{{0, 0, 0}, {1, 1, 1}}));
  x += Point{1, 0, 0};
  x += Point{0, 1, 0};
  x += Point{0, 0, 1};
  BOOST_TEST(!intersects(x, Box{}));
  BOOST_TEST(intersects(x, Box{{0, 0, 0}, {1, 1, 1}}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(intersects_point, KDOP_t, KDOP_types)
{
  {
    KDOP_t x;
    BOOST_TEST(!intersects(Point{1, 1, 1}, x));
  }
  {
    KDOP_t x; // unit cube with (1,1,0)--(1,1,1) edge chopped away
    // bottom face
    x += Point{0, 0, 0};
    x += Point{1, 0, 0};
    x += Point{1, 0.75f, 0};
    x += Point{0.75f, 1, 0};
    x += Point{0, 1, 0};
    // top
    x += Point{0, 0, 1};
    x += Point{1, 0, 1};
    x += Point{1, 0.75f, 1};
    x += Point{0.75f, 1, 1};
    x += Point{0, 1, 1};
    // test intersection with point on the missing edge
    BOOST_TEST(intersects(Point{1, 1, 0.5}, x) ==
               (std::is_same<KDOP_t, KDOP<6>>::value ||
                std::is_same<KDOP_t, KDOP<14>>::value));
    BOOST_TEST(intersects(Point{1, 1, 0.625}, x) ==
               (std::is_same<KDOP_t, KDOP<6>>::value ||
                std::is_same<KDOP_t, KDOP<14>>::value));
    BOOST_TEST(intersects(Point{1, 1, 0.375}, x) ==
               (std::is_same<KDOP_t, KDOP<6>>::value ||
                std::is_same<KDOP_t, KDOP<14>>::value));
    BOOST_TEST(intersects(Point{1, 1, 0.875}, x) ==
               (std::is_same<KDOP_t, KDOP<6>>::value));
    BOOST_TEST(intersects(Point{1, 1, 0.125}, x) ==
               (std::is_same<KDOP_t, KDOP<6>>::value));
    // with both ends of the edge
    BOOST_TEST(intersects(Point{1, 1, 0}, x) ==
               (std::is_same<KDOP_t, KDOP<6>>::value));
    BOOST_TEST(intersects(Point{1, 1, 1}, x) ==
               (std::is_same<KDOP_t, KDOP<6>>::value));
    // with centroid of unit cube
    BOOST_TEST(intersects(Point{0.5, 0.5, 0.5}, x));
    // with some point outside the unit cube
    BOOST_TEST(!intersects(Point{1, 2, 3}, x));
  }
  {
    KDOP_t x; // unit cube with (1,1,1) corner cut off
    // bottom face
    x += Point{0, 0, 0};
    x += Point{1, 0, 0};
    x += Point{1, 1, 0};
    x += Point{0, 1, 0};
    // top
    x += Point{0, 0, 1};
    x += Point{1, 0, 1};
    x += Point{1, 0.75f, 1};
    x += Point{0.75f, 1, 1};
    x += Point{0, 1, 1};
    // test intersection with center of the missing corner
    BOOST_TEST(intersects(Point{1, 1, 1}, x) ==
               (std::is_same<KDOP_t, KDOP<6>>::value ||
                std::is_same<KDOP_t, KDOP<18>>::value));
    // test with points on the edges out of that corner
    BOOST_TEST(intersects(Point{0.5, 1, 1}, x));
    BOOST_TEST(intersects(Point{0.875, 1, 1}, x) ==
               (std::is_same<KDOP_t, KDOP<6>>::value ||
                std::is_same<KDOP_t, KDOP<18>>::value));
    BOOST_TEST(intersects(Point{1, 0.5, 1}, x));
    BOOST_TEST(intersects(Point{1, 0.875, 1}, x) ==
               (std::is_same<KDOP_t, KDOP<6>>::value ||
                std::is_same<KDOP_t, KDOP<18>>::value));
    BOOST_TEST(intersects(Point{1, 1, 0.5}, x));
    BOOST_TEST(intersects(Point{1, 1, 0.875}, x) ==
               (std::is_same<KDOP_t, KDOP<6>>::value ||
                std::is_same<KDOP_t, KDOP<18>>::value));
    // with centroid of unit cube
    BOOST_TEST(intersects(Point{0.5, 0.5, 0.5}, x));
    // with some point outside the unit cube
    BOOST_TEST(!intersects(Point{1, 2, 3}, x));
  }
}

BOOST_AUTO_TEST_SUITE_END()
