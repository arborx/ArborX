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

#include <ArborX_Box.hpp>
#include <ArborX_KDOP.hpp>
#include <ArborX_Point.hpp>
#include <details/ArborX_Algorithms.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>

#include <tuple>
#include <type_traits>

using ArborX::Experimental::KDOP;

BOOST_AUTO_TEST_SUITE(DiscreteOrientedPolytopes)

using KDOP_2D_types = std::tuple<KDOP<2, 4>, KDOP<2, 8>>;

BOOST_AUTO_TEST_CASE_TEMPLATE(intersects_kdop_kdop_2D, KDOP_t, KDOP_2D_types)
{
  using Point = ArborX::Point<2>;
  KDOP_t x;
  BOOST_TEST(!intersects(x, x));
  expand(x, Point{1, 0});
  expand(x, Point{0, 1});
  BOOST_TEST(intersects(x, x));
  BOOST_TEST(!intersects(x, KDOP_t{}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(intersects_kdop_box_2D, KDOP_t, KDOP_2D_types)
{
  using Point = ArborX::Point<2>;
  using Box = ArborX::Box<2>;

  KDOP_t x;
  BOOST_TEST(!intersects(x, Box{}));
  BOOST_TEST(!intersects(x, Box{{0, 0}, {1, 1}}));
  expand(x, Point{1, 0});
  expand(x, Point{0, 1});
  BOOST_TEST(!intersects(x, Box{}));
  BOOST_TEST(intersects(x, Box{{0, 0}, {1, 1}}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(intersects_point_kdop_2D, KDOP_t, KDOP_2D_types)
{
  using Point = ArborX::Point<2>;
  {
    KDOP_t x;
    BOOST_TEST(!intersects(Point{1, 1}, x));
  }
  {
    KDOP_t x; // rombus
    expand(x, Point{0.5f, 0});
    expand(x, Point{0.5f, 1});
    expand(x, Point{0, 0.5f});
    expand(x, Point{1, 0.5f});
    // unit square corners
    BOOST_TEST(intersects(Point{0, 0}, x) ==
               (std::is_same_v<KDOP_t, KDOP<2, 4>>));
    BOOST_TEST(intersects(Point{1, 0}, x) ==
               (std::is_same_v<KDOP_t, KDOP<2, 4>>));
    BOOST_TEST(intersects(Point{0, 1}, x) ==
               (std::is_same_v<KDOP_t, KDOP<2, 4>>));
    BOOST_TEST(intersects(Point{1, 1}, x) ==
               (std::is_same_v<KDOP_t, KDOP<2, 4>>));
    // rombus corners
    BOOST_TEST(intersects(Point{0.5f, 0}, x));
    BOOST_TEST(intersects(Point{0.5f, 1}, x));
    BOOST_TEST(intersects(Point{0, 0.5f}, x));
    BOOST_TEST(intersects(Point{1, 0.5f}, x));
    // unit square center
    BOOST_TEST(intersects(Point{0.5f, 0.5f}, x));
    // mid rombus diagonals
    BOOST_TEST(intersects(Point{0.75f, 0.25f}, x));
    BOOST_TEST(intersects(Point{0.25f, 0.25f}, x));
    BOOST_TEST(intersects(Point{0.25f, 0.75f}, x));
    BOOST_TEST(intersects(Point{0.75f, 0.75f}, x));
    // slightly outside of the diagonals
    BOOST_TEST(intersects(Point{0.8f, 0.2f}, x) ==
               (std::is_same_v<KDOP_t, KDOP<2, 4>>));
    BOOST_TEST(intersects(Point{0.2f, 0.2f}, x) ==
               (std::is_same_v<KDOP_t, KDOP<2, 4>>));
    BOOST_TEST(intersects(Point{0.2f, 0.8f}, x) ==
               (std::is_same_v<KDOP_t, KDOP<2, 4>>));
    BOOST_TEST(intersects(Point{0.8f, 0.8f}, x) ==
               (std::is_same_v<KDOP_t, KDOP<2, 4>>));
  }
}

using KDOP_3D_types =
    std::tuple<KDOP<3, 6>, KDOP<3, 14>, KDOP<3, 18>, KDOP<3, 26>>;

BOOST_AUTO_TEST_CASE_TEMPLATE(intersects_kdop_kdop_3D, KDOP_t, KDOP_3D_types)
{
  using Point = ArborX::Point<3>;

  KDOP_t x;
  BOOST_TEST(!intersects(x, x));
  expand(x, Point{1, 0, 0});
  expand(x, Point{0, 1, 0});
  expand(x, Point{0, 0, 1});
  BOOST_TEST(intersects(x, x));
  BOOST_TEST(!intersects(x, KDOP_t{}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(intersects_kdop_box_3D, KDOP_t, KDOP_3D_types)
{
  using Box = ArborX::Box<3>;
  using Point = ArborX::Point<3>;

  KDOP_t x;
  BOOST_TEST(!intersects(x, Box{}));
  BOOST_TEST(!intersects(x, Box{{0, 0, 0}, {1, 1, 1}}));
  expand(x, Point{1, 0, 0});
  expand(x, Point{0, 1, 0});
  expand(x, Point{0, 0, 1});
  BOOST_TEST(!intersects(x, Box{}));
  BOOST_TEST(intersects(x, Box{{0, 0, 0}, {1, 1, 1}}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(intersects_point_kdop, KDOP_t, KDOP_3D_types)
{
  using Point = ArborX::Point<3>;
  {
    KDOP_t x;
    BOOST_TEST(!intersects(Point{1, 1, 1}, x));
  }
  {
    KDOP_t x; // unit cube with (1,1,0)--(1,1,1) edge chopped away
    // bottom face
    expand(x, Point{0, 0, 0});
    expand(x, Point{1, 0, 0});
    expand(x, Point{1, 0.75, 0});
    expand(x, Point{0.75, 1, 0});
    expand(x, Point{0, 1, 0});
    // top
    expand(x, Point{0, 0, 1});
    expand(x, Point{1, 0, 1});
    expand(x, Point{1, 0.75, 1});
    expand(x, Point{0.75, 1, 1});
    expand(x, Point{0, 1, 1});
    // test intersection with point on the missing edge
    BOOST_TEST(intersects(Point{1, 1, 0.5}, x) ==
               (std::is_same<KDOP_t, KDOP<3, 6>>::value ||
                std::is_same<KDOP_t, KDOP<3, 14>>::value));
    BOOST_TEST(intersects(Point{1, 1, 0.625}, x) ==
               (std::is_same<KDOP_t, KDOP<3, 6>>::value ||
                std::is_same<KDOP_t, KDOP<3, 14>>::value));
    BOOST_TEST(intersects(Point{1, 1, 0.375}, x) ==
               (std::is_same<KDOP_t, KDOP<3, 6>>::value ||
                std::is_same<KDOP_t, KDOP<3, 14>>::value));
    BOOST_TEST(intersects(Point{1, 1, 0.875}, x) ==
               (std::is_same<KDOP_t, KDOP<3, 6>>::value));
    BOOST_TEST(intersects(Point{1, 1, 0.125}, x) ==
               (std::is_same<KDOP_t, KDOP<3, 6>>::value));
    // with both ends of the edge
    BOOST_TEST(intersects(Point{1, 1, 0}, x) ==
               (std::is_same<KDOP_t, KDOP<3, 6>>::value));
    BOOST_TEST(intersects(Point{1, 1, 1}, x) ==
               (std::is_same<KDOP_t, KDOP<3, 6>>::value));
    // with centroid of unit cube
    BOOST_TEST(intersects(Point{0.5, 0.5, 0.5}, x));
    // with some point outside the unit cube
    BOOST_TEST(!intersects(Point{1, 2, 3}, x));
  }
  {
    KDOP_t x; // unit cube with (1,1,1) corner cut off
    // bottom face
    expand(x, Point{0, 0, 0});
    expand(x, Point{1, 0, 0});
    expand(x, Point{1, 1, 0});
    expand(x, Point{0, 1, 0});
    // top
    expand(x, Point{0, 0, 1});
    expand(x, Point{1, 0, 1});
    expand(x, Point{1, 0.75, 1});
    expand(x, Point{0.75, 1, 1});
    expand(x, Point{0, 1, 1});
    // test intersection with center of the missing corner
    BOOST_TEST(intersects(Point{1, 1, 1}, x) ==
               (std::is_same<KDOP_t, KDOP<3, 6>>::value ||
                std::is_same<KDOP_t, KDOP<3, 18>>::value));
    // test with points on the edges out of that corner
    BOOST_TEST(intersects(Point{0.5, 1, 1}, x));
    BOOST_TEST(intersects(Point{0.875, 1, 1}, x) ==
               (std::is_same<KDOP_t, KDOP<3, 6>>::value ||
                std::is_same<KDOP_t, KDOP<3, 18>>::value));
    BOOST_TEST(intersects(Point{1, 0.5, 1}, x));
    BOOST_TEST(intersects(Point{1, 0.875, 1}, x) ==
               (std::is_same<KDOP_t, KDOP<3, 6>>::value ||
                std::is_same<KDOP_t, KDOP<3, 18>>::value));
    BOOST_TEST(intersects(Point{1, 1, 0.5}, x));
    BOOST_TEST(intersects(Point{1, 1, 0.875}, x) ==
               (std::is_same<KDOP_t, KDOP<3, 6>>::value ||
                std::is_same<KDOP_t, KDOP<3, 18>>::value));
    // with centroid of unit cube
    BOOST_TEST(intersects(Point{0.5, 0.5, 0.5}, x));
    // with some point outside the unit cube
    BOOST_TEST(!intersects(Point{1, 2, 3}, x));
  }
}

BOOST_AUTO_TEST_SUITE_END()
