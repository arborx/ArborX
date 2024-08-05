/****************************************************************************
 * Copyright (c) 2024 by the ArborX authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX_OBB.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>

#include <tuple>
#include <type_traits>

namespace tt = boost::test_tools;

using ArborX::Experimental::OBB;

BOOST_AUTO_TEST_SUITE(OrientedBoundingBox)

BOOST_AUTO_TEST_CASE(equals_obb)
{
  using ArborX::Details::equals;

  OBB<2, float> obb({{1, 0}, {0, 1}});
  BOOST_TEST(equals(obb, obb));

  OBB<2, float> obb1({{1, 0}, {0, 1}, {0, 0}});
  BOOST_TEST(!equals(obb, obb1));

  OBB<3, float> obb3({{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
  BOOST_TEST(equals(obb3, obb3));
}

BOOST_AUTO_TEST_CASE(expand_obb_2D)
{
  using Point = ArborX::ExperimentalHyperGeometry::Point<2>;
  using ArborX::Details::equals;
  using ArborX::Details::expand;

  Point points[3] = {{1, 0}, {0, 0}, {0, 1}};

  OBB<2, float> obb;
  expand(obb, points[0]);
  BOOST_TEST(equals(obb, OBB({points[0]})));

  expand(obb, OBB({points[1], points[2]}));
  BOOST_TEST(equals(obb, OBB({points[0], points[1], points[2]})));
}

BOOST_AUTO_TEST_CASE(expand_obb_3D)
{
  using Point = ArborX::ExperimentalHyperGeometry::Point<3>;
  using ArborX::Details::equals;
  using ArborX::Details::expand;

  Point points[5] = {{1, 0, 0}, {0, 0, 0}, {0, 2, 0}, {0, 0, 3}, {0, 2, 3}};

  OBB<3, float> obb;
  expand(obb, points[0]);
  BOOST_TEST(equals(obb, OBB({points[0]})));

  expand(obb, OBB({points[1], points[2], points[3], points[4]}));
  BOOST_TEST(equals(
      obb, OBB({points[0], points[1], points[2], points[3], points[4]})));
}

BOOST_AUTO_TEST_CASE(intersects_point_obb_2D)
{
  using Point = ArborX::ExperimentalHyperGeometry::Point<2>;
  using ArborX::Details::intersects;

  {
    OBB<2, float> obb({{1, 0}});
    BOOST_TEST(intersects(Point{1, 0}, obb));
    BOOST_TEST(!intersects(Point{0, 0}, obb));
  }
  {
    //   x
    //    \
    //     \
    //      x
    OBB<2, float> obb({{1, 0}, {0, 1}});
    BOOST_TEST(intersects(Point{0.5f, 0.5f}, obb));
    BOOST_TEST(intersects(Point{0, 1}, obb));
    BOOST_TEST(intersects(Point{1, 0}, obb));
    BOOST_TEST(!intersects(Point{0, 0}, obb));
  }
  {
    //   x
    //  / \
    //  \  \
    //   x  x
    //    \/
    OBB<2, float> obb({{1, 0}, {0, 1}, {0, 0}});
    BOOST_TEST(intersects(Point{-0.25f, 0.26f}, obb));
    BOOST_TEST(!intersects(Point{-0.5f, 0.49f}, obb));
    BOOST_TEST(intersects(Point{-0.25f, 0.74f}, obb));
    BOOST_TEST(intersects(Point{0.25f, -0.24f}, obb));
    BOOST_TEST(!intersects(Point{0.5f, 0.51f}, obb));
    BOOST_TEST(intersects(Point{0.75f, 0.24f}, obb));
  }
  {
    // OBB is not necessarily the tightest box around the points. Here's an
    // example where the covariance matrix is proportional to identity, and
    // we simply choose the default axes, i.e.,
    //
    //  This      Not this
    //  +-x-+        x
    //  |   |       / \
    //  x   x      x   x
    //  |   |       \ /
    //  +-x-+        x
    OBB<2, float> obb({{1, 0}, {0, 1}, {1, 2}, {2, 1}});
    BOOST_TEST(intersects(Point{0, 1.f}, obb));
    BOOST_TEST(intersects(Point{1.f, 2.f}, obb));
    BOOST_TEST(intersects(Point{0.49f, 0.49f}, obb));
    BOOST_TEST(intersects(Point{0.49f, 1.51f}, obb));
    BOOST_TEST(intersects(Point{1.51f, 1.51f}, obb));
    BOOST_TEST(intersects(Point{1.51f, 0.49f}, obb));
  }
}

BOOST_AUTO_TEST_CASE(distance_point_obb_2D)
{
  using Point = ArborX::ExperimentalHyperGeometry::Point<2>;
  using ArborX::Details::distance;

  {
    OBB<2, float> obb({{1, 0}});
    BOOST_TEST(distance(Point{1, 0}, obb) == 0);
    BOOST_TEST(distance(Point{0, 0}, obb) == 1);
    BOOST_TEST(distance(Point{0, 1}, obb) == std::sqrt(2.f));
  }
  {
    //   x
    //    \
    //     \
    //      x
    OBB<2, float> obb({{1, 0}, {0, 1}});
    BOOST_TEST(distance(Point{0.5f, 0.5f}, obb) == 0);
    BOOST_TEST(distance(Point{-1, 0}, obb) == std::sqrt(2.f));
  }
  {
    //   x
    //  / \
    //  \  \
    //   x  x
    //    \/
    OBB<2, float> obb({{1, 0}, {0, 1}, {0, 0}});
    BOOST_TEST(distance(Point{1, 1}, obb) == std::sqrt(0.5f));
    BOOST_TEST(distance(Point{0.5f, -1.f}, obb) == 0.5f, tt::tolerance(1e-7f));
  }
}

BOOST_AUTO_TEST_CASE(centroid_obb_2D, *boost::unit_test::tolerance(1e-7f))
{
  using ArborX::Details::returnCentroid;

  {
    OBB<2, float> obb({{1, 0}});
    auto center = returnCentroid(obb);
    BOOST_TEST(center[0] == 1);
    BOOST_TEST(center[1] == 0);
  }
  {
    //   x
    //    \
    //     \
    //      x
    OBB<2, float> obb({{1, 0}, {0, 1}});
    auto center = returnCentroid(obb);
    BOOST_TEST(center[0] == 0.5f);
    BOOST_TEST(center[1] == 0.5f);
  }
  {
    //   x
    //  / \
    //  \  \
    //   x  x
    //    \/
    OBB<2, float> obb({{1, 0}, {0, 1}, {0, 0}});
    auto center = returnCentroid(obb);
    BOOST_TEST(center[0] == 0.25f);
    BOOST_TEST(center[1] == 0.25f);
  }
}

BOOST_AUTO_TEST_SUITE_END()
