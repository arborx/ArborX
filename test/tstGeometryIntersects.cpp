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

#include <ArborX_Box.hpp>
#include <ArborX_Ellipsoid.hpp>
#include <ArborX_Sphere.hpp>
#include <ArborX_Tetrahedron.hpp>
#include <ArborX_Triangle.hpp>
#include <algorithms/ArborX_Intersects.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(intersects)
{
  using ArborX::Details::intersects;

  using Point = ArborX::Point<3>;
  using Box = ArborX::Box<3>;
  using Sphere = ArborX::Sphere<3>;
  using Tetrahedron = ArborX::ExperimentalHyperGeometry::Tetrahedron<>;

  // uninitialized box does not intersect with other boxes
  static_assert(!intersects(Box{}, Box{{{1.0, 2.0, 3.0}}, {{4.0, 5.0, 6.0}}}));
  // uninitialized box does not even intersect with itself
  static_assert(!intersects(Box{}, Box{}));
  // box with zero extent does
  static_assert(intersects(Box{{{0.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}},
                           Box{{{0.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}}));

  // point
  constexpr Point point{{1.0, 1.0, 1.0}};
  // point is contained in a box
  static_assert(intersects(point, Box{{{0.0, 0.0, 0.0}}, {{2.0, 2.0, 2.0}}}));
  static_assert(
      !intersects(point, Box{{{-1.0, -1.0, -1.0}}, {{0.0, 0.0, 0.0}}}));
  // point is on a side of a box
  static_assert(intersects(point, Box{{{0.0, 0.0, 0.0}}, {{2.0, 2.0, 1.0}}}));
  // point is a corner of a box
  static_assert(intersects(point, Box{{{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}}));

  // unit cube
  constexpr Box box{{{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}};
  static_assert(intersects(box, box));
  static_assert(!intersects(box, Box{}));
  // smaller box inside
  static_assert(
      intersects(box, Box{{{0.25, 0.25, 0.25}}, {{0.75, 0.75, 0.75}}}));
  // bigger box that contains it
  static_assert(intersects(box, Box{{{-1.0, -1.0, -1.0}}, {{2.0, 2.0, 2.0}}}));
  // couple boxes that do intersect
  static_assert(intersects(box, Box{{{0.5, 0.5, 0.5}}, {{1.5, 1.5, 1.5}}}));
  static_assert(intersects(box, Box{{{-0.5, -0.5, -0.5}}, {{0.5, 0.5, 0.5}}}));
  // couple boxes that do not
  static_assert(
      !intersects(box, Box{{{-2.0, -2.0, -2.0}}, {{-1.0, -1.0, -1.0}}}));
  static_assert(!intersects(box, Box{{{0.0, 0.0, 2.0}}, {{1.0, 1.0, 3.0}}}));
  // boxes intersect if faces touch
  static_assert(intersects(box, Box{{{1.0, 0.0, 0.0}}, {{2.0, 1.0, 1.0}}}));
  static_assert(intersects(box, Box{{{-0.5, -0.5, -0.5}}, {{0.5, 0.0, 0.5}}}));

  // unit sphere
  constexpr Sphere sphere{{{0., 0., 0.}}, 1.};
  BOOST_TEST(intersects(sphere, Box{{{0., 0., 0.}}, {{1., 1., 1.}}}));
  BOOST_TEST(!intersects(sphere, Box{{{1., 2., 3.}}, {{4., 5., 6.}}}));
  BOOST_TEST(intersects(sphere, Point{0., 0.5, 0.5}));
  BOOST_TEST(intersects(sphere, Point{0., 0., 1.0}));
  BOOST_TEST(intersects(Point{-1., 0., 0.}, sphere));
  BOOST_TEST(intersects(Point{-0.6, -0.8, 0.}, sphere));
  BOOST_TEST(!intersects(Point{-0.7, -0.8, 0.}, sphere));

  // triangle
  using Point2 = ArborX::Point<2>;
  constexpr ArborX::Triangle<2> triangle{{{0, 0}}, {{1, 0}}, {{0, 2}}};
  BOOST_TEST(intersects(Point2{{0, 0}}, triangle));
  BOOST_TEST(intersects(Point2{{1, 0}}, triangle));
  BOOST_TEST(intersects(Point2{{0, 2}}, triangle));
  BOOST_TEST(intersects(Point2{{0.5, 0}}, triangle));
  BOOST_TEST(intersects(Point2{{0.5, 1}}, triangle));
  BOOST_TEST(intersects(Point2{{0, 1}}, triangle));
  BOOST_TEST(intersects(Point2{{0.25, 0.5}}, triangle));
  BOOST_TEST(!intersects(Point2{{1, 1}}, triangle));
  BOOST_TEST(!intersects(Point2{{0.5, 1.1}}, triangle));
  BOOST_TEST(!intersects(Point2{{1.1, 0}}, triangle));
  BOOST_TEST(!intersects(Point2{{-0.1, 0}}, triangle));

  // triangle box
  constexpr ArborX::Triangle<3> triangle3{
      {{1, 0, 0}}, {{0, 1, 0}}, {{0, 0, 1}}};
  constexpr Box unit_box{{{0, 0, 0}}, {{1, 1, 1}}};
  BOOST_TEST(intersects(triangle3, Box{{{0., 0., 0.}}, {{1., 1., 1.}}}));
  BOOST_TEST(intersects(triangle3, Box{{{.2, .25, .25}}, {{.4, .3, .5}}}));
  BOOST_TEST(!intersects(triangle3, Box{{{.1, .2, .3}}, {{.2, .3, .4}}}));
  BOOST_TEST(intersects(triangle3, Box{{{0, 0, 0}}, {{.5, .25, .25}}}));
  BOOST_TEST(intersects(
      ArborX::Triangle<3>{{{0, 0, 0}}, {{0, 1, 0}}, {{1, 0, 0}}}, unit_box));
  BOOST_TEST(intersects(
      ArborX::Triangle<3>{{{0, 0, 0}}, {{0, 1, 0}}, {{-1, 0, 0}}}, unit_box));
  BOOST_TEST(intersects(
      ArborX::Triangle<3>{{{.1, .1, .1}}, {{.1, .9, .1}}, {{.9, .1, .1}}},
      unit_box));

  // tetrahedron
  constexpr Tetrahedron tet{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  BOOST_TEST(intersects(Point{0, 0, 0}, tet));
  BOOST_TEST(intersects(Point{1, 0, 0}, tet));
  BOOST_TEST(intersects(Point{0, 1, 0}, tet));
  BOOST_TEST(intersects(Point{0, 0, 1}, tet));
  BOOST_TEST(intersects(Point{0.2, 0.2, 0.1}, tet));
  BOOST_TEST(!intersects(Point{-0.1, 0, 0}, tet));
  BOOST_TEST(!intersects(Point{0, -0.1, 0}, tet));
  BOOST_TEST(!intersects(Point{0, 0, 1.1}, tet));
  BOOST_TEST(!intersects(Point{0.5, 0.5, 0.5}, tet));
  BOOST_TEST(!intersects(Point{-0.5, 0.5, 0.5}, tet));

  // segment
  using Segment2 = ArborX::Experimental::Segment<2>;
  constexpr Segment2 seg{{1, 1}, {2, 2}};
  BOOST_TEST(intersects(Segment2{{2, 2}, {3, 3}}, seg));
  BOOST_TEST(intersects(Segment2{{1.5, 1.5}, {1.7, 1.7}}, seg));
  BOOST_TEST(intersects(Segment2{{0, 0}, {1, 1}}, seg));
  BOOST_TEST(intersects(Segment2{{1, 2}, {2, 1}}, seg));
  BOOST_TEST(intersects(Segment2{{2, 0}, {0, 2}}, seg));
  BOOST_TEST(intersects(Segment2{{1, 3}, {3, 1}}, seg));
  BOOST_TEST(!intersects(Segment2{{0, 0}, {0.9, 0.9}}, seg));
  BOOST_TEST(!intersects(Segment2{{1.1, 1}, {2, 1}}, seg));
  BOOST_TEST(!intersects(Segment2{{1, 0}, {2, 1}}, seg));
  BOOST_TEST(!intersects(Segment2{{1, 3}, {3, 1.1}}, seg));

  constexpr ArborX::Box<2> box2{{{0.0, 0.0}}, {{1.0, 1.0}}};
  BOOST_TEST(intersects(Segment2{{0, 0}, {0, 0}}, box2));
  BOOST_TEST(intersects(Segment2{{-1, 1}, {1, -1}}, box2));
  BOOST_TEST(intersects(Segment2{{-1, 0}, {2, 0}}, box2));
  BOOST_TEST(intersects(Segment2{{-1, 0.5}, {0.5, 0.5}}, box2));
  BOOST_TEST(intersects(Segment2{{0.5, 0.5}, {0.5, 2}}, box2));
  BOOST_TEST(intersects(Segment2{{-1, 2}, {2, -1}}, box2));
  BOOST_TEST(intersects(Segment2{{0.5, 2}, {0.5, -1}}, box2));
  BOOST_TEST(intersects(Segment2{{0.4, 0.4}, {0.6, 0.6}}, box2));
  BOOST_TEST(!intersects(Segment2{{0, -1}, {1, -1}}, box2));
  BOOST_TEST(!intersects(Segment2{{0.5, 1.6}, {2, 0}}, box2));
  BOOST_TEST(intersects(box2, Segment2{{-1, 2}, {2, -1}}));
  BOOST_TEST(!intersects(Segment2{{0.5, 1.6}, {2, 0}}, box2));

  // ellipsoid [2x^2 - 3xy + 2y^2 <= 1]
  using Ellipse = ArborX::Experimental::Ellipsoid<2>;
  constexpr Ellipse ellipse{{1.f, 1.f}, {{2.f, -1.5f}, {-1.5f, 2.f}}};
  BOOST_TEST(intersects(ellipse, Point2{1.f, 1.f}));
  BOOST_TEST(intersects(ellipse, Point2{0.f, 0.f}));
  BOOST_TEST(!intersects(ellipse, Point2{-0.01f, -0.01f}));
  BOOST_TEST(intersects(ellipse, Point2{0.f, 0.5f}));
  BOOST_TEST(!intersects(ellipse, Point2{0.f, 0.6f}));
  BOOST_TEST(intersects(ellipse, Point2{2.f, 2.f}));
  BOOST_TEST(!intersects(ellipse, Point2{0.5f, 1.5f}));
  BOOST_TEST(intersects(ellipse, Point2{1.f, 0.3f}));
  BOOST_TEST(!intersects(ellipse, Point2{1.f, 0.29f}));
  BOOST_TEST(intersects(ellipse, Point2{1.f, 1.69f}));
  BOOST_TEST(intersects(ellipse, Point2{1.f, 1.70f}));

  BOOST_TEST(intersects(ellipse, Segment2{{-1, 1}, {1, -1}}));
  BOOST_TEST(!intersects(ellipse, Segment2{{-1.1, 1}, {1, -1}}));
  BOOST_TEST(intersects(ellipse, Segment2{{0, 0}, {0, 1}}));
  BOOST_TEST(intersects(ellipse, Segment2{{0.5, 0.5}, {1.5, 1.5}}));
  BOOST_TEST(intersects(ellipse, Segment2{{0.0, 1.9}, {3.0, 1.9}}));
  BOOST_TEST(!intersects(ellipse, Segment2{{2.1, 0}, {2.1, 3}}));

  using Box2 = ArborX::Box<2>;
  BOOST_TEST(intersects(ellipse, Box2{{-10, -10}, {10, 10}}));
  BOOST_TEST(intersects(ellipse, Box2{{0.5, 0.5}, {1.0, 1.0}}));
  BOOST_TEST(intersects(ellipse, Box2{{-1, -1}, {0, 0}}));
  BOOST_TEST(intersects(ellipse, Box2{{2, 2}, {3, 3}}));
  BOOST_TEST(intersects(ellipse, Box2{{-1, -1}, {0, 2}}));
  BOOST_TEST(intersects(ellipse, Box2{{-1, -1}, {2, 0}}));
  BOOST_TEST(intersects(ellipse, Box2{{2, 1}, {3, 3}}));
  BOOST_TEST(intersects(ellipse, Box2{{1, 2}, {3, 3}}));
  BOOST_TEST(!intersects(ellipse, Box2{{1.5, 0}, {2, 0.5}}));
  BOOST_TEST(!intersects(ellipse, Box2{{-1, -1}, {-0.1, -0.1}}));
  BOOST_TEST(!intersects(ellipse, Box2{{0, 1.5}, {0.5, 2}}));
  BOOST_TEST(!intersects(ellipse, Box2{{2.1, 2.1}, {3, 3}}));
}

BOOST_AUTO_TEST_CASE(intsersects_segment_triangle)
{
  using ArborX::Details::intersects;
  using Segment = ArborX::Experimental::Segment<3>;
  using Triangle = ArborX::Triangle<3>;

  constexpr Triangle triangle{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

  BOOST_TEST(intersects(Segment{{0, 0, 0}, {1, 1, 1}}, triangle));
  BOOST_TEST(intersects(Segment{{0, 0, 0}, {1, 0, 0}}, triangle));
  BOOST_TEST(intersects(Segment{{0, 0, 0}, {0, 1, 0}}, triangle));
  BOOST_TEST(intersects(Segment{{0, 0, 0}, {0, 0, 1}}, triangle));
  BOOST_TEST(intersects(Segment{{0, 0, 0}, {0.5, 0.25, 0.25}}, triangle));
  BOOST_TEST(!intersects(Segment{{0, 0, 0}, {0.45, 0.25, 0.25}}, triangle));
  BOOST_TEST(!intersects(Segment{{0.9, 0, 0}, {0, 0, 0.9}}, triangle));
}
