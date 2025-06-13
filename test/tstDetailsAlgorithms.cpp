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
#include <ArborX_Point.hpp>
#include <ArborX_Segment.hpp>
#include <ArborX_Sphere.hpp>
#include <ArborX_Tetrahedron.hpp>
#include <ArborX_Triangle.hpp>
#include <algorithms/ArborX_Centroid.hpp>
#include <algorithms/ArborX_Convert.hpp>
#include <algorithms/ArborX_Distance.hpp>
#include <algorithms/ArborX_Equals.hpp>
#include <algorithms/ArborX_Expand.hpp>
#include <algorithms/ArborX_Intersects.hpp>
#include <algorithms/ArborX_Valid.hpp>
#include <kokkos_ext/ArborX_KokkosExtArithmeticTraits.hpp>

#include <boost/mpl/list.hpp>

#define BOOST_TEST_MODULE Geometry
#include <boost/test/unit_test.hpp>

using Point = ArborX::Point<3>;
using Box = ArborX::Box<3>;
using Sphere = ArborX::Sphere<3>;
using Triangle = ArborX::Triangle<3>;
using Tetrahedron = ArborX::ExperimentalHyperGeometry::Tetrahedron<>;

BOOST_AUTO_TEST_CASE(distance_point_point)
{
  using ArborX::Details::distance;
  BOOST_TEST(distance(Point{{1.0, 2.0, 3.0}}, Point{{1.0, 1.0, 1.0}}) ==
             std::sqrt(5.f));
}

BOOST_AUTO_TEST_CASE(distance_point_box)
{
  using ArborX::Details::distance;

  // box is unit cube
  constexpr Box box{{{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}};

  // distance is zero if the point is inside the box
  BOOST_TEST(distance(Point{{0.5, 0.5, 0.5}}, box) == 0.0);
  // or anywhere on the boundary
  BOOST_TEST(distance(Point{{0.0, 0.0, 0.5}}, box) == 0.0);
  // normal projection onto center of one face
  BOOST_TEST(distance(Point{{2.0, 0.5, 0.5}}, box) == 1.0);
  // projection onto edge
  BOOST_TEST(distance(Point{{2.0, 0.75, -1.0}}, box) == std::sqrt(2.f));
  // projection onto corner node
  BOOST_TEST(distance(Point{{-1.0, 2.0, 2.0}}, box) == std::sqrt(3.f));
}

BOOST_AUTO_TEST_CASE(distance_point_sphere)
{
  using ArborX::Details::distance;

  // unit sphere
  constexpr Sphere sphere{{{0., 0., 0.}}, 1.};
  BOOST_TEST(distance(Point{{.5, .5, .5}}, sphere) == 0.);
  BOOST_TEST(distance(Point{{2., 0., 0.}}, sphere) == 1.);
  BOOST_TEST(distance(Point{{1., 1., 1.}}, sphere) == std::sqrt(3.f) - 1.f);
  BOOST_TEST(distance(sphere, Point{2., 0., 0.}) == 1.);
}

BOOST_AUTO_TEST_CASE(distance_point_segment)
{
  using ArborX::Details::distance;

  using ArborX::Point;
  using ArborX::Experimental::Segment;

  constexpr Segment segment0{{0.f, 0.f}, {0.f, 0.f}};
  BOOST_TEST(distance(Point{0.f, 0.f}, segment0) == 0.f);
  BOOST_TEST(distance(Point{1.f, 0.f}, segment0) == 1.f);
  BOOST_TEST(distance(Point{-1.f, 0.f}, segment0) == 1.f);

  constexpr Segment segment1{{0.f, 0.f}, {1.f, 1.f}};
  BOOST_TEST(distance(Point{0.f, 0.f}, segment1) == 0.f);
  BOOST_TEST(distance(Point{1.f, 1.f}, segment1) == 0.f);
  BOOST_TEST(distance(Point{0.5f, 0.5f}, segment1) == 0.f);
  BOOST_TEST(distance(Point{1.0f, 0.f}, segment1) == std::sqrt(0.5f));
  BOOST_TEST(distance(Point{0.f, -1.f}, segment1) == 1.f);
  BOOST_TEST(distance(Point{1.5f, 1.f}, segment1) == 0.5f);
}

BOOST_AUTO_TEST_CASE(distance_point_triangle)
{
  using ArborX::Details::distance;

  /* Zones
         \ 2/
          \/
      5   /\b  6
         /  \
        /    \
    \  /   0  \  /
     \/a______c\/
    1 |    4   | 3
      |        |
  */
  using Point2 = ArborX::Point<2>;
  constexpr ArborX::Triangle triangle2{Point2{-1, 0}, Point2{1, 0},
                                       Point2{0, 1}};

  // vertices
  BOOST_TEST(distance(Point2{-1, 0}, triangle2) == 0);
  BOOST_TEST(distance(Point2{1, 0}, triangle2) == 0);
  BOOST_TEST(distance(Point2{0, 1}, triangle2) == 0);
  // mid edges
  BOOST_TEST(distance(Point2{-0.5f, 0.5f}, triangle2) == 0);
  BOOST_TEST(distance(Point2{0.5f, 0.5f}, triangle2) == 0);
  BOOST_TEST(distance(Point2{0, 0}, triangle2) == 0);
  // inside
  BOOST_TEST(distance(Point2{0, 0.5f}, triangle2) == 0);
  // outside zone 1
  BOOST_TEST(distance(Point2{-2, 0}, triangle2) == 1);
  // outside zone 2
  BOOST_TEST(distance(Point2{0, 2}, triangle2) == 1);
  // outside zone 3
  BOOST_TEST(distance(Point2{2, 0}, triangle2) == 1);
  // outside zone 4
  BOOST_TEST(distance(Point2{0, -1}, triangle2) == 1);
  // outside zone 5
  BOOST_TEST(distance(Point2{-1, 1}, triangle2) == std::sqrt(2.f) / 2);
  // outside zone 6
  BOOST_TEST(distance(Point2{1, 1}, triangle2) == std::sqrt(2.f) / 2);

  using Point3 = ArborX::Point<3>;
  constexpr ArborX::Triangle triangle3{Point3{1, 0, 0}, Point3{0, 1, 0},
                                       Point3{0, 0, 0}};

  // same plane
  BOOST_TEST(distance(Point3{2, 0, 0}, triangle3) == 1);
  BOOST_TEST(distance(Point3{0.5f, -0.5f, 0}, triangle3) == 0.5f);
  BOOST_TEST(distance(Point3{-0.5f, 0.5f, 0}, triangle3) == 0.5f);
  // projected to inside
  BOOST_TEST(distance(Point3{0, 0, 1}, triangle3) == 1);
  BOOST_TEST(distance(Point3{1, 0, -1}, triangle3) == 1);
  BOOST_TEST(distance(Point3{0, 1, 2}, triangle3) == 2);
  BOOST_TEST(distance(Point3{0.25f, 0.25f, 2.f}, triangle3) == 2);
  // projected outside
  BOOST_TEST(distance(Point3{-1, 0, 1}, triangle3) == std::sqrt(2.f));
  BOOST_TEST(distance(Point3{0, -1, -1}, triangle3) == std::sqrt(2.f));
  BOOST_TEST(distance(Point3{2, -1, -1}, triangle3) == std::sqrt(3.f));

  constexpr ArborX::Triangle triangle3_2{Point3{0, 0, 0}, Point3{0, 1, 0},
                                         Point3{0, 0, 1}};
  BOOST_TEST(distance(Point3{-1, 0, 1}, triangle3_2) == 1);
  BOOST_TEST(distance(Point3{0, -1, -1}, triangle3_2) == std::sqrt(2.f));
  BOOST_TEST(distance(Point3{1, -1, -1}, triangle3_2) == std::sqrt(3.f));
}

BOOST_AUTO_TEST_CASE(distance_box_box)
{
  using ArborX::Details::distance;
  namespace KokkosExt = ArborX::Details::KokkosExt;

  constexpr Box unit_box{{{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}};

  // distance to self
  BOOST_TEST(distance(unit_box, unit_box) == 0);
  // distance to another unit box translated along one axis
  BOOST_TEST(distance(unit_box, Box{{{2, 0, 0}}, {{3, 1, 1}}}) == 1);
  BOOST_TEST(distance(unit_box, Box{{{0, -3, 0}}, {{1, -2, 1}}}) == 2);
  BOOST_TEST(distance(unit_box, Box{{{0, 0, 4}}, {{1, 1, 5}}}) == 3);
  // distance to another unit box translated along a plane
  BOOST_TEST(distance(unit_box, Box{{{-4, -4, 0}}, {{-3, -3, 1}}}) ==
             3 * std::sqrt(2.f));
  BOOST_TEST(distance(unit_box, Box{{{0, -2, 3}}, {{1, -1, 4}}}) ==
             std::sqrt(5.f));
  BOOST_TEST(distance(unit_box, Box{{{5, 0, 7}}, {{6, 1, 8}}}) ==
             2 * std::sqrt(13.f));

  // distance to another box that contains the box
  BOOST_TEST(distance(unit_box, Box{{{-1, -2, -3}}, {{4, 5, 6}}}) == 0);
  // distance to another box within the unit box
  BOOST_TEST(distance(unit_box, Box{{{.1, .2, .3}}, {{.4, .5, .6}}}) == 0);
  // distance to another box that overlaps with the unit box
  BOOST_TEST(distance(unit_box, Box{{{.1, .2, .3}}, {{4, 5, 6}}}) == 0);

  // distance to empty boxes
  auto infinity = KokkosExt::ArithmeticTraits::infinity<float>::value;
  BOOST_TEST(distance(unit_box, Box{}) == infinity);
  BOOST_TEST(distance(Box{}, unit_box) == infinity);
  BOOST_TEST(distance(Box{}, Box{}) == infinity);
}

BOOST_AUTO_TEST_CASE(distance_sphere_box)
{
  using ArborX::Details::distance;
  namespace KokkosExt = ArborX::Details::KokkosExt;

  auto infinity = KokkosExt::ArithmeticTraits::infinity<float>::value;

  // unit sphere
  constexpr Sphere sphere{{{0., 0., 0.}}, 1.};
  // distance between a sphere and a box no intersection
  BOOST_TEST(distance(sphere, Box{{2.0, 3.0, 4.0}, {2.5, 3.5, 4.5}}) ==
             std::sqrt(29.f) - 1.f);
  // distance between a sphere and a box with intersection
  BOOST_TEST(distance(sphere, Box{{0.5, 0.5, 0.5}, {2.5, 3.5, 4.5}}) == 0.f);
  // distance between a sphere included in a box and that box
  BOOST_TEST(distance(sphere, Box{{-2., -2., -2.}, {2., 2., 2.}}) == 0.f);
  // distance between a sphere and a box included in that sphere
  BOOST_TEST(distance(sphere, Box{{0., 0., 0.}, {0.1, 0.2, 0.3}}) == 0.f);
  // distance to empty box
  BOOST_TEST(distance(sphere, Box{}) == infinity);
}

BOOST_AUTO_TEST_CASE(intersects)
{
  using ArborX::Details::intersects;

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

BOOST_AUTO_TEST_CASE(equals)
{
  using ArborX::Details::equals;
  // points
  static_assert(equals(Point{{0., 0., 0.}}, {{0., 0., 0.}}));
  static_assert(!equals(Point{{0., 0., 0.}}, {{1., 1., 1.}}));
  // boxes
  static_assert(equals(Box{{{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}},
                       {{{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}}));
  static_assert(!equals(Box{{{0.0, 0.0, 0.0}}, {{1.0, 0.0, 1.0}}},
                        {{{-1.0, -1.0, -1.0}}, {{1.0, 1.0, 1.0}}}));
  // spheres
  static_assert(equals(Sphere{{{0., 0., 0.}}, 1.}, {{{0., 0., 0.}}, 1.}));
  static_assert(!equals(Sphere{{{0., 0., 0.}}, 1.}, {{{0., 1., 2.}}, 1.}));
  static_assert(!equals(Sphere{{{0., 0., 0.}}, 1.}, {{{0., 0., 0.}}, 2.}));
}

BOOST_AUTO_TEST_CASE(expand)
{
  using ArborX::Details::equals;
  using ArborX::Details::expand;
  Box box;

  // expand box with points
  expand(box, Point{{0.0, 0.0, 0.0}});
  BOOST_TEST(equals(box, Box{{{0.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}}));
  expand(box, Point{{1.0, 1.0, 1.0}});
  BOOST_TEST(equals(box, Box{{{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}}));
  expand(box, Point{{0.25, 0.75, 0.25}});
  BOOST_TEST(equals(box, Box{{{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}}));
  expand(box, Point{{-1.0, -1.0, -1.0}});
  BOOST_TEST(equals(box, Box{{{-1.0, -1.0, -1.0}}, {{1.0, 1.0, 1.0}}}));

  // expand box with boxes
  expand(box, Box{{{0.25, 0.25, 0.25}}, {{0.75, 0.75, 0.75}}});
  BOOST_TEST(equals(box, {{{-1.0, -1.0, -1.0}}, {{1.0, 1.0, 1.0}}}));
  expand(box, Box{{{10.0, 10.0, 10.0}}, {{11.0, 11.0, 11.0}}});
  BOOST_TEST(equals(box, {{{-1.0, -1.0, -1.0}}, {{11.0, 11.0, 11.0}}}));

  // expand box with spheres
  expand(box, Sphere{{{0., 1., 2.}}, 3.});
  BOOST_TEST(equals(box, Box{{{-3., -2., -1.}}, {{11., 11., 11.}}}));
  expand(box, Sphere{{{0., 0., 0.}}, 1.});
  BOOST_TEST(equals(box, Box{{{-3., -2., -1.}}, {{11., 11., 11.}}}));
  expand(box, Sphere{{{0., 0., 0.}}, 24.});
  BOOST_TEST(equals(box, Box{{{-24., -24., -24.}}, {{24., 24., 24.}}}));

  // expand box with triangles
  expand(box, Triangle{{{-1, -1, 0}}, {{2, 2, 2}}, {{1, 1, 0}}});
  BOOST_TEST(equals(box, Box{{{-24., -24., -24.}}, {{24., 24., 24.}}}));
  expand(box, Triangle{{{0, 0, 0}}, {{48, 0, 0}}, {{0, 48, 0}}});
  BOOST_TEST(equals(box, Box{{{-24., -24., -24.}}, {{48., 48., 24.}}}));

  // expand box with tetrahedrons
  box = Box{};
  expand(box, Tetrahedron{{-1, -2, 3}, {1, 3, 2}, {0, 3, 7}, {-5, 4, 7}});
  BOOST_TEST(equals(box, Box{{-5, -2, 2}, {1, 4, 7}}));
  expand(box, Tetrahedron{{-3, -5, 2}, {2, 6, -1}, {3, 2, 3}, {5, 8, -3}});
  BOOST_TEST(equals(box, Box{{-5, -5, -3}, {5, 8, 7}}));

  // expand box with segments
  using ArborX::Experimental::Segment;
  box = Box{};
  expand(box, Segment{{0.f, 0.f, 0.f}, {1.f, 1.f, 1.f}});
  BOOST_TEST(equals(box, Box{{0, 0, 0}, {1, 1, 1}}));
  expand(box, Segment{{-1.f, 3.0f, 0.0f}, {2.f, 1.f, 0.f}});
  BOOST_TEST(equals(box, Box{{-1, 0, 0}, {2, 3, 1}}));
}

BOOST_AUTO_TEST_CASE(convert)
{
  using ArborX::Point;
  using ArborX::Details::convert;
  using ArborX::Details::equals;
  BOOST_TEST(equals(convert<Point<2, double>>(Point{3.f, 2.f}), Point{3., 2.}));
  BOOST_TEST(
      equals(convert<Point<2, float>>(Point{3.f, 2.f}), Point{3.f, 2.f}));
  BOOST_TEST(
      !equals(convert<Point<2, float>>(Point{3.f, 2.f}), Point{2.f, 2.f}));
}

BOOST_AUTO_TEST_CASE(centroid)
{
  using ArborX::Details::equals;
  using ArborX::Details::returnCentroid;

  Box box{{{-10.0, 0.0, 10.0}}, {{0.0, 10.0, 20.0}}};
  auto center = returnCentroid(box);
  BOOST_TEST(center[0] == -5.0);
  BOOST_TEST(center[1] == 5.0);
  BOOST_TEST(center[2] == 15.0);

  Triangle tri2{{{-1, -0.5}}, {{1, -0.5}}, {{0, 1}}};
  auto tri2_center = returnCentroid(tri2);
  BOOST_TEST(tri2_center[0] == 0);
  BOOST_TEST(tri2_center[1] == 0);

  Triangle tri3{{{0, 0, -2}}, {{3, 0, 1}}, {{0, 3, 1}}};
  auto tri3_center = returnCentroid(tri3);
  BOOST_TEST(tri3_center[0] == 1);
  BOOST_TEST(tri3_center[1] == 1);
  BOOST_TEST(tri3_center[2] == 0);

  Tetrahedron tet{{{0, 0, -2}}, {{-4, 3, 4}}, {{1, 9, 5}}, {{-1, 0, 1}}};
  auto tet_center = returnCentroid(tet);
  BOOST_TEST(tet_center[0] == -1);
  BOOST_TEST(tet_center[1] == 3);
  BOOST_TEST(tet_center[2] == 2);

  using ArborX::Experimental::Segment;
  Segment segment{{-1.f, -1.f}, {3.f, 3.f}};
  auto seg_center = returnCentroid(segment);
  BOOST_TEST(equals(seg_center, ArborX::Point{1.f, 1.f}));

  using ArborX::Experimental::Ellipsoid;
  Ellipsoid ellipse{{1.f, 0.f}, {{2.f, 1.f}, {1.f, 2.f}}};
  auto ell_center = returnCentroid(ellipse);
  BOOST_TEST(equals(ell_center, ArborX::Point{1.f, 0.f}));
}

BOOST_AUTO_TEST_CASE(is_valid)
{
  using ArborX::Details::isValid;

  auto infty = std::numeric_limits<float>::infinity();

  BOOST_TEST(isValid(Point{{1., 2., 3.}}));
  BOOST_TEST(!isValid(Point{{0., infty, 0.}}));

  BOOST_TEST(isValid(Box{{{1., 2., 3.}}, {{4., 5., 6.}}}));
  BOOST_TEST(!isValid(Box{{{0., 0., 0.}}, {{0., 0., 0.}}}));
  BOOST_TEST(!isValid(Box{{{0., 0., -infty}}, {{0., 0., 0.}}}));
  BOOST_TEST(!isValid(Box{{{0., 0., 0.}}, {{+infty, 0., 0.}}}));
  BOOST_TEST(!isValid(Box{}));
  BOOST_TEST(!isValid(Box{{{1., 1., 1.}}, {{0., 0., 0.}}}));
  BOOST_TEST(!isValid(Box{{{0., 0., 1.}}, {{0., 0., 0.}}}));
  BOOST_TEST(!isValid(Box{{{0., 0., 0.}}, {{-1, 0., 0.}}}));

  BOOST_TEST(isValid(Sphere{{{1., 2., 3.}}, 4.}));
  BOOST_TEST(isValid(Sphere{{{0., 0., 0.}}, 0.}));
  BOOST_TEST(!isValid(Sphere{{{1., 2., 3.}}, -1.}));
  BOOST_TEST(!isValid(Sphere{{{0., 0., 0.}}, +infty}));
  BOOST_TEST(!isValid(Sphere{{{0., -infty, 0.}}, +1.}));
  BOOST_TEST(isValid(Sphere{}));
}
