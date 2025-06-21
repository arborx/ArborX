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
#include <ArborX_Segment.hpp>
#include <ArborX_Sphere.hpp>
#include <ArborX_Tetrahedron.hpp>
#include <ArborX_Triangle.hpp>
#include <algorithms/ArborX_Distance.hpp>

#include <boost/test/unit_test.hpp>

using CoordinatesList = std::tuple<float, double>;

BOOST_AUTO_TEST_CASE_TEMPLATE(distance_point_point, Coordinate, CoordinatesList)
{
  using ArborX::Details::distance;
  using Point = ArborX::Point<3, Coordinate>;

  BOOST_TEST(distance(Point{1, 2, 3}, Point{1, 1, 1}) ==
             std::sqrt((Coordinate)5));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(distance_point_box, Coordinate, CoordinatesList)
{
  using ArborX::Details::distance;
  using Point = ArborX::Point<3, Coordinate>;
  using Box = ArborX::Box<3, Coordinate>;

  // box is unit cube
  constexpr Box box{{{0, 0, 0}}, {{1, 1, 1}}};

  // distance is zero if the point is inside the box
  BOOST_TEST(distance(Point{0.5, 0.5, 0.5}, box) == 0);
  // or anywhere on the boundary
  BOOST_TEST(distance(Point{0.0, 0.0, 0.5}, box) == 0);
  // normal projection onto center of one face
  BOOST_TEST(distance(Point{2.0, 0.5, 0.5}, box) == 1);
  // projection onto edge
  BOOST_TEST(distance(Point{2.0, 0.75, -1.0}, box) == std::sqrt((Coordinate)2));
  // projection onto corner node
  BOOST_TEST(distance(Point{{-1.0, 2.0, 2.0}}, box) ==
             std::sqrt((Coordinate)3));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(distance_point_sphere, Coordinate,
                              CoordinatesList)
{
  using ArborX::Details::distance;
  using Point = ArborX::Point<3, Coordinate>;
  using Sphere = ArborX::Sphere<3, Coordinate>;

  // unit sphere
  constexpr Sphere sphere{{{0, 0, 0}}, 1};
  BOOST_TEST(distance(Point{{.5, .5, .5}}, sphere) == 0);
  BOOST_TEST(distance(Point{{2, 0, 0}}, sphere) == 1);
  BOOST_TEST(distance(Point{{1, 1, 1}}, sphere) ==
             std::sqrt((Coordinate)3) - 1);

  BOOST_TEST(distance(sphere, Point{2, 0, 0}) == 1);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(distance_point_segment, Coordinate,
                              CoordinatesList)
{
  using ArborX::Details::distance;

  using Point = ArborX::Point<2, Coordinate>;
  using Segment = ArborX::Experimental::Segment<2, Coordinate>;

  constexpr Segment segment0{{0, 0}, {0, 0}};
  BOOST_TEST(distance(Point{0, 0}, segment0) == 0);
  BOOST_TEST(distance(Point{1, 0}, segment0) == 1);
  BOOST_TEST(distance(Point{-1, 0}, segment0) == 1);

  constexpr Segment segment1{{0, 0}, {1, 1}};
  BOOST_TEST(distance(Point{0, 0}, segment1) == 0);
  BOOST_TEST(distance(Point{1, 1}, segment1) == 0);
  BOOST_TEST(distance(Point{0.5, 0.5}, segment1) == 0);
  BOOST_TEST(distance(Point{1.0f, 0}, segment1) == std::sqrt((Coordinate)0.5));
  BOOST_TEST(distance(Point{0, -1}, segment1) == 1);
  BOOST_TEST(distance(Point{1.5f, 1}, segment1) == (Coordinate)0.5);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(distance_point_triangle, Coordinate,
                              CoordinatesList)
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
  using Point2 = ArborX::Point<2, Coordinate>;
  constexpr ArborX::Triangle triangle2{Point2{-1, 0}, Point2{1, 0},
                                       Point2{0, 1}};

  // vertices
  BOOST_TEST(distance(Point2{-1, 0}, triangle2) == 0);
  BOOST_TEST(distance(Point2{1, 0}, triangle2) == 0);
  BOOST_TEST(distance(Point2{0, 1}, triangle2) == 0);
  // mid edges
  BOOST_TEST(distance(Point2{-0.5, 0.5}, triangle2) == 0);
  BOOST_TEST(distance(Point2{0.5, 0.5}, triangle2) == 0);
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
  BOOST_TEST(distance(Point2{-1, 1}, triangle2) ==
             std::sqrt((Coordinate)2) / 2);
  // outside zone 6
  BOOST_TEST(distance(Point2{1, 1}, triangle2) == std::sqrt((Coordinate)2) / 2);

  using Point3 = ArborX::Point<3, Coordinate>;
  constexpr ArborX::Triangle<3, Coordinate> triangle3{
      Point3{1, 0, 0}, Point3{0, 1, 0}, Point3{0, 0, 0}};

  // same plane
  BOOST_TEST(distance(Point3{2, 0, 0}, triangle3) == 1);
  BOOST_TEST(distance(Point3{0.5, -0.5, 0}, triangle3) == (Coordinate)0.5);
  BOOST_TEST(distance(Point3{-0.5, 0.5, 0}, triangle3) == (Coordinate)0.5);
  // projected to inside
  BOOST_TEST(distance(Point3{0, 0, 1}, triangle3) == 1);
  BOOST_TEST(distance(Point3{1, 0, -1}, triangle3) == 1);
  BOOST_TEST(distance(Point3{0, 1, 2}, triangle3) == 2);
  BOOST_TEST(distance(Point3{0.25, 0.25, 2.}, triangle3) == 2);
  // projected outside
  BOOST_TEST(distance(Point3{-1, 0, 1}, triangle3) == std::sqrt((Coordinate)2));
  BOOST_TEST(distance(Point3{0, -1, -1}, triangle3) ==
             std::sqrt((Coordinate)2));
  BOOST_TEST(distance(Point3{2, -1, -1}, triangle3) ==
             std::sqrt((Coordinate)3));

  constexpr ArborX::Triangle triangle3_2{Point3{0, 0, 0}, Point3{0, 1, 0},
                                         Point3{0, 0, 1}};
  BOOST_TEST(distance(Point3{-1, 0, 1}, triangle3_2) == 1);
  BOOST_TEST(distance(Point3{0, -1, -1}, triangle3_2) ==
             std::sqrt((Coordinate)2));
  BOOST_TEST(distance(Point3{1, -1, -1}, triangle3_2) ==
             std::sqrt((Coordinate)3));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(distance_point_tetrahedron, Coordinate,
                              CoordinatesList)
{
  using ArborX::Details::distance;

  using Point = ArborX::Point<3, Coordinate>;
  using Tetrahedron =
      ArborX::ExperimentalHyperGeometry::Tetrahedron<Coordinate>;

  constexpr Tetrahedron tet{Point{0, 0, 0}, Point{1, 0, 0}, Point{0, 1, 0},
                            Point{0, 0, 1}};

  // vertices
  BOOST_TEST(distance(Point{0, 0, 0}, tet) == 0);
  BOOST_TEST(distance(Point{1, 0, 0}, tet) == 0);
  BOOST_TEST(distance(Point{0, 1, 0}, tet) == 0);
  BOOST_TEST(distance(Point{0, 0, 1}, tet) == 0);

  // inside
  BOOST_TEST(distance(Point{0.4, 0.4, 0.1}, tet) == 0);
  BOOST_TEST(distance(Point{0.8, 0.05, 0.05}, tet) == 0);

  // same plane as some side
  BOOST_TEST(distance(Point{2, 0, 0}, tet) == 1);
  BOOST_TEST(distance(Point{0.5, -0.5, 0}, tet) == (Coordinate)0.5);
  BOOST_TEST(distance(Point{-0.5, 0.5, 0}, tet) == (Coordinate)0.5);
  BOOST_TEST(distance(Point{0, 0, 2}, tet) == 1);
  BOOST_TEST(distance(Point{0, -0.5, 0}, tet) == (Coordinate).5);

  // outside
  BOOST_TEST(distance(Point{-1, -1, -1}, tet) == std::sqrt((Coordinate)3));
  BOOST_TEST(distance(Point{-1, -1, 2}, tet) == std::sqrt((Coordinate)3));
  BOOST_TEST(distance(Point{1.5, 1.5, -1}, tet) == std::sqrt((Coordinate)3));
  BOOST_TEST(distance(Point{1.5, 1.5, 0.5}, tet) == (Coordinate)1.5);

  Point v[4] = {Point{0, 0, 0}, Point{1, 0, 0}, Point{0, 1, 0},
                Point{-2, 0, 1}};
  Point p{-1.5, 0, -0.5};
  BOOST_TEST(distance(p, Tetrahedron{v[0], v[1], v[2], v[3]}) ==
             std::sqrt((Coordinate)1.25));
  BOOST_TEST(distance(p, Tetrahedron{v[1], v[2], v[3], v[0]}) ==
             std::sqrt((Coordinate)1.25));
  BOOST_TEST(distance(p, Tetrahedron{v[2], v[3], v[0], v[1]}) ==
             std::sqrt((Coordinate)1.25));
  BOOST_TEST(distance(p, Tetrahedron{v[3], v[0], v[1], v[2]}) ==
             std::sqrt((Coordinate)1.25));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(distance_box_box, Coordinate, CoordinatesList)
{
  using ArborX::Details::distance;
  namespace KokkosExt = ArborX::Details::KokkosExt;
  using Box = ArborX::Box<3, Coordinate>;

  constexpr Box unit_box{{{0, 0, 0}}, {{1, 1, 1}}};

  // distance to self
  BOOST_TEST(distance(unit_box, unit_box) == 0);
  // distance to another unit box translated along one axis
  BOOST_TEST(distance(unit_box, Box{{{2, 0, 0}}, {{3, 1, 1}}}) == 1);
  BOOST_TEST(distance(unit_box, Box{{{0, -3, 0}}, {{1, -2, 1}}}) == 2);
  BOOST_TEST(distance(unit_box, Box{{{0, 0, 4}}, {{1, 1, 5}}}) == 3);
  // distance to another unit box translated along a plane
  BOOST_TEST(distance(unit_box, Box{{{-4, -4, 0}}, {{-3, -3, 1}}}) ==
             std::sqrt((Coordinate)18));
  BOOST_TEST(distance(unit_box, Box{{{0, -2, 3}}, {{1, -1, 4}}}) ==
             std::sqrt((Coordinate)5));
  BOOST_TEST(distance(unit_box, Box{{{5, 0, 7}}, {{6, 1, 8}}}) ==
             std::sqrt((Coordinate)52));

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

BOOST_AUTO_TEST_CASE_TEMPLATE(distance_sphere_box, Coordinate, CoordinatesList)
{
  using ArborX::Details::distance;
  namespace KokkosExt = ArborX::Details::KokkosExt;
  using Box = ArborX::Box<3, Coordinate>;
  using Sphere = ArborX::Sphere<3, Coordinate>;

  auto infinity = KokkosExt::ArithmeticTraits::infinity<Coordinate>::value;

  // unit sphere
  constexpr Sphere sphere{{{0., 0., 0.}}, 1.};
  Box box;
  // distance between a sphere and a box no intersection
  box = Box{{2.0, 3.0, 4.0}, {2.5, 3.5, 4.5}};
  BOOST_TEST(distance(sphere, box) == std::sqrt((Coordinate)29) - 1);
  BOOST_TEST(distance(box, sphere) == std::sqrt((Coordinate)29) - 1);
  // distance between a sphere and a box with intersection
  BOOST_TEST(distance(sphere, Box{{0.5, 0.5, 0.5}, {2.5, 3.5, 4.5}}) == 0);
  // distance between a sphere included in a box and that box
  BOOST_TEST(distance(sphere, Box{{-2., -2., -2.}, {2., 2., 2.}}) == 0);
  // distance between a sphere and a box included in that sphere
  box = Box{{0., 0., 0.}, {0.1, 0.2, 0.3}};
  BOOST_TEST(distance(sphere, box) == 0);
  BOOST_TEST(distance(box, sphere) == 0);
  // distance to empty box
  BOOST_TEST(distance(sphere, Box{}) == infinity);
}
