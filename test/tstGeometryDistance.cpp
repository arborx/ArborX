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

BOOST_AUTO_TEST_CASE(distance_point_point)
{
  using ArborX::Details::distance;
  using Point = ArborX::Point<3>;

  BOOST_TEST(distance(Point{{1.0, 2.0, 3.0}}, Point{{1.0, 1.0, 1.0}}) ==
             std::sqrt(5.f));
}

BOOST_AUTO_TEST_CASE(distance_point_box)
{
  using ArborX::Details::distance;
  using Point = ArborX::Point<3>;
  using Box = ArborX::Box<3>;

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
  using Point = ArborX::Point<3>;
  using Sphere = ArborX::Sphere<3>;

  // unit sphere
  constexpr Sphere sphere{{{0., 0., 0.}}, 1.};
  BOOST_TEST(distance(Point{{.5, .5, .5}}, sphere) == 0.);
  BOOST_TEST(distance(Point{{2., 0., 0.}}, sphere) == 1.);
  BOOST_TEST(distance(Point{{1., 1., 1.}}, sphere) == std::sqrt(3.f) - 1.f);

  BOOST_TEST(distance(sphere, Point{2, 0, 0}) == 1);
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

BOOST_AUTO_TEST_CASE(distance_point_tetrahedron)
{
  using ArborX::Details::distance;

  using Point = ArborX::Point<3>;
  using Tetrahedron = ArborX::ExperimentalHyperGeometry::Tetrahedron<>;

  constexpr Tetrahedron tet{Point{0, 0, 0}, Point{1, 0, 0}, Point{0, 1, 0},
                            Point{0, 0, 1}};

  // vertices
  BOOST_TEST(distance(Point{0, 0, 0}, tet) == 0);
  BOOST_TEST(distance(Point{1, 0, 0}, tet) == 0);
  BOOST_TEST(distance(Point{0, 1, 0}, tet) == 0);
  BOOST_TEST(distance(Point{0, 0, 1}, tet) == 0);

  // inside
  BOOST_TEST(distance(Point{0.4f, 0.4f, 0.1f}, tet) == 0);
  BOOST_TEST(distance(Point{0.8f, 0.05f, 0.05f}, tet) == 0);

  // same plane as some side
  BOOST_TEST(distance(Point{2, 0, 0}, tet) == 1);
  BOOST_TEST(distance(Point{0.5f, -0.5f, 0}, tet) == 0.5f);
  BOOST_TEST(distance(Point{-0.5f, 0.5f, 0}, tet) == 0.5f);
  BOOST_TEST(distance(Point{0, 0, 2}, tet) == 1);
  BOOST_TEST(distance(Point{0, -0.5f, 0}, tet) == 0.5f);

  // outside
  BOOST_TEST(distance(Point{-1, -1, -1}, tet) == std::sqrt(3.f));
  BOOST_TEST(distance(Point{-1, -1, 2}, tet) == std::sqrt(3.f));
  BOOST_TEST(distance(Point{1.5f, 1.5f, -1}, tet) == std::sqrt(3.f));
  BOOST_TEST(distance(Point{1.5f, 1.5f, 0.5f}, tet) == 1.5f);

  Point v[4] = {Point{0, 0, 0}, Point{1, 0, 0}, Point{0, 1, 0},
                Point{-2, 0, 1}};
  Point p{-1.5f, 0, -0.5f};
  BOOST_TEST(distance(p, Tetrahedron{v[0], v[1], v[2], v[3]}) ==
             std::sqrt(1.25f));
  BOOST_TEST(distance(p, Tetrahedron{v[1], v[2], v[3], v[0]}) ==
             std::sqrt(1.25f));
  BOOST_TEST(distance(p, Tetrahedron{v[2], v[3], v[0], v[1]}) ==
             std::sqrt(1.25f));
  BOOST_TEST(distance(p, Tetrahedron{v[3], v[0], v[1], v[2]}) ==
             std::sqrt(1.25f));
}

BOOST_AUTO_TEST_CASE(distance_box_box)
{
  using ArborX::Details::distance;
  namespace KokkosExt = ArborX::Details::KokkosExt;
  using Box = ArborX::Box<3>;

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
  using Box = ArborX::Box<3>;
  using Sphere = ArborX::Sphere<3>;

  auto infinity = KokkosExt::ArithmeticTraits::infinity<float>::value;

  // unit sphere
  constexpr Sphere sphere{{{0., 0., 0.}}, 1.};
  Box box;
  // distance between a sphere and a box no intersection
  box = Box{{2.0, 3.0, 4.0}, {2.5, 3.5, 4.5}};
  BOOST_TEST(distance(sphere, box) == std::sqrt(29.f) - 1.f);
  BOOST_TEST(distance(box, sphere) == std::sqrt(29.f) - 1.f);
  // distance between a sphere and a box with intersection
  BOOST_TEST(distance(sphere, Box{{0.5, 0.5, 0.5}, {2.5, 3.5, 4.5}}) == 0.f);
  // distance between a sphere included in a box and that box
  BOOST_TEST(distance(sphere, Box{{-2., -2., -2.}, {2., 2., 2.}}) == 0.f);
  // distance between a sphere and a box included in that sphere
  box = Box{{0., 0., 0.}, {0.1, 0.2, 0.3}};
  BOOST_TEST(distance(sphere, box) == 0.f);
  BOOST_TEST(distance(box, sphere) == 0.f);
  // distance to empty box
  BOOST_TEST(distance(sphere, Box{}) == infinity);
}
