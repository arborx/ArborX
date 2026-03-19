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
#include <algorithms/ArborX_ClosestPoint.hpp>
#include <algorithms/ArborX_Equals.hpp>

#include <boost/test/unit_test.hpp>

using CoordinatesList = std::tuple<float, double>;

BOOST_AUTO_TEST_CASE_TEMPLATE(closest_point_point_box, Coordinate,
                              CoordinatesList)
{
  using ArborX::Details::equals;
  using ArborX::Experimental::closestPoint;
  using Point = ArborX::Point<2, Coordinate>;
  using Box = ArborX::Box<2, Coordinate>;

  Box box{{-1, -1}, {2, 1}};
  BOOST_TEST(equals(closestPoint(Point{0, 0}, box), {0, 0}));
  BOOST_TEST(equals(closestPoint(Point{-1, -1}, box), {-1, -1}));
  BOOST_TEST(equals(closestPoint(Point{-1.5, 0}, box), {-1, 0}));
  BOOST_TEST(equals(closestPoint(Point{-1.5, -1.5}, box), {-1, -1}));
  BOOST_TEST(equals(closestPoint(Point{2.5, -1.5}, box), {2, -1}));
  BOOST_TEST(equals(closestPoint(Point{0, 2}, box), {0, 1}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(closest_point_point_triangle, Coordinate,
                              CoordinatesList)
{
  using ArborX::Details::equals;
  using ArborX::Experimental::closestPoint;
  using Point2 = ArborX::Point<2, Coordinate>;
  using Point3 = ArborX::Point<3, Coordinate>;

  ArborX::Triangle<2, Coordinate> tri2{{1, 1}, {2, 1}, {1, 2}};

  BOOST_TEST(equals(closestPoint(Point2{0, 0}, tri2), Point2{1, 1}));
  BOOST_TEST(equals(closestPoint(Point2{0, 1.5}, tri2), Point2{1, 1.5}));
  BOOST_TEST(equals(closestPoint(Point2{0, 2.5}, tri2), Point2{1, 2}));
  BOOST_TEST(equals(closestPoint(Point2{2, 2}, tri2), Point2{1.5, 1.5}));
  BOOST_TEST(equals(closestPoint(Point2{3, 0}, tri2), Point2{2, 1}));
  BOOST_TEST(equals(closestPoint(Point2{1.5, 0}, tri2), Point2{1.5, 1}));

  ArborX::Triangle<3, Coordinate> tri3{{1, 1, 1}, {2, 1, 1}, {1, 2, 1}};

  BOOST_TEST(equals(closestPoint(Point3{0, 0, 0}, tri3), Point3{1, 1, 1}));
  BOOST_TEST(equals(closestPoint(Point3{0, 1.5, 1}, tri3), Point3{1, 1.5, 1}));
  BOOST_TEST(equals(closestPoint(Point3{0, 1.5, 2}, tri3), Point3{1, 1.5, 1}));
  BOOST_TEST(
      equals(closestPoint(Point3{1.25, 1.25, 0}, tri3), Point3{1.25, 1.25, 1}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(closest_point_point_segment, Coordinate,
                              CoordinatesList)
{
  using ArborX::Details::equals;
  using ArborX::Experimental::closestPoint;
  using Point2 = ArborX::Point<2, Coordinate>;
  using Point3 = ArborX::Point<3, Coordinate>;

  ArborX::Experimental::Segment<2, Coordinate> seg2{{1, 1}, {2, 2}};

  BOOST_TEST(equals(closestPoint(Point2{0, 0}, seg2), {1, 1}));
  BOOST_TEST(equals(closestPoint(Point2{1, 1}, seg2), {1, 1}));
  BOOST_TEST(equals(closestPoint(Point2{0, 1}, seg2), {1, 1}));
  BOOST_TEST(equals(closestPoint(Point2{1, 0}, seg2), {1, 1}));
  BOOST_TEST(equals(closestPoint(Point2{3, 3}, seg2), {2, 2}));
  BOOST_TEST(equals(closestPoint(Point2{1, 2}, seg2), {1.5, 1.5}));
  BOOST_TEST(equals(closestPoint(Point2{2, 1}, seg2), {1.5, 1.5}));

  ArborX::Experimental::Segment<3, Coordinate> seg3{{1, 1, 1}, {2, 2, 2}};
  BOOST_TEST(equals(closestPoint(Point3{0, 0, -1}, seg3), {1, 1, 1}));
  BOOST_TEST(equals(closestPoint(Point3{3, 3, 4}, seg3), {2, 2, 2}));
  BOOST_TEST(equals(closestPoint(Point3{1, 2, 1}, seg3),
                    {(Coordinate)4 / 3, (Coordinate)4 / 3, (Coordinate)4 / 3}));
}
