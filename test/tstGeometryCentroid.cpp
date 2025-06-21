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
#include <ArborX_Tetrahedron.hpp>
#include <ArborX_Triangle.hpp>
#include <algorithms/ArborX_Centroid.hpp>
#include <algorithms/ArborX_Equals.hpp>

#include <boost/test/unit_test.hpp>

using CoordinatesList = std::tuple<float, double>;

BOOST_AUTO_TEST_CASE_TEMPLATE(centroid, Coordinate, CoordinatesList)
{
  using ArborX::Details::equals;
  using ArborX::Details::returnCentroid;

  ArborX::Box<3, Coordinate> box{{-10.f, 0.f, 10.f}, {0.f, 10.f, 20.f}};
  auto center = returnCentroid(box);
  BOOST_TEST(center[0] == -5.0);
  BOOST_TEST(center[1] == 5.0);
  BOOST_TEST(center[2] == 15.0);

  ArborX::Triangle<2, Coordinate> tri2{{-1, -0.5}, {1, -0.5}, {0, 1}};
  BOOST_TEST(equals(returnCentroid(tri2), {0, 0}));

  ArborX::Triangle<3, Coordinate> tri3{{0, 0, -2}, {3, 0, 1}, {0, 3, 1}};
  BOOST_TEST(equals(returnCentroid(tri3), {1, 1, 0}));

  ArborX::ExperimentalHyperGeometry::Tetrahedron<Coordinate> tet{
      {0, 0, -2}, {-4, 3, 4}, {1, 9, 5}, {-1, 0, 1}};
  BOOST_TEST(equals(returnCentroid(tet), {-1, 3, 2}));

  ArborX::Experimental::Segment<2, Coordinate> segment{{-1, -1}, {3, 3}};
  BOOST_TEST(equals(returnCentroid(segment), {1, 1}));

  ArborX::Experimental::Ellipsoid<2, Coordinate> ellipse{{1, 0},
                                                         {{2, 1}, {1, 2}}};
  BOOST_TEST(equals(returnCentroid(ellipse), {1, 0}));
}
