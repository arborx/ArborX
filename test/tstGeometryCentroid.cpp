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

BOOST_AUTO_TEST_CASE(centroid)
{
  using ArborX::Details::equals;
  using ArborX::Details::returnCentroid;

  ArborX::Box box{{-10.f, 0.f, 10.f}, {0.f, 10.f, 20.f}};
  auto center = returnCentroid(box);
  BOOST_TEST(center[0] == -5.0);
  BOOST_TEST(center[1] == 5.0);
  BOOST_TEST(center[2] == 15.0);

  ArborX::Triangle tri2{{-1.f, -0.5f}, {1.f, -0.5f}, {0.f, 1.f}};
  BOOST_TEST(equals(returnCentroid(tri2), ArborX::Point{0.f, 0.f}));

  ArborX::Triangle tri3{{0.f, 0.f, -2.f}, {3.f, 0.f, 1.f}, {0.f, 3.f, 1.f}};
  BOOST_TEST(equals(returnCentroid(tri3), ArborX::Point{1.f, 1.f, 0.f}));

  ArborX::ExperimentalHyperGeometry::Tetrahedron tet{
      {0.f, 0.f, -2.f}, {-4.f, 3.f, 4.f}, {1.f, 9.f, 5.f}, {-1.f, 0.f, 1.f}};
  BOOST_TEST(equals(returnCentroid(tet), ArborX::Point{-1.f, 3.f, 2.f}));

  ArborX::Experimental::Segment segment{{-1.f, -1.f}, {3.f, 3.f}};
  BOOST_TEST(equals(returnCentroid(segment), ArborX::Point{1.f, 1.f}));

  ArborX::Experimental::Ellipsoid ellipse{{1.f, 0.f}, {{2.f, 1.f}, {1.f, 2.f}}};
  BOOST_TEST(equals(returnCentroid(ellipse), ArborX::Point{1.f, 0.f}));
}
