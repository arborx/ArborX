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
#include <algorithms/ArborX_Equals.hpp>
#include <algorithms/ArborX_Expand.hpp>

#include <boost/test/unit_test.hpp>

using CoordinatesList = std::tuple<float, double>;

BOOST_AUTO_TEST_CASE_TEMPLATE(expand, Coordinate, CoordinatesList)
{
  using ArborX::Details::equals;
  using ArborX::Details::expand;

  using Point = ArborX::Point<3, Coordinate>;
  using Box = ArborX::Box<3, Coordinate>;
  using Sphere = ArborX::Sphere<3, Coordinate>;
  using Triangle = ArborX::Triangle<3, Coordinate>;
  using Segment = ArborX::Experimental::Segment<3, Coordinate>;
  using Tetrahedron =
      ArborX::ExperimentalHyperGeometry::Tetrahedron<Coordinate>;

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
  box = Box{};
  expand(box, Segment{{0.f, 0.f, 0.f}, {1.f, 1.f, 1.f}});
  BOOST_TEST(equals(box, Box{{0, 0, 0}, {1, 1, 1}}));
  expand(box, Segment{{-1.f, 3.0f, 0.0f}, {2.f, 1.f, 0.f}});
  BOOST_TEST(equals(box, Box{{-1, 0, 0}, {2, 3, 1}}));
}
