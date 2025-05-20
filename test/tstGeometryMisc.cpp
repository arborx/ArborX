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
#include <algorithms/ArborX_Convert.hpp>
#include <algorithms/ArborX_Equals.hpp>
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
