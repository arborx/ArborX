/****************************************************************************
 * Copyright (c) 2012-2019 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include "ArborX_BoostGeometryAdapters.hpp"
#include <ArborX_DetailsAlgorithms.hpp>

#include <boost/test/unit_test.hpp>

#define BOOST_TEST_MODULE BoostGeometryAdapters

namespace bg = boost::geometry;
namespace details = ArborX::Details;
// Conveniently importing Point and Box in ArborX::Details:: namespace
// and declaring type aliases within boost::geometry:: so that we are able to
// just use details:: and bg:: to specify what geometry or algorithm we mean.
namespace ArborX
{
namespace Details
{
using ArborX::Box;
using ArborX::Point;
} // namespace Details
} // namespace ArborX
namespace boost
{
namespace geometry
{
using Point = model::point<double, 3, cs::cartesian>;
using Box = model::box<Point>;
} // namespace geometry
} // namespace boost

BOOST_AUTO_TEST_CASE(equals)
{
  details::Point point = {{1., 2., 3.}};
  assert(details::equals(point, {{1., 2., 3.}}));
  assert(!details::equals(point, {{0., 0., 0.}}));
  assert(bg::equals(point, bg::make<details::Point>(1., 2., 3.)));
  assert(!bg::equals(point, bg::make<details::Point>(4., 5., 6.)));
  assert(bg::equals(point, bg::make<bg::Point>(1., 2., 3.)));
  assert(!bg::equals(point, bg::make<bg::Point>(0., 0., 0.)));

  details::Box box = {{{1., 2., 3.}}, {{4., 5., 6.}}};
  assert(details::equals(box, {{{1., 2., 3.}}, {{4., 5., 6.}}}));
  assert(!details::equals(box, {{{0., 0., 0.}}, {{1., 1., 1.}}}));
  assert(bg::equals(box, details::Box{{{1., 2., 3.}}, {{4., 5., 6.}}}));
  assert(!bg::equals(box, details::Box{{{0., 0., 0.}}, {{1., 1., 1.}}}));
  assert(bg::equals(box, bg::Box(bg::make<bg::Point>(1., 2., 3.),
                                 bg::make<bg::Point>(4., 5., 6.))));
  assert(!bg::equals(box, bg::Box(bg::make<bg::Point>(0., 0., 0.),
                                  bg::make<bg::Point>(1., 1., 1.))));
  // Now just for fun compare the ArborX box to a Boost.Geometry box composed
  // of ArborX points.
  assert(bg::equals(
      box, bg::model::box<details::Point>({{1., 2., 3.}}, {{4., 5., 6.}})));
  assert(!bg::equals(
      box, bg::model::box<details::Point>({{0., 0., 0.}}, {{0., 0., 0.}})));
}

BOOST_AUTO_TEST_CASE(distance)
{
  // NOTE unsure if should test for floating point equality here
  details::Point a = {{0., 0., 0.}};
  details::Point b = {{0., 1., 0.}};
  assert(details::distance(a, b) == 1.);
  assert(bg::distance(a, b) == 1.);

  std::tie(a, b) = std::make_pair<details::Point, details::Point>(
      {{0., 0., 0.}}, {{1., 1., 1.}});
  assert(details::distance(a, b) == std::sqrt(3.));
  assert(bg::distance(a, b) == std::sqrt(3.));

  assert(details::distance(a, a) == 0.);
  assert(bg::distance(a, a) == 0.);

  details::Box unit_box = {{{0., 0., 0.}}, {{1., 1., 1.}}};
  details::Point p = {{.5, .5, .5}};
  // NOTE ArborX has no overload distance( Box, Point )
  assert(details::distance(p, unit_box) == 0.);
  // assert( details::distance( unit_box, p ) == 0. );
  assert(bg::distance(p, unit_box) == 0.);
  assert(bg::distance(unit_box, p) == 0.);

  p = {{-1., -1., -1.}};
  assert(details::distance(p, unit_box) == std::sqrt(3.));
  assert(bg::distance(p, unit_box) == std::sqrt(3.));

  p = {{-1., .5, -1.}};
  assert(details::distance(p, unit_box) == std::sqrt(2.));
  assert(bg::distance(p, unit_box) == std::sqrt(2.));

  p = {{-1., .5, .5}};
  assert(details::distance(p, unit_box) == 1.);
  assert(bg::distance(p, unit_box) == 1.);
}

BOOST_AUTO_TEST_CASE(expand)
{
  using details::equals;
  details::Box box;
  // NOTE even though not considered valid, default constructed ArborX box can
  // be expanded using Boost.Geometry algorithm.
  assert(!bg::is_valid(box));
  bg::expand(box, details::Point{{0., 0., 0.}});
  details::expand(box, {{1., 1., 1.}});
  assert(equals(box, {{{0., 0., 0.}}, {{1., 1., 1.}}}));
  bg::expand(box, details::Box{{{1., 2., 3.}}, {{4., 5., 6.}}});
  details::expand(box, {{{-1., -2., -3.}}, {{0., 0., 0.}}});
  assert(equals(box, {{{-1., -2., -3.}}, {{4., 5., 6.}}}));
}

BOOST_AUTO_TEST_CASE(centroid)
{
  using details::equals;

  // Interestingly enough, even though for Boost.Geometry, the ArborX default
  // constructed "empty" box is not valid, it will still calculate its
  // centroid.  Admittedly, the result (centroid at origin) is garbage :)
  details::Box empty_box = {};
  assert(!bg::is_valid(empty_box));
  assert(
      equals(bg::return_centroid<details::Point>(empty_box), {{0., 0., 0.}}));
  assert(equals(details::returnCentroid(empty_box), {{0., 0., 0.}}));

  details::Box unit_box = {{{0., 0., 0.}}, {{1., 1., 1.}}};
  assert(equals(bg::return_centroid<details::Point>(unit_box), {{.5, .5, .5}}));
  assert(equals(details::returnCentroid(unit_box), {{.5, .5, .5}}));

  details::Point a_point = {{1., 2., 3.}};
  assert(equals(bg::return_centroid<details::Point>(a_point), a_point));
  assert(equals(details::returnCentroid(a_point), a_point));
}

BOOST_AUTO_TEST_CASE(is_valid)
{
  // NOTE "empty" box is considered as valid in ArborX but it is
  // not according to Boost.Geometry
  details::Box empty_box = {};
  assert(details::isValid(empty_box));
  std::string message;
  assert(!bg::is_valid(empty_box, message));
  assert(message == "Box has corners in wrong order");

  // Same issue with infinitesimal box around a point (here the origin)
  details::Box a_box = {{{0., 0., 0.}}, {{0., 0., 0.}}};
  assert(details::isValid(a_box));
  assert(!bg::is_valid(a_box, message));
  assert(message == "Geometry has wrong topological dimension");

  details::Box unit_box = {{{0., 0., 0.}}, {{1., 1., 1.}}};
  assert(details::isValid(unit_box));
  assert(bg::is_valid(unit_box));

  details::Box invalid_box = {{{1., 2., 3.}}, {{4., NAN, 6.}}};
  assert(!details::isValid(invalid_box));
  assert(!bg::is_valid(invalid_box, message));
  assert(message == "Geometry has point(s) with invalid coordinate(s)");

  details::Point a_point = {{1., 2., 3.}};
  assert(details::isValid(a_point));
  assert(bg::is_valid(a_point));

  auto const infty = std::numeric_limits<double>::infinity();
  details::Point invalid_point = {{infty, 1.41, 3.14}};
  assert(!details::isValid(invalid_point));
  assert(!bg::is_valid(invalid_point, message));
  assert(message == "Geometry has point(s) with invalid coordinate(s)");

  // Also Boost.Geometry has a is_empty() algorithm but it has a different
  // meaning, it checks whether a geometry is an empty set and always returns
  // false for a point or a box.
  assert(!bg::is_empty(empty_box));
  assert(!bg::is_empty(a_box));
  assert(!bg::is_empty(unit_box));
}

BOOST_AUTO_TEST_CASE(intersects)
{
  details::Box const unit_box = {{{0., 0., 0.}}, {{1., 1., 1.}}};

  // self-intersection
  assert(details::intersects(unit_box, unit_box));
  assert(bg::intersects(unit_box, unit_box));

  // share a corner
  details::Box other_box = {{{1., 1., 1.}}, {{2., 2., 2.}}};
  assert(details::intersects(unit_box, other_box));
  assert(bg::intersects(unit_box, other_box));

  // share an edge
  other_box = {{{1., 0., 1.}}, {{2., 1., 2.}}};
  assert(details::intersects(unit_box, other_box));
  assert(bg::intersects(unit_box, other_box));

  // share a face
  other_box = {{{0., -1., 0.}}, {{1., 0., 1.}}};
  assert(details::intersects(unit_box, other_box));
  assert(bg::intersects(unit_box, other_box));

  // contains the other box
  other_box = {{{.3, .3, .3}}, {{.6, .6, .6}}};
  assert(details::intersects(unit_box, other_box));
  assert(bg::intersects(unit_box, other_box));

  // within the other box
  other_box = {{{-1., -1., -1.}}, {{2., 2., 2.}}};
  assert(details::intersects(unit_box, other_box));
  assert(bg::intersects(unit_box, other_box));

  // overlapping
  other_box = {{{.5, .5, .5}}, {{2., 2., 2.}}};
  assert(details::intersects(unit_box, other_box));
  assert(bg::intersects(unit_box, other_box));

  // disjoint
  other_box = {{{1., 2., 3.}}, {{4., 5., 6.}}};
  assert(!details::intersects(unit_box, other_box));
  assert(!bg::intersects(unit_box, other_box));
}
