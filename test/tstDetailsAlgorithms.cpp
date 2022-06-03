/****************************************************************************
 * Copyright (c) 2017-2022 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX_DetailsAlgorithms.hpp>
#include <ArborX_DetailsKokkosExtArithmeticTraits.hpp>

#define BOOST_TEST_MODULE Geometry
#include <boost/test/unit_test.hpp>

// NOTE: message is not required anymore with C++17
#define STATIC_ASSERT(cond) static_assert(cond, "")

using ArborX::Box;
using ArborX::Point;
using ArborX::Sphere;

BOOST_AUTO_TEST_CASE(distance)
{
  using ArborX::Details::distance;
  BOOST_TEST(distance({{1.0, 2.0, 3.0}}, {{1.0, 1.0, 1.0}}) == std::sqrt(5.f));

  // box is unit cube
  constexpr Box box{{{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}};

  // distance is zero if the point is inside the box
  BOOST_TEST(distance({{0.5, 0.5, 0.5}}, box) == 0.0);
  // or anywhere on the boundary
  BOOST_TEST(distance({{0.0, 0.0, 0.5}}, box) == 0.0);
  // normal projection onto center of one face
  BOOST_TEST(distance({{2.0, 0.5, 0.5}}, box) == 1.0);
  // projection onto edge
  BOOST_TEST(distance({{2.0, 0.75, -1.0}}, box) == std::sqrt(2.f));
  // projection onto corner node
  BOOST_TEST(distance({{-1.0, 2.0, 2.0}}, box) == std::sqrt(3.f));

  // unit sphere
  constexpr Sphere sphere{{{0., 0., 0.}}, 1.};
  BOOST_TEST(distance({{.5, .5, .5}}, sphere) == 0.);
  BOOST_TEST(distance({{2., 0., 0.}}, sphere) == 1.);
  BOOST_TEST(distance({{1., 1., 1.}}, sphere) == std::sqrt(3.f) - 1.f);
}

BOOST_AUTO_TEST_CASE(distance_box_box)
{
  using ArborX::Details::distance;

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
  STATIC_ASSERT(!intersects(Box{}, {{{1.0, 2.0, 3.0}}, {{4.0, 5.0, 6.0}}}));
  // uninitialized box does not even intersect with itself
  STATIC_ASSERT(!intersects(Box{}, Box{}));
  // box with zero extent does
  STATIC_ASSERT(intersects(Box{{{0.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}},
                           Box{{{0.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}}));

  // point
  constexpr Point point{{1.0, 1.0, 1.0}};
  // point is contained in a box
  STATIC_ASSERT(intersects(point, {{{0.0, 0.0, 0.0}}, {{2.0, 2.0, 2.0}}}));
  STATIC_ASSERT(!intersects(point, {{{-1.0, -1.0, -1.0}}, {{0.0, 0.0, 0.0}}}));
  // point is on a side of a box
  STATIC_ASSERT(intersects(point, {{{0.0, 0.0, 0.0}}, {{2.0, 2.0, 1.0}}}));
  // point is a corner of a box
  STATIC_ASSERT(intersects(point, {{{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}}));

  // unit cube
  constexpr Box box{{{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}};
  STATIC_ASSERT(intersects(box, box));
  STATIC_ASSERT(!intersects(box, Box{}));
  // smaller box inside
  STATIC_ASSERT(intersects(box, {{{0.25, 0.25, 0.25}}, {{0.75, 0.75, 0.75}}}));
  // bigger box that contains it
  STATIC_ASSERT(intersects(box, {{{-1.0, -1.0, -1.0}}, {{2.0, 2.0, 2.0}}}));
  // couple boxes that do intersect
  STATIC_ASSERT(intersects(box, {{{0.5, 0.5, 0.5}}, {{1.5, 1.5, 1.5}}}));
  STATIC_ASSERT(intersects(box, {{{-0.5, -0.5, -0.5}}, {{0.5, 0.5, 0.5}}}));
  // couple boxes that do not
  STATIC_ASSERT(!intersects(box, {{{-2.0, -2.0, -2.0}}, {{-1.0, -1.0, -1.0}}}));
  STATIC_ASSERT(!intersects(box, {{{0.0, 0.0, 2.0}}, {{1.0, 1.0, 3.0}}}));
  // boxes intersect if faces touch
  STATIC_ASSERT(intersects(box, {{{1.0, 0.0, 0.0}}, {{2.0, 1.0, 1.0}}}));
  STATIC_ASSERT(intersects(box, {{{-0.5, -0.5, -0.5}}, {{0.5, 0.0, 0.5}}}));

  // unit sphere
  constexpr Sphere sphere{{{0., 0., 0.}}, 1.};
  BOOST_TEST(intersects(sphere, Box{{{0., 0., 0.}}, {{1., 1., 1.}}}));
  BOOST_TEST(!intersects(sphere, Box{{{1., 2., 3.}}, {{4., 5., 6.}}}));
  BOOST_TEST(intersects(sphere, Point{0., 0.5, 0.5}));
  BOOST_TEST(intersects(sphere, Point{0., 0., 1.0}));
  BOOST_TEST(intersects(Point{-1., 0., 0.}, sphere));
  BOOST_TEST(intersects(Point{-0.6, -0.8, 0.}, sphere));
  BOOST_TEST(!intersects(Point{-0.7, -0.8, 0.}, sphere));
}

BOOST_AUTO_TEST_CASE(equals)
{
  using ArborX::Details::equals;
  // points
  STATIC_ASSERT(equals(Point{{0., 0., 0.}}, {{0., 0., 0.}}));
  STATIC_ASSERT(!equals(Point{{0., 0., 0.}}, {{1., 1., 1.}}));
  // boxes
  STATIC_ASSERT(equals(Box{{{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}},
                       {{{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}}));
  STATIC_ASSERT(!equals(Box{{{0.0, 0.0, 0.0}}, {{1.0, 0.0, 1.0}}},
                        {{{-1.0, -1.0, -1.0}}, {{1.0, 1.0, 1.0}}}));
  // spheres
  STATIC_ASSERT(equals({{{0., 0., 0.}}, 1.}, {{{0., 0., 0.}}, 1.}));
  STATIC_ASSERT(!equals({{{0., 0., 0.}}, 1.}, {{{0., 1., 2.}}, 1.}));
  STATIC_ASSERT(!equals({{{0., 0., 0.}}, 1.}, {{{0., 0., 0.}}, 2.}));
}

BOOST_AUTO_TEST_CASE(expand)
{
  using ArborX::Details::equals;
  using ArborX::Details::expand;
  Box box;

  // expand box with points
  expand(box, {{0.0, 0.0, 0.0}});
  BOOST_TEST(equals(box, {{{0.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}}));
  expand(box, {{1.0, 1.0, 1.0}});
  BOOST_TEST(equals(box, {{{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}}));
  expand(box, {{0.25, 0.75, 0.25}});
  BOOST_TEST(equals(box, {{{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}}));
  expand(box, {{-1.0, -1.0, -1.0}});
  BOOST_TEST(equals(box, {{{-1.0, -1.0, -1.0}}, {{1.0, 1.0, 1.0}}}));

  // expand box with boxes
  expand(box, {{{0.25, 0.25, 0.25}}, {{0.75, 0.75, 0.75}}});
  BOOST_TEST(equals(box, {{{-1.0, -1.0, -1.0}}, {{1.0, 1.0, 1.0}}}));
  expand(box, {{{10.0, 10.0, 10.0}}, {{11.0, 11.0, 11.0}}});
  BOOST_TEST(equals(box, {{{-1.0, -1.0, -1.0}}, {{11.0, 11.0, 11.0}}}));

  // expand box with spheres
  expand(box, {{{0., 1., 2.}}, 3.});
  BOOST_TEST(equals(box, {{{-3., -2., -1.}}, {{11., 11., 11.}}}));
  expand(box, {{{0., 0., 0.}}, 1.});
  BOOST_TEST(equals(box, {{{-3., -2., -1.}}, {{11., 11., 11.}}}));
  expand(box, {{{0., 0., 0.}}, 24.});
  BOOST_TEST(equals(box, {{{-24., -24., -24.}}, {{24., 24., 24.}}}));
}

BOOST_AUTO_TEST_CASE(centroid)
{
  using ArborX::Details::centroid;
  Box box{{{-10.0, 0.0, 10.0}}, {{0.0, 10.0, 20.0}}};
  Point center;
  centroid(box, center);
  BOOST_TEST(center[0] == -5.0);
  BOOST_TEST(center[1] == 5.0);
  BOOST_TEST(center[2] == 15.0);
}

BOOST_AUTO_TEST_CASE(is_valid)
{
  using ArborX::Details::isValid;

  auto infty = std::numeric_limits<double>::infinity();

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
