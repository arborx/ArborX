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
#include <ArborX_DetailsAlgorithms.hpp>

#include <boost/test/unit_test.hpp>

#define BOOST_TEST_MODULE DetailsAlgorithms

namespace details = ArborX::Details;

BOOST_AUTO_TEST_CASE(distance)
{
  BOOST_TEST(
      details::distance({{1.0, 2.0, 3.0}}, {{1.0, 1.0, 1.0}}).to_double() ==
      std::sqrt(5.0));

  // box is unit cube
  ArborX::Box box = {{{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}};

  // distance is zero if the point is inside the box
  BOOST_TEST(details::distance({{0.5, 0.5, 0.5}}, box).to_double() == 0.0);
  // or anywhere on the boundary
  BOOST_TEST(details::distance({{0.0, 0.0, 0.5}}, box).to_double() == 0.0);
  // normal projection onto center of one face
  BOOST_TEST(details::distance({{2.0, 0.5, 0.5}}, box).to_double() == 1.0);
  // projection onto edge
  BOOST_TEST(details::distance({{2.0, 0.75, -1.0}}, box).to_double() ==
             std::sqrt(2.0));
  // projection onto corner node
  BOOST_TEST(details::distance({{-1.0, 2.0, 2.0}}, box).to_double() ==
             std::sqrt(3.0));

  // unit sphere
  ArborX::Sphere sphere = {{{0., 0., 0.}}, 1.};
  BOOST_TEST(details::distance({{.5, .5, .5}}, sphere).to_double() == 0.);

  BOOST_TEST(details::distance({{2., 0., 0.}}, sphere).to_double() == 1.);

  BOOST_TEST(details::distance({{1., 1., 1.}}, sphere).to_double() ==
             std::sqrt(3.) - 1.);
}

BOOST_AUTO_TEST_CASE(overlaps)
{
  ArborX::Box box;
  // uninitialized box does not even overlap with itself
  BOOST_TEST(!details::intersects(box, box));
  // box with zero extent does
  box = {{{0.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}};
  BOOST_TEST(details::intersects(box, box));
  BOOST_TEST(!details::intersects(box, ArborX::Box()));
  // overlap with box that contains it
  BOOST_TEST(
      details::intersects(box, {{{-1.0, -1.0, -1.0}}, {{1.0, 1.0, 1.0}}}));
  // does not overlap with some other box
  BOOST_TEST(!details::intersects(box, {{{1.0, 1.0, 1.0}}, {{2.0, 2.0, 2.0}}}));
  // overlap when only touches another
  BOOST_TEST(details::intersects(box, {{{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}}));
  // unit cube
  box = {{{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}};
  BOOST_TEST(details::intersects(box, box));
  BOOST_TEST(!details::intersects(box, ArborX::Box()));
  // smaller box inside
  BOOST_TEST(
      details::intersects(box, {{{0.25, 0.25, 0.25}}, {{0.75, 0.75, 0.75}}}));
  // bigger box that contains it
  BOOST_TEST(
      details::intersects(box, {{{-1.0, -1.0, -1.0}}, {{2.0, 2.0, 2.0}}}));
  // couple boxes that do overlap
  BOOST_TEST(details::intersects(box, {{{0.5, 0.5, 0.5}}, {{1.5, 1.5, 1.5}}}));
  BOOST_TEST(
      details::intersects(box, {{{-0.5, -0.5, -0.5}}, {{0.5, 0.5, 0.5}}}));
  // couple boxes that do not
  BOOST_TEST(
      !details::intersects(box, {{{-2.0, -2.0, -2.0}}, {{-1.0, -1.0, -1.0}}}));
  BOOST_TEST(!details::intersects(box, {{{0.0, 0.0, 2.0}}, {{1.0, 1.0, 3.0}}}));
  // boxes overlap if faces touch
  BOOST_TEST(details::intersects(box, {{{1.0, 0.0, 0.0}}, {{2.0, 1.0, 1.0}}}));
  BOOST_TEST(
      details::intersects(box, {{{-0.5, -0.5, -0.5}}, {{0.5, 0.0, 0.5}}}));
}

BOOST_AUTO_TEST_CASE(intersects)
{
  ArborX::Sphere sphere = {{{0., 0., 0.}}, 1.};
  BOOST_TEST(details::intersects(sphere, {{{0., 0., 0.}}, {{1., 1., 1.}}}));
  BOOST_TEST(!details::intersects(sphere, {{{1., 2., 3.}}, {{4., 5., 6.}}}));
}

BOOST_AUTO_TEST_CASE(equals)
{
  // points
  BOOST_TEST(details::equals({{0., 0., 0.}}, {{0., 0., 0.}}));
  BOOST_TEST(!details::equals({{0., 0., 0.}}, {{1., 1., 1.}}));
  // boxes
  BOOST_TEST(details::equals({{{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}},
                             {{{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}}));
  BOOST_TEST(!details::equals({{{0.0, 0.0, 0.0}}, {{1.0, 0.0, 1.0}}},
                              {{{-1.0, -1.0, -1.0}}, {{1.0, 1.0, 1.0}}}));
  // spheres
  BOOST_TEST(details::equals({{{0., 0., 0.}}, 1.}, {{{0., 0., 0.}}, 1.}));
  BOOST_TEST(!details::equals({{{0., 0., 0.}}, 1.}, {{{0., 1., 2.}}, 1.}));
  BOOST_TEST(!details::equals({{{0., 0., 0.}}, 1.}, {{{0., 0., 0.}}, 2.}));
}

BOOST_AUTO_TEST_CASE(expand)
{
  ArborX::Box box;

  // expand box with points
  details::expand(box, {{0.0, 0.0, 0.0}});
  BOOST_TEST(details::equals(box, {{{0.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}}));
  details::expand(box, {{1.0, 1.0, 1.0}});
  BOOST_TEST(details::equals(box, {{{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}}));
  details::expand(box, {{0.25, 0.75, 0.25}});
  BOOST_TEST(details::equals(box, {{{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}}));
  details::expand(box, {{-1.0, -1.0, -1.0}});
  BOOST_TEST(details::equals(box, {{{-1.0, -1.0, -1.0}}, {{1.0, 1.0, 1.0}}}));

  // expand box with boxes
  details::expand(box, {{{0.25, 0.25, 0.25}}, {{0.75, 0.75, 0.75}}});
  BOOST_TEST(details::equals(box, {{{-1.0, -1.0, -1.0}}, {{1.0, 1.0, 1.0}}}));
  details::expand(box, {{{10.0, 10.0, 10.0}}, {{11.0, 11.0, 11.0}}});
  BOOST_TEST(
      details::equals(box, {{{-1.0, -1.0, -1.0}}, {{11.0, 11.0, 11.0}}}));

  // expand box with spheres
  details::expand(box, {{{0., 1., 2.}}, 3.});
  BOOST_TEST(details::equals(box, {{{-3., -2., -1.}}, {{11., 11., 11.}}}));
  details::expand(box, {{{0., 0., 0.}}, 1.});
  BOOST_TEST(details::equals(box, {{{-3., -2., -1.}}, {{11., 11., 11.}}}));
  details::expand(box, {{{0., 0., 0.}}, 24.});
  BOOST_TEST(details::equals(box, {{{-24., -24., -24.}}, {{24., 24., 24.}}}));
}

BOOST_AUTO_TEST_CASE(centroid)
{
  ArborX::Box box = {{{-10.0, 0.0, 10.0}}, {{0.0, 10.0, 20.0}}};
  ArborX::Point center;
  details::centroid(box, center);
  BOOST_TEST(center[0] == -5.0);
  BOOST_TEST(center[1] == 5.0);
  BOOST_TEST(center[2] == 15.0);
}

BOOST_AUTO_TEST_CASE(is_valid)
{
  using ArborX::Box;
  using ArborX::Point;
  using ArborX::Sphere;
  using ArborX::Details::isValid;

  auto const infty = std::numeric_limits<double>::infinity();

  BOOST_TEST(isValid(Point{{1., 2., 3.}}));
  BOOST_TEST(!isValid(Point{{0., infty, 0.}}));

  BOOST_TEST(isValid(Box{{{1., 2., 3.}}, {{4., 5., 6.}}}));
  BOOST_TEST(isValid(Box{{{0., 0., 0.}}, {{0., 0., 0.}}}));
  BOOST_TEST(!isValid(Box{{{0., 0., -infty}}, {{0., 0., 0.}}}));
  BOOST_TEST(!isValid(Box{{{0., 0., 0.}}, {{+infty, 0., 0.}}}));
  BOOST_TEST(isValid(Box{}));

  BOOST_TEST(isValid(Sphere{{{1., 2., 3.}}, 4.}));
  BOOST_TEST(isValid(Sphere{{{0., 0., 0.}}, 0.}));
  BOOST_TEST(!isValid(Sphere{{{1., 2., 3.}}, -1.}));
  BOOST_TEST(!isValid(Sphere{{{0., 0., 0.}}, +infty}));
  BOOST_TEST(!isValid(Sphere{{{0., -infty, 0.}}, +1.}));
  BOOST_TEST(isValid(Sphere{}));
}
