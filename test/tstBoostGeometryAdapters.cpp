/****************************************************************************
 * Copyright (c) 2012-2019 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include "DTK_BoostGeometryAdapters.hpp"

#include <DTK_DetailsAlgorithms.hpp>

#include <boost/test/unit_test.hpp>

#define BOOST_TEST_MODULE BoostGeometryAdapters

namespace bg = boost::geometry;
namespace dtk = DataTransferKit::Details;
// Conveniently importing Point and Box in DataTransferKit::Details:: namespace
// and declaring type aliases within boost::geometry:: so that we are able to
// just use dtk:: and bg:: to specify what geometry or algorithm we mean.
namespace DataTransferKit
{
namespace Details
{
using DataTransferKit::Box;
using DataTransferKit::Point;
} // namespace Details
} // namespace DataTransferKit
namespace boost
{
namespace geometry
{
using Point = model::point<double, 3, cs::cartesian>;
using Box = model::box<Point>;
} // namespace geometry
} // namespace boost

BOOST_AUTO_TEST_CASE( equals )
{
    dtk::Point point = {{1., 2., 3.}};
    BOOST_TEST( dtk::equals( point, {{1., 2., 3.}} ) );
    BOOST_TEST( !dtk::equals( point, {{0., 0., 0.}} ) );
    BOOST_TEST( bg::equals( point, bg::make<dtk::Point>( 1., 2., 3. ) ) );
    BOOST_TEST( !bg::equals( point, bg::make<dtk::Point>( 4., 5., 6. ) ) );
    BOOST_TEST( bg::equals( point, bg::make<bg::Point>( 1., 2., 3. ) ) );
    BOOST_TEST( !bg::equals( point, bg::make<bg::Point>( 0., 0., 0. ) ) );

    dtk::Box box = {{{1., 2., 3.}}, {{4., 5., 6.}}};
    BOOST_TEST( dtk::equals( box, {{{1., 2., 3.}}, {{4., 5., 6.}}} ) );
    BOOST_TEST( !dtk::equals( box, {{{0., 0., 0.}}, {{1., 1., 1.}}} ) );
    BOOST_TEST( bg::equals( box, dtk::Box{{{1., 2., 3.}}, {{4., 5., 6.}}} ) );
    BOOST_TEST( !bg::equals( box, dtk::Box{{{0., 0., 0.}}, {{1., 1., 1.}}} ) );
    BOOST_TEST(
        bg::equals( box, bg::Box( bg::make<bg::Point>( 1., 2., 3. ),
                                  bg::make<bg::Point>( 4., 5., 6. ) ) ) );
    BOOST_TEST(
        !bg::equals( box, bg::Box( bg::make<bg::Point>( 0., 0., 0. ),
                                   bg::make<bg::Point>( 1., 1., 1. ) ) ) );
    // Now just for fun compare the DTK box to a Boost.Geometry box composed of
    // DTK points.
    BOOST_TEST( bg::equals(
        box, bg::model::box<dtk::Point>( {{1., 2., 3.}}, {{4., 5., 6.}} ) ) );
    BOOST_TEST( !bg::equals(
        box, bg::model::box<dtk::Point>( {{0., 0., 0.}}, {{0., 0., 0.}} ) ) );
}

BOOST_AUTO_TEST_CASE( distance )
{
    // NOTE unsure if should test for floating point equality here
    dtk::Point a = {{0., 0., 0.}};
    dtk::Point b = {{0., 1., 0.}};
    BOOST_TEST( dtk::distance( a, b ) == 1. );
    BOOST_TEST( bg::distance( a, b ) == 1. );

    std::tie( a, b ) = std::make_pair<dtk::Point, dtk::Point>( {{0., 0., 0.}},
                                                               {{1., 1., 1.}} );
    BOOST_TEST( dtk::distance( a, b ) == std::sqrt( 3. ) );
    BOOST_TEST( bg::distance( a, b ) == std::sqrt( 3. ) );

    BOOST_TEST( dtk::distance( a, a ) == 0. );
    BOOST_TEST( bg::distance( a, a ) == 0. );

    dtk::Box unit_box = {{{0., 0., 0.}}, {{1., 1., 1.}}};
    dtk::Point p = {{.5, .5, .5}};
    // NOTE DTK has no overload distance( Box, Point )
    BOOST_TEST( dtk::distance( p, unit_box ) == 0. );
    // BOOST_TEST( dtk::distance( unit_box, p ) == 0. );
    BOOST_TEST( bg::distance( p, unit_box ) == 0. );
    BOOST_TEST( bg::distance( unit_box, p ) == 0. );

    p = {{-1., -1., -1.}};
    BOOST_TEST( dtk::distance( p, unit_box ) == std::sqrt( 3. ) );
    BOOST_TEST( bg::distance( p, unit_box ) == std::sqrt( 3. ) );

    p = {{-1., .5, -1.}};
    BOOST_TEST( dtk::distance( p, unit_box ) == std::sqrt( 2. ) );
    BOOST_TEST( bg::distance( p, unit_box ) == std::sqrt( 2. ) );

    p = {{-1., .5, .5}};
    BOOST_TEST( dtk::distance( p, unit_box ) == 1. );
    BOOST_TEST( bg::distance( p, unit_box ) == 1. );
}

BOOST_AUTO_TEST_CASE( expand )
{
    using dtk::equals;
    dtk::Box box;
    // NOTE even though not considered valid, default constructed DTK box can be
    // expanded using Boost.Geometry algorithm.
    BOOST_TEST( !bg::is_valid( box ) );
    bg::expand( box, dtk::Point{{0., 0., 0.}} );
    dtk::expand( box, {{1., 1., 1.}} );
    BOOST_TEST( equals( box, {{{0., 0., 0.}}, {{1., 1., 1.}}} ) );
    bg::expand( box, dtk::Box{{{1., 2., 3.}}, {{4., 5., 6.}}} );
    dtk::expand( box, {{{-1., -2., -3.}}, {{0., 0., 0.}}} );
    BOOST_TEST( equals( box, {{{-1., -2., -3.}}, {{4., 5., 6.}}} ) );
}

BOOST_AUTO_TEST_CASE( centroid )
{
    using dtk::equals;

    // Interestingly enough, even though for Boost.Geometry, the DTK default
    // constructed "empty" box is not valid, it will still calculate its
    // centroid.  Admittedly, the result (centroid at origin) is garbage :)
    dtk::Box empty_box = {};
    BOOST_TEST( !bg::is_valid( empty_box ) );
    BOOST_TEST( equals( bg::return_centroid<dtk::Point>( empty_box ),
                        {{0., 0., 0.}} ) );
    BOOST_TEST( equals( dtk::returnCentroid( empty_box ), {{0., 0., 0.}} ) );

    dtk::Box unit_box = {{{0., 0., 0.}}, {{1., 1., 1.}}};
    BOOST_TEST(
        equals( bg::return_centroid<dtk::Point>( unit_box ), {{.5, .5, .5}} ) );
    BOOST_TEST( equals( dtk::returnCentroid( unit_box ), {{.5, .5, .5}} ) );

    dtk::Point a_point = {{1., 2., 3.}};
    BOOST_TEST( equals( bg::return_centroid<dtk::Point>( a_point ), a_point ) );
    BOOST_TEST( equals( dtk::returnCentroid( a_point ), a_point ) );
}

BOOST_AUTO_TEST_CASE( is_valid )
{
    // NOTE "empty" box is considered as valid in DataTransferKit but it is
    // not according to Boost.Geometry
    dtk::Box empty_box = {};
    BOOST_TEST( dtk::isValid( empty_box ) );
    std::string message;
    BOOST_TEST( !bg::is_valid( empty_box, message ) );
    BOOST_TEST( message == "Box has corners in wrong order" );

    // Same issue with infinitesimal box around a point (here the origin)
    dtk::Box a_box = {{{0., 0., 0.}}, {{0., 0., 0.}}};
    BOOST_TEST( dtk::isValid( a_box ) );
    BOOST_TEST( !bg::is_valid( a_box, message ) );
    BOOST_TEST( message == "Geometry has wrong topological dimension" );

    dtk::Box unit_box = {{{0., 0., 0.}}, {{1., 1., 1.}}};
    BOOST_TEST( dtk::isValid( unit_box ) );
    BOOST_TEST( bg::is_valid( unit_box ) );

    dtk::Box invalid_box = {{{1., 2., 3.}}, {{4., NAN, 6.}}};
    BOOST_TEST( !dtk::isValid( invalid_box ) );
    BOOST_TEST( !bg::is_valid( invalid_box, message ) );
    BOOST_TEST( message == "Geometry has point(s) with invalid coordinate(s)" );

    dtk::Point a_point = {{1., 2., 3.}};
    BOOST_TEST( dtk::isValid( a_point ) );
    BOOST_TEST( bg::is_valid( a_point ) );

    auto const infty = std::numeric_limits<double>::infinity();
    dtk::Point invalid_point = {{infty, 1.41, 3.14}};
    BOOST_TEST( !dtk::isValid( invalid_point ) );
    BOOST_TEST( !bg::is_valid( invalid_point, message ) );
    BOOST_TEST( message == "Geometry has point(s) with invalid coordinate(s)" );

    // Also Boost.Geometry has a is_empty() algorithm but it has a different
    // meaning, it checks whether a geometry is an empty set and always returns
    // false for a point or a box.
    BOOST_TEST( !bg::is_empty( empty_box ) );
    BOOST_TEST( !bg::is_empty( a_box ) );
    BOOST_TEST( !bg::is_empty( unit_box ) );
}

BOOST_AUTO_TEST_CASE( intersects )
{
    dtk::Box const unit_box = {{{0., 0., 0.}}, {{1., 1., 1.}}};

    // self-intersection
    BOOST_TEST( dtk::intersects( unit_box, unit_box ) );
    BOOST_TEST( bg::intersects( unit_box, unit_box ) );

    // share a corner
    dtk::Box other_box = {{{1., 1., 1.}}, {{2., 2., 2.}}};
    BOOST_TEST( dtk::intersects( unit_box, other_box ) );
    BOOST_TEST( bg::intersects( unit_box, other_box ) );

    // share an edge
    other_box = {{{1., 0., 1.}}, {{2., 1., 2.}}};
    BOOST_TEST( dtk::intersects( unit_box, other_box ) );
    BOOST_TEST( bg::intersects( unit_box, other_box ) );

    // share a face
    other_box = {{{0., -1., 0.}}, {{1., 0., 1.}}};
    BOOST_TEST( dtk::intersects( unit_box, other_box ) );
    BOOST_TEST( bg::intersects( unit_box, other_box ) );

    // contains the other box
    other_box = {{{.3, .3, .3}}, {{.6, .6, .6}}};
    BOOST_TEST( dtk::intersects( unit_box, other_box ) );
    BOOST_TEST( bg::intersects( unit_box, other_box ) );

    // within the other box
    other_box = {{{-1., -1., -1.}}, {{2., 2., 2.}}};
    BOOST_TEST( dtk::intersects( unit_box, other_box ) );
    BOOST_TEST( bg::intersects( unit_box, other_box ) );

    // overlapping
    other_box = {{{.5, .5, .5}}, {{2., 2., 2.}}};
    BOOST_TEST( dtk::intersects( unit_box, other_box ) );
    BOOST_TEST( bg::intersects( unit_box, other_box ) );

    // disjoint
    other_box = {{{1., 2., 3.}}, {{4., 5., 6.}}};
    BOOST_TEST( !dtk::intersects( unit_box, other_box ) );
    BOOST_TEST( !bg::intersects( unit_box, other_box ) );
}
