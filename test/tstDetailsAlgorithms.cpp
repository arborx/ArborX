/****************************************************************************
 * Copyright (c) 2012-2018 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
#include <DTK_DetailsAlgorithms.hpp>

#include <Teuchos_UnitTestHarness.hpp>

namespace dtk = DataTransferKit::Details;

TEUCHOS_UNIT_TEST( DetailsAlgorithms, distance )
{
    TEST_EQUALITY( dtk::distance( {{1.0, 2.0, 3.0}}, {{1.0, 1.0, 1.0}} ),
                   std::sqrt( 5.0 ) );

    // box is unit cube
    DataTransferKit::Box box = {{{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}};

    // distance is zero if the point is inside the box
    TEST_EQUALITY( dtk::distance( {{0.5, 0.5, 0.5}}, box ), 0.0 );
    // or anywhere on the boundary
    TEST_EQUALITY( dtk::distance( {{0.0, 0.0, 0.5}}, box ), 0.0 );
    // normal projection onto center of one face
    TEST_EQUALITY( dtk::distance( {{2.0, 0.5, 0.5}}, box ), 1.0 );
    // projection onto edge
    TEST_EQUALITY( dtk::distance( {{2.0, 0.75, -1.0}}, box ),
                   std::sqrt( 2.0 ) );
    // projection onto corner node
    TEST_EQUALITY( dtk::distance( {{-1.0, 2.0, 2.0}}, box ), std::sqrt( 3.0 ) );

    // unit sphere
    DataTransferKit::Sphere sphere = {{{0., 0., 0.}}, 1.};
    TEST_EQUALITY( dtk::distance( {{.5, .5, .5}}, sphere ), 0. );
    TEST_EQUALITY( dtk::distance( {{2., 0., 0.}}, sphere ), 1. );
    TEST_EQUALITY( dtk::distance( {{1., 1., 1.}}, sphere ),
                   std::sqrt( 3. ) - 1. );
}

TEUCHOS_UNIT_TEST( DetailsAlgorithms, overlaps )
{
    DataTransferKit::Box box;
    // uninitialized box does not even overlap with itself
    TEST_ASSERT( !dtk::intersects( box, box ) );
    // box with zero extent does
    box = {{{0.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}};
    TEST_ASSERT( dtk::intersects( box, box ) );
    TEST_ASSERT( !dtk::intersects( box, DataTransferKit::Box() ) );
    // overlap with box that contains it
    TEST_ASSERT(
        dtk::intersects( box, {{{-1.0, -1.0, -1.0}}, {{1.0, 1.0, 1.0}}} ) );
    // does not overlap with some other box
    TEST_ASSERT(
        !dtk::intersects( box, {{{1.0, 1.0, 1.0}}, {{2.0, 2.0, 2.0}}} ) );
    // overlap when only touches another
    TEST_ASSERT(
        dtk::intersects( box, {{{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}} ) );
    // unit cube
    box = {{{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}};
    TEST_ASSERT( dtk::intersects( box, box ) );
    TEST_ASSERT( !dtk::intersects( box, DataTransferKit::Box() ) );
    // smaller box inside
    TEST_ASSERT(
        dtk::intersects( box, {{{0.25, 0.25, 0.25}}, {{0.75, 0.75, 0.75}}} ) );
    // bigger box that contains it
    TEST_ASSERT(
        dtk::intersects( box, {{{-1.0, -1.0, -1.0}}, {{2.0, 2.0, 2.0}}} ) );
    // couple boxes that do overlap
    TEST_ASSERT(
        dtk::intersects( box, {{{0.5, 0.5, 0.5}}, {{1.5, 1.5, 1.5}}} ) );
    TEST_ASSERT(
        dtk::intersects( box, {{{-0.5, -0.5, -0.5}}, {{0.5, 0.5, 0.5}}} ) );
    // couple boxes that do not
    TEST_ASSERT(
        !dtk::intersects( box, {{{-2.0, -2.0, -2.0}}, {{-1.0, -1.0, -1.0}}} ) );
    TEST_ASSERT(
        !dtk::intersects( box, {{{0.0, 0.0, 2.0}}, {{1.0, 1.0, 3.0}}} ) );
    // boxes overlap if faces touch
    TEST_ASSERT(
        dtk::intersects( box, {{{1.0, 0.0, 0.0}}, {{2.0, 1.0, 1.0}}} ) );
    TEST_ASSERT(
        dtk::intersects( box, {{{-0.5, -0.5, -0.5}}, {{0.5, 0.0, 0.5}}} ) );
}

TEUCHOS_UNIT_TEST( DetailsAlgorithms, intersects )
{
    DataTransferKit::Sphere sphere = {{{0., 0., 0.}}, 1.};
    TEST_ASSERT( dtk::intersects( sphere, {{{0., 0., 0.}}, {{1., 1., 1.}}} ) );
    TEST_ASSERT( !dtk::intersects( sphere, {{{1., 2., 3.}}, {{4., 5., 6.}}} ) );
}

TEUCHOS_UNIT_TEST( DetailsAlgorithms, equals )
{
    // points
    TEST_ASSERT( dtk::equals( {{0., 0., 0.}}, {{0., 0., 0.}} ) );
    TEST_ASSERT( !dtk::equals( {{0., 0., 0.}}, {{1., 1., 1.}} ) );
    // boxes
    TEST_ASSERT( dtk::equals( {{{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}},
                              {{{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}} ) );
    TEST_ASSERT( !dtk::equals( {{{0.0, 0.0, 0.0}}, {{1.0, 0.0, 1.0}}},
                               {{{-1.0, -1.0, -1.0}}, {{1.0, 1.0, 1.0}}} ) );
    // spheres
    TEST_ASSERT( dtk::equals( {{{0., 0., 0.}}, 1.}, {{{0., 0., 0.}}, 1.} ) );
    TEST_ASSERT( !dtk::equals( {{{0., 0., 0.}}, 1.}, {{{0., 1., 2.}}, 1.} ) );
    TEST_ASSERT( !dtk::equals( {{{0., 0., 0.}}, 1.}, {{{0., 0., 0.}}, 2.} ) );
}

TEUCHOS_UNIT_TEST( DetailsAlgorithms, expand )
{
    DataTransferKit::Box box;

    // expand box with points
    dtk::expand( box, {{0.0, 0.0, 0.0}} );
    TEST_ASSERT( dtk::equals( box, {{{0.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}} ) );
    dtk::expand( box, {{1.0, 1.0, 1.0}} );
    TEST_ASSERT( dtk::equals( box, {{{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}} ) );
    dtk::expand( box, {{0.25, 0.75, 0.25}} );
    TEST_ASSERT( dtk::equals( box, {{{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}} ) );
    dtk::expand( box, {{-1.0, -1.0, -1.0}} );
    TEST_ASSERT(
        dtk::equals( box, {{{-1.0, -1.0, -1.0}}, {{1.0, 1.0, 1.0}}} ) );

    // expand box with boxes
    dtk::expand( box, {{{0.25, 0.25, 0.25}}, {{0.75, 0.75, 0.75}}} );
    TEST_ASSERT(
        dtk::equals( box, {{{-1.0, -1.0, -1.0}}, {{1.0, 1.0, 1.0}}} ) );
    dtk::expand( box, {{{10.0, 10.0, 10.0}}, {{11.0, 11.0, 11.0}}} );
    TEST_ASSERT(
        dtk::equals( box, {{{-1.0, -1.0, -1.0}}, {{11.0, 11.0, 11.0}}} ) );

    // expand box with spheres
    dtk::expand( box, {{{0., 1., 2.}}, 3.} );
    TEST_ASSERT( dtk::equals( box, {{{-3., -2., -1.}}, {{11., 11., 11.}}} ) );
    dtk::expand( box, {{{0., 0., 0.}}, 1.} );
    TEST_ASSERT( dtk::equals( box, {{{-3., -2., -1.}}, {{11., 11., 11.}}} ) );
    dtk::expand( box, {{{0., 0., 0.}}, 24.} );
    TEST_ASSERT(
        dtk::equals( box, {{{-24., -24., -24.}}, {{24., 24., 24.}}} ) );
}

TEUCHOS_UNIT_TEST( DetailsAlgorithms, centroid )
{
    DataTransferKit::Box box = {{{-10.0, 0.0, 10.0}}, {{0.0, 10.0, 20.0}}};
    DataTransferKit::Point centroid;
    dtk::centroid( box, centroid );
    TEST_EQUALITY( centroid[0], -5.0 );
    TEST_EQUALITY( centroid[1], 5.0 );
    TEST_EQUALITY( centroid[2], 15.0 );
}
