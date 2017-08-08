/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/

// TODO: get rid of this one
#include <DTK_DetailsTreeTraversal.hpp>

#include <DTK_DistributedSearchTree.hpp>

#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_UnitTestHarness.hpp>

#include <algorithm>
#include <bitset>
#include <iostream>
#include <random>
#include <tuple>

namespace details = DataTransferKit::Details;

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( DistributedSearchTree, hello_world,
                                   DeviceType )
{
    Teuchos::RCP<const Teuchos::Comm<int>> comm =
        Teuchos::DefaultComm<int>::getComm();
    int const comm_rank = Teuchos::rank( *comm );
    int const comm_size = Teuchos::size( *comm );

    DataTransferKit::DistributedSearchTreeImpl<DeviceType>::epsilon = 0.5;
    int const n = 4;
    Kokkos::View<DataTransferKit::Box *, DeviceType> boxes( "boxes", n );
    auto boxes_host = Kokkos::create_mirror_view( boxes );
    // [  rank 0       [  rank 1       [  rank 2       [  rank 3       [
    // x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---
    // ^   ^   ^   ^
    // 0   1   2   3   ^   ^   ^   ^
    //                 0   1   2   3   ^   ^   ^   ^
    //                                 0   1   2   3   ^   ^   ^   ^
    //                                                 0   1   2   3
    for ( int i = 0; i < n; ++i )
    {
        DataTransferKit::Box box;
        DataTransferKit::Point point = {{(double)i / n + comm_rank, 0., 0.}};
        DataTransferKit::Details::expand( box, point );
        boxes_host( i ) = box;
    }
    Kokkos::deep_copy( boxes, boxes_host );

    DataTransferKit::DistributedSearchTree<DeviceType> tree( comm, boxes );

    // 0---0---0---0---1---1---1---1---2---2---2---2---3---3---3---3---
    // |               |               |               |               |
    // |               |               |               x   x   x   x   |
    // |               |               |               |<------0------>|
    // |               |               x   x   x   x   x               |
    // |               |               |<------1------>|               |
    // |               x   x   x   x   x               |               |
    // |               |<------2------>|               |               |
    // x   x   x   x   x               |               |               |
    // |<------3------>|               |               |               |
    // |               |               |               |               |
    Kokkos::View<DataTransferKit::Details::Within *, DeviceType> queries(
        "queries", 1 );
    auto queries_host = Kokkos::create_mirror_view( queries );
    queries_host( 0 ) = DataTransferKit::Details::within(
        {{0.5 + comm_size - 1 - comm_rank, 0., 0.}}, 0.5 );
    deep_copy( queries, queries_host );

    // 0---0---0---0---1---1---1---1---2---2---2---2---3---3---3---3---
    // |               |               |               |               |
    // |               |               |           x   x   x           |
    // |               |           x   x   x        <--0-->            |
    // |           x   x   x        <--1-->            |               |
    // x   x        <--2-->            |               |               |
    // 3-->            |               |               |               |
    // |               |               |               |               |
    Kokkos::View<DataTransferKit::Details::Nearest *, DeviceType>
        nearest_queries( "nearest_queries", 1 );
    auto nearest_queries_host = Kokkos::create_mirror_view( nearest_queries );
    nearest_queries_host( 0 ) = DataTransferKit::Details::nearest(
        {{0.0 + comm_size - 1 - comm_rank, 0., 0.}},
        comm_rank < comm_size - 1 ? 3 : 2 );
    deep_copy( nearest_queries, nearest_queries_host );

    Kokkos::View<int *, DeviceType> offset( "offset" );
    Kokkos::View<int *, DeviceType> indices( "indices" );
    Kokkos::View<int *, DeviceType> ranks( "ranks" );
    tree.query( queries, indices, offset, ranks );

    auto indices_host = Kokkos::create_mirror_view( indices );
    Kokkos::deep_copy( indices_host, indices );
    auto ranks_host = Kokkos::create_mirror_view( ranks );
    Kokkos::deep_copy( ranks_host, ranks );
    auto offset_host = Kokkos::create_mirror_view( offset );
    Kokkos::deep_copy( offset_host, offset );

    TEST_EQUALITY( offset_host.extent( 0 ), 2 );
    TEST_EQUALITY( offset_host( 0 ), 0 );
    TEST_EQUALITY( offset_host( 1 ), indices_host.extent_int( 0 ) );
    TEST_EQUALITY( indices_host.extent( 0 ), ranks_host.extent( 0 ) );
    TEST_EQUALITY( indices_host.extent( 0 ), comm_rank > 0 ? n + 1 : n );
    for ( int i = 0; i < n; ++i )
    {
        TEST_EQUALITY( n - 1 - i, indices_host( i ) );
        TEST_EQUALITY( comm_size - 1 - comm_rank, ranks_host( i ) );
    }
    if ( comm_rank > 0 )
    {
        TEST_EQUALITY( indices_host( n ), 0 );
        TEST_EQUALITY( ranks_host( n ), comm_size - comm_rank );
    }

    tree.query( nearest_queries, indices, offset, ranks );

    indices_host = Kokkos::create_mirror_view( indices );
    ranks_host = Kokkos::create_mirror_view( ranks );
    offset_host = Kokkos::create_mirror_view( offset );
    Kokkos::deep_copy( indices_host, indices );
    Kokkos::deep_copy( ranks_host, ranks );
    Kokkos::deep_copy( offset_host, offset );

    TEST_COMPARE( n, >, 2 );
    TEST_EQUALITY( offset_host.extent( 0 ), 2 );
    TEST_EQUALITY( offset_host( 0 ), 0 );
    TEST_EQUALITY( offset_host( 1 ), indices_host.extent_int( 0 ) );
    TEST_EQUALITY( indices_host.extent( 0 ),
                   comm_rank < comm_size - 1 ? 3 : 2 );

    TEST_EQUALITY( indices_host( 0 ), 0 );
    TEST_EQUALITY( ranks_host( 0 ), comm_size - 1 - comm_rank );
    if ( comm_rank < comm_size - 1 )
    {
        TEST_EQUALITY( indices_host( 1 ), n - 1 );
        TEST_EQUALITY( ranks_host( 1 ), comm_size - 2 - comm_rank );
        TEST_EQUALITY( indices_host( 2 ), 1 );
        TEST_EQUALITY( ranks_host( 2 ), comm_size - 1 - comm_rank );
    }
    else
    {
        TEST_EQUALITY( indices_host( 1 ), 1 );
        TEST_EQUALITY( ranks_host( 1 ), comm_size - 1 - comm_rank );
    }
}

std::vector<std::array<double, 3>>
make_random_cloud( double const Lx, double const Ly, double const Lz,
                   int const n, double const seed )
{
    std::vector<std::array<double, 3>> cloud( n );
    std::default_random_engine generator( seed );
    std::uniform_real_distribution<double> distribution_x( 0.0, Lz );
    std::uniform_real_distribution<double> distribution_y( 0.0, Ly );
    std::uniform_real_distribution<double> distribution_z( 0.0, Lz );
    for ( int i = 0; i < n; ++i )
    {
        double x = distribution_x( generator );
        double y = distribution_y( generator );
        double z = distribution_z( generator );
        cloud[i] = {{x, y, z}};
    }
    return cloud;
}

// Include the test macros.
#include "DataTransferKitSearch_ETIHelperMacros.h"

// Create the test group
#define UNIT_TEST_GROUP( NODE )                                                \
    using DeviceType##NODE = typename NODE::device_type;                       \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( DistributedSearchTree, hello_world,  \
                                          DeviceType##NODE )

// Demangle the types
DTK_ETI_MANGLING_TYPEDEFS()

// Instantiate the tests
DTK_INSTANTIATE_N( UNIT_TEST_GROUP )
