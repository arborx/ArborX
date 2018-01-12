/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/

#include <DTK_DetailsDistributedSearchTreeImpl.hpp>

#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Tpetra_Distributor.hpp>

#include <algorithm> // fill
#include <set>
#include <vector>

TEUCHOS_UNIT_TEST( DetailsTeuchosSerializationTraits, geometries )
{
    using DataTransferKit::Box;
    using DataTransferKit::Details::equals;
    using DataTransferKit::Point;
    using DataTransferKit::Sphere;

    Teuchos::RCP<const Teuchos::Comm<int>> comm =
        Teuchos::DefaultComm<int>::getComm();
    int const comm_rank = comm->getRank();
    int const comm_size = comm->getSize();

    Point p = {{(double)comm_rank, (double)comm_rank, (double)comm_rank}};
    std::vector<Point> all_p( comm_size );
    Teuchos::gatherAll( *comm, 1, &p, comm_size, all_p.data() );

    for ( int i = 0; i < comm_size; ++i )
        TEST_ASSERT(
            equals( all_p[i], {{(double)i, (double)i, double( i )}} ) );

    Box b;
    if ( comm_rank == 0 )
        b = {{{0., 0., 0.}}, {{1., 1., 1.}}};
    Teuchos::broadcast( *comm, 0, 1, &b );
    TEST_ASSERT( equals( b, {{{0., 0., 0.}}, {{1., 1., 1.}}} ) );

    Sphere s = {{{0., 0., 0.}}, (double)comm_size - (double)comm_rank};
    std::vector<Sphere> all_s( comm_size );
    Teuchos::gatherAll( *comm, 1, &s, comm_size, all_s.data() );
    for ( int i = 0; i < comm_size; ++i )
        TEST_ASSERT( equals(
            all_s[i], {{{0., 0., 0.}}, (double)comm_size - (double)i} ) );
}

TEUCHOS_UNIT_TEST( DetailsTeuchosSerializationTraits, predicates )
{
    using DataTransferKit::Box;
    using DataTransferKit::Details::Intersects;
    using DataTransferKit::Details::equals;
    using DataTransferKit::Details::nearest;
    using DataTransferKit::Point;

    Teuchos::RCP<const Teuchos::Comm<int>> comm =
        Teuchos::DefaultComm<int>::getComm();
    int const comm_rank = comm->getRank();
    int const comm_size = comm->getSize();

    Point p = {{(double)comm_rank, (double)comm_rank, (double)comm_rank}};
    auto nearest_query = nearest( p, comm_size );
    std::vector<decltype( nearest_query )> all_nearest_queries( comm_size );
    Teuchos::gatherAll( *comm, 1, &nearest_query, comm_size,
                        all_nearest_queries.data() );
    for ( int i = 0; i < comm_size; ++i )
    {
        TEST_ASSERT( equals( all_nearest_queries[i]._geometry,
                             {{(double)i, (double)i, (double)i}} ) );
        TEST_EQUALITY( all_nearest_queries[i]._k, comm_size );
    }

    Box b = {{{0., 0., 0.}}, p};
    Intersects<Box> intersects_query( b );
    Teuchos::broadcast( *comm, comm_size - 1, 1, &intersects_query );
    TEST_ASSERT(
        equals( intersects_query._geometry.minCorner(), {{0., 0., 0.}} ) );
    TEST_ASSERT( equals( intersects_query._geometry.maxCorner(),
                         {{(double)comm_size - 1, (double)comm_size - 1,
                           (double)comm_size - 1}} ) );
}

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( DetailsDistributedSearchTreeImpl, recv_from,
                                   DeviceType )
{
    // Checking that it is not necessary to send ranks because it can be
    // inferred from getProcsFrom() and getLengthsFrom().
    Teuchos::RCP<const Teuchos::Comm<int>> comm =
        Teuchos::DefaultComm<int>::getComm();
    int const comm_rank = comm->getRank();
    int const comm_size = comm->getSize();

    std::vector<int> tn( comm_size + 1 );
    for ( int i = 0; i < comm_size + 1; ++i )
        tn[i] = i * ( i - 1 ) / 2;

    // First use send buffer to set up the communication plan.  Sending 0
    // packet to rank 1, 1 packet to rank 1, 2 packets to rank 2, etc.
    int const n_exports = tn[comm_size];
    std::vector<int> exports( n_exports );
    for ( int i = 0; i < comm_size; ++i )
        for ( int j = tn[i]; j < tn[i + 1]; ++j )
            exports[j] = i;

    Tpetra::Distributor distributor( comm );
    int const n_imports = distributor.createFromSends(
        Teuchos::ArrayView<int>( exports.data(), exports.size() ) );
    TEST_EQUALITY( n_imports, comm_rank * comm_size );

    std::vector<int> imports( n_imports );
    distributor.doPostsAndWaits(
        Teuchos::ArrayView<int const>( exports.data(), exports.size() ), 1,
        Teuchos::ArrayView<int>( imports.data(), imports.size() ) );

    TEST_COMPARE_ARRAYS( imports, std::vector<int>( n_imports, comm_rank ) );

    // Then fill buffer with rank of the process that is sending packets.
    std::fill( exports.begin(), exports.end(), comm_rank );
    distributor.doPostsAndWaits(
        Teuchos::ArrayView<int const>( exports.data(), exports.size() ), 1,
        Teuchos::ArrayView<int>( imports.data(), imports.size() ) );

    auto procs_from = distributor.getProcsFrom();
    auto lengths_from = distributor.getLengthsFrom();
    TEST_EQUALITY( procs_from.size(), lengths_from.size() );
    std::vector<int> recv_from( n_imports, -1 );
    int count = 0;
    for ( auto i = 0; i < procs_from.size(); ++i )
        for ( size_t j = 0; j < lengths_from[i]; ++j )
            recv_from[count++] = procs_from[i];
    TEST_EQUALITY( count, n_imports );
    TEST_COMPARE_ARRAYS( imports, recv_from );
}

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( DetailsDistributedSearchTreeImpl,
                                   sort_results, DeviceType )
{
    std::vector<int> ids_ = {4, 3, 2, 1, 4, 3, 2, 4, 3, 4};
    std::vector<int> sorted_ids = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4};
    std::vector<int> offset = {0, 1, 3, 6, 10};
    int const n = 10;
    int const m = 4;
    TEST_EQUALITY( ids_.size(), n );
    TEST_EQUALITY( sorted_ids.size(), n );
    TEST_EQUALITY( offset.size(), m + 1 );
    std::vector<int> results_ = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<std::set<int>> sorted_results = {
        {3},
        {6, 2},
        {8, 5, 1},
        {9, 7, 4, 0},
    };
    std::vector<int> ranks_ = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    std::vector<std::set<int>> sorted_ranks = {
        {13},
        {16, 12},
        {18, 15, 11},
        {19, 17, 14, 10},
    };
    TEST_EQUALITY( results_.size(), n );
    TEST_EQUALITY( ranks_.size(), n );

    Kokkos::View<int *, DeviceType> ids( "query_ids", n );
    auto ids_host = Kokkos::create_mirror_view( ids );
    for ( int i = 0; i < n; ++i )
        ids_host( i ) = ids_[i];
    Kokkos::deep_copy( ids, ids_host );

    Kokkos::View<int *, DeviceType> results( "results", n );
    auto results_host = Kokkos::create_mirror_view( results );
    for ( int i = 0; i < n; ++i )
        results_host( i ) = results_[i];
    Kokkos::deep_copy( results, results_host );

    Kokkos::View<int *, DeviceType> ranks( "ranks", n );
    auto ranks_host = Kokkos::create_mirror_view( ranks );
    for ( int i = 0; i < n; ++i )
        ranks_host( i ) = ranks_[i];
    Kokkos::deep_copy( ranks, ranks_host );

    DataTransferKit::DistributedSearchTreeImpl<DeviceType>::sortResults(
        ids, results, ranks );

    // COMMENT: ids are untouched
    Kokkos::deep_copy( ids_host, ids );
    TEST_COMPARE_ARRAYS( ids_host, ids_ );

    Kokkos::deep_copy( results_host, results );
    Kokkos::deep_copy( ranks_host, ranks );
    for ( int q = 0; q < m; ++q )
        for ( int i = offset[q]; i < offset[q + 1]; ++i )
        {
            TEST_EQUALITY( sorted_results[q].count( results_host[i] ), 1 );
            TEST_EQUALITY( sorted_ranks[q].count( ranks_host[i] ), 1 );
        }

    Kokkos::View<int *, DeviceType> not_sized_properly( "", m );
    TEST_THROW(
        DataTransferKit::DistributedSearchTreeImpl<DeviceType>::sortResults(
            ids, not_sized_properly ),
        DataTransferKit::DataTransferKitException );
}

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( DetailsDistributedSearchTreeImpl,
                                   count_results, DeviceType )
{
    std::vector<int> ids_ref = {4, 3, 2, 1, 4, 3, 2, 4, 3, 4};
    std::vector<int> offset_ref = {
        0, 0, 1, 3, 6, 10,
    };
    int const m = 5;
    int const nnz = 10;
    TEST_EQUALITY( ids_ref.size(), nnz );
    TEST_EQUALITY( offset_ref.size(), m + 1 );

    Kokkos::View<int *, DeviceType> ids( "query_ids", nnz );
    auto ids_host = Kokkos::create_mirror_view( ids );
    for ( int i = 0; i < nnz; ++i )
        ids_host( i ) = ids_ref[i];
    Kokkos::deep_copy( ids, ids_host );

    Kokkos::View<int *, DeviceType> offset( "offset" );

    DataTransferKit::DistributedSearchTreeImpl<DeviceType>::countResults(
        m, ids, offset );

    auto offset_host = Kokkos::create_mirror_view( offset );
    Kokkos::deep_copy( offset_host, offset );
    TEST_COMPARE_ARRAYS( offset_host, offset_ref );
}

// Include the test macros.
#include "DataTransferKitSearch_ETIHelperMacros.h"

// Create the test group
#define UNIT_TEST_GROUP( NODE )                                                \
    using DeviceType##NODE = typename NODE::device_type;                       \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( DetailsDistributedSearchTreeImpl,    \
                                          recv_from, DeviceType##NODE )        \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( DetailsDistributedSearchTreeImpl,    \
                                          sort_results, DeviceType##NODE )     \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( DetailsDistributedSearchTreeImpl,    \
                                          count_results, DeviceType##NODE )

// Demangle the types
DTK_ETI_MANGLING_TYPEDEFS()

// Instantiate the tests
DTK_INSTANTIATE_N( UNIT_TEST_GROUP )
