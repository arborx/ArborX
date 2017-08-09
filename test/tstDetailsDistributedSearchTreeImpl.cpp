/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/

#include <details/DTK_DetailsDistributedSearchTreeImpl.hpp>

#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Tpetra_Distributor.hpp>

#include <algorithm>
#include <bitset>
#include <iostream>
#include <random>
#include <tuple>

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
    TEUCHOS_ASSERT_EQUALITY( n_imports, comm_rank * comm_size );

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
    auto lengths_form = distributor.getLengthsFrom();
    TEST_EQUALITY( procs_from.size(), lengths_form.size() );
    std::vector<int> recv_from( n_imports, -1 );
    int count = 0;
    for ( auto i = 0; i < procs_from.size(); ++i )
        for ( size_t j = 0; j < lengths_form[i]; ++j )
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
        {3}, {6, 2}, {8, 5, 1}, {9, 7, 4, 0},
    };
    std::vector<int> ranks_ = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    std::vector<std::set<int>> sorted_ranks = {
        {13}, {16, 12}, {18, 15, 11}, {19, 17, 14, 10},
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

    DataTransferKit::DistributedSearchTreeImpl<DeviceType>::sort_results(
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

    DataTransferKit::DistributedSearchTreeImpl<DeviceType>::count_results(
        m, ids, offset );

    auto offset_host = Kokkos::create_mirror_view( offset );
    Kokkos::deep_copy( offset_host, offset );
    TEST_COMPARE_ARRAYS( offset_host, offset_ref );
}

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( DetailsDistributedSearchTreeImpl,
                                   tpetra_fixme, DeviceType )
{
    Teuchos::RCP<const Teuchos::Comm<int>> comm =
        Teuchos::DefaultComm<int>::getComm();
    int const comm_rank = comm->getRank();
    int const comm_size = comm->getSize();

    Tpetra::Distributor distributor( comm );
    int const n = 3 * comm_size;
    Kokkos::View<int *, DeviceType> proc_ids( "proc_ids", n );
    int const n_exports = proc_ids.extent( 0 );
    using ExecutionSpace = typename DeviceType::execution_space;
    Kokkos::parallel_for(
        "fill_proc_ids", Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
        KOKKOS_LAMBDA( int i ) { proc_ids( i ) = i % comm_size; } );
    Kokkos::fence();
    auto proc_ids_host = Kokkos::create_mirror_view( proc_ids );
    Kokkos::deep_copy( proc_ids_host, proc_ids );
    int const n_imports = distributor.createFromSends(
        Teuchos::ArrayView<int const>( proc_ids_host.data(), n_exports ) );
    Kokkos::View<int *, DeviceType> exports( "exports", n_exports );
    Kokkos::parallel_for(
        "fill_exports", Kokkos::RangePolicy<ExecutionSpace>( 0, n_exports ),
        KOKKOS_LAMBDA( int i ) { exports( i ) = comm_rank; } );
    Kokkos::fence();

    Kokkos::View<int *, DeviceType> imports( "imports", n_imports );
// See https://github.com/trilinos/Trilinos/issues/1454
// The code compiles with the patch that was submitted.  Sticking with the
// workaround for now until we figure out what version of Trilinos goes into out
// Docker image.
#define WORKAROUND 1
#ifndef WORKAROUND
    distributor.doPostsAndWaits( exports, 1, imports );
    auto imports_host = Kokkos::create_mirror_view( imports );
    Kokkos::deep_copy( imports_host, imports );
#else
    auto exports_host = Kokkos::create_mirror_view( exports );
    Kokkos::deep_copy( exports_host, exports );
    auto imports_host = Kokkos::create_mirror_view( imports );
    distributor.doPostsAndWaits(
        Teuchos::ArrayView<int const>( exports_host.data(), n_exports ), 1,
        Teuchos::ArrayView<int>( imports_host.data(), n_imports ) );
    Kokkos::deep_copy( imports, imports_host );
#endif

    for ( int i = 0; i < n_imports; ++i )
        TEUCHOS_ASSERT_EQUALITY( imports_host( i ), i / 3 );
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
                                          count_results, DeviceType##NODE )    \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( DetailsDistributedSearchTreeImpl,    \
                                          tpetra_fixme, DeviceType##NODE )

// Demangle the types
DTK_ETI_MANGLING_TYPEDEFS()

// Instantiate the tests
DTK_INSTANTIATE_N( UNIT_TEST_GROUP )
