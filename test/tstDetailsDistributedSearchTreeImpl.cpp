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

#include <DTK_DetailsDistributedSearchTreeImpl.hpp>

#include <Teuchos_UnitTestHarness.hpp>

#include <algorithm> // fill
#include <set>
#include <vector>

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

    DataTransferKit::Details::DistributedSearchTreeImpl<
        DeviceType>::sortResults( ids, results, ranks );

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
    TEST_THROW( DataTransferKit::Details::DistributedSearchTreeImpl<
                    DeviceType>::sortResults( ids, not_sized_properly ),
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

    DataTransferKit::Details::DistributedSearchTreeImpl<
        DeviceType>::countResults( m, ids, offset );

    auto offset_host = Kokkos::create_mirror_view( offset );
    Kokkos::deep_copy( offset_host, offset );
    TEST_COMPARE_ARRAYS( offset_host, offset_ref );
}

template <typename View1, typename View2>
inline void checkViewWasNotAllocated( View1 const &v1, View2 const &v2,
                                      bool &success,
                                      Teuchos::FancyOStream &out )
{
    // NOTE: cannot use operator== here because array layout may "change" for
    // rank-1 views
    TEST_EQUALITY( v1.data(), v2.data() );
    TEST_EQUALITY( v1.span(), v2.span() );

    TEST_EQUALITY( (int)View1::rank, (int)View2::rank );
    TEST_ASSERT( ( std::is_same<typename View1::const_value_type,
                                typename View2::const_value_type>::value ) );
    TEST_ASSERT( ( std::is_same<typename View1::memory_space,
                                typename View2::memory_space>::value ) );

    TEST_EQUALITY( v1.dimension_0(), v2.dimension_0() );
    TEST_EQUALITY( v1.dimension_1(), v2.dimension_1() );
    TEST_EQUALITY( v1.dimension_2(), v2.dimension_2() );
    TEST_EQUALITY( v1.dimension_3(), v2.dimension_3() );
    TEST_EQUALITY( v1.dimension_4(), v2.dimension_4() );
    TEST_EQUALITY( v1.dimension_5(), v2.dimension_5() );
    TEST_EQUALITY( v1.dimension_6(), v2.dimension_6() );
    TEST_EQUALITY( v1.dimension_7(), v2.dimension_7() );
}

template <typename View1, typename View2>
inline void checkNewViewWasAllocated( View1 const &v1, View2 const &v2,
                                      bool &success,
                                      Teuchos::FancyOStream &out )
{
    TEST_INEQUALITY( v1.data(), v2.data() );

    TEST_EQUALITY( (int)View1::rank, (int)View2::rank );
    TEST_ASSERT( ( std::is_same<typename View1::const_value_type,
                                typename View2::const_value_type>::value ) );

    TEST_EQUALITY( v1.dimension_0(), v2.dimension_0() );
    TEST_EQUALITY( v1.dimension_1(), v2.dimension_1() );
    TEST_EQUALITY( v1.dimension_2(), v2.dimension_2() );
    TEST_EQUALITY( v1.dimension_3(), v2.dimension_3() );
    TEST_EQUALITY( v1.dimension_4(), v2.dimension_4() );
    TEST_EQUALITY( v1.dimension_5(), v2.dimension_5() );
    TEST_EQUALITY( v1.dimension_6(), v2.dimension_6() );
    TEST_EQUALITY( v1.dimension_7(), v2.dimension_7() );
}

TEUCHOS_UNIT_TEST( DetailsDistributedSearchTreeImpl,
                   create_layout_right_mirror_view )
{
    using DataTransferKit::Details::create_layout_right_mirror_view;
    using Kokkos::ALL;
    using Kokkos::LayoutLeft;
    using Kokkos::LayoutRight;
    using Kokkos::make_pair;
    using Kokkos::subview;
    using Kokkos::View;

    // rank-1 and not strided -> do not allocate
    View<int *, LayoutLeft> u( "u", 255 );
    auto u_h = create_layout_right_mirror_view( u );
    checkViewWasNotAllocated( u, u_h, success, out );

    // right layout -> do not allocate
    View<int **, LayoutRight> v( "v", 2, 3 );
    auto v_h = create_layout_right_mirror_view( v );
    checkViewWasNotAllocated( v, v_h, success, out );

    // left layout and rank > 1 -> allocate
    View<int **, LayoutLeft> w( "w", 4, 5 );
    auto w_h = create_layout_right_mirror_view( w );
    checkNewViewWasAllocated( w, w_h, success, out );

    // strided layout -> allocate
    auto x = subview( v, ALL, 0 );
    auto x_h = create_layout_right_mirror_view( x );
    checkNewViewWasAllocated( x, x_h, success, out );

    // subview is rank-1 and not strided -> do not allocate
    auto y = subview( u, make_pair( 8, 16 ) );
    auto y_h = create_layout_right_mirror_view( y );
    checkViewWasNotAllocated( y, y_h, success, out );
}

void checkBufferLayout( std::vector<int> const &ranks,
                        std::vector<int> const &permute_ref,
                        std::vector<int> const &unique_ref,
                        std::vector<int> const &counts_ref,
                        std::vector<int> const &offsets_ref, bool &success,
                        Teuchos::FancyOStream &out )
{
    std::vector<int> permute( ranks.size() );
    std::vector<int> unique;
    std::vector<int> counts;
    std::vector<int> offsets;
    DataTransferKit::Details::sortAndDetermineBufferLayout(
        Kokkos::View<int const *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>( ranks.data(),
                                                               ranks.size() ),
        Kokkos::View<int *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>( permute.data(),
                                                               permute.size() ),
        unique, counts, offsets );
    TEST_COMPARE_ARRAYS( permute_ref, permute );
    TEST_COMPARE_ARRAYS( unique_ref, unique );
    TEST_COMPARE_ARRAYS( counts_ref, counts );
    TEST_COMPARE_ARRAYS( offsets_ref, offsets );
}

TEUCHOS_UNIT_TEST( DetailsDistributor, sort_and_determine_buffer_layout )
{
    checkBufferLayout( {}, {}, {}, {}, {0}, success, out );
    checkBufferLayout( {2, 2}, {0, 1}, {2}, {2}, {0, 2}, success, out );
    checkBufferLayout( {3, 3, 2, 3, 2, 1}, {0, 1, 3, 2, 4, 5}, {3, 2, 1},
                       {3, 2, 1}, {0, 3, 5, 6}, success, out );
    checkBufferLayout( {1, 2, 3, 2, 3, 3}, {5, 3, 0, 4, 1, 2}, {3, 2, 1},
                       {3, 2, 1}, {0, 3, 5, 6}, success, out );
    checkBufferLayout( {0, 1, 2, 3}, {3, 2, 1, 0}, {3, 2, 1, 0}, {1, 1, 1, 1},
                       {0, 1, 2, 3, 4}, success, out );
}

// Include the test macros.
#include "DataTransferKit_ETIHelperMacros.h"

// Create the test group
#define UNIT_TEST_GROUP( NODE )                                                \
    using DeviceType##NODE = typename NODE::device_type;                       \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( DetailsDistributedSearchTreeImpl,    \
                                          sort_results, DeviceType##NODE )     \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( DetailsDistributedSearchTreeImpl,    \
                                          count_results, DeviceType##NODE )

// Demangle the types
DTK_ETI_MANGLING_TYPEDEFS()

// Instantiate the tests
DTK_INSTANTIATE_N( UNIT_TEST_GROUP )
