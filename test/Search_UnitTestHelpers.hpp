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

#ifndef DTK_SEARCH_TEST_HELPERS_HPP
#define DTK_SEARCH_TEST_HELPERS_HPP

#include <DTK_DistributedSearchTree.hpp>
#include <DTK_LinearBVH.hpp>

#include <Kokkos_View.hpp>

#include <Teuchos_Comm.hpp>
#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_LocalTestingHelpers.hpp>
#include <Teuchos_RCP.hpp>

#include <vector>

// The `out` and `success` parameters come from the Teuchos unit testing macros
// expansion.
template <typename Query, typename DeviceType>
void checkResults( DataTransferKit::BVH<DeviceType> const &bvh,
                   Kokkos::View<Query *, DeviceType> const &queries,
                   std::vector<int> const &indices_ref,
                   std::vector<int> const &offset_ref, bool &success,
                   Teuchos::FancyOStream &out )
{
    Kokkos::View<int *, DeviceType> indices( "indices" );
    Kokkos::View<int *, DeviceType> offset( "offset" );
    bvh.query( queries, indices, offset );

    auto indices_host = Kokkos::create_mirror_view( indices );
    deep_copy( indices_host, indices );
    auto offset_host = Kokkos::create_mirror_view( offset );
    deep_copy( offset_host, offset );

    TEST_COMPARE_ARRAYS( indices_host, indices_ref );
    TEST_COMPARE_ARRAYS( offset_host, offset_ref );
}

// Same as above except that we get the distances out of the queries and
// compare them to the reference solution passed as argument.  Templated type
// `Query` is pretty much a nearest predicate in this case.
template <typename Query, typename DeviceType>
void checkResults( DataTransferKit::BVH<DeviceType> const &bvh,
                   Kokkos::View<Query *, DeviceType> const &queries,
                   std::vector<int> const &indices_ref,
                   std::vector<int> const &offset_ref,
                   std::vector<double> const &distances_ref, bool &success,
                   Teuchos::FancyOStream &out )
{
    Kokkos::View<int *, DeviceType> indices( "indices" );
    Kokkos::View<int *, DeviceType> offset( "offset" );
    Kokkos::View<double *, DeviceType> distances( "distances" );
    bvh.query( queries, indices, offset, distances );

    auto indices_host = Kokkos::create_mirror_view( indices );
    deep_copy( indices_host, indices );
    auto offset_host = Kokkos::create_mirror_view( offset );
    deep_copy( offset_host, offset );
    auto distances_host = Kokkos::create_mirror_view( distances );
    deep_copy( distances_host, distances );

    TEST_COMPARE_ARRAYS( indices_host, indices_ref );
    TEST_COMPARE_ARRAYS( offset_host, offset_ref );
    TEST_COMPARE_FLOATING_ARRAYS( distances_host, distances_ref, 1e-14 );
}

// The `out` and `success` parameters come from the Teuchos unit testing macros
// expansion.
template <typename Query, typename DeviceType>
void checkResults(
    DataTransferKit::DistributedSearchTree<DeviceType> const &tree,
    Kokkos::View<Query *, DeviceType> const &queries,
    std::vector<int> const &indices_ref, std::vector<int> const &offset_ref,
    std::vector<int> const &ranks_ref, bool &success,
    Teuchos::FancyOStream &out )
{
    Kokkos::View<int *, DeviceType> indices( "indices" );
    Kokkos::View<int *, DeviceType> offset( "offset" );
    Kokkos::View<int *, DeviceType> ranks( "ranks" );
    tree.query( queries, indices, offset, ranks );

    auto indices_host = Kokkos::create_mirror_view( indices );
    deep_copy( indices_host, indices );
    auto offset_host = Kokkos::create_mirror_view( offset );
    deep_copy( offset_host, offset );
    auto ranks_host = Kokkos::create_mirror_view( ranks );
    deep_copy( ranks_host, ranks );

    TEST_COMPARE_ARRAYS( indices_host, indices_ref );
    TEST_COMPARE_ARRAYS( offset_host, offset_ref );
    TEST_COMPARE_ARRAYS( ranks_host, ranks_ref );
}

template <typename Query, typename DeviceType>
void checkResults(
    DataTransferKit::DistributedSearchTree<DeviceType> const &tree,
    Kokkos::View<Query *, DeviceType> const &queries,
    std::vector<int> const &indices_ref, std::vector<int> const &offset_ref,
    std::vector<int> const &ranks_ref, std::vector<double> const &distances_ref,
    bool &success, Teuchos::FancyOStream &out )
{
    Kokkos::View<int *, DeviceType> indices( "indices" );
    Kokkos::View<int *, DeviceType> offset( "offset" );
    Kokkos::View<int *, DeviceType> ranks( "ranks" );
    Kokkos::View<double *, DeviceType> distances( "distances" );
    tree.query( queries, indices, offset, ranks, distances );

    auto indices_host = Kokkos::create_mirror_view( indices );
    deep_copy( indices_host, indices );
    auto offset_host = Kokkos::create_mirror_view( offset );
    deep_copy( offset_host, offset );
    auto ranks_host = Kokkos::create_mirror_view( ranks );
    deep_copy( ranks_host, ranks );
    auto distances_host = Kokkos::create_mirror_view( distances );
    deep_copy( distances_host, distances );

    TEST_COMPARE_ARRAYS( indices_host, indices_ref );
    TEST_COMPARE_ARRAYS( offset_host, offset_ref );
    TEST_COMPARE_ARRAYS( ranks_host, ranks_ref );
    TEST_COMPARE_FLOATING_ARRAYS( distances_host, distances_ref, 1e-14 );
}

template <typename DeviceType>
DataTransferKit::BVH<DeviceType>
makeBvh( std::vector<DataTransferKit::Box> const &b )
{
    int const n = b.size();
    Kokkos::View<DataTransferKit::Box *, DeviceType> boxes( "boxes", n );
    auto boxes_host = Kokkos::create_mirror_view( boxes );
    for ( int i = 0; i < n; ++i )
        boxes_host( i ) = b[i];
    Kokkos::deep_copy( boxes, boxes_host );
    return DataTransferKit::BVH<DeviceType>( boxes );
}

template <typename DeviceType>
DataTransferKit::DistributedSearchTree<DeviceType>
makeDistributedSearchTree( Teuchos::RCP<const Teuchos::Comm<int>> const &comm,
                           std::vector<DataTransferKit::Box> const &b )
{
    int const n = b.size();
    Kokkos::View<DataTransferKit::Box *, DeviceType> boxes( "boxes", n );
    auto boxes_host = Kokkos::create_mirror_view( boxes );
    for ( int i = 0; i < n; ++i )
        boxes_host( i ) = b[i];
    Kokkos::deep_copy( boxes, boxes_host );
    return DataTransferKit::DistributedSearchTree<DeviceType>( comm, boxes );
}

template <typename DeviceType>
Kokkos::View<DataTransferKit::Details::Overlap *, DeviceType>
makeOverlapQueries( std::vector<DataTransferKit::Box> const &boxes )
{
    int const n = boxes.size();
    Kokkos::View<DataTransferKit::Details::Overlap *, DeviceType> queries(
        "overlap_queries", n );
    auto queries_host = Kokkos::create_mirror_view( queries );
    for ( int i = 0; i < n; ++i )
        queries_host( i ) = DataTransferKit::Details::overlap( boxes[i] );
    Kokkos::deep_copy( queries, queries_host );
    return queries;
}

template <typename DeviceType>
Kokkos::View<DataTransferKit::Details::Nearest<DataTransferKit::Point> *,
             DeviceType>
makeNearestQueries(
    std::vector<std::pair<DataTransferKit::Point, int>> const &points )
{
    // NOTE: `points` is not a very descriptive name here. It stores both the
    // actual point and the number k of neighbors to query for.
    int const n = points.size();
    Kokkos::View<DataTransferKit::Details::Nearest<DataTransferKit::Point> *,
                 DeviceType>
        queries( "nearest_queries", n );
    auto queries_host = Kokkos::create_mirror_view( queries );
    for ( int i = 0; i < n; ++i )
        queries_host( i ) = DataTransferKit::Details::nearest(
            points[i].first, points[i].second );
    Kokkos::deep_copy( queries, queries_host );
    return queries;
}

template <typename DeviceType>
Kokkos::View<DataTransferKit::Details::Within *, DeviceType> makeWithinQueries(
    std::vector<std::pair<DataTransferKit::Point, double>> const &points )
{
    // NOTE: `points` is not a very descriptive name here. It stores both the
    // actual point and the radius for the search around that point.
    int const n = points.size();
    Kokkos::View<DataTransferKit::Details::Within *, DeviceType> queries(
        "within_queries", n );
    auto queries_host = Kokkos::create_mirror_view( queries );
    for ( int i = 0; i < n; ++i )
        queries_host( i ) = DataTransferKit::Details::within(
            points[i].first, points[i].second );
    Kokkos::deep_copy( queries, queries_host );
    return queries;
}

template <typename View, typename Value = typename View::value_type>
std::vector<Value> extractAndSort( View const &v, int begin, int end )
{
    std::vector<Value> r( v.data() + begin, v.data() + end );
    std::sort( r.begin(), r.end() );
    return r;
};

template <typename InputView1, typename InputView2>
void validateResults( std::tuple<InputView1, InputView1> const &reference,
                      std::tuple<InputView2, InputView2> const &other,
                      bool &success, Teuchos::FancyOStream &out )
{
    TEST_COMPARE_ARRAYS( std::get<0>( reference ), std::get<0>( other ) );
    auto const offset = std::get<0>( reference );
    auto const n_queries = offset.extent_int( 0 ) - 1;
    for ( int i = 0; i < n_queries; ++i )
        TEST_COMPARE_ARRAYS( extractAndSort( std::get<1>( reference ),
                                             offset( i ), offset( i + 1 ) ),
                             extractAndSort( std::get<1>( other ), offset( i ),
                                             offset( i + 1 ) ) );
}

#endif
