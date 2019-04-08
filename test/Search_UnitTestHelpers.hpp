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

#ifndef DTK_SEARCH_TEST_HELPERS_HPP
#define DTK_SEARCH_TEST_HELPERS_HPP

#include <DTK_DetailsKokkosExt.hpp> // is_accessible_from
#include <DTK_DistributedSearchTree.hpp>
#include <DTK_LinearBVH.hpp>

#include <Kokkos_View.hpp>

#include "DTK_EnableViewComparison.hpp"

#include <boost/test/unit_test.hpp>

#include <tuple>
#include <vector>

namespace tt = boost::test_tools;

template <typename Query, typename DeviceType>
void checkResults( DataTransferKit::BVH<DeviceType> const &bvh,
                   Kokkos::View<Query *, DeviceType> const &queries,
                   std::vector<int> const &indices_ref,
                   std::vector<int> const &offset_ref )
{
    Kokkos::View<int *, DeviceType> indices( "indices" );
    Kokkos::View<int *, DeviceType> offset( "offset" );
    bvh.query( queries, indices, offset );

    auto indices_host = Kokkos::create_mirror_view( indices );
    deep_copy( indices_host, indices );
    auto offset_host = Kokkos::create_mirror_view( offset );
    deep_copy( offset_host, offset );

    BOOST_TEST( indices_host == indices_ref, tt::per_element() );
    BOOST_TEST( offset_host == offset_ref, tt::per_element() );
}

// Same as above except that we get the distances out of the queries and
// compare them to the reference solution passed as argument.  Templated type
// `Query` is pretty much a nearest predicate in this case.
template <typename Query, typename DeviceType>
void checkResults( DataTransferKit::BVH<DeviceType> const &bvh,
                   Kokkos::View<Query *, DeviceType> const &queries,
                   std::vector<int> const &indices_ref,
                   std::vector<int> const &offset_ref,
                   std::vector<double> const &distances_ref )
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

    BOOST_TEST( indices_host == indices_ref, tt::per_element() );
    BOOST_TEST( offset_host == offset_ref, tt::per_element() );
    BOOST_TEST( distances_host == distances_ref, tt::per_element() );
}

template <typename Query, typename DeviceType>
void checkResults(
    DataTransferKit::DistributedSearchTree<DeviceType> const &tree,
    Kokkos::View<Query *, DeviceType> const &queries,
    std::vector<int> const &indices_ref, std::vector<int> const &offset_ref,
    std::vector<int> const &ranks_ref )
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

    BOOST_TEST( offset_host == offset_ref, tt::per_element() );
    auto const m = offset_host.extent_int( 0 ) - 1;
    for ( int i = 0; i < m; ++i )
    {
        std::vector<std::tuple<int, int>> l;
        std::vector<std::tuple<int, int>> r;
        for ( int j = offset_ref[i]; j < offset_ref[i + 1]; ++j )
        {
            l.push_back( std::make_tuple( ranks_host[j], indices_host[j] ) );
            r.push_back( std::make_tuple( ranks_ref[j], indices_ref[j] ) );
        }
        sort( l.begin(), l.end() );
        sort( r.begin(), r.end() );
        BOOST_TEST( l.size() == r.size() );
        int const n = l.size();
        BOOST_TEST( n == offset_ref[i + 1] - offset_ref[i] );
        for ( int j = 0; j < n; ++j )
        {
            // FIXME_BOOST would be nice if we could compare tuples
            BOOST_TEST( std::get<0>( l[j] ) == std::get<0>( r[j] ) );
            BOOST_TEST( std::get<1>( l[j] ) == std::get<1>( r[j] ) );
        }
    }
}

template <typename Query, typename DeviceType>
void checkResults(
    DataTransferKit::DistributedSearchTree<DeviceType> const &tree,
    Kokkos::View<Query *, DeviceType> const &queries,
    std::vector<int> const &indices_ref, std::vector<int> const &offset_ref,
    std::vector<int> const &ranks_ref,
    std::vector<double> const &distances_ref )
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

    BOOST_TEST( indices_host == indices_ref, tt::per_element() );
    BOOST_TEST( offset_host == offset_ref, tt::per_element() );
    BOOST_TEST( ranks_host == ranks_ref, tt::per_element() );
    BOOST_TEST( distances_host != distances_ref, tt::per_element() );
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
makeDistributedSearchTree( MPI_Comm comm,
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
Kokkos::View<DataTransferKit::Overlap *, DeviceType>
makeOverlapQueries( std::vector<DataTransferKit::Box> const &boxes )
{
    int const n = boxes.size();
    Kokkos::View<DataTransferKit::Overlap *, DeviceType> queries(
        "overlap_queries", n );
    auto queries_host = Kokkos::create_mirror_view( queries );
    for ( int i = 0; i < n; ++i )
        queries_host( i ) = DataTransferKit::overlap( boxes[i] );
    Kokkos::deep_copy( queries, queries_host );
    return queries;
}

template <typename DeviceType>
Kokkos::View<DataTransferKit::Nearest<DataTransferKit::Point> *, DeviceType>
makeNearestQueries(
    std::vector<std::pair<DataTransferKit::Point, int>> const &points )
{
    // NOTE: `points` is not a very descriptive name here. It stores both the
    // actual point and the number k of neighbors to query for.
    int const n = points.size();
    Kokkos::View<DataTransferKit::Nearest<DataTransferKit::Point> *, DeviceType>
        queries( "nearest_queries", n );
    auto queries_host = Kokkos::create_mirror_view( queries );
    for ( int i = 0; i < n; ++i )
        queries_host( i ) =
            DataTransferKit::nearest( points[i].first, points[i].second );
    Kokkos::deep_copy( queries, queries_host );
    return queries;
}

template <typename DeviceType>
Kokkos::View<DataTransferKit::Within *, DeviceType> makeWithinQueries(
    std::vector<std::pair<DataTransferKit::Point, double>> const &points )
{
    // NOTE: `points` is not a very descriptive name here. It stores both the
    // actual point and the radius for the search around that point.
    int const n = points.size();
    Kokkos::View<DataTransferKit::Within *, DeviceType> queries(
        "within_queries", n );
    auto queries_host = Kokkos::create_mirror_view( queries );
    for ( int i = 0; i < n; ++i )
        queries_host( i ) =
            DataTransferKit::within( points[i].first, points[i].second );
    Kokkos::deep_copy( queries, queries_host );
    return queries;
}

template <typename InputView1, typename InputView2>
void validateResults( std::tuple<InputView1, InputView1> const &reference,
                      std::tuple<InputView2, InputView2> const &other )
{
    static_assert( KokkosExt::is_accessible_from_host<InputView1>::value, "" );
    static_assert( KokkosExt::is_accessible_from_host<InputView2>::value, "" );
    BOOST_TEST( std::get<0>( reference ) == std::get<0>( other ),
                tt::per_element() );
    auto const offset = std::get<0>( reference );
    auto const m = offset.extent_int( 0 ) - 1;
    for ( int i = 0; i < m; ++i )
    {
        std::vector<int> l;
        std::vector<int> r;
        for ( int j = offset[i]; j < offset[i + 1]; ++j )
        {
            l.push_back( std::get<1>( other )[j] );
            r.push_back( std::get<1>( reference )[j] );
        }
        std::sort( l.begin(), l.end() );
        std::sort( r.begin(), r.end() );
        BOOST_TEST( l.size() == r.size() );
        int const n = l.size();
        BOOST_TEST( n == offset[i + 1] - offset[i] );
        BOOST_TEST( l == r, tt::per_element() );
    }
}

template <typename InputView1, typename InputView2>
void validateResults(
    std::tuple<InputView1, InputView1, InputView1> const &reference,
    std::tuple<InputView2, InputView2, InputView2> const &other )
{
    static_assert( KokkosExt::is_accessible_from_host<InputView1>::value, "" );
    static_assert( KokkosExt::is_accessible_from_host<InputView2>::value, "" );
    BOOST_TEST( std::get<0>( reference ) == std::get<0>( other ),
                tt::per_element() );
    auto const offset = std::get<0>( reference );
    auto const m = offset.extent_int( 0 ) - 1;
    for ( int i = 0; i < m; ++i )
    {
        std::vector<std::tuple<int, int>> l;
        std::vector<std::tuple<int, int>> r;
        for ( int j = offset[i]; j < offset[i + 1]; ++j )
        {
            l.emplace_back( std::get<1>( other )[j], std::get<2>( other )[j] );
            r.emplace_back( std::get<1>( reference )[j],
                            std::get<2>( reference )[j] );
        }
        std::sort( l.begin(), l.end() );
        std::sort( r.begin(), r.end() );
        // somehow can't use TEST_COMPARE_ARRAY() so doing it myself
        BOOST_TEST( l.size() == r.size() );
        int const n = l.size();
        BOOST_TEST( n == offset( i + 1 ) - offset( i ) );
        for ( int j = 0; j < n; ++j )
        {
            // FIXME_BOOST would be nice if we could compare tuples
            BOOST_TEST( std::get<0>( l[j] ) == std::get<0>( r[j] ) );
            BOOST_TEST( std::get<1>( l[j] ) == std::get<1>( r[j] ) );
        }
    }
}

#endif
