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

#ifndef DTK_BOOST_RTREE_HELPERS_HPP
#define DTK_BOOST_RTREE_HELPERS_HPP

#include <DTK_DetailsBox.hpp>
#include <DTK_DetailsPoint.hpp>
#include <DTK_DetailsPredicate.hpp>

#include "DTK_BoostGeometryAdapters.hpp"
#include "DTK_BoostRangeAdapters.hpp"
#include <boost/range/adaptors.hpp>

namespace BoostRTreeHelpers
{
// NOTE: The balancing algorithm does not really matter since the tree is
// created using packing algorithm and there is no later insertion of new
// objects into the tree.  The values of the maximum and the minimum number
// of elements could be adjusted in principle but I didn't bother setting
// this type as a template argument.
using Parameter = boost::geometry::index::linear<16>;

template <typename Value, typename Index>
struct PairMaker
{
    template <typename T>
    inline std::pair<Value, Index> operator()( T const &v ) const
    {
        return std::make_pair( v.value(), v.index() );
    }
};

template <typename Indexable>
using RTree =
    boost::geometry::index::rtree<std::pair<Indexable, int>, Parameter>;

template <typename View>
static RTree<typename View::value_type> makeRTree( View const &objects )
{
    using Indexable = typename View::value_type;
    return RTree<Indexable>(
        objects | boost::adaptors::indexed() |
        boost::adaptors::transformed( PairMaker<Indexable, int>() ) );
}

// NOTE: The trailing return type is required with C++11 :(
template <typename Value>
struct UnaryPredicate
{
    using Function = std::function<bool( Value const & )>;
    UnaryPredicate( Function pred )
        : _pred( pred )
    {
    }
    inline bool operator()( Value const &val ) const { return _pred( val ); }
    static UnaryPredicate makeAlwaysFalse()
    {
        return UnaryPredicate( []( Value const & ) { return false; } );
    }
    Function _pred;
};

template <typename Value>
static auto translate(
    DataTransferKit::Details::Intersects<DataTransferKit::Sphere> const &query )
    -> decltype( boost::geometry::index::intersects( DataTransferKit::Box() ) &&
                 boost::geometry::index::satisfies(
                     UnaryPredicate<Value>::makeAlwaysFalse() ) )
{
    auto const sphere = query._geometry;
    auto const radius = sphere.radius();
    auto const centroid = sphere.centroid();
    DataTransferKit::Box box;
    DataTransferKit::Details::expand( box, sphere );
    return boost::geometry::index::intersects( box ) &&
           boost::geometry::index::satisfies(
               UnaryPredicate<Value>( [centroid, radius]( Value const &val ) {
                   boost::geometry::index::indexable<Value> indexableGetter;
                   auto const &geometry = indexableGetter( val );
                   return boost::geometry::distance( centroid, geometry ) <=
                          radius;
               } ) );
}

template <typename Value, typename Geometry>
static auto
translate( DataTransferKit::Details::Nearest<Geometry> const &query )
    -> decltype( boost::geometry::index::nearest( Geometry(), 0 ) )
{
    auto const geometry = query._geometry;
    auto const k = query._k;
    return boost::geometry::index::nearest( geometry, k );
}

template <typename Indexable, typename InputView,
          typename OutputView = Kokkos::View<int *, Kokkos::HostSpace>>
static std::tuple<OutputView, OutputView>
performQueries( RTree<Indexable> const &rtree, InputView const &queries )
{
    using Value = typename RTree<Indexable>::value_type;
    auto const n_queries = queries.extent_int( 0 );
    OutputView offset( "offset", n_queries + 1 );
    std::vector<Value> returned_values;
    for ( int i = 0; i < n_queries; ++i )
        offset( i ) = rtree.query( translate<Value>( queries( i ) ),
                                   std::back_inserter( returned_values ) );
    DataTransferKit::exclusivePrefixSum( offset );
    auto const n_results = DataTransferKit::lastElement( offset );
    OutputView indices( "indices", n_results );
    for ( int i = 0; i < n_queries; ++i )
        for ( int j = offset( i ); j < offset( i + 1 ); ++j )
            indices( j ) = returned_values[j].second;
    return std::make_tuple( offset, indices );
}
} // end namespace BoostRTreeHelpers

#endif
