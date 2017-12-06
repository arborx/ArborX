/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/
#ifndef DTK_PREDICATE_HPP
#define DTK_PREDICATE_HPP

#include <DTK_DetailsAlgorithms.hpp>
#include <DTK_DetailsNode.hpp>

namespace DataTransferKit
{
namespace Details
{

struct NearestPredicateTag
{
};
struct SpatialPredicateTag
{
};

// COMMENT: Default constructor and assignment operator are required to be able
// to declare a Kokkos::View of a predicate type and fill it with a
// Kokkos::for_parallel.

struct Nearest
{
    using Tag = NearestPredicateTag;

    KOKKOS_INLINE_FUNCTION
    Nearest()
        : _query_point( {{0., 0., 0.}} )
        , _k( 0 )
    {
    }

    KOKKOS_INLINE_FUNCTION Nearest &operator=( Nearest const &other )
    {
        _query_point = other._query_point;
        _k = other._k;
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    Nearest( Point const &query_point, int k )
        : _query_point( query_point )
        , _k( k )
    {
    }

    Point _query_point;
    int _k;
};

template <typename Geometry>
struct Intersects
{
    using Tag = SpatialPredicateTag;

    KOKKOS_INLINE_FUNCTION Intersects() = default;

    KOKKOS_INLINE_FUNCTION Intersects( Geometry const &geometry )
        : _geometry( geometry )
    {
    }

    KOKKOS_INLINE_FUNCTION
    bool operator()( Node const *node ) const
    {
        return intersects( _geometry, node->bounding_box );
    }

    Geometry _geometry;
};

using Within = Intersects<Sphere>;
using Overlap = Intersects<Box>;

KOKKOS_INLINE_FUNCTION
Nearest nearest( Point const &p, int k = 1 ) { return Nearest( p, k ); }

KOKKOS_INLINE_FUNCTION
Within within( Point const &p, double r ) { return Within( {p, r} ); }

KOKKOS_INLINE_FUNCTION
Overlap overlap( Box const &b ) { return Overlap( b ); }

} // end namespace Details
} // end namespace DataTransferKit

#endif
