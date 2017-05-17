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
#include <DTK_Node.hpp>

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

struct Nearest
{
    using Tag = NearestPredicateTag;
    KOKKOS_INLINE_FUNCTION
    Nearest( Point const &query_point, int k )
        : _query_point( query_point )
        , _k( k )
    {
    }

    Point const _query_point;
    int const _k;
};

class Within
{
  public:
    using Tag = SpatialPredicateTag;
    KOKKOS_INLINE_FUNCTION
    Within( Point const &query_point, double const radius )
        : _query_point( query_point )
        , _radius( radius )
    {
    }

    KOKKOS_INLINE_FUNCTION
    bool operator()( Node const *node ) const
    {
        double node_distance = distance( _query_point, node->bounding_box );
        return ( node_distance <= _radius ) ? true : false;
    }

  private:
    Point const _query_point;
    double const _radius;
};

KOKKOS_INLINE_FUNCTION
Nearest nearest( Point const &g, int k = 1 ) { return Nearest( g, k ); }
}
}

#endif
