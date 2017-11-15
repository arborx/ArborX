/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/
#ifndef DTK_DETAILS_ALGORITHMS_HPP
#define DTK_DETAILS_ALGORITHMS_HPP

#include <DTK_DetailsBox.hpp>
#include <DTK_DetailsPoint.hpp>
#include <DTK_KokkosHelpers.hpp>

#include <Kokkos_Macros.hpp>

namespace DataTransferKit
{
namespace Details
{
// distance point-point
KOKKOS_INLINE_FUNCTION
double distance( Point const &a, Point const &b )
{
    double distance_squared = 0.0;
    for ( int d = 0; d < 3; ++d )
    {
        double tmp = b[d] - a[d];
        distance_squared += tmp * tmp;
    }
    return std::sqrt( distance_squared );
}

// distance point-box
KOKKOS_INLINE_FUNCTION
double distance( Point const &point, Box const &box )
{
    Point projected_point;
    for ( int d = 0; d < 3; ++d )
    {
        if ( point[d] < box.minCorner()[d] )
            projected_point[d] = box.minCorner()[d];
        else if ( point[d] > box.maxCorner()[d] )
            projected_point[d] = box.maxCorner()[d];
        else
            projected_point[d] = point[d];
    }
    return distance( point, projected_point );
}

// expand an axis-aligned bounding box to include a point
KOKKOS_INLINE_FUNCTION
void expand( Box &box, Point const &point )
{
    for ( int d = 0; d < 3; ++d )
    {
        if ( point[d] < box.minCorner()[d] )
            box.minCorner()[d] = point[d];
        if ( point[d] > box.maxCorner()[d] )
            box.maxCorner()[d] = point[d];
    }
}

// expand an axis-aligned bounding box to include another box
KOKKOS_INLINE_FUNCTION
void expand( Box &box, Box const &other )
{
    for ( int d = 0; d < 3; ++d )
    {
        if ( box.minCorner()[d] > other.minCorner()[d] )
            box.minCorner()[d] = other.minCorner()[d];
        if ( box.maxCorner()[d] < other.maxCorner()[d] )
            box.maxCorner()[d] = other.maxCorner()[d];
    }
}

// check if two axis-aligned bounding boxes overlap
KOKKOS_INLINE_FUNCTION
bool overlaps( Box const &box, Box const &other )
{
    for ( int d = 0; d < 3; ++d )
        if ( box.minCorner()[d] > other.maxCorner()[d] ||
             box.maxCorner()[d] < other.minCorner()[d] )
            return false;
    return true;
}

// calculate the centroid of a box
KOKKOS_INLINE_FUNCTION
void centroid( Box const &box, Point &c )
{
    for ( int d = 0; d < 3; ++d )
        c[d] = 0.5 * ( box.minCorner()[d] + box.maxCorner()[d] );
}

// FIXME: use expand()
template <typename DeviceType>
class ExpandBoxWithBoxFunctor
{
  public:
    ExpandBoxWithBoxFunctor(
        Kokkos::View<Box const *, DeviceType> bounding_boxes )
        : _greatest( Kokkos::ArithTraits<double>::max() )
        , _lowest( -_greatest )
        , _bounding_boxes( bounding_boxes )
    {
    }

    KOKKOS_INLINE_FUNCTION
    void init( Box &box ) const
    {
        for ( int d = 0; d < 3; ++d )
        {
            box.minCorner()[d] = _greatest;
            box.maxCorner()[d] = _lowest;
        }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( int const i, Box &box ) const
    {
        for ( int d = 0; d < 3; ++d )
        {
            box.minCorner()[d] = KokkosHelpers::min(
                _bounding_boxes[i].minCorner()[d], box.minCorner()[d] );
            box.maxCorner()[d] = KokkosHelpers::max(
                _bounding_boxes[i].maxCorner()[d], box.maxCorner()[d] );
        }
    }

    KOKKOS_INLINE_FUNCTION
    void join( volatile Box &dst, volatile Box const &src ) const
    {
        for ( int d = 0; d < 3; ++d )
        {
            dst.minCorner()[d] =
                KokkosHelpers::min( src.minCorner()[d], dst.minCorner()[d] );
            dst.maxCorner()[d] =
                KokkosHelpers::max( src.maxCorner()[d], dst.maxCorner()[d] );
        }
    }

  private:
    double const _greatest;
    double const _lowest;
    Kokkos::View<Box const *, DeviceType> _bounding_boxes;
};

} // end namespace Details
} // end namespace DataTransferKit

#endif
