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
        if ( point[d] < box[2 * d + 0] )
            projected_point[d] = box[2 * d + 0];
        else if ( point[d] > box[2 * d + 1] )
            projected_point[d] = box[2 * d + 1];
        else
            projected_point[d] = point[d];
    }
    return distance( point, projected_point );
}

// expand an axis-aligned bounding box to include a point
void expand( Box &box, Point const &point );

// expand an axis-aligned bounding box to include another box
KOKKOS_INLINE_FUNCTION
void expand( Box &box, Box const &other )
{
    for ( int d = 0; d < 3; ++d )
    {
        if ( box[2 * d + 0] > other[2 * d + 0] )
            box[2 * d + 0] = other[2 * d + 0];
        if ( box[2 * d + 1] < other[2 * d + 1] )
            box[2 * d + 1] = other[2 * d + 1];
    }
}

// check if two axis-aligned bounding boxes overlap
KOKKOS_INLINE_FUNCTION
bool overlaps( Box const &box, Box const &other )
{
    for ( int d = 0; d < 3; ++d )
        if ( box[2 * d + 0] > other[2 * d + 1] ||
             box[2 * d + 1] < other[2 * d + 0] )
            return false;
    return true;
}

// calculate the centroid of a box
KOKKOS_INLINE_FUNCTION
void centroid( Box const &box, Point &c )
{
    for ( int d = 0; d < 3; ++d )
        c[d] = 0.5 * ( box[2 * d + 0] + box[2 * d + 1] );
}

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
KOKKOS_INLINE_FUNCTION
unsigned int expandBits( unsigned int v )
{
    v = ( v * 0x00010001u ) & 0xFF0000FFu;
    v = ( v * 0x00000101u ) & 0x0F00F00Fu;
    v = ( v * 0x00000011u ) & 0xC30C30C3u;
    v = ( v * 0x00000005u ) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
KOKKOS_INLINE_FUNCTION
unsigned int morton3D( double x, double y, double z )
{
    x = KokkosHelpers::min( KokkosHelpers::max( x * 1024.0, 0.0 ), 1023.0 );
    y = KokkosHelpers::min( KokkosHelpers::max( y * 1024.0, 0.0 ), 1023.0 );
    z = KokkosHelpers::min( KokkosHelpers::max( z * 1024.0, 0.0 ), 1023.0 );
    unsigned int xx = expandBits( (unsigned int)x );
    unsigned int yy = expandBits( (unsigned int)y );
    unsigned int zz = expandBits( (unsigned int)z );
    return xx * 4 + yy * 2 + zz;
}

#define __clz( x ) __builtin_clz( x )
// TODO: this is a mess
// we need a default impl
//#define __clz( x ) __builtin_clz( x )
// default implementation if nothing else is available
// Taken from:
// http://stackoverflow.com/questions/23856596/counting-leading-zeros-in-a-32-bit-unsigned-integer-with-best-algorithm-in-c-pro
KOKKOS_INLINE_FUNCTION
int clz( uint32_t x )
{
    if ( x == 0 )
        return 32;
    static const char debruijn32[32] = {
        0, 31, 9, 30, 3, 8,  13, 29, 2,  5,  7,  21, 12, 24, 28, 19,
        1, 10, 4, 14, 6, 22, 25, 20, 11, 15, 23, 26, 16, 27, 17, 18};
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x++;
    return debruijn32[x * 0x076be629 >> 27];
}

// TODO: use preprocessor directive to select an implementation
// it turns out NVDIA's implementation of int __clz(unsigned int x) is
// slightly different than GCC __builtin_clz
// this caused a bug in an early implementation of the function that compute
// the common prefixes between two keys (NB: when i == j)
KOKKOS_INLINE_FUNCTION
int countLeadingZeros( unsigned int x )
{
#if defined __CUDACC__
    // intrinsic function that is only supported in device code
    // COMMENT: not sure how I am supposed to use it then...
    // TODO for now don't use the builtin functions
    // return __clz( x );
    return clz( x );

#elif defined __GNUC__
    // int __builtin_clz(unsigned int x) result is undefined if x is 0
    // TODO for now don't use the builtin functions
    // return x != 0 ? __builtin_clz( x ) : 32;
    return x != 0 ? clz( x ) : 32;
#else
    // similar problem with the default implementation
    return x != 0 ? clz( x ) : 32;
#endif
}

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
            box[2 * d] = _greatest;
            box[2 * d + 1] = _lowest;
        }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( int const i, Box &box ) const
    {
        for ( int d = 0; d < 3; ++d )
        {
            box[2 * d] =
                KokkosHelpers::min( _bounding_boxes[i][2 * d], box[2 * d] );
            box[2 * d + 1] = KokkosHelpers::max( _bounding_boxes[i][2 * d + 1],
                                                 box[2 * d + 1] );
        }
    }

    KOKKOS_INLINE_FUNCTION
    void join( volatile Box &dst, volatile Box const &src ) const
    {
        for ( int d = 0; d < 3; ++d )
        {
            dst[2 * d] = KokkosHelpers::min( src[2 * d], dst[2 * d] );
            dst[2 * d + 1] =
                KokkosHelpers::max( src[2 * d + 1], dst[2 * d + 1] );
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
