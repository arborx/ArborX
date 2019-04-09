/****************************************************************************
 * Copyright (c) 2012-2019 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_DETAILS_MORTON_CODE_UTILS_HPP
#define ARBORX_DETAILS_MORTON_CODE_UTILS_HPP

#include <ArborX_Exception.hpp>

#include <ArborX_DetailsKokkosExt.hpp> // min. max

namespace ArborX
{

namespace Details
{

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
    using KokkosExt::max;
    using KokkosExt::min;

    // The interval [0,1] is subdivided into 1024 bins (in each direction).
    // If we were to use more bits to encode the Morton code, we would need
    // to reflect these changes in expandBits() as well as in the clz()
    // function that returns the number of leading zero bits since it
    // currently assumes that the code can be represented by a 32 bit
    // integer.
    x = min( max( x * 1024.0, 0.0 ), 1023.0 );
    y = min( max( y * 1024.0, 0.0 ), 1023.0 );
    z = min( max( z * 1024.0, 0.0 ), 1023.0 );
    unsigned int xx = expandBits( (unsigned int)x );
    unsigned int yy = expandBits( (unsigned int)y );
    unsigned int zz = expandBits( (unsigned int)z );
    return xx * 4 + yy * 2 + zz;
}

} // namespace Details

} // namespace ArborX

#endif
