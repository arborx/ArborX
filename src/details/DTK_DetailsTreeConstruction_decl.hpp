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

#ifndef DTK_DETAILS_TREE_CONSTRUCTION_DECL_HPP
#define DTK_DETAILS_TREE_CONSTRUCTION_DECL_HPP

#include "DTK_ConfigDefs.hpp"

#include <DTK_DetailsBox.hpp>
#include <DTK_DetailsNode.hpp>
#include <DTK_KokkosHelpers.hpp> // clz, min. max

#include <Kokkos_Core.hpp>
#include <Kokkos_Pair.hpp>

namespace DataTransferKit
{
namespace Details
{
/**
 * This structure contains all the functions used to build the BVH. All the
 * functions are static.
 */
template <typename DeviceType>
struct TreeConstruction
{
  public:
    using ExecutionSpace = typename DeviceType::execution_space;

    static void calculateBoundingBoxOfTheScene(
        Kokkos::View<Box const *, DeviceType> bounding_boxes,
        Box &scene_bounding_box );

    // to assign the Morton code for a given object, we use the centroid point
    // of its bounding box, and express it relative to the bounding box of the
    // scene.
    static void
    assignMortonCodes( Kokkos::View<Box const *, DeviceType> bounding_boxes,
                       Kokkos::View<unsigned int *, DeviceType> morton_codes,
                       Box const &scene_bounding_box );

    static void
    sortObjects( Kokkos::View<unsigned int *, DeviceType> morton_codes,
                 Kokkos::View<int *, DeviceType> object_ids );

    static Node *generateHierarchy(
        Kokkos::View<unsigned int *, DeviceType> sorted_morton_codes,
        Kokkos::View<Node *, DeviceType> leaf_nodes,
        Kokkos::View<Node *, DeviceType> internal_nodes );

    static void
    calculateBoundingBoxes( Kokkos::View<Node *, DeviceType> leaf_nodes,
                            Kokkos::View<Node *, DeviceType> internal_nodes );

    KOKKOS_INLINE_FUNCTION
    static int
    commonPrefix( Kokkos::View<unsigned int *, DeviceType> morton_codes, int i,
                  int j )
    {
        using KokkosHelpers::clz;

        int const n = morton_codes.extent( 0 );
        if ( j < 0 || j > n - 1 )
            return -1;

        // our construction algorithm relies on keys being unique so we handle
        // explicitly case of duplicate Morton codes by augmenting each key by
        // a bit representation of its index.
        if ( morton_codes[i] == morton_codes[j] )
        {
            // clz( k[i] ^ k[j] ) == 32
            return 32 + clz( i ^ j );
        }
        return clz( morton_codes[i] ^ morton_codes[j] );
    }

    // Expands a 10-bit integer into 30 bits
    // by inserting 2 zeros after each bit.
    KOKKOS_INLINE_FUNCTION
    static unsigned int expandBits( unsigned int v )
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
    static unsigned int morton3D( double x, double y, double z )
    {
        using KokkosHelpers::max;
        using KokkosHelpers::min;

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

    KOKKOS_FUNCTION
    static int
    findSplit( Kokkos::View<unsigned int *, DeviceType> sorted_morton_codes,
               int first, int last );

    KOKKOS_FUNCTION
    static Kokkos::pair<int, int> determineRange(
        Kokkos::View<unsigned int *, DeviceType> sorted_morton_codes, int i );
};
}
}

#endif
