/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/

#ifndef DTK_TREECONSTRUCTION_DECL_HPP
#define DTK_TREECONSTRUCTION_DECL_HPP

#include "DTK_ConfigDefs.hpp"
#include <details/DTK_DetailsAlgorithms.hpp>

#include <DTK_Box.hpp>
#include <DTK_Node.hpp>
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
template <typename NO>
struct TreeConstruction
{
  public:
    using DeviceType = typename NO::device_type;
    using ExecutionSpace = typename DeviceType::execution_space;

    static void calculateBoundingBoxOfTheScene(
        Kokkos::View<BBox const *, DeviceType> bounding_boxes,
        BBox &scene_bounding_box );

    // to assign the Morton code for a given object, we use the centroid point
    // of its bounding box, and express it relative to the bounding box of the
    // scene.
    static void
    assignMortonCodes( Kokkos::View<BBox const *, DeviceType> bounding_boxes,
                       Kokkos::View<unsigned int *, DeviceType> morton_codes,
                       BBox const &scene_bounding_box );

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
        int const n = morton_codes.extent( 0 );
        if ( j < 0 || j > n - 1 )
            return -1;

        // our construction algorithm relies on keys being unique so we handle
        // explicitly case of duplicate Morton codes by augmenting each key by
        // a bit representation of its index.
        if ( morton_codes[i] == morton_codes[j] )
        {
            // countLeadingZeros( k[i] ^ k[j] ) == 32
            return 32 + Details::countLeadingZeros( i ^ j );
        }
        return Details::countLeadingZeros( morton_codes[i] ^ morton_codes[j] );
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
