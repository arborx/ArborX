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

#include <DTK_Box.hpp>
#include <DTK_DetailsNode.hpp>
#include <DTK_KokkosHelpers.hpp> // clz

#include <Kokkos_Macros.hpp>
#include <Kokkos_Pair.hpp>
#include <Kokkos_View.hpp>

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
    initializeLeafNodes( Kokkos::View<size_t const *, DeviceType> indices,
                         Kokkos::View<Box const *, DeviceType> bounding_boxes,
                         Kokkos::View<Node *, DeviceType> leaf_nodes );

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

    KOKKOS_FUNCTION
    static int
    findSplit( Kokkos::View<unsigned int *, DeviceType> sorted_morton_codes,
               int first, int last );

    KOKKOS_FUNCTION
    static Kokkos::pair<int, int> determineRange(
        Kokkos::View<unsigned int *, DeviceType> sorted_morton_codes, int i );
};
} // namespace Details
} // namespace DataTransferKit

#endif
