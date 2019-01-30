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

#ifndef DTK_DETAILS_TREE_CONSTRUCTION_DEF_HPP
#define DTK_DETAILS_TREE_CONSTRUCTION_DEF_HPP

#include "DTK_ConfigDefs.hpp"

#include <DTK_DBC.hpp>
#include <DTK_DetailsAlgorithms.hpp>
#include <DTK_DetailsUtils.hpp> // iota

#include <DTK_KokkosHelpers.hpp> // sgn, min, max

#include <Kokkos_Atomic.hpp>
#include <Kokkos_Sort.hpp>

#include <cassert>

namespace DataTransferKit
{
namespace Details
{

template <typename DeviceType>
class CalculateInternalNodesBoundingVolumesFunctor
{
  public:
    CalculateInternalNodesBoundingVolumesFunctor(
        Node *root, Kokkos::View<int const *, DeviceType> parents,
        size_t n_internal_nodes )
        : _root( root )
        , _flags( Kokkos::ViewAllocateWithoutInitializing( "flags" ),
                  n_internal_nodes )
        , _parents( parents )
    {
        // Initialize flags to zero
        Kokkos::deep_copy( _flags, 0 );
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( int const i ) const
    {
        Node *node = _root + _parents( i );
        // Walk toward the root but do not actually process it because its
        // bounding box has already been computed (bounding box of the scene)
        while ( node != _root )
        {
            // Use an atomic flag per internal node to terminate the first
            // thread that enters it, while letting the second one through.
            // This ensures that every node gets processed only once, and not
            // before both of its children are processed.
            if ( Kokkos::atomic_compare_exchange_strong(
                     &_flags( node - _root ), 0, 1 ) )
                break;

            // Internal node bounding boxes are unitialized hence the
            // assignment operator below.
            // FIXME: accessing Node::bounding_box is not ideal but I was
            // reluctant to pass the bounding volume hierarchy to
            // generateHierarchy()
            node->bounding_box = ( node->children.first )->bounding_box;
            expand( node->bounding_box,
                    ( node->children.second )->bounding_box );

            node = _root + _parents( node - _root );
        }
        // NOTE: could check that bounding box of the root node is indeed the
        // union of the two children.
    }

  private:
    Node *_root;
    // Use int instead of bool because CAS (Compare And Swap) on CUDA does not
    // support boolean
    Kokkos::View<int *, DeviceType> _flags;
    Kokkos::View<int const *, DeviceType> _parents;
};

template <typename DeviceType>
void TreeConstruction<DeviceType>::calculateInternalNodesBoundingVolumes(
    Kokkos::View<Node const *, DeviceType> leaf_nodes,
    Kokkos::View<Node *, DeviceType> internal_nodes,
    Kokkos::View<int const *, DeviceType> parents )
{
    auto const first = internal_nodes.extent( 0 );
    auto const last = first + leaf_nodes.extent( 0 );
    Node *root = internal_nodes.data();
    Kokkos::parallel_for(
        DTK_MARK_REGION( "calculate_bounding_boxes" ),
        Kokkos::RangePolicy<ExecutionSpace>( first, last ),
        CalculateInternalNodesBoundingVolumesFunctor<DeviceType>( root, parents,
                                                                  first ) );
    Kokkos::fence();
}
} // namespace Details
} // namespace DataTransferKit

// Explicit instantiation macro
#define DTK_TREECONSTRUCTION_INSTANT( NODE )                                   \
    namespace Details                                                          \
    {                                                                          \
    template struct TreeConstruction<typename NODE::device_type>;              \
    }

#endif
