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

#ifndef DTK_LINEAR_BVH_DEF_HPP
#define DTK_LINEAR_BVH_DEF_HPP

#include "DTK_ConfigDefs.hpp"

#include <DTK_DetailsAlgorithms.hpp>
#include <DTK_DetailsTreeConstruction.hpp>
#include <DTK_KokkosHelpers.hpp>

#include <Kokkos_ArithTraits.hpp>

namespace DataTransferKit
{
template <typename DeviceType>
BoundingVolumeHierarchy<DeviceType>::BoundingVolumeHierarchy(
    Kokkos::View<Box const *, DeviceType> bounding_boxes )
    : _leaf_nodes( Kokkos::ViewAllocateWithoutInitializing( "leaf_nodes" ),
                   bounding_boxes.extent( 0 ) )
    , _internal_nodes(
          Kokkos::ViewAllocateWithoutInitializing( "internal_nodes" ),
          bounding_boxes.extent( 0 ) > 0 ? bounding_boxes.extent( 0 ) - 1 : 0 )
    , _indices( Kokkos::ViewAllocateWithoutInitializing( "sorted_indices" ),
                bounding_boxes.extent( 0 ) )
{
    using ExecutionSpace = typename DeviceType::execution_space;

    if ( empty() )
    {
        return;
    }

    if ( size() == 1 )
    {
        iota( _indices );
        Details::TreeConstruction<DeviceType>::initializeLeafNodes(
            _indices, bounding_boxes, _leaf_nodes );
        return;
    }

    // determine the bounding box of the scene
    Details::TreeConstruction<DeviceType>::calculateBoundingBoxOfTheScene(
        bounding_boxes, _internal_nodes[0].bounding_box );

    // calculate morton code of all objects
    int const n = bounding_boxes.extent( 0 );
    Kokkos::View<unsigned int *, DeviceType> morton_indices(
        Kokkos::ViewAllocateWithoutInitializing( "morton" ), n );
    Details::TreeConstruction<DeviceType>::assignMortonCodes(
        bounding_boxes, morton_indices, _internal_nodes[0].bounding_box );

    // sort them along the Z-order space-filling curve
    iota( _indices );
    Details::TreeConstruction<DeviceType>::sortObjects( morton_indices,
                                                        _indices );

    Details::TreeConstruction<DeviceType>::initializeLeafNodes(
        _indices, bounding_boxes, _leaf_nodes );

    // generate bounding volume hierarchy
    Details::TreeConstruction<DeviceType>::generateHierarchy(
        morton_indices, _leaf_nodes, _internal_nodes );

    // calculate bounding box for each internal node by walking the hierarchy
    // toward the root
    Details::TreeConstruction<DeviceType>::calculateBoundingBoxes(
        _leaf_nodes, _internal_nodes );
}

} // namespace DataTransferKit

// Explicit instantiation macro
#define DTK_LINEAR_BVH_INSTANT( NODE )                                         \
    template class BoundingVolumeHierarchy<typename NODE::device_type>;

#endif
