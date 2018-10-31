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

#ifndef DTK_LINEAR_BVH_DECL_HPP
#define DTK_LINEAR_BVH_DECL_HPP

#include <DTK_Box.hpp>
#include <DTK_DetailsBoundingVolumeHierarchyImpl.hpp>
#include <DTK_DetailsNode.hpp>
#include <DTK_DetailsSortUtils.hpp>
#include <DTK_DetailsTreeConstruction.hpp>

#include <Kokkos_Macros.hpp>
#include <Kokkos_View.hpp>

namespace DataTransferKit
{

template <typename DeviceType>
class BoundingVolumeHierarchy
{
  public:
    using device_type = DeviceType;
    using bounding_volume_type = Box;
    using size_type = typename Kokkos::View<Node *, DeviceType>::size_type;

    BoundingVolumeHierarchy() = default; // build an empty tree

    template <typename Primitives>
    BoundingVolumeHierarchy( Primitives const &primitives );

    KOKKOS_INLINE_FUNCTION
    size_type size() const { return _leaf_nodes.extent( 0 ); }

    KOKKOS_INLINE_FUNCTION
    bool empty() const { return size() == 0; }

    KOKKOS_INLINE_FUNCTION
    bounding_volume_type bounds() const
    {
        // NOTE should default constructor initialize to an invalid geometry?
        if ( empty() )
            return bounding_volume_type();
        // FIXME this->getBoundingVolume( this->getRoot() );
        return ( size() > 1 ? _internal_nodes : _leaf_nodes )[0].bounding_box;
    }

    template <typename Predicates, typename... Args>
    inline void query( Predicates const &predicates, Args &&... args ) const
    {
        // FIXME lame placeholder for concept check
        static_assert( Kokkos::is_view<Predicates>::value, "must pass a view" );
        using Tag = typename Predicates::value_type::Tag;
        Details::BoundingVolumeHierarchyImpl<DeviceType>::queryDispatch(
            Tag{}, *this, predicates, std::forward<Args>( args )... );
    }

  private:
    friend struct Details::TreeTraversal<DeviceType>;

    Kokkos::View<Node *, DeviceType> _leaf_nodes;
    Kokkos::View<Node *, DeviceType> _internal_nodes;
};

template <typename DeviceType>
using BVH = BoundingVolumeHierarchy<DeviceType>;

template <typename DeviceType>
template <typename Primitives>
BoundingVolumeHierarchy<DeviceType>::BoundingVolumeHierarchy(
    Primitives const &primitives )
    : _leaf_nodes( Kokkos::ViewAllocateWithoutInitializing( "leaf_nodes" ),
                   primitives.extent( 0 ) )
    , _internal_nodes(
          Kokkos::ViewAllocateWithoutInitializing( "internal_nodes" ),
          primitives.extent( 0 ) > 0 ? primitives.extent( 0 ) - 1 : 0 )
{
    // FIXME lame placeholder for concept check
    static_assert( Kokkos::is_view<Primitives>::value, "must pass a view" );

    if ( empty() )
    {
        return;
    }

    if ( size() == 1 )
    {
        Kokkos::View<size_t *, DeviceType> permutation_indices( "permute", 1 );
        Details::TreeConstruction<DeviceType>::initializeLeafNodes(
            permutation_indices, primitives, _leaf_nodes );
        return;
    }

    // determine the bounding box of the scene
    Details::TreeConstruction<DeviceType>::calculateBoundingBoxOfTheScene(
        // FIXME this->getBoundingVolume( this->getRoot() );
        primitives, _internal_nodes[0].bounding_box );

    // calculate morton code of all objects
    auto const n = primitives.extent( 0 );
    Kokkos::View<unsigned int *, DeviceType> morton_indices(
        Kokkos::ViewAllocateWithoutInitializing( "morton" ), n );
    Details::TreeConstruction<DeviceType>::assignMortonCodes(
        // FIXME this->getBoundingVolume( this->getRoot() );
        primitives, morton_indices, _internal_nodes[0].bounding_box );

    // sort them along the Z-order space-filling curve
    auto permutation_indices = Details::sortObjects( morton_indices );

    Details::TreeConstruction<DeviceType>::initializeLeafNodes(
        permutation_indices, primitives, _leaf_nodes );

    // generate bounding volume hierarchy
    Details::TreeConstruction<DeviceType>::generateHierarchy(
        morton_indices, _leaf_nodes, _internal_nodes );

    // calculate bounding box for each internal node by walking the hierarchy
    // toward the root
    Details::TreeConstruction<DeviceType>::calculateBoundingBoxes(
        _leaf_nodes, _internal_nodes );
}

} // namespace DataTransferKit

#endif
