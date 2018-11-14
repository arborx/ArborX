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
    size_type size() const { return _size; }

    KOKKOS_INLINE_FUNCTION
    bool empty() const { return size() == 0; }

    KOKKOS_INLINE_FUNCTION
    bounding_volume_type bounds() const
    {
        // NOTE should default constructor initialize to an invalid geometry?
        if ( empty() )
            return bounding_volume_type();
        return getBoundingVolume( getRoot() );
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

    Kokkos::View<Node *, DeviceType> getInternalNodes()
    {
        assert( !empty() );
        return Kokkos::subview( _internal_and_leaf_nodes,
                                std::make_pair( size_type{0}, size() - 1 ) );
    }

    Kokkos::View<Node *, DeviceType> getLeafNodes()
    {
        assert( !empty() );
        return Kokkos::subview( _internal_and_leaf_nodes,
                                std::make_pair( size() - 1, 2 * size() - 1 ) );
    }

    KOKKOS_INLINE_FUNCTION
    Node const *getRoot() const { return _internal_and_leaf_nodes.data(); }

    KOKKOS_INLINE_FUNCTION
    Node *getRoot() { return _internal_and_leaf_nodes.data(); }

    KOKKOS_INLINE_FUNCTION
    bounding_volume_type const &getBoundingVolume( Node const *node ) const
    {
        return node->bounding_box;
    }

    KOKKOS_INLINE_FUNCTION
    bounding_volume_type &getBoundingVolume( Node *node )
    {
        return node->bounding_box;
    }

    KOKKOS_INLINE_FUNCTION
    size_t getLeafPermutationIndex( Node const *leaf ) const
    {
        static_assert( sizeof( size_t ) == sizeof( Node * ),
                       "Conversion is a bad idea if these sizes do not match" );
        return reinterpret_cast<size_t>( leaf->children.second );
    }

    KOKKOS_INLINE_FUNCTION
    bool isLeaf( Node const *node ) const
    {
        return ( node->children.first == nullptr );
    }

    size_t _size;
    Kokkos::View<Node *, DeviceType> _internal_and_leaf_nodes;
};

template <typename DeviceType>
using BVH = BoundingVolumeHierarchy<DeviceType>;

template <typename DeviceType>
template <typename Primitives>
BoundingVolumeHierarchy<DeviceType>::BoundingVolumeHierarchy(
    Primitives const &primitives )
    : _size( primitives.extent( 0 ) )
    , _internal_and_leaf_nodes(
          Kokkos::ViewAllocateWithoutInitializing( "internal_and_leaf_nodes" ),
          _size > 0 ? 2 * _size - 1 : 0 )
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
            permutation_indices, primitives, getLeafNodes() );
        return;
    }

    // determine the bounding box of the scene
    Details::TreeConstruction<DeviceType>::calculateBoundingBoxOfTheScene(
        primitives, getBoundingVolume( getRoot() ) );

    // calculate morton code of all objects
    Kokkos::View<unsigned int *, DeviceType> morton_indices(
        Kokkos::ViewAllocateWithoutInitializing( "morton" ), size() );
    Details::TreeConstruction<DeviceType>::assignMortonCodes(
        primitives, morton_indices, getBoundingVolume( getRoot() ) );

    // sort them along the Z-order space-filling curve
    auto permutation_indices = Details::sortObjects( morton_indices );
    Details::TreeConstruction<DeviceType>::initializeLeafNodes(
        permutation_indices, primitives, getLeafNodes() );

    // generate bounding volume hierarchy
    Kokkos::View<int *, DeviceType> parents(
        Kokkos::ViewAllocateWithoutInitializing( "parents" ), 2 * size() - 1 );
    Details::TreeConstruction<DeviceType>::generateHierarchy(
        morton_indices, getLeafNodes(), getInternalNodes(), parents );

    // calculate bounding volume for each internal node by walking the
    // hierarchy toward the root
    Details::TreeConstruction<
        DeviceType>::calculateInternalNodesBoundingVolumes( getLeafNodes(),
                                                            getInternalNodes(),
                                                            parents );
}

} // namespace DataTransferKit

#endif
