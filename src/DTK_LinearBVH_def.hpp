/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/

#ifndef DTK_LINEARBVH_DEF_HPP
#define DTK_LINEARBVH_DEF_HPP

#include "DTK_ConfigDefs.hpp"

#include <DTK_DetailsAlgorithms.hpp>
#include <DTK_DetailsTreeConstruction.hpp>
#include <DTK_KokkosHelpers.hpp>

#include <Kokkos_ArithTraits.hpp>

namespace DataTransferKit
{
template <typename NO>
class SetBoundingBoxesFunctor
{
  public:
    using DeviceType = typename NO::device_type;
    using ExecutionSpace = typename DeviceType::execution_space;

    SetBoundingBoxesFunctor(
        Kokkos::View<Node *, DeviceType> leaf_nodes,
        Kokkos::View<int *, DeviceType> indices,
        Kokkos::View<Box const *, DeviceType> bounding_boxes )
        : _leaf_nodes( leaf_nodes )
        , _indices( indices )
        , _bounding_boxes( bounding_boxes )
    {
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( int const i ) const
    {
        _leaf_nodes[i].bounding_box = _bounding_boxes[_indices[i]];
    }

  private:
    Kokkos::View<Node *, DeviceType> _leaf_nodes;
    Kokkos::View<int *, DeviceType> _indices;
    Kokkos::View<Box const *, DeviceType> _bounding_boxes;
};

template <typename NO>
BVH<NO>::BVH( Kokkos::View<Box const *, DeviceType> bounding_boxes )
    : _leaf_nodes( "leaf_nodes", bounding_boxes.extent( 0 ) )
    , _internal_nodes( "internal_nodes", bounding_boxes.extent( 0 ) - 1 )
    , _indices( "sorted_indices", bounding_boxes.extent( 0 ) )
{
    using ExecutionSpace = typename DeviceType::execution_space;

    // determine the bounding box of the scene
    Details::TreeConstruction<NO>::calculateBoundingBoxOfTheScene(
        bounding_boxes, _internal_nodes[0].bounding_box );

    // calculate morton code of all objects
    int const n = bounding_boxes.extent( 0 );
    Kokkos::View<unsigned int *, DeviceType> morton_indices( "morton", n );
    Details::TreeConstruction<NO>::assignMortonCodes(
        bounding_boxes, morton_indices, _internal_nodes[0].bounding_box );

    // sort them along the Z-order space-filling curve
    Iota<NO> iota_functor( _indices );
    Kokkos::parallel_for( "set_indices",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
                          iota_functor );
    Kokkos::fence();
    Details::TreeConstruction<NO>::sortObjects( morton_indices, _indices );

    // generate bounding volume hierarchy
    SetBoundingBoxesFunctor<NO> set_bounding_boxes_functor(
        _leaf_nodes, _indices, bounding_boxes );
    Kokkos::parallel_for( "set_bounding_boxes",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
                          set_bounding_boxes_functor );
    Kokkos::fence();
    Details::TreeConstruction<NO>::generateHierarchy(
        morton_indices, _leaf_nodes, _internal_nodes );

    // calculate bounding box for each internal node by walking the hierarchy
    // toward the root
    Details::TreeConstruction<NO>::calculateBoundingBoxes( _leaf_nodes,
                                                           _internal_nodes );
}

// template <typename SC, typename LO, typename GO, typename NO>
// template <typename Predicate>
// int BVH<SC,LO,GO,NO>::query( Predicate const &predicates,
//                Kokkos::View<int *, BVH::DeviceType> out ) const
//{
//    using Tag = typename Predicate::Tag;
//    return Details::query_dispatch( this, predicates, out, Tag{} );
//}
//
// template int BVH::query( Details::Nearest<Details::Point> const &,
//                         Kokkos::View<int *, BVH::DeviceType> ) const;
//
// template int BVH::query( Details::Within<Details::Point> const &,
//                         Kokkos::View<int *, BVH::DeviceType> ) const;
//
} // end namespace DataTransferKit

// Explicit instantiation macro
#define DTK_LINEARBVH_INSTANT( NODE ) template class BVH<NODE>;

#endif
