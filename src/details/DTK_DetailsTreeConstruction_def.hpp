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

#ifndef DTK_DETAILS_TREE_CONSTRUCTION_DEF_HPP
#define DTK_DETAILS_TREE_CONSTRUCTION_DEF_HPP

#include "DTK_ConfigDefs.hpp"

#include <DTK_DetailsAlgorithms.hpp>
#include <DTK_KokkosHelpers.hpp> // sgn, min, max

#include <Kokkos_Atomic.hpp>
#include <Kokkos_Sort.hpp>

#include <cassert>

namespace DataTransferKit
{
namespace Details
{

template <typename DeviceType>
class CalculateBoundingBoxOfTheSceneFunctor
{
  public:
    CalculateBoundingBoxOfTheSceneFunctor(
        Kokkos::View<Box const *, DeviceType> bounding_boxes )
        : _bounding_boxes( bounding_boxes )
    {
    }

    KOKKOS_INLINE_FUNCTION
    void init( Box &box ) const { box = Box(); }

    KOKKOS_INLINE_FUNCTION
    void operator()( int const i, Box &box ) const
    {
        expand( box, _bounding_boxes( i ) );
    }

    KOKKOS_INLINE_FUNCTION
    void join( volatile Box &dst, volatile Box const &src ) const
    {
        expand( dst, src );
    }

  private:
    Kokkos::View<Box const *, DeviceType> _bounding_boxes;
};

template <typename DeviceType>
class AssignMortonCodesFunctor
{
  public:
    AssignMortonCodesFunctor(
        Kokkos::View<Box const *, DeviceType> bounding_boxes,
        Kokkos::View<unsigned int *, DeviceType> morton_codes,
        Box const &scene_bounding_box )
        : _bounding_boxes( bounding_boxes )
        , _morton_codes( morton_codes )
        , _scene_bounding_box( scene_bounding_box )
    {
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( int const i ) const
    {
        Point xyz;
        double a, b;
        centroid( _bounding_boxes[i], xyz );
        // scale coordinates with respect to bounding box of the scene
        for ( int d = 0; d < 3; ++d )
        {
            a = _scene_bounding_box.minCorner()[d];
            b = _scene_bounding_box.maxCorner()[d];
            xyz[d] = ( a != b ? ( xyz[d] - a ) / ( b - a ) : 0 );
        }
        _morton_codes[i] =
            TreeConstruction<DeviceType>::morton3D( xyz[0], xyz[1], xyz[2] );
    }

  private:
    Kokkos::View<Box const *, DeviceType> _bounding_boxes;
    Kokkos::View<unsigned int *, DeviceType> _morton_codes;
    Box const &_scene_bounding_box;
};

template <typename DeviceType>
class GenerateHierarchyFunctor
{
  public:
    GenerateHierarchyFunctor(
        Kokkos::View<unsigned int *, DeviceType> sorted_morton_codes,
        Kokkos::View<Node *, DeviceType> leaf_nodes,
        Kokkos::View<Node *, DeviceType> internal_nodes )
        : _sorted_morton_codes( sorted_morton_codes )
        , _leaf_nodes( leaf_nodes )
        , _internal_nodes( internal_nodes )
    {
    }

    // from "Thinking Parallel, Part III: Tree Construction on the GPU" by
    // Karras
    KOKKOS_INLINE_FUNCTION
    void operator()( int const i ) const
    {
        // Construct internal nodes.
        // Find out which range of objects the node corresponds to.
        // (This is where the magic happens!)

        auto range = TreeConstruction<DeviceType>::determineRange(
            _sorted_morton_codes, i );
        int first = range.first;
        int last = range.second;

        // Determine where to split the range.

        int split = TreeConstruction<DeviceType>::findSplit(
            _sorted_morton_codes, first, last );

        // Select childA.

        Node *childA;
        if ( split == first )
            childA = &_leaf_nodes[split];
        else
            childA = &_internal_nodes[split];

        // Select childB.

        Node *childB;
        if ( split + 1 == last )
            childB = &_leaf_nodes[split + 1];
        else
            childB = &_internal_nodes[split + 1];

        // Record parent-child relationships.

        _internal_nodes[i].children.first = childA;
        _internal_nodes[i].children.second = childB;
        childA->parent = &_internal_nodes[i];
        childB->parent = &_internal_nodes[i];
    }

  private:
    Kokkos::View<unsigned int *, DeviceType> _sorted_morton_codes;
    Kokkos::View<Node *, DeviceType> _leaf_nodes;
    Kokkos::View<Node *, DeviceType> _internal_nodes;
};

template <typename DeviceType>
class CalculateBoundingBoxesFunctor
{
  public:
    CalculateBoundingBoxesFunctor( Kokkos::View<Node *, DeviceType> leaf_nodes,
                                   Node *root )
        : _leaf_nodes( leaf_nodes )
        , _root( root )
        , _flags( "flags", leaf_nodes.extent( 0 ) - 1 )
    {
        // Initialize flags to zero
        Kokkos::deep_copy( _flags, 0 );
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( int const i ) const
    {
        Node *node = _leaf_nodes( i ).parent;
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
            for ( Node *child : {node->children.first, node->children.second} )
                expand( node->bounding_box, child->bounding_box );
            node = node->parent;
        }
        // NOTE: could check that bounding box of the root node is indeed the
        // union of the two children.
    }

  private:
    Kokkos::View<Node *, DeviceType> _leaf_nodes;
    Node *_root;
    // Use int instead of bool because CAS (Compare And Swap) on CUDA does not
    // support boolean
    Kokkos::View<int *, DeviceType> _flags;
};

template <typename DeviceType>
void TreeConstruction<DeviceType>::calculateBoundingBoxOfTheScene(
    Kokkos::View<Box const *, DeviceType> bounding_boxes,
    Box &scene_bounding_box )
{
    auto const n = bounding_boxes.extent( 0 );
    Kokkos::parallel_reduce(
        DTK_MARK_REGION( "calculate_bounding_box_of_the_scene" ),
        Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
        CalculateBoundingBoxOfTheSceneFunctor<DeviceType>( bounding_boxes ),
        scene_bounding_box );
    Kokkos::fence();
}

template <typename DeviceType>
void TreeConstruction<DeviceType>::assignMortonCodes(
    Kokkos::View<Box const *, DeviceType> bounding_boxes,
    Kokkos::View<unsigned int *, DeviceType> morton_codes,
    Box const &scene_bounding_box )
{
    auto const n = morton_codes.extent( 0 );
    Kokkos::parallel_for(
        DTK_MARK_REGION( "assign_morton_codes" ),
        Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
        AssignMortonCodesFunctor<DeviceType>( bounding_boxes, morton_codes,
                                              scene_bounding_box ) );
    Kokkos::fence();
}

template <typename DeviceType>
void TreeConstruction<DeviceType>::sortObjects(
    Kokkos::View<unsigned int *, DeviceType> morton_codes,
    Kokkos::View<int *, DeviceType> object_ids )
{
    int const n = morton_codes.extent( 0 );

    typedef Kokkos::BinOp1D<Kokkos::View<unsigned int *, DeviceType>> CompType;

    Kokkos::Experimental::MinMaxScalar<unsigned int> result;
    Kokkos::Experimental::MinMax<unsigned int> reducer( result );
    parallel_reduce(
        DTK_MARK_REGION( "find_min_max_morton_codes" ),
        Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
        Kokkos::Impl::min_max_functor<Kokkos::View<unsigned int *, DeviceType>>(
            morton_codes ),
        reducer );
    if ( result.min_val == result.max_val )
        return;
    Kokkos::BinSort<Kokkos::View<unsigned int *, DeviceType>, CompType>
        bin_sort( morton_codes,
                  CompType( n / 2, result.min_val, result.max_val ), true );
    bin_sort.create_permute_vector();
    bin_sort.sort( morton_codes );
    // TODO: We might be able to just use `bin_sort.get_permute_vector()`
    // instead of initializing the indices with iota() and sorting the vector
    bin_sort.sort( object_ids );
    Kokkos::fence();
}

template <typename DeviceType>
Node *TreeConstruction<DeviceType>::generateHierarchy(
    Kokkos::View<unsigned int *, DeviceType> sorted_morton_codes,
    Kokkos::View<Node *, DeviceType> leaf_nodes,
    Kokkos::View<Node *, DeviceType> internal_nodes )
{
    auto const n = sorted_morton_codes.extent( 0 );
    Kokkos::parallel_for(
        DTK_MARK_REGION( "generate_hierarchy" ),
        Kokkos::RangePolicy<ExecutionSpace>( 0, n - 1 ),
        GenerateHierarchyFunctor<DeviceType>( sorted_morton_codes, leaf_nodes,
                                              internal_nodes ) );
    Kokkos::fence();
    // returns a pointer to the root node of the tree
    return internal_nodes.data();
}

template <typename DeviceType>
void TreeConstruction<DeviceType>::calculateBoundingBoxes(
    Kokkos::View<Node *, DeviceType> leaf_nodes,
    Kokkos::View<Node *, DeviceType> internal_nodes )
{
    auto const n = leaf_nodes.extent( 0 );
    Node *root = internal_nodes.data();
    Kokkos::parallel_for(
        DTK_MARK_REGION( "calculate_bounding_boxes" ),
        Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
        CalculateBoundingBoxesFunctor<DeviceType>( leaf_nodes, root ) );
    Kokkos::fence();
}

template <typename DeviceType>
int TreeConstruction<DeviceType>::findSplit(
    Kokkos::View<unsigned int *, DeviceType> sorted_morton_codes, int first,
    int last )
{
    // Calculate the number of highest bits that are the same
    // for all objects, using the count-leading-zeros intrinsic.

    int common_prefix = commonPrefix( sorted_morton_codes, first, last );

    // Use binary search to find where the next bit differs.
    // Specifically, we are looking for the highest object that
    // shares more than commonPrefix bits with the first one.

    int split = first; // initial guess
    int step = last - first;

    do
    {
        step = ( step + 1 ) >> 1;     // exponential decrease
        int new_split = split + step; // proposed new position

        if ( new_split < last )
        {
            if ( commonPrefix( sorted_morton_codes, first, new_split ) >
                 common_prefix )
                split = new_split; // accept proposal
        }
    } while ( step > 1 );

    return split;
}

template <typename DeviceType>
Kokkos::pair<int, int> TreeConstruction<DeviceType>::determineRange(
    Kokkos::View<unsigned int *, DeviceType> sorted_morton_codes, int i )
{
    using KokkosHelpers::max;
    using KokkosHelpers::min;
    using KokkosHelpers::sgn;

    // determine direction of the range (+1 or -1)
    int direction = sgn( commonPrefix( sorted_morton_codes, i, i + 1 ) -
                         commonPrefix( sorted_morton_codes, i, i - 1 ) );
    assert( direction == +1 || direction == -1 );

    // compute upper bound for the length of the range
    int max_step = 2;
    int common_prefix = commonPrefix( sorted_morton_codes, i, i - direction );
    while ( commonPrefix( sorted_morton_codes, i, i + direction * max_step ) >
            common_prefix )
    {
        max_step = max_step << 1;
    }

    // find the other end using binary search
    int split = 0;
    int step = max_step;
    do
    {
        step = step >> 1;
        if ( commonPrefix( sorted_morton_codes, i,
                           i + ( split + step ) * direction ) > common_prefix )
            split += step;
    } while ( step > 1 );
    int j = i + split * direction;

    return {min( i, j ), max( i, j )};
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
