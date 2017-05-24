/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/

#ifndef DTK_DETAILSTREECONSTRUCTION_DEF_HPP
#define DTK_DETAILSTREECONSTRUCTION_DEF_HPP

#include "DTK_ConfigDefs.hpp"

#include <DTK_DetailsAlgorithms.hpp>
#include <DTK_KokkosHelpers.hpp>

#include <Kokkos_Atomic.hpp>
#include <Kokkos_Sort.hpp>

#include <cassert>

namespace DataTransferKit
{
namespace Details
{

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
            a = _scene_bounding_box[2 * d];
            b = _scene_bounding_box[2 * d + 1];
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
                                   Node *root,
                                   Kokkos::View<int *, DeviceType> ready_flags )
        : _leaf_nodes( leaf_nodes )
        , _root( root )
        , _ready_flags( ready_flags )
    {
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( int const i ) const
    {
        Node *node = _leaf_nodes[i].parent;
        while ( node != _root )
        {
            if ( Kokkos::atomic_compare_exchange_strong(
                     &_ready_flags[node - _root], 0, 1 ) )
                break;
            for ( Node *child : {node->children.first, node->children.second} )
                expand( node->bounding_box, child->bounding_box );
            node = node->parent;
        }
        // NOTE: could stop at node != root and then just check that what we
        // computed earlier (bounding box of the scene) is indeed the union of
        // the two children.
    }

  private:
    Kokkos::View<Node *, DeviceType> _leaf_nodes;
    Node *_root;
    Kokkos::View<int *, DeviceType> _ready_flags;
};

template <typename DeviceType>
void TreeConstruction<DeviceType>::calculateBoundingBoxOfTheScene(
    Kokkos::View<Box const *, DeviceType> bounding_boxes,
    Box &scene_bounding_box )
{
    int const n = bounding_boxes.extent( 0 );
    ExpandBoxWithBoxFunctor<DeviceType> functor( bounding_boxes );
    Kokkos::parallel_reduce( "calculate_bouding_of_the_scene",
                             Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
                             functor, scene_bounding_box );
    Kokkos::fence();
}

template <typename DeviceType>
void TreeConstruction<DeviceType>::assignMortonCodes(
    Kokkos::View<Box const *, DeviceType> bounding_boxes,
    Kokkos::View<unsigned int *, DeviceType> morton_codes,
    Box const &scene_bounding_box )
{
    int const n = morton_codes.extent( 0 );
    AssignMortonCodesFunctor<DeviceType> functor( bounding_boxes, morton_codes,
                                                  scene_bounding_box );
    Kokkos::parallel_for( "assign_morton_codes",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
                          functor );
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
    // instead of initializing the indices with Iota and sorting the vector
    bin_sort.sort( object_ids );
}

template <typename DeviceType>
Node *TreeConstruction<DeviceType>::generateHierarchy(
    Kokkos::View<unsigned int *, DeviceType> sorted_morton_codes,
    Kokkos::View<Node *, DeviceType> leaf_nodes,
    Kokkos::View<Node *, DeviceType> internal_nodes )
{
    GenerateHierarchyFunctor<DeviceType> functor( sorted_morton_codes,
                                                  leaf_nodes, internal_nodes );

    int const n = sorted_morton_codes.extent( 0 );
    Kokkos::parallel_for( "generate_hierarchy",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n - 1 ),
                          functor );
    Kokkos::fence();

    // Node 0 is the root.
    return &( internal_nodes.data()[0] );
}

template <typename DeviceType>
void TreeConstruction<DeviceType>::calculateBoundingBoxes(
    Kokkos::View<Node *, DeviceType> leaf_nodes,
    Kokkos::View<Node *, DeviceType> internal_nodes )
{
    int const n = leaf_nodes.extent( 0 );

    // Use int instead of bool because CAS on CUDA does not support boolean
    Kokkos::View<int *, DeviceType> ready_flags( "ready_flags", n - 1 );
    Kokkos::parallel_for( "fill_ready_flags",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n - 1 ),
                          KOKKOS_LAMBDA( int i ) { ready_flags[i] = 0; } );
    Kokkos::fence();

    Node *root = &internal_nodes[0];

    CalculateBoundingBoxesFunctor<DeviceType> calc_functor( leaf_nodes, root,
                                                            ready_flags );
    Kokkos::parallel_for( "calculate_bounding_boxes",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
                          calc_functor );
    Kokkos::fence();
}

template <typename DeviceType>
int TreeConstruction<DeviceType>::findSplit(
    Kokkos::View<unsigned int *, DeviceType> sorted_morton_codes, int first,
    int last )
{
    // Identical Morton codes => split the range in the middle.

    unsigned int first_code = sorted_morton_codes[first];
    unsigned int last_code = sorted_morton_codes[last];

    if ( first_code == last_code )
        return ( first + last ) >> 1;

    // Calculate the number of highest bits that are the same
    // for all objects, using the count-leading-zeros intrinsic.

    int common_prefix = KokkosHelpers::clz( first_code ^ last_code );

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
            unsigned int split_code = sorted_morton_codes[new_split];
            int split_prefix = KokkosHelpers::clz( first_code ^ split_code );
            if ( split_prefix > common_prefix )
                split = new_split; // accept proposal
        }
    } while ( step > 1 );

    return split;
}

template <typename DeviceType>
Kokkos::pair<int, int> TreeConstruction<DeviceType>::determineRange(
    Kokkos::View<unsigned int *, DeviceType> sorted_morton_codes, int i )
{
    // determine direction of the range (+1 or -1)
    int direction =
        KokkosHelpers::sgn( commonPrefix( sorted_morton_codes, i, i + 1 ) -
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

    return {KokkosHelpers::min( i, j ), KokkosHelpers::max( i, j )};
}
}
}

// Explicit instantiation macro
#define DTK_TREECONSTRUCTION_INSTANT( NODE )                                   \
    namespace Details                                                          \
    {                                                                          \
    template struct TreeConstruction<typename NODE::device_type>;              \
    }

#endif
