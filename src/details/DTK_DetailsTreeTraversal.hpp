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
#ifndef DTK_DETAILS_TREE_TRAVERSAL_HPP
#define DTK_DETAILS_TREE_TRAVERSAL_HPP

#include <DTK_DBC.hpp>

#include <DTK_DetailsAlgorithms.hpp>
#include <DTK_DetailsNode.hpp>
#include <DTK_DetailsPriorityQueue.hpp>
#include <DTK_DetailsStack.hpp>
#include <DTK_Predicates.hpp>

namespace DataTransferKit
{

template <typename DeviceType>
class BoundingVolumeHierarchy;

namespace Details
{
template <typename DeviceType>
struct TreeTraversal
{
  public:
    using ExecutionSpace = typename DeviceType::execution_space;

    template <typename Predicate, typename Insert>
    KOKKOS_INLINE_FUNCTION static int
    query( BoundingVolumeHierarchy<DeviceType> const &bvh,
           Predicate const &pred, Insert const &insert )
    {
        using Tag = typename Predicate::Tag;
        return queryDispatch( bvh, pred, insert, Tag{} );
    }

    /**
     * Return true if the node is a leaf.
     */
    KOKKOS_INLINE_FUNCTION
    static bool isLeaf( Node const *node )
    {
        return ( node->children.first == nullptr ) &&
               ( node->children.second == nullptr );
    }

    /**
     * Return the index of the leaf node.
     */
    KOKKOS_INLINE_FUNCTION
    static int getIndex( BoundingVolumeHierarchy<DeviceType> const &bvh,
                         Node const *leaf )
    {
        return bvh._indices[leaf - bvh._leaf_nodes.data()];
    }

    /**
     * Return the root node of the BVH.
     */
    KOKKOS_INLINE_FUNCTION
    static Node const *getRoot( BoundingVolumeHierarchy<DeviceType> const &bvh )
    {
        if ( bvh.empty() )
            return nullptr;
        return ( bvh.size() > 1 ? bvh._internal_nodes : bvh._leaf_nodes )
            .data();
    }
};

// There are two (related) families of search: one using a spatial predicate and
// one using nearest neighbours query (see boost::geometry::queries
// documentation).
template <typename DeviceType, typename Predicate, typename Insert>
KOKKOS_FUNCTION int
spatialQuery( BoundingVolumeHierarchy<DeviceType> const &bvh,
              Predicate const &predicate, Insert const &insert )
{
    if ( bvh.empty() )
        return 0;

    if ( bvh.size() == 1 )
    {
        Node const *leaf = TreeTraversal<DeviceType>::getRoot( bvh );
        if ( predicate( leaf ) )
        {
            int const leaf_index =
                TreeTraversal<DeviceType>::getIndex( bvh, leaf );
            insert( leaf_index );
            return 1;
        }
        else
            return 0;
    }

    Stack<Node const *> stack;

    Node const *root = TreeTraversal<DeviceType>::getRoot( bvh );
    stack.push( root );
    int count = 0;

    while ( !stack.empty() )
    {
        Node const *node = stack.top();
        stack.pop();

        if ( TreeTraversal<DeviceType>::isLeaf( node ) )
        {
            insert( TreeTraversal<DeviceType>::getIndex( bvh, node ) );
            count++;
        }
        else
        {
            for ( Node const *child :
                  {node->children.first, node->children.second} )
            {
                if ( predicate( child ) )
                {
                    stack.push( child );
                }
            }
        }
    }
    return count;
}

// query k nearest neighbours
template <typename DeviceType, typename Distance, typename Insert>
KOKKOS_FUNCTION int
nearestQuery( BoundingVolumeHierarchy<DeviceType> const &bvh,
              Distance const &distance, int k, Insert const &insert )
{
    if ( bvh.empty() || k < 1 )
        return 0;

    if ( bvh.size() == 1 )
    {
        Node const *leaf = TreeTraversal<DeviceType>::getRoot( bvh );
        int const leaf_index = TreeTraversal<DeviceType>::getIndex( bvh, leaf );
        double const leaf_distance = distance( leaf );
        insert( leaf_index, leaf_distance );
        return 1;
    }

    using PairNodePtrDistance = Kokkos::pair<Node const *, double>;

    struct CompareDistance
    {
        KOKKOS_INLINE_FUNCTION bool
        operator()( PairNodePtrDistance const &lhs,
                    PairNodePtrDistance const &rhs ) const
        {
            // reverse order (larger distance means lower priority)
            return lhs.second > rhs.second;
        }
    };

    PriorityQueue<PairNodePtrDistance, CompareDistance> queue;
    // priority does not matter for the root since the node will be
    // processed directly and removed from the priority queue we don't even
    // bother computing the distance to it.
    Node const *root = TreeTraversal<DeviceType>::getRoot( bvh );
    queue.push( root, 0. );
    int count = 0;

    while ( !queue.empty() && count < k )
    {
        // get the node that is on top of the priority list (i.e. is the
        // closest to the query point)
        Node const *node = queue.top().first;
        double const node_distance = queue.top().second;
        // NOTE: it would be nice to be able to do something like
        // tie( node, node_distance = queue.top();

        // NOTE: not calling queue.pop() here so that it can be combined with
        // the next push in case the node is internal (thus sparing a bubble-up
        // operation)
        if ( TreeTraversal<DeviceType>::isLeaf( node ) )
        {
            queue.pop();
            insert( TreeTraversal<DeviceType>::getIndex( bvh, node ),
                    node_distance );
            count++;
        }
        else
        {
            // insert children of the node in the priority list

            auto const left_child = node->children.first;
            auto const right_child = node->children.second;
            queue.pop_push( left_child, distance( left_child ) );
            queue.push( right_child, distance( right_child ) );
        }
    }
    return count;
}

template <typename DeviceType, typename Predicate, typename Insert>
KOKKOS_INLINE_FUNCTION int
queryDispatch( BoundingVolumeHierarchy<DeviceType> const &bvh,
               Predicate const &pred, Insert const &insert,
               SpatialPredicateTag )
{
    return spatialQuery( bvh, pred, insert );
}

template <typename DeviceType, typename Predicate, typename Insert>
KOKKOS_INLINE_FUNCTION int
queryDispatch( BoundingVolumeHierarchy<DeviceType> const &bvh,
               Predicate const &pred, Insert const &insert,
               NearestPredicateTag )
{
    auto const geometry = pred._geometry;
    auto const k = pred._k;
    return nearestQuery( bvh,
                         [geometry]( Node const *node ) {
                             return distance( geometry, node->bounding_box );
                         },
                         k, insert );
}

} // namespace Details
} // namespace DataTransferKit

#endif
